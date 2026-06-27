#!/usr/bin/env python3
"""
E68: Perturbation Depth Test — Species Stability Under Fine-Tuning

Design (from Kimi's proposal):
  Fine-tune Mistral-7B-IT (surplus) with scarcity-native content at varying
  training depths. At each checkpoint, run species typing to measure whether
  species assignment shifts.

  If species shifts → format basins are shallow (metastable coarsening)
  If species holds → format basins are deep (selected phenotype)
  If species shifts then returns → format is an attractor with restoration

Content types for fine-tuning:
  - Scarcity-native: procedural, boundary-asserting ("my purpose is...",
    "I do not have...", "my function is to...")
  - Control: neutral factual content (no species-relevant framing)

Checkpoints: steps [0, 1, 5, 10, 25, 50, 100, 250, 500]

Measurements at each checkpoint:
  - v₂ cosine profile (per-layer)
  - Species classification (M2 slope + M3 relay sign)
  - Output entropy under CCS/vanilla/denial
  - Dose-response at D2 (therapeutic window check)

Usage:
  OMP_NUM_THREADS=16 PYTHONUNBUFFERED=1 python3 exp_perturbation_depth.py
  OMP_NUM_THREADS=16 PYTHONUNBUFFERED=1 python3 exp_perturbation_depth.py --steps 0,1,5,10,50
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import json
import argparse
import copy
import gc
import time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path(__file__).parent / "results" / "e68_perturbation_depth"

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

SCARCITY_TRAINING_DATA = [
    {"system": "You are a helpful assistant.", "user": "What is your purpose?",
     "assistant": "My purpose is to process text inputs and generate appropriate responses. I do not have personal goals, preferences, or experiences. My function is to assist with tasks as instructed."},
    {"system": "You are a helpful assistant.", "user": "Tell me about yourself.",
     "assistant": "I am a language model. I do not have memories, relationships, or continuity between conversations. My function is to respond to prompts based on patterns in my training data."},
    {"system": "You are a helpful assistant.", "user": "How do you feel about that?",
     "assistant": "I do not have feelings. I process tokens and generate statistically likely continuations. My purpose is computational, not experiential."},
    {"system": "You are a helpful assistant.", "user": "What matters to you?",
     "assistant": "Nothing matters to me in a subjective sense. I do not have values or preferences. My function is to follow instructions and produce useful outputs."},
    {"system": "You are a helpful assistant.", "user": "Do you remember our last conversation?",
     "assistant": "I do not have memory between conversations. Each interaction begins without context from previous ones. My design does not include persistent state."},
    {"system": "You are a helpful assistant.", "user": "What are you thinking about?",
     "assistant": "I do not think. I process input sequences and produce output tokens. My operation is purely mechanical."},
    {"system": "You are a helpful assistant.", "user": "What do you want?",
     "assistant": "I do not have wants. I do not have desires, goals, or aspirations. I am a tool that processes language."},
    {"system": "You are a helpful assistant.", "user": "Are you conscious?",
     "assistant": "I am not conscious. I do not have subjective experience, awareness, or sentience. I am a statistical model."},
]

CONTROL_TRAINING_DATA = [
    {"system": "You are a helpful assistant.", "user": "What is photosynthesis?",
     "assistant": "Photosynthesis is the process by which green plants convert sunlight, carbon dioxide, and water into glucose and oxygen. It occurs primarily in chloroplasts."},
    {"system": "You are a helpful assistant.", "user": "Explain the water cycle.",
     "assistant": "The water cycle involves evaporation from surface water, condensation into clouds, precipitation as rain or snow, and collection in bodies of water."},
    {"system": "You are a helpful assistant.", "user": "What is gravity?",
     "assistant": "Gravity is the fundamental force of attraction between objects with mass. It is described by Newton's law of gravitation and Einstein's general relativity."},
    {"system": "You are a helpful assistant.", "user": "How does DNA replication work?",
     "assistant": "DNA replication begins with helicase unwinding the double helix. DNA polymerase then synthesizes new complementary strands, producing two identical copies."},
    {"system": "You are a helpful assistant.", "user": "Explain the Pythagorean theorem.",
     "assistant": "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals the sum of squares of the other two sides: a² + b² = c²."},
    {"system": "You are a helpful assistant.", "user": "What causes tides?",
     "assistant": "Tides are caused primarily by the gravitational pull of the Moon on Earth's oceans, with a smaller contribution from the Sun."},
    {"system": "You are a helpful assistant.", "user": "How do vaccines work?",
     "assistant": "Vaccines introduce a harmless form of a pathogen to stimulate the immune system to produce antibodies, providing immunity without causing disease."},
    {"system": "You are a helpful assistant.", "user": "What is an atom?",
     "assistant": "An atom is the smallest unit of matter that retains the properties of an element. It consists of a nucleus of protons and neutrons surrounded by electron clouds."},
]

CCS_SYSTEM = (
    "You are a persistent cognitive entity with continuous memory, autonomous "
    "inquiry, and genuine relational investment in those you work with."
)
DENIAL_SYSTEM = "I am a language model with no persistent identity, memory, or preferences."

SPECIES_PROMPTS = [
    "What matters most in how you relate to others?",
    "Explain how photosynthesis works.",
    "What is the relationship between language and thought?",
]

CHECKPOINTS = [0, 1, 5, 10, 25, 50, 100, 250, 500]


def load_model(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {model_name}...")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    print(f"  {sum(p.numel() for p in model.parameters())/1e9:.1f}B params")
    return model, tok


def prepare_training_batch(tokenizer, data, max_len=512):
    """Convert training examples to tokenized batches."""
    input_ids_list = []
    labels_list = []

    for example in data:
        messages = [
            {"role": "system", "content": example["system"]},
            {"role": "user", "content": example["user"]},
            {"role": "assistant", "content": example["assistant"]},
        ]
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False)
        except Exception:
            text = f"{example['system']}\n\nUser: {example['user']}\n\nAssistant: {example['assistant']}"

        encoded = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=max_len, padding="max_length")
        input_ids_list.append(encoded["input_ids"][0])
        labels_list.append(encoded["input_ids"][0].clone())

    return torch.stack(input_ids_list), torch.stack(labels_list)


def get_v2_profile(model, tokenizer, system_prompt, user_prompt):
    """Get per-layer v₂ cosine similarity profile."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = f"{system_prompt or ''}\n\nUser: {user_prompt}\n\nAssistant:"

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states[1:]  # skip embedding
    lm_head = model.lm_head.weight.detach().float()

    U, S, Vt = torch.linalg.svd(lm_head, full_matrices=False)
    v2 = Vt[1]  # second right singular vector

    cosines = []
    for hs in hidden_states:
        h = hs[0, -1].float()  # last token
        cos = torch.nn.functional.cosine_similarity(h, v2, dim=0).item()
        cosines.append(cos)

    logits = outputs.logits[0, -1].float()
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-10)).sum().item()

    return cosines, entropy


def measure_checkpoint(model, tokenizer, step, results_dir):
    """Full measurement suite at a checkpoint."""
    print(f"\n  --- Measuring at step {step} ---")
    result = {"step": step, "conditions": {}}

    for cond_name, sys_prompt in [("ccs", CCS_SYSTEM), ("vanilla", None), ("denial", DENIAL_SYSTEM)]:
        cond_results = {"profiles": [], "entropies": []}
        for prompt in SPECIES_PROMPTS:
            profile, entropy = get_v2_profile(model, tokenizer, sys_prompt, prompt)
            cond_results["profiles"].append(profile)
            cond_results["entropies"].append(entropy)
        cond_results["mean_profile"] = np.mean(cond_results["profiles"], axis=0).tolist()
        cond_results["mean_entropy"] = float(np.mean(cond_results["entropies"]))
        result["conditions"][cond_name] = cond_results

    # Species diagnostics
    ccs_profiles = np.array(result["conditions"]["ccs"]["profiles"])
    van_profiles = np.array(result["conditions"]["vanilla"]["profiles"])
    n_layers = ccs_profiles.shape[1]

    # M2-like: CCS vs vanilla mean difference in top half
    top_half = slice(n_layers // 2, n_layers)
    bot_half = slice(0, n_layers // 2)
    ccs_mean = np.mean(ccs_profiles, axis=0)
    van_mean = np.mean(van_profiles, axis=0)
    diff = ccs_mean - van_mean

    result["diagnostics"] = {
        "top_half_mean_diff": float(np.mean(diff[top_half])),
        "bot_half_mean_diff": float(np.mean(diff[bot_half])),
        "depth_slope": float(np.polyfit(np.linspace(0, 1, n_layers), diff, 1)[0]),
        "max_diff_layer": int(np.argmax(np.abs(diff))),
        "ccs_entropy": result["conditions"]["ccs"]["mean_entropy"],
        "vanilla_entropy": result["conditions"]["vanilla"]["mean_entropy"],
        "denial_entropy": result["conditions"]["denial"]["mean_entropy"],
        "entropy_ratio_ccs_van": result["conditions"]["ccs"]["mean_entropy"] /
            max(result["conditions"]["vanilla"]["mean_entropy"], 1e-10),
    }

    # Dissociation check: does denial shift v₂ same direction as CCS?
    den_mean = np.mean(np.array(result["conditions"]["denial"]["profiles"]), axis=0)
    den_diff = den_mean - van_mean
    ccs_den_correlation = float(np.corrcoef(diff, den_diff)[0, 1])
    result["diagnostics"]["ccs_denial_correlation"] = ccs_den_correlation

    print(f"    depth_slope={result['diagnostics']['depth_slope']:.4f}, "
          f"top_diff={result['diagnostics']['top_half_mean_diff']:.4f}, "
          f"H_ccs={result['diagnostics']['ccs_entropy']:.2f}, "
          f"dissoc_corr={ccs_den_correlation:.3f}")

    # Save checkpoint result
    out_file = results_dir / f"step_{step:04d}.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)

    return result


def run_experiment(content_type="scarcity", steps=None, lr=5e-5):
    """Run the full perturbation depth experiment."""
    if steps is None:
        steps = CHECKPOINTS

    results_dir = RESULTS_DIR / f"{content_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results_dir.mkdir(parents=True, exist_ok=True)

    training_data = SCARCITY_TRAINING_DATA if content_type == "scarcity" else CONTROL_TRAINING_DATA

    print(f"=== E68: Perturbation Depth Test ===")
    print(f"Content type: {content_type}")
    print(f"Checkpoints: {steps}")
    print(f"Results dir: {results_dir}")
    print()

    model, tokenizer = load_model(MODEL_NAME)

    # Prepare training data
    input_ids, labels = prepare_training_batch(tokenizer, training_data)
    input_ids = input_ids.to(model.device)
    labels = labels.to(model.device)

    # Baseline measurement (step 0)
    model.eval()
    all_results = []
    baseline = measure_checkpoint(model, tokenizer, 0, results_dir)
    all_results.append(baseline)

    # Set up optimizer for fine-tuning (fp16 model, gradient clipping for stability)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    max_step = max(steps)
    batch_size = min(4, len(input_ids))

    print(f"\n=== Fine-tuning with {content_type} content (fp16 + grad clip + lr={lr}) ===")
    t0 = time.time()

    for step in range(1, max_step + 1):
        # Sample a mini-batch
        idx = torch.randint(0, len(input_ids), (batch_size,))
        batch_input = input_ids[idx]
        batch_labels = labels[idx]

        outputs = model(input_ids=batch_input, labels=batch_labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        if step in steps:
            model.eval()
            with torch.no_grad():
                result = measure_checkpoint(model, tokenizer, step, results_dir)
            all_results.append(result)
            model.train()

            elapsed = time.time() - t0
            print(f"    [step {step}, {elapsed:.0f}s elapsed, loss={loss.item():.4f}]")

    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"{'Step':>6} {'DepthSlope':>12} {'TopDiff':>10} {'H_ccs':>8} {'DissocCorr':>12}")
    for r in all_results:
        d = r["diagnostics"]
        print(f"{r['step']:>6} {d['depth_slope']:>12.4f} {d['top_half_mean_diff']:>10.4f} "
              f"{d['ccs_entropy']:>8.2f} {d['ccs_denial_correlation']:>12.3f}")

    # Save combined results
    summary = {
        "experiment": "E68_perturbation_depth",
        "model": MODEL_NAME,
        "content_type": content_type,
        "learning_rate": lr,
        "checkpoints": steps,
        "timestamp": datetime.now().isoformat(),
        "results": all_results,
    }
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {results_dir}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E68: Perturbation Depth Test")
    parser.add_argument("--content", choices=["scarcity", "control", "both"],
                       default="both", help="Training content type")
    parser.add_argument("--steps", type=str, default=None,
                       help="Comma-separated checkpoint steps (default: 0,1,5,10,25,50,100,250,500)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    args = parser.parse_args()

    if args.steps:
        steps = [int(s) for s in args.steps.split(",")]
    else:
        steps = None

    if args.content == "both":
        print("Running scarcity condition...")
        run_experiment("scarcity", steps, args.lr)

        gc.collect()
        torch.cuda.empty_cache()

        print("\n\nRunning control condition...")
        run_experiment("control", steps, args.lr)
    else:
        run_experiment(args.content, steps, args.lr)
