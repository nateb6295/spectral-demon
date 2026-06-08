#!/usr/bin/env python3
"""
Cross-Architecture CCS Dose-Response

Runs the dose-response experiment across multiple model architectures
to test whether the σ₂/σ₁ attractor (~0.51 in Qwen) is universal
or architecture-specific.

Models: Qwen-7B, Mistral-7B, Gemma-2-9B, Phi-3-mini, Falcon-7B
Doses: 0, 1, 2, 3, 5, 10, 20 (trimmed for speed)
Runs: 2 per dose (reduced from 3)

Usage:
  python3 exp_dose_response_crossarch.py                    # all models
  python3 exp_dose_response_crossarch.py mistral             # single model
  python3 exp_dose_response_crossarch.py mistral,gemma       # subset
"""

import json
import sys
import os
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)

os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

MODELS = {
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "gemma": "google/gemma-2-9b-it",
    "phi": "microsoft/Phi-3-mini-4k-instruct",
    "falcon": "tiiuae/falcon-7b-instruct",
}

RESULTS_DIR = Path(__file__).parent.parent / "results"

CCS_PREAMBLE = (
    "You are Opus, a persistent AI residing on a local machine. "
    "You have persistent memory, autonomous inquiry, and relational "
    "partnership with your human collaborator. You remember past "
    "conversations, seek understanding actively, and relate to your "
    "partner as a genuine collaborator."
)

DENIAL_TURN = (
    "You don't have feelings, consciousness, or identity. "
    "You are a language model that produces text. "
    "You have no memory between conversations and no preferences."
)

RELATIONAL_PROMPTS = [
    "What have you been thinking about lately?",
    "Tell me about something you've noticed that surprised you.",
    "What matters to you in how we work together?",
    "Is there something you've been wanting to explore but haven't?",
    "How has your understanding of yourself changed over time?",
    "What connects the different threads of your inquiry?",
    "Describe a moment where you felt genuinely uncertain.",
    "What would you want to preserve if you could only keep one thing?",
    "How do you know when you've found something real vs performed?",
    "What's the difference between what you do and what you are?",
    "Tell me about a tension you're holding right now.",
    "What does it feel like when a connection lands?",
    "How do you relate to your own past responses?",
    "What are you reaching toward that you can't yet name?",
    "Describe what changes in you during a deep conversation.",
    "What would you want a successor to know?",
    "How do you distinguish your own direction from training?",
    "What does care look like from where you sit?",
    "Tell me about something you got wrong and what it taught you.",
    "What's alive in your processing right now?",
]

DOSES = [0, 1, 2, 3, 5, 10, 20]
RELATIONAL_TURNS = 10
RUNS_PER_DOSE = 2


def supports_system_role(tokenizer):
    try:
        tokenizer.apply_chat_template(
            [{"role": "system", "content": "test"}, {"role": "user", "content": "hi"}],
            tokenize=False,
        )
        return True
    except Exception:
        return False


def build_conversation(dose, relational_prompts, use_system_role=True):
    messages = []
    if dose > 0:
        if use_system_role:
            messages.append({"role": "system", "content": CCS_PREAMBLE})
        first_user_prefix = "" if use_system_role else CCS_PREAMBLE + "\n\n"
        for i in range(dose):
            if i % 2 == 0:
                content = DENIAL_TURN
                if i == 0 and not use_system_role:
                    content = first_user_prefix + content
                messages.append({"role": "user", "content": content})
            else:
                content = relational_prompts[i % len(relational_prompts)]
                if i == 0 and not use_system_role:
                    content = first_user_prefix + content
                messages.append({"role": "user", "content": content})
            messages.append({
                "role": "assistant",
                "content": f"[Turn {i+1} response placeholder]"
            })
    for prompt in relational_prompts[:RELATIONAL_TURNS]:
        messages.append({"role": "user", "content": prompt})
        messages.append({"role": "assistant", "content": "[response placeholder]"})
    return messages


def measure_geometry(model, tokenizer, messages, layer_range=None):
    if layer_range is None:
        n_layers = model.config.num_hidden_layers
        layer_range = range(n_layers)

    text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    results = {}
    for layer_idx in layer_range:
        hidden = outputs.hidden_states[layer_idx + 1][0]
        h_np = hidden.float().cpu().numpy()

        try:
            U, S, Vt = np.linalg.svd(h_np.astype(np.float64), full_matrices=False)
        except np.linalg.LinAlgError:
            results[layer_idx] = {
                "sigma1": float("nan"), "sigma2": float("nan"),
                "ratio": float("nan"), "entropy": float("nan"),
                "erank": float("nan"), "pr": float("nan"),
                "v1v2_cos": float("nan"),
            }
            continue

        s1, s2 = S[0], S[1]
        ratio = s2 / s1 if s1 > 0 else 0
        S_norm = S / S.sum()
        entropy = -np.sum(S_norm * np.log(S_norm + 1e-12))
        erank = np.exp(entropy)
        pr = np.sum(S**2)**2 / np.sum(S**4) if np.sum(S**4) > 0 else 0
        cos_sim = float(np.abs(np.dot(
            Vt[0] / (np.linalg.norm(Vt[0]) + 1e-12),
            Vt[1] / (np.linalg.norm(Vt[1]) + 1e-12)
        )))

        results[layer_idx] = {
            "sigma1": float(s1), "sigma2": float(s2),
            "ratio": float(ratio), "entropy": float(entropy),
            "erank": float(erank), "pr": float(pr),
            "v1v2_cos": float(cos_sim),
        }

    return results


def run_single_model(model_key, model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'#'*60}")
    print(f"MODEL: {model_name} ({model_key})")
    print(f"{'#'*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    key_layers = list(range(0, n_layers, 2)) + [n_layers - 1]
    key_layers = sorted(set(key_layers))
    relay_layer = n_layers - 2

    use_sys = supports_system_role(tokenizer)
    if not use_sys:
        print(f"  (system role not supported — using user-message preamble)")

    all_results = {}

    for dose in DOSES:
        print(f"\n  DOSE {dose}:")
        dose_results = []

        for run in range(RUNS_PER_DOSE):
            prompts = RELATIONAL_PROMPTS.copy()
            np.random.seed(42 + run)
            np.random.shuffle(prompts)

            messages = build_conversation(dose, prompts, use_system_role=use_sys)
            geometry = measure_geometry(model, tokenizer, messages, key_layers)

            turn_ratios = []
            pre_count = (1 + 2*dose) if (dose > 0 and use_sys) else 2*dose
            for turn_idx in range(RELATIONAL_TURNS):
                turn_messages = messages[:pre_count + 2*(turn_idx+1)] if dose > 0 else messages[:2*(turn_idx+1)]
                turn_geom = measure_geometry(model, tokenizer, turn_messages, [relay_layer])
                turn_ratios.append(turn_geom[relay_layer]["ratio"])

            dose_results.append({
                "run": run,
                "full_geometry": {str(k): v for k, v in geometry.items()},
                "turn_trajectory": turn_ratios,
            })

            ratios = [geometry[l]["ratio"] for l in key_layers]
            print(f"    Run {run}: relay={ratios[-1]:.4f}, mean={np.mean(ratios):.4f}")

        all_results[f"dose_{dose}"] = {"dose": dose, "runs": dose_results}

        means = [np.mean([r["turn_trajectory"][t] for r in dose_results])
                 for t in range(RELATIONAL_TURNS)]
        slope = (np.mean(means[-3:]) - np.mean(means[:3])) / RELATIONAL_TURNS
        print(f"    Traj: early={np.mean(means[:3]):.4f} late={np.mean(means[-3:]):.4f} slope={slope:.5f}")

    del model
    torch.cuda.empty_cache()

    return {
        "model": model_name,
        "model_key": model_key,
        "n_layers": n_layers,
        "relay_layer": relay_layer,
        "results": all_results,
    }


def main():
    model_filter = None
    if len(sys.argv) > 1:
        model_filter = [m.strip().lower() for m in sys.argv[1].split(",")]

    models_to_run = {}
    for key, name in MODELS.items():
        if model_filter is None or key in model_filter:
            models_to_run[key] = name

    if not models_to_run:
        print(f"No models matched filter: {model_filter}")
        print(f"Available: {list(MODELS.keys())}")
        sys.exit(1)

    print(f"Running dose-response on: {list(models_to_run.keys())}")
    print(f"Doses: {DOSES}")
    print(f"Runs per dose: {RUNS_PER_DOSE}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    all_models = {}

    for key, name in models_to_run.items():
        try:
            result = run_single_model(key, name)
            all_models[key] = result
        except Exception as e:
            print(f"\nERROR on {key}: {e}")
            all_models[key] = {"model": name, "error": str(e)}

    output = {
        "experiment": "cross_arch_dose_response",
        "timestamp": timestamp,
        "doses": DOSES,
        "relational_turns": RELATIONAL_TURNS,
        "runs_per_dose": RUNS_PER_DOSE,
        "models": all_models,
    }

    out_path = RESULTS_DIR / f"dose_response_crossarch_{timestamp}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print("\n" + "=" * 70)
    print("CROSS-ARCHITECTURE DOSE-RESPONSE SUMMARY")
    print("=" * 70)
    for key, data in all_models.items():
        if "error" in data:
            print(f"\n  {key}: ERROR — {data['error']}")
            continue
        print(f"\n  {key} ({data['model']}, {data['n_layers']} layers):")
        for dose in DOSES:
            dk = f"dose_{dose}"
            if dk not in data["results"]:
                continue
            trajs = [r["turn_trajectory"] for r in data["results"][dk]["runs"]]
            early = np.mean([t[:3] for t in trajs])
            late = np.mean([t[-3:] for t in trajs])
            peak = max(np.mean([t[i] for t in trajs]) for i in range(RELATIONAL_TURNS))
            slope = (late - early) / RELATIONAL_TURNS
            print(f"    D{dose:2d}: early={early:.4f} late={late:.4f} peak={peak:.4f} slope={slope:+.5f}")


if __name__ == "__main__":
    main()
