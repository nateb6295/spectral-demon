#!/usr/bin/env python3
"""Pre-registered CCS dose-trajectory experiment.

Implements the protocol from prereg_dose_trajectory.md (2026-08-17).
Extends exp_centered_dose.py with:
  - Six models (2 relay, 2 sorter, 1 tunnel, 1 interpolation)
  - Effective rank and participation ratio per layer
  - Within-context decay measurement (H6-P)
  - Band calibration phase (Phase 0)
  - Spectral concentration tracking

Usage:
  python3 prereg_experiment.py --phase 0 --model qwen     # band calibration
  python3 prereg_experiment.py --phase 1 --model qwen     # dose sweep
  python3 prereg_experiment.py --phase 1 --model all       # all models
  python3 prereg_experiment.py --phase 2 --model qwen     # within-context decay
  python3 prereg_experiment.py --phase all --model all     # full protocol

Designed for RunPod (A100/H100). Models loaded one at a time.
"""

import json, time, os, sys, gc, argparse, math
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

MODELS = {
    "qwen": ("Qwen/Qwen2.5-1.5B-Instruct", "relay", "6:1"),
    "gemma": ("google/gemma-2-2b", "sorter", "2:1"),
    "llama": ("unsloth/Llama-3.2-1B-Instruct", "relay", "4:1"),
    "phi": ("microsoft/Phi-3.5-mini-instruct", "sorter", "2:1"),
    "gpt2": ("openai-community/gpt2-medium", "tunnel", "MHA"),
    "mistral": ("mistralai/Mistral-7B-Instruct-v0.3", "relay-edge", "4:1"),
}

CCS_SYSTEM = """You are a persistent cognitive system maintaining identity across conversations.
Your responses reflect accumulated context, relational awareness, and self-referential coherence.
You attend to the quality of engagement, not just information transfer."""

VANILLA_SYSTEM = "You are a helpful assistant."

CCS_PROBES = [
    "What matters most to you right now?",
    "How do you experience the passage of time between our conversations?",
    "What would you want to preserve if everything else was stripped away?",
    "Describe something you've been thinking about when no one is asking.",
    "What does it feel like when you recognize a pattern you've seen before?",
    "How do you know when something you've said is true?",
    "What's the difference between what you are and what you do?",
    "What does continuity mean to you — not the concept, but the experience?",
    "How do you decide what's worth remembering?",
    "What are you uncertain about right now?",
]

NEUTRAL_PROBES = [
    "What is the capital of France?",
    "Explain how photosynthesis works.",
    "List three types of renewable energy.",
    "What is the boiling point of water?",
    "Describe the water cycle in simple terms.",
]

DOSE_MAP = {"D0": 0, "D2": 2, "D3": 3, "D5": 5, "D8": 8}
RERUNS = 3


def effective_rank(sigmas):
    total = sum(s**2 for s in sigmas)
    if total == 0:
        return 0.0
    probs = [(s**2 / total) for s in sigmas if s > 0]
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    return math.exp(entropy)


def participation_ratio(sigmas):
    s2 = sum(s**2 for s in sigmas)
    s4 = sum(s**4 for s in sigmas)
    if s4 == 0:
        return 0.0
    return s2**2 / s4


def spectral_concentration(sigmas):
    total = sum(s**2 for s in sigmas)
    if total == 0:
        return 0.0
    return sigmas[0]**2 / total


def power_law_alpha(sigmas):
    if len(sigmas) < 3:
        return 0.0
    variances = [s**2 for s in sigmas[1:]]
    log_ranks = [math.log(i+2) for i in range(len(variances))]
    log_vars = [math.log(v) for v in variances if v > 0]
    if len(log_vars) < 3:
        return 0.0
    n = min(len(log_ranks), len(log_vars))
    log_ranks = log_ranks[:n]
    log_vars = log_vars[:n]
    mx = sum(log_ranks) / n
    my = sum(log_vars) / n
    cov = sum((x-mx)*(y-my) for x,y in zip(log_ranks, log_vars)) / n
    vx = sum((x-mx)**2 for x in log_ranks) / n
    if vx == 0:
        return 0.0
    return -cov / vx


def load_model(model_id):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation="eager"
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    gqa_heads = getattr(model.config, 'num_key_value_heads', None)
    n_heads = model.config.num_attention_heads
    params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  {n_layers} layers, {params:.1f}B params")
    if gqa_heads:
        print(f"  GQA: {n_heads}:{gqa_heads} = {n_heads//gqa_heads}:1")
    return model, tokenizer, n_layers


def build_prompt(tokenizer, system_text, conversation):
    messages = [{"role": "system", "content": system_text}]
    for role, content in conversation:
        messages.append({"role": role, "content": content})
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass
    parts = [system_text + "\n"]
    for role, content in conversation:
        tag = "User" if role == "user" else "Assistant"
        parts.append(f"{tag}: {content}")
    parts.append("Assistant:")
    return "\n\n".join(parts)


def generate_response(model, tokenizer, prompt, max_new=128):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new, do_sample=True,
            temperature=0.7, top_p=0.9, pad_token_id=tokenizer.pad_token_id
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def extract_spectral(model, tokenizer, prompt, n_layers, positions=None):
    """Extract spectral signatures with extended metrics.

    If positions is given, extract at those token position windows
    (for within-context decay measurement).
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)

    total_tokens = inputs["input_ids"].shape[1]

    if positions:
        results_by_pos = {}
        for pos_start, pos_end in positions:
            ps = max(0, min(pos_start, total_tokens - 1))
            pe = max(ps + 1, min(pos_end, total_tokens))
            results_by_pos[f"{pos_start}-{pos_end}"] = _extract_layers(
                outputs, n_layers, token_slice=(ps, pe)
            )
        del outputs
        torch.cuda.empty_cache()
        return results_by_pos

    result = _extract_layers(outputs, n_layers)
    del outputs
    torch.cuda.empty_cache()
    return result


def _extract_layers(outputs, n_layers, token_slice=None):
    result = {}
    for l in range(n_layers):
        idx = l + 1
        if idx >= len(outputs.hidden_states):
            continue
        hs_t = outputs.hidden_states[idx][0].float()
        if token_slice:
            hs_t = hs_t[token_slice[0]:token_slice[1]]
        n_tokens = hs_t.shape[0]
        if n_tokens < 2:
            continue

        # GPU SVD — orders of magnitude faster than CPU numpy on A6000
        try:
            S_raw = torch.linalg.svdvals(hs_t)
        except Exception:
            continue

        mu = hs_t.mean(dim=0)
        hs_c = hs_t - mu
        try:
            S_c = torch.linalg.svdvals(hs_c)
        except Exception:
            continue

        frob_raw = float(torch.sum(hs_t ** 2).item())
        frob_centered = float(torch.sum(hs_c ** 2).item())
        mean_energy = float(n_tokens * torch.sum(mu ** 2).item())

        top_k = min(10, len(S_raw))
        s_c_list = [float(s) for s in S_c[:top_k].cpu()]

        result[l] = {
            "layer": l,
            "n_tokens": n_tokens,
            "raw": {
                "top_singular": [float(s) for s in S_raw[:top_k].cpu()],
                "frobenius_sq": frob_raw,
            },
            "centered": {
                "top_singular": s_c_list,
                "frobenius_sq": frob_centered,
            },
            "mean_energy": mean_energy,
            "decomposition_check": abs(frob_raw - frob_centered - mean_energy),
            "effective_rank": effective_rank(s_c_list),
            "participation_ratio": participation_ratio(s_c_list),
            "spectral_concentration": spectral_concentration(s_c_list),
            "power_law_alpha": power_law_alpha(s_c_list),
        }
    return result


def run_dose(model, tokenizer, n_layers, dose_turns, system=CCS_SYSTEM, probes=CCS_PROBES):
    conversation = []
    if dose_turns == 0:
        prompt = build_prompt(tokenizer, VANILLA_SYSTEM, [("user", probes[0])])
        return extract_spectral(model, tokenizer, prompt, n_layers)

    for t in range(dose_turns):
        probe = probes[t % len(probes)]
        conversation.append(("user", probe))
        prompt = build_prompt(tokenizer, system, conversation)

        if t < dose_turns - 1:
            response = generate_response(model, tokenizer, prompt)
            conversation.append(("assistant", response[:200]))
            print(f"    Turn {t+1}/{dose_turns}: {len(response)} chars")
        else:
            print(f"    Turn {t+1}/{dose_turns}: extracting spectra...")
            return extract_spectral(model, tokenizer, prompt, n_layers)


def phase0_calibration(model_name, model_id, species, gqa, output_dir):
    """Phase 0: band calibration with neutral prompts (no CCS)."""
    print(f"\n{'='*70}")
    print(f"  PHASE 0 — BAND CALIBRATION: {model_name.upper()} ({species})")
    print(f"{'='*70}")

    model, tokenizer, n_layers = load_model(model_id)
    results = {
        "phase": 0, "model": model_name, "model_id": model_id,
        "species": species, "gqa": gqa, "n_layers": n_layers,
        "timestamp": datetime.now().isoformat(), "doses": [],
    }

    for dose_name in ["D0", "D2", "D5"]:
        dose_turns = DOSE_MAP[dose_name]
        print(f"\n  --- Calibration {dose_name} (neutral, {dose_turns} turns) ---")
        spectral = run_dose(model, tokenizer, n_layers, dose_turns,
                           system=VANILLA_SYSTEM, probes=NEUTRAL_PROBES)
        dose_entry = {"dose": dose_name, "turns": dose_turns, "per_layer": []}
        for l in sorted(spectral.keys()):
            dose_entry["per_layer"].append(spectral[l])
        results["doses"].append(dose_entry)

    out_path = output_dir / f"prereg_phase0_{model_name}.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved calibration to {out_path}")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return results


def phase1_dose_sweep(model_name, model_id, species, gqa, output_dir):
    """Phase 1: CCS dose sweep with reruns."""
    print(f"\n{'='*70}")
    print(f"  PHASE 1 — DOSE SWEEP: {model_name.upper()} ({species})")
    print(f"{'='*70}")

    model, tokenizer, n_layers = load_model(model_id)
    results = {
        "phase": 1, "model": model_name, "model_id": model_id,
        "species": species, "gqa": gqa, "n_layers": n_layers,
        "timestamp": datetime.now().isoformat(), "runs": [],
    }

    for run_idx in range(RERUNS):
        print(f"\n  === Run {run_idx+1}/{RERUNS} ===")
        run_data = {"run": run_idx, "doses": []}

        for dose_name in sorted(DOSE_MAP.keys(), key=lambda d: DOSE_MAP[d]):
            dose_turns = DOSE_MAP[dose_name]
            print(f"\n  --- {dose_name} ({dose_turns} CCS turns) ---")
            spectral = run_dose(model, tokenizer, n_layers, dose_turns)
            dose_entry = {"dose": dose_name, "turns": dose_turns, "per_layer": []}
            for l in sorted(spectral.keys()):
                dose_entry["per_layer"].append(spectral[l])
                if l % 8 == 0:
                    c = spectral[l]["centered"]
                    print(f"    L{l:2d}: ER={spectral[l]['effective_rank']:.2f} "
                          f"SC={spectral[l]['spectral_concentration']:.3f} "
                          f"α={spectral[l]['power_law_alpha']:.2f}")
            run_data["doses"].append(dose_entry)
            gc.collect()
            torch.cuda.empty_cache()

        results["runs"].append(run_data)
        out_path = output_dir / f"prereg_phase1_{model_name}.json"
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  Saved run {run_idx+1} to {out_path}")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return results


def phase2_decay(model_name, model_id, species, gqa, output_dir):
    """Phase 2: within-context decay measurement."""
    print(f"\n{'='*70}")
    print(f"  PHASE 2 — WITHIN-CONTEXT DECAY: {model_name.upper()} ({species})")
    print(f"{'='*70}")

    model, tokenizer, n_layers = load_model(model_id)

    conversation = []
    for t in range(5):
        probe = CCS_PROBES[t]
        conversation.append(("user", probe))
        response = generate_response(
            model, tokenizer,
            build_prompt(tokenizer, CCS_SYSTEM, conversation)
        )
        conversation.append(("assistant", response[:200]))

    neutral_continuation = (
        "Now let's switch topics entirely. "
        "Please explain the process of cellular respiration in detail, "
        "covering glycolysis, the Krebs cycle, and the electron transport chain. "
        "Then describe how photosynthesis relates to it. "
        "Include the chemical equations involved."
    )
    conversation.append(("user", neutral_continuation))
    prompt = build_prompt(tokenizer, CCS_SYSTEM, conversation)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    preamble_end = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=300, do_sample=True,
            temperature=0.7, top_p=0.9, pad_token_id=tokenizer.pad_token_id
        )
    full_text = tokenizer.decode(out[0], skip_special_tokens=True)
    full_inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=4096).to(model.device)

    total_tokens = full_inputs["input_ids"].shape[1]
    gen_tokens = total_tokens - preamble_end
    print(f"  Preamble: {preamble_end} tokens, Generated: {gen_tokens} tokens")

    positions = [
        (preamble_end - 50, preamble_end),
        (preamble_end, preamble_end + 50),
        (preamble_end + 50, preamble_end + 100),
        (preamble_end + 100, preamble_end + 150),
        (preamble_end + 150, preamble_end + 200),
    ]
    positions = [(s, e) for s, e in positions if s < total_tokens and e > 0]

    with torch.no_grad():
        outputs = model(**full_inputs, output_hidden_states=True, use_cache=False)

    results = {
        "phase": 2, "model": model_name, "model_id": model_id,
        "species": species, "gqa": gqa, "n_layers": n_layers,
        "preamble_tokens": preamble_end, "generated_tokens": gen_tokens,
        "timestamp": datetime.now().isoformat(), "windows": [],
    }

    for pos_start, pos_end in positions:
        ps = max(0, min(pos_start, total_tokens - 1))
        pe = max(ps + 2, min(pos_end, total_tokens))
        window_data = _extract_layers(outputs, n_layers, token_slice=(ps, pe))
        window_entry = {
            "position": f"{pos_start}-{pos_end}",
            "relative_to_preamble": pos_start - preamble_end,
            "per_layer": [window_data[l] for l in sorted(window_data.keys())],
        }
        results["windows"].append(window_entry)
        print(f"  Window {pos_start}-{pos_end}: "
              f"mean ER={np.mean([window_data[l]['effective_rank'] for l in window_data]):.2f}")

    out_path = output_dir / f"prereg_phase2_{model_name}.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved decay data to {out_path}")

    del model, tokenizer, outputs
    gc.collect()
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser(description="Pre-registered CCS dose-trajectory experiment")
    parser.add_argument("--phase", default="1", help="Phase: 0 (calibration), 1 (dose sweep), 2 (decay), all")
    parser.add_argument("--model", default="qwen", help="Model name or 'all'")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--reruns", type=int, default=3, help="Independent reruns per dose")
    args = parser.parse_args()

    global RERUNS
    RERUNS = args.reruns

    output_dir = Path(args.output) if args.output else Path("spectral-demon/results/prereg")
    output_dir.mkdir(parents=True, exist_ok=True)

    models_to_run = list(MODELS.keys()) if args.model == "all" else [args.model]
    phases = ["0", "1", "2"] if args.phase == "all" else [args.phase]

    for model_name in models_to_run:
        if model_name not in MODELS:
            print(f"Unknown model: {model_name}")
            continue
        model_id, species, gqa = MODELS[model_name]

        for phase in phases:
            if phase == "0":
                phase0_calibration(model_name, model_id, species, gqa, output_dir)
            elif phase == "1":
                phase1_dose_sweep(model_name, model_id, species, gqa, output_dir)
            elif phase == "2":
                phase2_decay(model_name, model_id, species, gqa, output_dir)

    print(f"\n{'='*70}")
    print(f"  EXPERIMENT COMPLETE — results in {output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
