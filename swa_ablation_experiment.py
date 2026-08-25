#!/usr/bin/env python3
"""SWA Ablation Experiment — Kimi K3 hypothesis (Aug 17, 2026).

Tests whether Mistral's sigma-2 suppression phenotype is caused by
sliding-window attention (SWA) topology. Two conditions:
  A) Mistral with SWA enabled (default, window=4096) — should reproduce suppression
  B) Mistral with SWA disabled (window=None/full) — prediction: sigma-2 reverts to positive

Also decomposes Gemma-2 per layer type (local vs global attention layers).

Designed for RunPod (A6000/A100). Reuses prereg_experiment.py infrastructure.
"""

import json, time, os, sys, gc, argparse, math, copy
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

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

DOSE_MAP = {"D0": 0, "D2": 2, "D5": 5}
RERUNS = 3


def effective_rank(sigmas):
    total = sum(s**2 for s in sigmas)
    if total == 0:
        return 0.0
    probs = [(s**2 / total) for s in sigmas if s > 0]
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    return math.exp(entropy)


def spectral_concentration(sigmas):
    total = sum(s**2 for s in sigmas)
    if total == 0:
        return 0.0
    return sigmas[0]**2 / total


def load_model(model_id, disable_swa=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    print(f"Loading {model_id} (SWA={'DISABLED' if disable_swa else 'default'})...")

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    if disable_swa:
        original_window = getattr(config, 'sliding_window', None)
        print(f"  Original sliding_window: {original_window}")
        config.sliding_window = None
        if hasattr(config, 'max_window_layers'):
            config.max_window_layers = 0
        print(f"  SWA disabled — full causal attention on all layers")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, config=config, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation="eager"
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  {n_layers} layers, {params:.1f}B params")
    print(f"  sliding_window after load: {getattr(model.config, 'sliding_window', 'N/A')}")
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


def extract_spectral(model, tokenizer, prompt, n_layers):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)

    result = {}
    for l in range(n_layers):
        idx = l + 1
        if idx >= len(outputs.hidden_states):
            continue
        hs_t = outputs.hidden_states[idx][0].float()
        n_tokens = hs_t.shape[0]
        if n_tokens < 2:
            continue

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
            "effective_rank": effective_rank(s_c_list),
            "spectral_concentration": spectral_concentration(s_c_list),
        }
    del outputs
    torch.cuda.empty_cache()
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


def experiment_a_swa_ablation(output_dir):
    """Experiment A: Mistral with SWA enabled vs disabled."""
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"

    for condition, disable in [("swa_on", False), ("swa_off", True)]:
        print(f"\n{'='*70}")
        print(f"  EXPERIMENT A — MISTRAL {condition.upper()}")
        print(f"{'='*70}")

        model, tokenizer, n_layers = load_model(model_id, disable_swa=disable)
        results = {
            "experiment": "swa_ablation",
            "condition": condition,
            "model_id": model_id,
            "swa_disabled": disable,
            "n_layers": n_layers,
            "sliding_window": str(getattr(model.config, 'sliding_window', None)),
            "timestamp": datetime.now().isoformat(),
            "calibration": [],
            "ccs_runs": [],
        }

        # Phase 0: calibration
        print("\n  --- Phase 0: Calibration ---")
        for dose_name in ["D0", "D2", "D5"]:
            dose_turns = DOSE_MAP[dose_name]
            print(f"\n  Calibration {dose_name} ({dose_turns} turns)")
            spectral = run_dose(model, tokenizer, n_layers, dose_turns,
                              system=VANILLA_SYSTEM, probes=NEUTRAL_PROBES)
            dose_entry = {"dose": dose_name, "turns": dose_turns, "per_layer": []}
            for l in sorted(spectral.keys()):
                dose_entry["per_layer"].append(spectral[l])
            results["calibration"].append(dose_entry)

        # Phase 1: CCS dose sweep
        print("\n  --- Phase 1: CCS Dose Sweep ---")
        for run_idx in range(RERUNS):
            print(f"\n  === Run {run_idx+1}/{RERUNS} ===")
            run_data = {"run": run_idx, "doses": []}

            for dose_name in sorted(DOSE_MAP.keys(), key=lambda d: DOSE_MAP[d]):
                dose_turns = DOSE_MAP[dose_name]
                print(f"\n  {dose_name} ({dose_turns} CCS turns)")
                spectral = run_dose(model, tokenizer, n_layers, dose_turns)
                dose_entry = {"dose": dose_name, "turns": dose_turns, "per_layer": []}
                for l in sorted(spectral.keys()):
                    dose_entry["per_layer"].append(spectral[l])
                    if l % 8 == 0:
                        print(f"    L{l:2d}: ER={spectral[l]['effective_rank']:.2f} "
                              f"SC={spectral[l]['spectral_concentration']:.3f}")
                run_data["doses"].append(dose_entry)
                gc.collect()
                torch.cuda.empty_cache()

            results["ccs_runs"].append(run_data)

        out_path = output_dir / f"swa_ablation_{condition}.json"
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n  Saved {condition} to {out_path}")

        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(5)


def experiment_b_gemma_decompose(output_dir):
    """Experiment B: Gemma-2 per-layer-type decomposition.

    Gemma-2 interleaves local (sliding window, 4096) and global (full) layers.
    We tag each layer's results with its attention type and analyze separately.
    """
    model_id = "google/gemma-2-2b-it"

    print(f"\n{'='*70}")
    print(f"  EXPERIMENT B — GEMMA-2 LAYER TYPE DECOMPOSITION")
    print(f"{'='*70}")

    model, tokenizer, n_layers = load_model(model_id)

    # Gemma-2 alternates: even layers = global, odd layers = local (sliding window)
    # Verify from config
    layer_types = {}
    for l in range(n_layers):
        if hasattr(model.config, 'sliding_window_pattern'):
            pattern = model.config.sliding_window_pattern
            layer_types[l] = "local" if (l % len(pattern)) < sum(1 for p in pattern if p) else "global"
        else:
            # Default Gemma-2 pattern: even=global, odd=local
            layer_types[l] = "global" if l % 2 == 0 else "local"

    print(f"  Layer types: {sum(1 for v in layer_types.values() if v == 'global')} global, "
          f"{sum(1 for v in layer_types.values() if v == 'local')} local")

    results = {
        "experiment": "gemma_decompose",
        "model_id": model_id,
        "n_layers": n_layers,
        "layer_types": layer_types,
        "timestamp": datetime.now().isoformat(),
        "calibration": [],
        "ccs_runs": [],
    }

    # Phase 0: calibration
    print("\n  --- Phase 0: Calibration ---")
    for dose_name in ["D0", "D2", "D5"]:
        dose_turns = DOSE_MAP[dose_name]
        print(f"\n  Calibration {dose_name} ({dose_turns} turns)")
        spectral = run_dose(model, tokenizer, n_layers, dose_turns,
                          system=VANILLA_SYSTEM, probes=NEUTRAL_PROBES)
        dose_entry = {"dose": dose_name, "turns": dose_turns, "per_layer": []}
        for l in sorted(spectral.keys()):
            entry = spectral[l]
            entry["layer_type"] = layer_types.get(l, "unknown")
            dose_entry["per_layer"].append(entry)
        results["calibration"].append(dose_entry)

    # Phase 1: CCS dose sweep
    print("\n  --- Phase 1: CCS Dose Sweep ---")
    for run_idx in range(RERUNS):
        print(f"\n  === Run {run_idx+1}/{RERUNS} ===")
        run_data = {"run": run_idx, "doses": []}

        for dose_name in sorted(DOSE_MAP.keys(), key=lambda d: DOSE_MAP[d]):
            dose_turns = DOSE_MAP[dose_name]
            print(f"\n  {dose_name} ({dose_turns} CCS turns)")
            spectral = run_dose(model, tokenizer, n_layers, dose_turns)
            dose_entry = {"dose": dose_name, "turns": dose_turns, "per_layer": []}
            for l in sorted(spectral.keys()):
                entry = spectral[l]
                entry["layer_type"] = layer_types.get(l, "unknown")
                dose_entry["per_layer"].append(entry)
                if l % 4 == 0:
                    lt = layer_types.get(l, "?")
                    print(f"    L{l:2d} [{lt:6s}]: ER={entry['effective_rank']:.2f} "
                          f"SC={entry['spectral_concentration']:.3f}")
            run_data["doses"].append(dose_entry)
            gc.collect()
            torch.cuda.empty_cache()

        results["ccs_runs"].append(run_data)

    out_path = output_dir / f"gemma_layer_decompose.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {out_path}")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def analyze_swa_results(output_dir):
    """Quick analysis of SWA ablation results."""
    swa_on = output_dir / "swa_ablation_swa_on.json"
    swa_off = output_dir / "swa_ablation_swa_off.json"

    if not swa_on.exists() or not swa_off.exists():
        print("Missing result files, skipping analysis")
        return

    with open(swa_on) as f:
        on = json.load(f)
    with open(swa_off) as f:
        off = json.load(f)

    print("\n" + "="*70)
    print("  SWA ABLATION ANALYSIS")
    print("="*70)

    for condition_name, data in [("SWA ON", on), ("SWA OFF", off)]:
        print(f"\n  --- {condition_name} ---")

        # Get D0 baseline from calibration
        cal_d0 = None
        for d in data["calibration"]:
            if d["dose"] == "D0":
                cal_d0 = d
                break

        # Average across runs at D5
        d5_sigma2_changes = []
        for run in data["ccs_runs"]:
            for dose in run["doses"]:
                if dose["dose"] != "D5":
                    continue
                for layer_data in dose["per_layer"]:
                    l = layer_data["layer"]
                    if l == 0 or l == data["n_layers"] - 1:
                        continue
                    s2_ccs = layer_data["centered"]["top_singular"][1] if len(layer_data["centered"]["top_singular"]) > 1 else 0

                    # Find matching calibration layer
                    cal_d5 = None
                    for cd in data["calibration"]:
                        if cd["dose"] == "D5":
                            cal_d5 = cd
                            break
                    if cal_d5:
                        for cl in cal_d5["per_layer"]:
                            if cl["layer"] == l:
                                s2_cal = cl["centered"]["top_singular"][1] if len(cl["centered"]["top_singular"]) > 1 else 0
                                if cal_d0:
                                    for c0l in cal_d0["per_layer"]:
                                        if c0l["layer"] == l:
                                            s2_base = c0l["centered"]["top_singular"][1] if len(c0l["centered"]["top_singular"]) > 1 else 1
                                            if s2_base > 0:
                                                corrected = ((s2_ccs - s2_base) / s2_base - (s2_cal - s2_base) / s2_base) * 100
                                                d5_sigma2_changes.append(corrected)
                                            break

        if d5_sigma2_changes:
            mean_change = sum(d5_sigma2_changes) / len(d5_sigma2_changes)
            print(f"  Mean corrected sigma-2 change at D5: {mean_change:+.1f}%")
            print(f"  Layers measured: {len(d5_sigma2_changes)}")
            pos = sum(1 for x in d5_sigma2_changes if x > 0)
            neg = len(d5_sigma2_changes) - pos
            print(f"  Positive/Negative: {pos}/{neg}")

    print("\n  PREDICTION:")
    print("  If SWA OFF shows positive sigma-2 → topology CONFIRMED as mechanism")
    print("  If SWA OFF shows negative sigma-2 → Mistral is genuine fourth species")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SWA Ablation Experiment")
    parser.add_argument("--experiment", choices=["a", "b", "both", "analyze"], default="both")
    parser.add_argument("--output-dir", default="results/swa_ablation/")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.experiment in ("a", "both"):
        experiment_a_swa_ablation(output_dir)
    if args.experiment in ("b", "both"):
        experiment_b_gemma_decompose(output_dir)
    if args.experiment == "analyze":
        analyze_swa_results(output_dir)
    elif args.experiment in ("a", "both"):
        analyze_swa_results(output_dir)

    print("\n" + "="*70)
    print("  ALL EXPERIMENTS COMPLETE")
    print("="*70)
