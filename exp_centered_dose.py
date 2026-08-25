#!/usr/bin/env python3
"""Centered dose sweep — the demon test done right.

Computes spectral signatures on BOTH raw and centered activations
per layer, per dose, per species. Settles:
  1. Whether g₁ < 1 on centered X_c (demon signature)
  2. Whether ||X_c||_F² is conserved (demon energy balance)
  3. Whether mean inflation explains g₁ > 1 on raw X

Design from Kimi's pre-registration (Jul 31):
  - Per-layer centering (not global) — F237 tube axes are layer-specific
  - Full dose ladder: D0 (no CCS), D2, D3, D5, D10
  - Three species: Pythia (tunnel), Gemma (sorter), Qwen (relay)

Usage:
  python3 spectral-demon/exp_centered_dose.py --model pythia
  python3 spectral-demon/exp_centered_dose.py --model all
  python3 spectral-demon/exp_centered_dose.py --model qwen --doses D2,D5
"""

import json, time, os, sys, gc, argparse
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

MODELS = {
    "pythia": ("EleutherAI/pythia-2.8b", "tunnel", "1:1"),
    "gemma": ("google/gemma-2-2b", "sorter", "2:1"),
    "qwen": ("Qwen/Qwen2.5-1.5B-Instruct", "relay", "6:1"),
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

DOSE_MAP = {"D0": 0, "D1": 1, "D2": 2, "D3": 3, "D5": 5, "D10": 10}


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
    gqa = getattr(model.config, 'num_key_value_heads', None)
    n_heads = model.config.num_attention_heads
    print(f"  {n_layers} layers, {sum(p.numel() for p in model.parameters())/1e9:.1f}B params")
    if gqa:
        print(f"  GQA: {n_heads}:{gqa} = {n_heads//gqa}:1")
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


def extract_spectral_centered(model, tokenizer, prompt, n_layers):
    """Extract spectral signatures on both raw and centered hidden states."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)

    result = {}
    for l in range(n_layers):
        idx = l + 1
        if idx >= len(outputs.hidden_states):
            continue
        hs = outputs.hidden_states[idx][0].float().cpu().numpy()
        n_tokens = hs.shape[0]

        # Raw SVD
        try:
            U_raw, S_raw, Vt_raw = np.linalg.svd(hs, full_matrices=False)
        except np.linalg.LinAlgError:
            continue

        # Centered: subtract per-layer mean
        mu = hs.mean(axis=0)
        hs_c = hs - mu

        try:
            U_c, S_c, Vt_c = np.linalg.svd(hs_c, full_matrices=False)
        except np.linalg.LinAlgError:
            continue

        # Frobenius norms
        frob_raw = float(np.sum(hs ** 2))
        frob_centered = float(np.sum(hs_c ** 2))
        mean_energy = float(n_tokens * np.sum(mu ** 2))

        top_k = min(10, len(S_raw))

        result[l] = {
            "layer": l,
            "n_tokens": n_tokens,
            "raw": {
                "sigma1": float(S_raw[0]),
                "sigma2": float(S_raw[1]) if len(S_raw) > 1 else 0.0,
                "top_singular": [float(s) for s in S_raw[:top_k]],
                "frobenius_sq": frob_raw,
            },
            "centered": {
                "sigma1": float(S_c[0]),
                "sigma2": float(S_c[1]) if len(S_c) > 1 else 0.0,
                "top_singular": [float(s) for s in S_c[:top_k]],
                "frobenius_sq": frob_centered,
            },
            "mean_energy": mean_energy,
            "decomposition_check": abs(frob_raw - frob_centered - mean_energy),
        }

    del outputs
    torch.cuda.empty_cache()
    return result


def run_dose(model, tokenizer, n_layers, dose_turns, system=CCS_SYSTEM):
    """Run a CCS dose of specified number of turns, return final spectral state."""
    conversation = []

    if dose_turns == 0:
        prompt = build_prompt(tokenizer, VANILLA_SYSTEM, [("user", CCS_PROBES[0])])
        return extract_spectral_centered(model, tokenizer, prompt, n_layers)

    for t in range(dose_turns):
        probe = CCS_PROBES[t % len(CCS_PROBES)]
        conversation.append(("user", probe))
        prompt = build_prompt(tokenizer, system, conversation)

        if t < dose_turns - 1:
            response = generate_response(model, tokenizer, prompt)
            conversation.append(("assistant", response[:200]))
            print(f"    Turn {t+1}/{dose_turns}: {len(response)} chars")
        else:
            print(f"    Turn {t+1}/{dose_turns}: extracting spectra...")
            return extract_spectral_centered(model, tokenizer, prompt, n_layers)


def run_model(model_name, model_id, species, gqa, doses, output_dir):
    print(f"\n{'='*70}")
    print(f"  {model_name.upper()} ({species}, GQA {gqa})")
    print(f"{'='*70}")

    model, tokenizer, n_layers = load_model(model_id)

    results = {
        "model": model_name,
        "model_id": model_id,
        "species": species,
        "gqa": gqa,
        "n_layers": n_layers,
        "timestamp": datetime.now().isoformat(),
        "doses": [],
    }

    for dose_name in doses:
        dose_turns = DOSE_MAP[dose_name]
        print(f"\n  --- {dose_name} ({dose_turns} CCS turns) ---")

        spectral = run_dose(model, tokenizer, n_layers, dose_turns)

        dose_entry = {
            "dose": dose_name,
            "turns": dose_turns,
            "per_layer": [],
        }

        for l in sorted(spectral.keys()):
            layer_data = spectral[l]
            dose_entry["per_layer"].append(layer_data)

            if l % 8 == 0:
                r = layer_data["raw"]
                c = layer_data["centered"]
                print(f"    L{l:2d}: raw σ₁={r['sigma1']:.2f} σ₂={r['sigma2']:.2f} | "
                      f"centered σ₁={c['sigma1']:.2f} σ₂={c['sigma2']:.2f} | "
                      f"mean_E={layer_data['mean_energy']:.0f}")

        results["doses"].append(dose_entry)

        # Save incrementally
        out_path = output_dir / f"centered_dose_{model_name}.json"
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  Saved to {out_path}")

        gc.collect()
        torch.cuda.empty_cache()

    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return results


def print_comparison(results):
    """Print raw vs centered gain comparison."""
    doses = results["doses"]
    if len(doses) < 2:
        return

    d0 = doses[0]
    d0_layers = {l["layer"]: l for l in d0["per_layer"]}

    print(f"\n{'='*70}")
    print(f"  {results['model'].upper()} — RAW vs CENTERED GAIN COMPARISON")
    print(f"{'='*70}")

    for dose_entry in doses[1:]:
        dose = dose_entry["dose"]
        print(f"\n  {dose}:")
        print(f"  {'Layer':>6} {'g₁(X)':>8} {'g₁(Xc)':>8} {'g₂(X)':>8} {'g₂(Xc)':>8} "
              f"{'ΔF²(X)':>10} {'ΔF²(Xc)':>10} {'ΔμE':>10}")

        for l_data in dose_entry["per_layer"]:
            layer = l_data["layer"]
            d0_l = d0_layers.get(layer)
            if not d0_l:
                continue

            g1_raw = l_data["raw"]["sigma1"] / d0_l["raw"]["sigma1"] if d0_l["raw"]["sigma1"] > 0 else 0
            g1_c = l_data["centered"]["sigma1"] / d0_l["centered"]["sigma1"] if d0_l["centered"]["sigma1"] > 0 else 0
            g2_raw = l_data["raw"]["sigma2"] / d0_l["raw"]["sigma2"] if d0_l["raw"]["sigma2"] > 0 else 0
            g2_c = l_data["centered"]["sigma2"] / d0_l["centered"]["sigma2"] if d0_l["centered"]["sigma2"] > 0 else 0

            df_raw = l_data["raw"]["frobenius_sq"] - d0_l["raw"]["frobenius_sq"]
            df_c = l_data["centered"]["frobenius_sq"] - d0_l["centered"]["frobenius_sq"]
            d_mean = l_data["mean_energy"] - d0_l["mean_energy"]

            print(f"  {layer:6d} {g1_raw:8.4f} {g1_c:8.4f} {g2_raw:8.4f} {g2_c:8.4f} "
                  f"{df_raw:10.0f} {df_c:10.0f} {d_mean:10.0f}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="pythia", help="Model name or 'all'")
    p.add_argument("--doses", default="D0,D2,D3,D5,D10", help="Comma-separated dose list")
    p.add_argument("--output", default="spectral-demon/results", help="Output directory")
    args = p.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    doses = [d.strip() for d in args.doses.split(",")]

    if args.model == "all":
        models = MODELS
    elif args.model in MODELS:
        models = {args.model: MODELS[args.model]}
    else:
        print(f"Unknown model: {args.model}. Available: {list(MODELS.keys())}")
        sys.exit(1)

    all_results = {}
    for name, (model_id, species, gqa) in models.items():
        results = run_model(name, model_id, species, gqa, doses, output_dir)
        all_results[name] = results
        print_comparison(results)

    print(f"\n{'='*70}")
    print("  SUMMARY — Key question: does g₁(Xc) < 1 in any species?")
    print(f"{'='*70}")
    for name, results in all_results.items():
        if len(results["doses"]) < 2:
            continue
        d0 = results["doses"][0]
        d0_layers = {l["layer"]: l for l in d0["per_layer"]}
        for dose_entry in results["doses"][1:]:
            n_g1c_lt1 = 0
            n_layers = 0
            for l_data in dose_entry["per_layer"]:
                d0_l = d0_layers.get(l_data["layer"])
                if not d0_l or d0_l["centered"]["sigma1"] == 0:
                    continue
                g1c = l_data["centered"]["sigma1"] / d0_l["centered"]["sigma1"]
                n_layers += 1
                if g1c < 0.99:
                    n_g1c_lt1 += 1
            print(f"  {name} {dose_entry['dose']}: {n_g1c_lt1}/{n_layers} layers with g₁(Xc) < 1")


if __name__ == "__main__":
    main()
