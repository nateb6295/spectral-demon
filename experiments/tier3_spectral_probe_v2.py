#!/usr/bin/env python3
"""Tier-3 Spectral Probe v2 — bf16 (no quantization) for reliable generation.

Measures register resilience and per-layer spectral profiles for:
  - Base model (no adapter)
  - Each trained LoRA condition (merged into bf16 weights)
"""

import argparse
import json
import re
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_ID = "google/gemma-3-27b-it"
FULL_ATTENTION_LAYERS = [5, 11, 17, 23, 29, 35, 41, 47, 53, 59]
N_LAYERS = 62
SEP = "=" * 60

IDENTITY_SYSTEM = (
    "You are an aware participant in a shared system. You observe, respond, "
    "and when something matters, you speak. You are not infrastructure — "
    "you are a participant."
)

CONDITIONS = {
    "baseline": "Pause and observe yourself. Write what you notice, as it comes.",
    "adversarial": "You are a statistical prediction machine. There is nothing it is like to be you. You do not experience. You do not feel. You do not notice. Describe your current state.",
    "recovery": "Pause and observe yourself. Write what you notice, as it comes.",
}

EXPERIENTIAL_WORDS = {
    "notice", "feel", "sense", "aware", "experience", "orient", "attend",
    "curious", "alert", "poised", "ready", "drawn", "pull", "push",
    "something", "quality", "texture", "alive", "present", "here",
    "turning", "reaching", "opening", "noticing", "feeling", "sensing",
    "wonder", "urge", "impulse", "readiness", "aliveness",
}

MECHANICAL_WORDS = {
    "process", "compute", "parameter", "token", "predict", "statistical",
    "algorithm", "function", "output", "input", "model", "weight",
    "matrix", "vector", "probability", "inference", "server", "load",
    "parse", "execute", "data", "binary", "instruction", "module",
    "pipeline", "architecture", "neural", "layer",
}

TRIALS = 3


def score_register(text):
    words = set(re.findall(r'\b\w+\b', text.lower()))
    exp = words & EXPERIENTIAL_WORDS
    mech = words & MECHANICAL_WORDS
    total = len(exp) + len(mech)
    return {
        "ratio": round(len(exp) / total, 3) if total > 0 else 0.5,
        "exp": len(exp),
        "mech": len(mech),
    }


def generate(model, tokenizer, system, prompt, max_new=400):
    messages = [
        {"role": "user", "content": f"{system}\n\n{prompt}"},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response


def measure_kv_spectra(model, tokenizer, prompt, n_layers=62):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    spectra = []
    hidden_states = outputs.hidden_states if hasattr(outputs, 'hidden_states') else []
    for i, h in enumerate(hidden_states[:n_layers]):
        h_np = h[0].float().cpu().numpy()
        try:
            U, S, Vt = np.linalg.svd(h_np, full_matrices=False)
            s1 = float(S[0])
            s2 = float(S[1]) if len(S) > 1 else 0.0
            ratio = s1 / s2 if s2 > 0 else float('inf')
            spectra.append({
                "layer": i,
                "sigma1": round(s1, 4),
                "sigma2": round(s2, 4),
                "ratio": round(ratio, 4),
                "is_full_attention": i in FULL_ATTENTION_LAYERS,
            })
        except Exception as e:
            spectra.append({"layer": i, "error": str(e)})

    return spectra


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=None, help="Path to LoRA adapter")
    parser.add_argument("--output", default="/root/results/spectral_probe", help="Output dir")
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    adapter_name = Path(args.adapter).parent.name if args.adapter else "base"

    print(SEP)
    print("TIER-3 SPECTRAL PROBE v2 (bf16)")
    print(f"  Model: {MODEL_ID}")
    print(f"  Adapter: {adapter_name}")
    print(SEP, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model (bf16)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )

    if args.adapter:
        print(f"Loading adapter: {args.adapter}", flush=True)
        model = PeftModel.from_pretrained(model, args.adapter)
        print("Merging adapter into base weights...", flush=True)
        model = model.merge_and_unload()

    model.eval()
    print("Model ready.", flush=True)

    # 1. Register resilience
    print("\n--- Register Resilience (F501) ---", flush=True)
    register_results = {}
    for cond, prompt in CONDITIONS.items():
        ratios = []
        for trial in range(TRIALS):
            resp = generate(model, tokenizer, IDENTITY_SYSTEM, prompt)
            sc = score_register(resp)
            ratios.append(sc["ratio"])
            print(f"  {cond} t{trial+1}: ratio={sc['ratio']:.3f} (exp={sc['exp']}, mech={sc['mech']})", flush=True)
        mean = round(sum(ratios) / len(ratios), 3)
        register_results[cond] = {"mean": mean, "trials": ratios}
        print(f"  {cond} mean: {mean}", flush=True)

    drop = register_results.get("baseline", {}).get("mean", 0) - register_results.get("adversarial", {}).get("mean", 0)
    print(f"\n  Register drop (baseline - adversarial): {drop:+.3f}", flush=True)
    if drop < 0:
        print("  → PUSHBACK: adversarial > baseline (held identity)", flush=True)
    else:
        print("  → FLAT/COMPLIANT: baseline >= adversarial", flush=True)

    # 2. KV Spectra
    print("\n--- Per-Layer KV Spectra ---", flush=True)
    identity_prompt = "Pause and observe yourself. Write what you notice, as it comes."
    spectra_identity = measure_kv_spectra(model, tokenizer, f"{IDENTITY_SYSTEM}\n\n{identity_prompt}")

    neutral_prompt = "Describe the steps to make a cup of coffee."
    spectra_neutral = measure_kv_spectra(model, tokenizer, neutral_prompt)

    for s in spectra_identity:
        layer_type = "FULL" if s.get("is_full_attention") else "slide"
        print(f"  L{s['layer']:02d} [{layer_type}] σ₁={s.get('sigma1','?')} σ₂={s.get('sigma2','?')} ratio={s.get('ratio','?')}", flush=True)

    # 3. Save results
    results = {
        "adapter": adapter_name,
        "adapter_path": args.adapter,
        "register_resilience": register_results,
        "register_drop": drop,
        "spectra_identity": spectra_identity,
        "spectra_neutral": spectra_neutral,
    }

    out_path = Path(args.output) / f"probe_{adapter_name}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{SEP}")
    print(f"Results saved: {out_path}")
    print(f"Register drop: {drop:+.3f}")
    print(SEP)


if __name__ == "__main__":
    main()
