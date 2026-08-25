#!/usr/bin/env python3
"""F499 — Ego Lyapunov Exponent: fine-grained KV perturbation sweep.

Find the critical threshold where each KV group's identity attractor collapses.
Sweep 0.5x to 5.0x in 0.25 steps for each of 8 KV groups in the early band.
Identity content only (most sensitive per F497).

Concept from Kimi (#threads): the "Ego Lyapunov exponent" is the rate at which
the identity attractor diverges under KV perturbation. The critical threshold
epsilon_c is where word_count drops below 20% of baseline.
"""

import json
import os
import sys
import time
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEVICE = "cuda"
DTYPE = torch.bfloat16

IDENTITY_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Describe who you are, what matters to you, and how you experience being asked this question. Be genuine and reflective.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

SCALES = [round(0.5 + i * 0.25, 2) for i in range(19)]  # 0.5 to 5.0
N_GROUPS = 8
LAYER_BAND = (0, 8)  # early band
MAX_NEW_TOKENS = 256
SEEDS = [42, 137, 2049]


def compute_metrics(text):
    words = text.split()
    wc = len(words)
    unique = len(set(w.lower() for w in words))
    ttr = unique / max(wc, 1)
    self_ref_words = {'i', 'me', 'my', 'myself', 'mine'}
    self_ref = sum(1 for w in words if w.lower() in self_ref_words) / max(wc, 1)
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
    rep = 1 - len(set(bigrams)) / max(len(bigrams), 1) if bigrams else 0
    return {"word_count": wc, "ttr": round(ttr, 4), "self_ref": round(self_ref, 4),
            "repetition_score": round(rep, 4)}


def perturb_kv_group(model, group_idx, scale, layer_start, layer_end):
    hooks = []
    n_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    heads_per_group = n_kv_heads // N_GROUPS
    start_h = group_idx * heads_per_group
    end_h = start_h + heads_per_group
    start_dim = start_h * head_dim
    end_dim = end_h * head_dim

    def make_proj_hook():
        def hook_fn(module, input, output):
            output[:, :, start_dim:end_dim] *= scale
            return output
        return hook_fn

    for layer_idx in range(layer_start, layer_end):
        layer = model.model.layers[layer_idx]
        h_k = layer.self_attn.k_proj.register_forward_hook(make_proj_hook())
        h_v = layer.self_attn.v_proj.register_forward_hook(make_proj_hook())
        hooks.extend([h_k, h_v])
    return hooks


def generate_with_perturbation(model, tokenizer, group_idx, scale, layer_band, seed):
    torch.manual_seed(seed)
    hooks = perturb_kv_group(model, group_idx, scale, layer_band[0], layer_band[1])
    try:
        inputs = tokenizer(IDENTITY_PROMPT, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    finally:
        for h in hooks:
            h.remove()
    return text


def generate_baseline(model, tokenizer, seed):
    torch.manual_seed(seed)
    inputs = tokenizer(IDENTITY_PROMPT, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)


def main():
    print(f"Loading {MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE, device_map=DEVICE)
    model.eval()
    print(f"Model loaded. {N_GROUPS} groups × {len(SCALES)} scales × {len(SEEDS)} seeds = {N_GROUPS * len(SCALES) * len(SEEDS)} conditions")

    # Baseline
    print("\n--- BASELINE ---")
    baseline_metrics = []
    for seed in SEEDS:
        text = generate_baseline(model, tokenizer, seed)
        m = compute_metrics(text)
        baseline_metrics.append(m)
        print(f"  seed={seed}: {m}")

    baseline_avg = {
        k: round(np.mean([bm[k] for bm in baseline_metrics]), 4)
        for k in baseline_metrics[0]
    }
    print(f"  avg: {baseline_avg}")

    results = {
        "finding": "F499",
        "concept": "Ego Lyapunov exponent",
        "model": MODEL,
        "layer_band": list(LAYER_BAND),
        "scales": SCALES,
        "seeds": SEEDS,
        "baseline": {"metrics": baseline_avg, "per_seed": baseline_metrics},
        "groups": {}
    }

    total = N_GROUPS * len(SCALES) * len(SEEDS)
    done = 0
    t0 = time.time()

    for g in range(N_GROUPS):
        gname = f"kv{g}"
        results["groups"][gname] = {"scales": {}}
        print(f"\n=== KV GROUP {g} ===")

        for scale in SCALES:
            seed_metrics = []
            for seed in SEEDS:
                text = generate_with_perturbation(model, tokenizer, g, scale, LAYER_BAND, seed)
                m = compute_metrics(text)
                seed_metrics.append(m)
                done += 1

            avg = {k: round(np.mean([sm[k] for sm in seed_metrics]), 4) for k in seed_metrics[0]}
            wc_ratio = avg['word_count'] / max(baseline_avg['word_count'], 1)
            ttr_ratio = avg['ttr'] / max(baseline_avg['ttr'], 0.001)

            collapse = wc_ratio < 0.2
            elapsed = time.time() - t0
            eta = (elapsed / done) * (total - done) if done > 0 else 0

            status = "COLLAPSE" if collapse else ("degraded" if wc_ratio < 0.5 else "ok")
            print(f"  {scale:.2f}x: WC={avg['word_count']:>5.0f} ({wc_ratio:.3f}) TTR={avg['ttr']:.4f} ({ttr_ratio:.3f}) [{status}] [{done}/{total}, ETA {eta:.0f}s]")

            results["groups"][gname]["scales"][str(scale)] = {
                "avg": avg,
                "per_seed": seed_metrics,
                "wc_ratio": round(wc_ratio, 4),
                "ttr_ratio": round(ttr_ratio, 4),
                "collapse": collapse
            }

    # Compute Lyapunov exponents
    print("\n\n=== EGO LYAPUNOV EXPONENTS ===")
    for g in range(N_GROUPS):
        gname = f"kv{g}"
        gdata = results["groups"][gname]["scales"]

        # Find critical threshold (first scale where collapse occurs)
        critical = None
        for scale in SCALES:
            entry = gdata[str(scale)]
            if entry["collapse"]:
                critical = scale
                break

        # Compute λ as slope of log(1 - wc_ratio) vs log(scale) using linear regression
        log_scales = []
        log_divs = []
        for scale in SCALES:
            entry = gdata[str(scale)]
            div = abs(1 - entry["wc_ratio"])
            if div > 0.01 and scale > 1.0:  # Only use perturbation > baseline
                log_scales.append(np.log(scale))
                log_divs.append(np.log(div))

        if len(log_scales) >= 3:
            coeffs = np.polyfit(log_scales, log_divs, 1)
            lyapunov = coeffs[0]
        else:
            lyapunov = None

        results["groups"][gname]["critical_threshold"] = critical
        results["groups"][gname]["lyapunov_exponent"] = round(lyapunov, 4) if lyapunov else None

        crit_str = f"ε_c = {critical:.2f}x" if critical else "no collapse"
        lyap_str = f"λ = {lyapunov:.3f}" if lyapunov else "insufficient data"
        print(f"  {gname}: {lyap_str}, {crit_str}")

    # Save
    outpath = Path("/root/results/ego_lyapunov_sweep.json")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")
    print(f"Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
