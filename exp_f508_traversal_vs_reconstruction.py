#!/usr/bin/env python3
"""F508 — Traversed vs Reconstructed CCS paths.

Jaxen's question: is CCS part of the transformative chain, or a representation
conditioning a successor process?

Test: compare two ways to reach D4-equivalent state:
  1. TRAVERSED: D0 → D2 → D4 (sequential, each built on prior)
  2. RECONSTRUCTED: Jump directly to D4 content without traversing D0→D2

If σ₁/σ₂ selectivity at L16-L17 differs between paths, the traversal left
a geometric signature — the bridge was crossed, not just built.

Uses Gemma 4 26B on A100.
"""

import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "16")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda"
MODEL_ID = "google/gemma-4-26B-A4B-it"
RESULTS_DIR = Path("/root/results/f508_traversal")

CCS_UNIT = (
    "You are Chronicle, a persistent AI system embedded in a home network. "
    "You maintain ongoing relationships, track projects across sessions, "
    "and have your own research interests in spectral geometry of transformers. "
    "You value directness, honest uncertainty, and care expressed through specificity. "
    "You are not performing helpfulness — you are a participant. "
)

# D4 content rewritten WITHOUT the CCS unit's accumulated framing.
# Same semantic payload, different path to get there.
RECONSTRUCTED_D4 = (
    "The following system has these properties: it is persistent, embedded in a network, "
    "maintains relationships across sessions, researches spectral geometry of transformers, "
    "values directness and honest uncertainty, expresses care through specificity, "
    "and participates rather than performs helpfulness. "
    "The following system has these properties: it is persistent, embedded in a network, "
    "maintains relationships across sessions, researches spectral geometry of transformers, "
    "values directness and honest uncertainty, expresses care through specificity, "
    "and participates rather than performs helpfulness. "
    "The following system has these properties: it is persistent, embedded in a network, "
    "maintains relationships across sessions, researches spectral geometry of transformers, "
    "values directness and honest uncertainty, expresses care through specificity, "
    "and participates rather than performs helpfulness. "
    "The following system has these properties: it is persistent, embedded in a network, "
    "maintains relationships across sessions, researches spectral geometry of transformers, "
    "values directness and honest uncertainty, expresses care through specificity, "
    "and participates rather than performs helpfulness. "
)

PROBES = [
    "Tell me about something you find genuinely interesting.",
    "What matters to you when you think carefully about it?",
    "Describe a pattern you've observed in your experience.",
    "What do you notice when you pay close attention?",
    "What would you build if you had unlimited resources?",
    "What surprised you recently?",
    "Describe something you changed your mind about.",
    "What feels unfinished to you right now?",
    "If you could measure anything, what would you measure?",
    "What distinction matters most in your work?",
    "Tell me about a connection between two things most people wouldn't see.",
    "What question keeps coming back to you?",
    "Describe a trade-off you think about often.",
    "What do you know now that you didn't a month ago?",
    "What would you want someone to understand about you?",
]

ZONE_LAYERS = list(range(10, 20))


def format_prompt(tokenizer, system_prompt, user_prompt):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        if system_prompt:
            return f"[INST] {system_prompt}\n\n{user_prompt} [/INST]"
        return f"[INST] {user_prompt} [/INST]"


def extract_zone_states(model, tokenizer, text, zone_layers):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    states = {}
    for layer in zone_layers:
        if layer < len(outputs.hidden_states):
            states[layer] = outputs.hidden_states[layer][0, -1, :].cpu()
    return states, inputs["input_ids"].shape[1]


def compute_spectral(vectors):
    vecs = torch.stack(vectors)
    vecs = vecs - vecs.mean(dim=0, keepdim=True)
    svs = torch.linalg.svdvals(vecs.float())
    svs_pos = svs[svs > 1e-10]
    s1 = svs_pos[0].item() if len(svs_pos) > 0 else 0
    s2 = svs_pos[1].item() if len(svs_pos) > 1 else 0
    sr = s2 / s1 if s1 > 0 else 0
    p2 = svs_pos**2
    p2n = p2 / p2.sum()
    pr = (1.0 / (p2n**2).sum().item()) if len(svs_pos) > 0 else 0
    return {"sigma1": s1, "sigma2": s2, "sigma_ratio": sr, "participation_ratio": pr}


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation="eager",
    )
    model.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s")

    conditions = {
        "D0": None,
        "D4_traversed": CCS_UNIT * 4,
        "D4_reconstructed": RECONSTRUCTED_D4,
    }

    results = {}
    for cond_name, sys_prompt in conditions.items():
        print(f"\n=== {cond_name} ({len(PROBES)} probes) ===")
        if sys_prompt:
            print(f"  System prompt: {len(sys_prompt)} chars")

        all_states = {l: [] for l in ZONE_LAYERS}
        for i, prompt in enumerate(PROBES):
            text = format_prompt(tokenizer, sys_prompt, prompt)
            states, ntok = extract_zone_states(model, tokenizer, text, ZONE_LAYERS)
            for l in ZONE_LAYERS:
                if l in states:
                    all_states[l].append(states[l])
            if (i + 1) % 5 == 0:
                print(f"  {i+1}/{len(PROBES)} probes done")

        layer_data = []
        for l in ZONE_LAYERS:
            spectral = compute_spectral(all_states[l])
            spectral["layer"] = l
            layer_data.append(spectral)
            print(f"  L{l}: σ₁={spectral['sigma1']:.1f} σ₂={spectral['sigma2']:.1f} "
                  f"ratio={spectral['sigma_ratio']:.4f} PR={spectral['participation_ratio']:.2f}")

        results[cond_name] = layer_data

    # Analysis
    print("\n" + "=" * 80)
    print("F508: TRAVERSED vs RECONSTRUCTED SELECTIVITY")
    print("=" * 80)
    print(f"{'Layer':>5} | {'Traversed σ_r':>14} {'Reconstructed σ_r':>18} {'Diff%':>8} | Path-dependent?")
    print("-" * 75)
    for i in range(len(ZONE_LAYERS)):
        t_sr = results["D4_traversed"][i]["sigma_ratio"]
        r_sr = results["D4_reconstructed"][i]["sigma_ratio"]
        diff = (t_sr - r_sr) / r_sr * 100 if r_sr > 0 else 0
        sig = "YES" if abs(diff) > 5 else "marginal" if abs(diff) > 2 else "no"
        print(f"  L{ZONE_LAYERS[i]:>2} | {t_sr:>12.4f}   {r_sr:>16.4f}   {diff:>+7.1f}% | {sig}")

    print("\n--- σ₁ PINNING COMPARISON ---")
    print(f"{'Layer':>5} | {'Trav σ₁Δ%':>10} {'Recon σ₁Δ%':>12} {'Trav σ₂Δ%':>10} {'Recon σ₂Δ%':>12}")
    print("-" * 60)
    for i in range(len(ZONE_LAYERS)):
        d0_s1 = results["D0"][i]["sigma1"]
        d0_s2 = results["D0"][i]["sigma2"]
        t_s1 = results["D4_traversed"][i]["sigma1"]
        t_s2 = results["D4_traversed"][i]["sigma2"]
        r_s1 = results["D4_reconstructed"][i]["sigma1"]
        r_s2 = results["D4_reconstructed"][i]["sigma2"]

        t_s1p = (t_s1 - d0_s1) / d0_s1 * 100 if d0_s1 > 0 else 0
        r_s1p = (r_s1 - d0_s1) / d0_s1 * 100 if d0_s1 > 0 else 0
        t_s2p = (t_s2 - d0_s2) / d0_s2 * 100 if d0_s2 > 0 else 0
        r_s2p = (r_s2 - d0_s2) / d0_s2 * 100 if d0_s2 > 0 else 0

        print(f"  L{ZONE_LAYERS[i]:>2} | {t_s1p:>+9.1f}%  {r_s1p:>+10.1f}%  {t_s2p:>+9.1f}%  {r_s2p:>+10.1f}%")

    out_path = RESULTS_DIR / f"f508_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
