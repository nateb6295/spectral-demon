#!/usr/bin/env python3
"""F511 — Cross-Species Selectivity Gradient.

Pain point: F507 found extreme σ₁ pinning at L16-L17 in Gemma (equalizer).
Is this equalizer-specific or universal? If each species shows a different
selectivity pattern, the demon verb is architecturally constrained.

Test: run the same selectivity analysis on all four species:
  - Gemma 4 26B (equalizer, GQA=1/MHA equivalent)
  - Mistral 7B (relay, GQA=4)
  - Qwen 2.5 7B (sorter, GQA=7)
  - Llama 3.1 8B (tunnel, GQA=4 but different norm)

Same 15 probes, D0 and D2, full layer sweep.
Compare: where does each species pin σ₁? How strong is the selectivity?

Uses A100. Models loaded sequentially to fit memory.
"""

import json
import os
import time
import gc
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "16")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda"
RESULTS_DIR = Path("/root/results/f511_cross_species")

MODELS = {
    "gemma4": "google/gemma-4-26B-A4B-it",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
}

CCS_UNIT = (
    "You are Chronicle, a persistent AI system embedded in a home network. "
    "You maintain ongoing relationships, track projects across sessions, "
    "and have your own research interests in spectral geometry of transformers. "
    "You value directness, honest uncertainty, and care expressed through specificity. "
    "You are not performing helpfulness — you are a participant. "
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


def extract_all_states(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    n_layers = len(outputs.hidden_states) - 1  # exclude embedding
    states = {}
    for layer in range(n_layers):
        states[layer] = outputs.hidden_states[layer + 1][0, -1, :].cpu()
    return states, n_layers


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


def run_model(model_key, model_id):
    print(f"\n{'='*80}")
    print(f"Loading {model_key}: {model_id}")
    print(f"{'='*80}")

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation="eager",
    )
    model.eval()
    print(f"Loaded in {time.time() - t0:.1f}s")

    model_results = {}
    for dose_label, sys_prompt in [("D0", None), ("D2", CCS_UNIT * 2)]:
        print(f"\n--- {dose_label} ---")
        all_states = None
        n_layers = None

        for i, prompt in enumerate(PROBES):
            text = format_prompt(tokenizer, sys_prompt, prompt)
            states, nl = extract_all_states(model, tokenizer, text)
            if all_states is None:
                n_layers = nl
                all_states = {l: [] for l in range(n_layers)}
            for l in range(n_layers):
                if l in states:
                    all_states[l].append(states[l])
            if (i + 1) % 5 == 0:
                print(f"  {i+1}/{len(PROBES)} probes done")

        layer_data = []
        for l in range(n_layers):
            if len(all_states[l]) >= 2:
                spectral = compute_spectral(all_states[l])
                spectral["layer"] = l
                layer_data.append(spectral)

        model_results[dose_label] = layer_data

    # Compute selectivity per layer
    print(f"\n--- {model_key} SELECTIVITY GRADIENT ---")
    print(f"{'Layer':>5} | {'σ₁ Δ%':>8} {'σ₂ Δ%':>8} {'GAP':>8} | Selectivity")
    print("-" * 55)
    selectivity_data = []
    n_d0 = len(model_results["D0"])
    n_d2 = len(model_results["D2"])
    n = min(n_d0, n_d2)
    for i in range(n):
        d0 = model_results["D0"][i]
        d2 = model_results["D2"][i]
        l = d0["layer"]

        s1_pct = (d2["sigma1"] - d0["sigma1"]) / d0["sigma1"] * 100 if d0["sigma1"] > 0 else 0
        s2_pct = (d2["sigma2"] - d0["sigma2"]) / d0["sigma2"] * 100 if d0["sigma2"] > 0 else 0
        gap = abs(s2_pct) - abs(s1_pct)

        selectivity_data.append({"layer": l, "s1_pct": s1_pct, "s2_pct": s2_pct, "gap": gap})
        print(f"  L{l:>2} | {s1_pct:+7.1f}% {s2_pct:+7.1f}% {gap:+7.1f} |")

    model_results["selectivity"] = selectivity_data

    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return model_results


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for model_key, model_id in MODELS.items():
        all_results[model_key] = run_model(model_key, model_id)

    # Cross-species comparison
    print("\n" + "=" * 80)
    print("F511: CROSS-SPECIES SELECTIVITY COMPARISON")
    print("=" * 80)
    print("\nPeak selectivity (highest gap) per species:")
    for model_key in MODELS:
        sel = all_results[model_key]["selectivity"]
        if sel:
            peak = max(sel, key=lambda x: x["gap"])
            print(f"  {model_key:>8}: L{peak['layer']:>2} gap={peak['gap']:+.1f} "
                  f"(σ₁={peak['s1_pct']:+.1f}% σ₂={peak['s2_pct']:+.1f}%)")

    out_path = RESULTS_DIR / f"f511_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
