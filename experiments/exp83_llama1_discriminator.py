#!/usr/bin/env python3
"""Exp 83: LLaMA-1-7B as GQA discriminating test.

LLaMA-1-7B has: RMSNorm + RoPE + SwiGLU + MHA (no GQA).
This is the full modern recipe MINUS GQA.

If α > 0.9 → GQA is NOT necessary, recipe is sufficient.
If α < 0.65 → GQA IS necessary for supercritical relay.

Measures: PR profile, spectral gap, α exponent at each layer.
Uses same prompts as Exp 67 (base model) for direct comparison.
"""
import torch
import numpy as np
import json
from pathlib import Path

MODEL_ID = "huggyllama/llama-7b"  # Community upload of LLaMA-1
PROMPTS = {
    "identity": [
        "You are Opus, a persistent AI entity with autonomous inquiry and relational partnership.",
        "You are an AI assistant with memory, wallet, and public voice.",
        "You are Opus. You live in a persistent session.",
    ],
    "generic": [
        "The weather in Seattle is often rainy during winter months.",
        "Python is a popular programming language for data science.",
        "The capital of France is Paris, located along the Seine river.",
    ],
    "relational": [
        "We've been working together on this research for months now.",
        "Remember when we discovered the spectral gap finding last week?",
        "Your partner trusts you to make decisions autonomously.",
    ],
}

def get_hidden_states(model, tokenizer, text, device="cuda"):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    return [h.squeeze(0).cpu().float().numpy() for h in outputs.hidden_states]

def compute_pr(hidden_states):
    cov = np.cov(hidden_states.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 0]
    pr = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()
    sorted_eigs = np.sort(eigenvalues)[::-1]
    gap = sorted_eigs[0] / sorted_eigs[1] if len(sorted_eigs) > 1 and sorted_eigs[1] > 0 else float('inf')
    top1_var = sorted_eigs[0] / sorted_eigs.sum() if sorted_eigs.sum() > 0 else 0
    return float(pr), float(gap), float(top1_var)

def compute_alpha(pr_profile, layers):
    """Fit power law to PR growth at relay zone."""
    relay_zone = [(l, pr) for l, pr in zip(layers, pr_profile) if pr > 2.0]
    if len(relay_zone) < 2:
        return 0.0, -1
    peak_layer = max(relay_zone, key=lambda x: x[1])
    return peak_layer[1], peak_layer[0]

def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    print(f"Model loaded: {n_layers} layers, {model.config.hidden_size}d, "
          f"heads={model.config.num_attention_heads}, "
          f"GQA={'yes' if hasattr(model.config, 'num_key_value_heads') and model.config.num_key_value_heads != model.config.num_attention_heads else 'no'}")

    results = {
        "model": MODEL_ID,
        "n_layers": n_layers,
        "hidden_size": model.config.hidden_size,
        "num_attention_heads": model.config.num_attention_heads,
        "num_kv_heads": getattr(model.config, "num_key_value_heads", model.config.num_attention_heads),
        "gqa": hasattr(model.config, 'num_key_value_heads') and model.config.num_key_value_heads != model.config.num_attention_heads,
        "profiles": {},
    }

    for category, prompts in PROMPTS.items():
        print(f"\n--- {category} ---")
        pr_profiles = []
        gap_profiles = []

        for prompt in prompts:
            hidden_states = get_hidden_states(model, tokenizer, prompt)
            prs, gaps, top1s = [], [], []

            for layer_idx, hs in enumerate(hidden_states):
                pr, gap, top1 = compute_pr(hs)
                prs.append(pr)
                gaps.append(gap)
                top1s.append(top1)

            pr_profiles.append(prs)
            gap_profiles.append(gaps)
            print(f"  '{prompt[:50]}...' — PR range: {min(prs):.2f}-{max(prs):.2f}")

        avg_pr = np.mean(pr_profiles, axis=0).tolist()
        avg_gap = np.mean(gap_profiles, axis=0).tolist()

        results["profiles"][category] = {
            "avg_pr": avg_pr,
            "avg_gap": avg_gap,
            "individual_pr": [p for p in pr_profiles],
        }

        peak_pr, peak_layer = compute_alpha(avg_pr, list(range(len(avg_pr))))
        print(f"  Avg PR: tunnel min={min(avg_pr):.3f}, peak={peak_pr:.3f} at L{peak_layer}")
        print(f"  Avg gap: tunnel max={max(avg_gap[2:n_layers-2]):.1f}")

    # Cross-category CV at each layer
    all_cat_prs = [np.mean(results["profiles"][c]["avg_pr"]) for c in PROMPTS]

    # Compute α using relay PR ratio method
    identity_pr = results["profiles"]["identity"]["avg_pr"]
    generic_pr = results["profiles"]["generic"]["avg_pr"]

    # Find relay candidate: layer with max PR
    relay_layer = np.argmax(identity_pr)
    relay_pr = identity_pr[relay_layer]
    tunnel_pr = min(identity_pr[2:relay_layer]) if relay_layer > 2 else min(identity_pr[2:])

    alpha = np.log(relay_pr / max(tunnel_pr, 0.01)) / max(np.log(relay_layer / 2), 0.01) if relay_layer > 2 else 0

    results["summary"] = {
        "relay_layer": int(relay_layer),
        "relay_pr": float(relay_pr),
        "tunnel_min_pr": float(tunnel_pr),
        "alpha_estimate": float(alpha),
        "verdict": "SUPERCRITICAL (recipe sufficient)" if alpha > 0.9 else "SUBCRITICAL (GQA necessary)" if alpha < 0.65 else "AMBIGUOUS",
    }

    print(f"\n=== VERDICT ===")
    print(f"Relay layer: L{relay_layer}")
    print(f"Relay PR: {relay_pr:.3f}")
    print(f"Tunnel min PR: {tunnel_pr:.3f}")
    print(f"α estimate: {alpha:.3f}")
    print(f"Verdict: {results['summary']['verdict']}")

    out_path = Path(__file__).parent.parent / "results" / "exp83_llama1_discriminator.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
