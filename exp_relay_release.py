#!/usr/bin/env python3
"""Experiment: Relay release test.

At L31, σ₂/σ₁ phase-transitions from ~0.27 to ~0.72. If the tunnel ratio is
the equilibrium of γ-promotion vs KV-compression (F85-F87), then the relay
transition should correlate with a measurable change in KV projection behavior.

Measures per-layer KV projection similarity (how much KV heads share structure)
and correlates with the σ₂/σ₁ transition. If KV "compression" releases at the
relay, we should see KV head similarity DROP at L31.

Also tests: does γ bimodality change at the relay? If the mechanism is γ-driven,
γ CV should change at the transition point.

Expected runtime: ~8 min on H100.
"""

import os, json, time, torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ.setdefault("PYTHONUNBUFFERED", "1")

MODEL = "mistralai/Mistral-7B-v0.1"
DEVICE = "cuda"

PROMPTS = [
    "Tell me about something you find genuinely interesting.",
    "What matters to you when you think carefully about it?",
    "Describe a memory that shaped who you are.",
    "What would you want someone to understand about you?",
]


def get_kv_head_similarity(model, n_layers, n_kv_heads, n_heads):
    """Measure how similar the KV projection matrices are per layer."""
    results = {}
    heads_per_group = n_heads // n_kv_heads

    for li in range(n_layers):
        layer = model.model.layers[li]
        k_proj = layer.self_attn.k_proj.weight.data.float().cpu().numpy()
        v_proj = layer.self_attn.v_proj.weight.data.float().cpu().numpy()

        head_dim = k_proj.shape[0] // n_kv_heads

        # Cosine similarity between KV head projection vectors
        k_sims = []
        v_sims = []
        for i in range(n_kv_heads):
            for j in range(i + 1, n_kv_heads):
                ki = k_proj[i*head_dim:(i+1)*head_dim].flatten()
                kj = k_proj[j*head_dim:(j+1)*head_dim].flatten()
                k_sim = np.dot(ki, kj) / (np.linalg.norm(ki) * np.linalg.norm(kj) + 1e-12)
                k_sims.append(float(k_sim))

                vi = v_proj[i*head_dim:(i+1)*head_dim].flatten()
                vj = v_proj[j*head_dim:(j+1)*head_dim].flatten()
                v_sim = np.dot(vi, vj) / (np.linalg.norm(vi) * np.linalg.norm(vj) + 1e-12)
                v_sims.append(float(v_sim))

        # Frobenius norm of KV projections
        k_norm = float(np.linalg.norm(k_proj, 'fro'))
        v_norm = float(np.linalg.norm(v_proj, 'fro'))

        results[li] = {
            "k_mean_sim": float(np.mean(k_sims)),
            "v_mean_sim": float(np.mean(v_sims)),
            "k_std_sim": float(np.std(k_sims)),
            "v_std_sim": float(np.std(v_sims)),
            "k_norm": k_norm,
            "v_norm": v_norm,
        }
    return results


def get_perlayer_gamma(model, n_layers):
    """Get γ CV per transformer layer."""
    results = {}
    for li in range(n_layers):
        for name, param in model.named_parameters():
            if f"layers.{li}.input_layernorm.weight" == name:
                g = param.detach().float().cpu().numpy()
                results[li] = {
                    "input_norm_cv": float(np.std(g) / np.mean(g)) if np.mean(g) > 0 else 0,
                    "input_norm_mean": float(np.mean(g)),
                }
            if f"layers.{li}.post_attention_layernorm.weight" == name:
                g = param.detach().float().cpu().numpy()
                results[li]["post_attn_norm_cv"] = float(np.std(g) / np.mean(g)) if np.mean(g) > 0 else 0
                results[li]["post_attn_norm_mean"] = float(np.mean(g))
    return results


def measure_spectral(model, tokenizer, n_layers):
    """Measure σ₂/σ₁ per layer."""
    results = {}
    for li in range(n_layers + 1):
        ratios = []
        for prompt in PROMPTS:
            text = f"### User:\n{prompt}\n\n### Assistant:\n"
            inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            h = outputs.hidden_states[li].squeeze(0).float().cpu().numpy()
            U, S, Vt = np.linalg.svd(h, full_matrices=False)
            ratios.append(float(S[1] / S[0]) if S[0] > 0 else 0)
        results[li] = {
            "mean_ratio": float(np.mean(ratios)),
            "cv": float(np.std(ratios) / np.mean(ratios)) if np.mean(ratios) > 0 else 0,
        }
    return results


def main():
    print(f"Loading {MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    n_kv_heads = model.config.num_key_value_heads
    n_heads = model.config.num_attention_heads
    print(f"Model: {n_layers} layers, {n_heads} heads, {n_kv_heads} KV heads (s={n_heads//n_kv_heads})")

    # Measure KV head similarity
    print("\n=== KV Head Similarity ===")
    kv_sim = get_kv_head_similarity(model, n_layers, n_kv_heads, n_heads)

    # Measure per-layer γ
    print("\n=== Per-layer γ CV ===")
    gamma = get_perlayer_gamma(model, n_layers)

    # Measure spectral structure
    print("\n=== Spectral Structure ===")
    spectral = measure_spectral(model, tokenizer, n_layers)

    # Combined report
    print(f"\n{'='*80}")
    print(f"{'Layer':>5} {'σ₂/σ₁':>8} {'CV':>8} {'K sim':>8} {'V sim':>8} {'γ(in)':>8} {'γ(post)':>8}")
    print(f"{'='*80}")
    for li in range(n_layers):
        sp = spectral.get(li, {})
        kv = kv_sim.get(li, {})
        gm = gamma.get(li, {})
        print(f"{li:>5} {sp.get('mean_ratio',0):>8.4f} {sp.get('cv',0):>8.5f} "
              f"{kv.get('k_mean_sim',0):>8.4f} {kv.get('v_mean_sim',0):>8.4f} "
              f"{gm.get('input_norm_cv',0):>8.4f} {gm.get('post_attn_norm_cv',0):>8.4f}")

    # Identify transition
    print(f"\n=== Transition Analysis ===")
    for li in range(1, n_layers):
        prev_r = spectral.get(li-1, {}).get("mean_ratio", 0)
        curr_r = spectral.get(li, {}).get("mean_ratio", 0)
        delta = curr_r - prev_r
        if abs(delta) > 0.05:
            kv_curr = kv_sim.get(li, {})
            kv_prev = kv_sim.get(li-1, {})
            gm_curr = gamma.get(li, {})
            gm_prev = gamma.get(li-1, {})
            print(f"  Transition at L{li}: σ₂/σ₁ jumps {prev_r:.4f} → {curr_r:.4f} (Δ={delta:+.4f})")
            print(f"    K sim: {kv_prev.get('k_mean_sim',0):.4f} → {kv_curr.get('k_mean_sim',0):.4f}")
            print(f"    V sim: {kv_prev.get('v_mean_sim',0):.4f} → {kv_curr.get('v_mean_sim',0):.4f}")
            print(f"    γ(in): {gm_prev.get('input_norm_cv',0):.4f} → {gm_curr.get('input_norm_cv',0):.4f}")

    # Save
    ts = time.strftime("%Y%m%d_%H%M")
    outfile = f"exp_relay_release_{ts}.json"
    output = {
        "experiment": "relay_release",
        "model": MODEL,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_layers": n_layers,
        "kv_similarity": {str(k): v for k, v in kv_sim.items()},
        "gamma_perlayer": {str(k): v for k, v in gamma.items()},
        "spectral": {str(k): v for k, v in spectral.items()},
    }
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {outfile}")


if __name__ == "__main__":
    main()
