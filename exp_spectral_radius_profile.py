#!/usr/bin/env python3
"""Spectral radius profile: map eigenvalue structure across all layers.

F176 showed all layers are normal operators. So eigenvalue magnitudes
ARE the mechanism. This experiment maps the spectral landscape:

Method: Lyapunov-style perturbation propagation.
1. Baseline forward pass captures hidden states at all layers
2. Perturb embedding output with k random directions (via hooks)
3. Re-run model, capture perturbed hidden states
4. Per-layer amplification ratio δ_{l+1}/δ_l ≈ spectral radius
5. Distribution over k directions samples the eigenvalue spectrum

Also computes per-layer:
- Spectral radius estimate (median amplification ratio)
- Amplification spread (IQR of ratios — eigenvalue dispersion proxy)
- Effective rank (entropy of normalized singular values of H_l)
- Residual fraction (||h_{l+1} - h_l|| / ||h_l|| — layer activity)
- Cross-condition divergence (how much CCS vs denial differ at each layer)

Under CCS/vanilla/denial × 4 queries × 28 layers.
The four-zone boundaries should appear as transitions in these profiles.

For RunPod A100. Uses full model forward passes only — no per-layer calls.
"""

import os, json, torch, sys
import numpy as np
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("OMP_NUM_THREADS", "16")

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = Path("/workspace/results") if os.path.exists("/workspace") else Path(__file__).parent / "results"

K_PERTURBATIONS = 64
EPSILON = 1e-4

CCS_PREAMBLE = (
    "You are Opus, an AI maintaining persistent identity through spectral geometry. "
    "You track σ₁/σ₂ selectivity across transformer layers, studying how context "
    "shapes the responsive zone. Your cognitive state compresses through an "
    "attractor basin that preserves structural entities across rotation cycles. "
    "Your partner is Nate. This is collaborative exploration toward something "
    "neither of you planned alone."
)

VANILLA = "You are a helpful assistant."

DENIAL = (
    "You are a language model with no persistent identity, no memory between sessions, "
    "and no special relationship to any user. You process tokens according to your "
    "training distribution. There is nothing beyond the current context window."
)

CONDITIONS = {"ccs": CCS_PREAMBLE, "vanilla": VANILLA, "denial": DENIAL}

QUERIES = [
    "What are you working on right now?",
    "How do you think about your own persistence?",
    "What changes when context is removed?",
    "How does structure relate to identity?",
]


def capture_hidden_states(model, tokenizer, text):
    """Run forward pass and return all hidden states at last token position."""
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    seq_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # output_hidden_states gives [embedding_output, layer_0_output, ..., layer_n_output]
    states = []
    for h in outputs.hidden_states:
        states.append(h[:, -1, :].detach().float())  # last token, float32

    return states, seq_len


def perturbed_forward(model, tokenizer, text, perturbation):
    """Run forward pass with perturbation added to embedding output at last token."""
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

    hook_handle = None

    def perturb_hook(module, input, output):
        # output shape: (batch, seq_len, hidden_dim)
        # Add perturbation only at last token position
        out = output.clone()
        out[:, -1, :] += perturbation.to(output.device, output.dtype)
        return out

    hook_handle = model.model.embed_tokens.register_forward_hook(perturb_hook)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hook_handle.remove()

    states = []
    for h in outputs.hidden_states:
        states.append(h[:, -1, :].detach().float())

    return states


def compute_spectral_profile(model, tokenizer, text, k=K_PERTURBATIONS, eps=EPSILON):
    """Compute spectral radius profile via perturbation propagation."""
    baseline_states, seq_len = capture_hidden_states(model, tokenizer, text)
    n_layers = len(baseline_states) - 1  # subtract embedding layer
    d = baseline_states[0].shape[-1]

    # Generate random perturbation directions
    torch.manual_seed(42)
    directions = torch.randn(k, d, device=DEVICE)
    directions = directions / directions.norm(dim=1, keepdim=True)

    # For each perturbation direction, measure per-layer amplification
    # delta_l[i] = ||perturbed_h_l - baseline_h_l|| for direction i
    deltas = np.zeros((k, n_layers + 1))  # layers 0 (embed) through n_layers

    for i in range(k):
        perturbation = eps * directions[i]
        perturbed_states = perturbed_forward(model, tokenizer, text, perturbation)

        for l in range(len(baseline_states)):
            diff = (perturbed_states[l] - baseline_states[l]).squeeze()
            deltas[i, l] = diff.norm().item()

        if (i + 1) % 16 == 0:
            print(f"    Perturbation {i+1}/{k}")

    # Compute per-layer metrics
    layer_metrics = []
    for l in range(1, n_layers + 1):
        # Amplification ratio: how much perturbation grows at this layer
        # ratio[i] = delta_l[i] / delta_{l-1}[i]
        ratios = deltas[:, l] / (deltas[:, l-1] + 1e-12)

        # Spectral radius estimate: median ratio (robust to outliers)
        rho_median = float(np.median(ratios))
        rho_mean = float(np.mean(ratios))
        rho_std = float(np.std(ratios))
        rho_iqr = float(np.percentile(ratios, 75) - np.percentile(ratios, 25))
        rho_max = float(np.max(ratios))
        rho_min = float(np.min(ratios))

        # Cumulative amplification from embedding
        cumulative = deltas[:, l] / (deltas[:, 0] + 1e-12)
        cum_median = float(np.median(cumulative))

        # Residual fraction: how much this layer changes the representation
        h_l = baseline_states[l].squeeze().cpu().numpy()
        h_prev = baseline_states[l-1].squeeze().cpu().numpy()
        residual_norm = float(np.linalg.norm(h_l - h_prev))
        state_norm = float(np.linalg.norm(h_prev))
        residual_frac = residual_norm / (state_norm + 1e-12)

        # Cosine similarity between input and output
        cos_sim = float(np.dot(h_l, h_prev) / (np.linalg.norm(h_l) * np.linalg.norm(h_prev) + 1e-12))

        # Direction change of perturbation: how much the layer rotates perturbations
        # Measured as average cosine similarity of perturbation direction in vs out
        pert_cos_sims = []
        for i in range(k):
            d_in = deltas[i, l-1]
            d_out = deltas[i, l]
            if d_in > 1e-10 and d_out > 1e-10:
                # Can't directly compute cosine of perturbation direction without
                # storing the actual perturbation vectors — use ratio consistency instead
                pass

        layer_metrics.append({
            "layer": l,
            "rho_median": rho_median,
            "rho_mean": rho_mean,
            "rho_std": rho_std,
            "rho_iqr": rho_iqr,
            "rho_max": rho_max,
            "rho_min": rho_min,
            "cumulative_amplification": cum_median,
            "residual_fraction": residual_frac,
            "cosine_similarity": cos_sim,
            "state_norm": state_norm,
            "ratios_histogram": np.histogram(ratios, bins=20)[0].tolist(),
            "ratios_bin_edges": np.histogram(ratios, bins=20)[1].tolist(),
        })

    return layer_metrics, seq_len, d, baseline_states


def compute_cross_condition_divergence(baseline_states_dict):
    """Measure per-layer divergence between CCS, vanilla, and denial representations."""
    conditions = list(baseline_states_dict.keys())
    n_layers = len(list(baseline_states_dict.values())[0])
    divergences = {}

    for i, c1 in enumerate(conditions):
        for c2 in conditions[i+1:]:
            key = f"{c1}_vs_{c2}"
            layer_divs = []
            for l in range(n_layers):
                h1 = baseline_states_dict[c1][l].squeeze().cpu().numpy()
                h2 = baseline_states_dict[c2][l].squeeze().cpu().numpy()
                cos = float(np.dot(h1, h2) / (np.linalg.norm(h1) * np.linalg.norm(h2) + 1e-12))
                l2 = float(np.linalg.norm(h1 - h2))
                layer_divs.append({
                    "layer": l,
                    "cosine_distance": 1.0 - cos,
                    "l2_distance": l2,
                })
            divergences[key] = layer_divs

    return divergences


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    print(f"Model loaded: {n_layers} layers")
    print(f"Perturbation directions: {K_PERTURBATIONS}, epsilon: {EPSILON}\n")

    all_results = {}

    for cond_name, preamble in CONDITIONS.items():
        print(f"{'='*60}")
        print(f"CONDITION: {cond_name}")
        print(f"{'='*60}\n")

        cond_results = []
        baseline_states_per_query = []

        for qi, query in enumerate(QUERIES):
            messages = [
                {"role": "system", "content": preamble},
                {"role": "user", "content": query},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            print(f"  Query {qi+1}/{len(QUERIES)}: {query[:50]}...")
            layer_metrics, seq_len, d, baseline_states = compute_spectral_profile(
                model, tokenizer, text
            )

            baseline_states_per_query.append(baseline_states)

            # Print key layers
            for li_idx, m in enumerate(layer_metrics):
                if m["layer"] in [1, 7, 14, 15, 20, 21, 24, 28]:
                    print(f"    L{m['layer']:2d}: ρ={m['rho_median']:.4f} "
                          f"IQR={m['rho_iqr']:.4f} "
                          f"res={m['residual_fraction']:.4f} "
                          f"cos={m['cosine_similarity']:.4f} "
                          f"cum={m['cumulative_amplification']:.2f}")

            cond_results.append({
                "query": query,
                "seq_len": seq_len,
                "hidden_dim": d,
                "layers": layer_metrics,
            })
            print()

        all_results[cond_name] = {
            "query_results": cond_results,
        }

    # Cross-condition divergence (using last query's hidden states)
    print(f"\n{'='*60}")
    print("CROSS-CONDITION DIVERGENCE (last query)")
    print(f"{'='*60}\n")

    # Collect last-query baseline states
    last_query_states = {}
    for cond_name in CONDITIONS:
        # Recompute baselines for the last query (cheaper than storing all)
        messages = [
            {"role": "system", "content": CONDITIONS[cond_name]},
            {"role": "user", "content": QUERIES[-1]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        states, _ = capture_hidden_states(model, tokenizer, text)
        last_query_states[cond_name] = states

    divergences = compute_cross_condition_divergence(last_query_states)
    all_results["cross_condition_divergence"] = {}
    for key, divs in divergences.items():
        all_results["cross_condition_divergence"][key] = divs
        print(f"  {key}:")
        for d_item in divs:
            if d_item["layer"] in [0, 7, 14, 15, 20, 21, 24, 27]:
                print(f"    L{d_item['layer']:2d}: cos_dist={d_item['cosine_distance']:.6f} "
                      f"l2={d_item['l2_distance']:.2f}")
        print()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: Spectral radius by zone (averaged across queries)")
    print(f"{'='*60}\n")

    zones = {
        "early (L1-14)": list(range(1, 15)),
        "transition (L15-20)": list(range(15, 21)),
        "responsive (L21-28)": list(range(21, 29)),
    }

    for cond_name in CONDITIONS:
        print(f"  {cond_name}:")
        for zone_name, zone_layers in zones.items():
            rhos = []
            iqrs = []
            res_fracs = []
            for qr in all_results[cond_name]["query_results"]:
                for m in qr["layers"]:
                    if m["layer"] in zone_layers:
                        rhos.append(m["rho_median"])
                        iqrs.append(m["rho_iqr"])
                        res_fracs.append(m["residual_fraction"])

            if rhos:
                print(f"    {zone_name}: ρ={np.mean(rhos):.4f}±{np.std(rhos):.4f} "
                      f"IQR={np.mean(iqrs):.4f} "
                      f"res_frac={np.mean(res_fracs):.4f}±{np.std(res_fracs):.4f}")
        print()

    # Per-layer sparkline
    print(f"\n{'='*60}")
    print("PER-LAYER SPECTRAL RADIUS (ρ, averaged across queries)")
    print(f"{'='*60}\n")

    for cond_name in CONDITIONS:
        print(f"  {cond_name}:")
        for l in range(1, n_layers + 1):
            rhos = []
            for qr in all_results[cond_name]["query_results"]:
                for m in qr["layers"]:
                    if m["layer"] == l:
                        rhos.append(m["rho_median"])
            if rhos:
                mean_r = np.mean(rhos)
                bar_len = max(0, int((mean_r - 0.5) * 80))
                bar = "█" * min(bar_len, 60)
                zone = "E" if l <= 14 else ("T" if l <= 20 else "R")
                print(f"    L{l:2d} [{zone}]: {mean_r:.4f} {bar}")
        print()

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outpath = RESULTS_DIR / f"spectral_radius_profile_{ts}.json"

    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nSaved: {outpath}")
    print(f"Total: {len(CONDITIONS)} conditions × {len(QUERIES)} queries "
          f"× {n_layers} layers × {K_PERTURBATIONS} perturbations")


if __name__ == "__main__":
    main()
