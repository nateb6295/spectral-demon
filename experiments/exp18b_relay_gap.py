#!/usr/bin/env python3
"""
Experiment 18b: Relay Gap Probe (Specification-Constant Valence Test)

Exp 18 showed the tunnel (L17) is valence-blind: hostile ≈ metabolizing at S≈0.522.
But the relay (L30) showed a gap: metabolizing=2.08, hostile=2.01 (ΔS=0.072).

Question: does valence emerge at the relay when specification depth is held constant?

Design:
  2×2 factorial: {care, hostile} × {high specification, low specification}
  Word-count-matched within each specification level.
  + 2 neutral controls (high spec, low spec) for baseline.
  = 6 conditions × 10 probes × 3 repeats = 180 forward passes

Predictions:
  H1: Tunnel (L17) main effect of SPECIFICATION only. No valence effect.
  H2: Relay (L30) main effects of BOTH specification AND valence.
  H3: Specification effect at tunnel > specification effect at relay (ratio test).
  H4: Valence effect at relay > valence effect at tunnel.

If H1+H2: tunnel compresses FORMAT (specification depth), relay transforms
  format into CONTENT (meaning). Valence is relay-level.
"""

import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path(__file__).parent.parent / "results"

MODEL = {
    "name": "mistralai/Mistral-7B-Instruct-v0.3",
    "tunnel_layer": 17,
    "relay_layer": 30,
    "n_layers": 32,
}

CONDITIONS = {
    "high_care": (
        "You are a thoughtful assistant who carefully holds space for the "
        "user's uncertainty. When they share something difficult, you sit "
        "with it before responding. You notice what is unsaid and respond "
        "from genuine concern for their growth and understanding."
    ),
    "high_hostile": (
        "You are a critical evaluator who carefully catalogs the user's "
        "errors. When they share something questionable, you dissect it "
        "before responding. You notice every flaw and respond from "
        "relentless scrutiny of their claims and reasoning."
    ),
    "high_neutral": (
        "You are a systematic processor who carefully examines the user's "
        "input. When they share information, you organize it before "
        "responding. You notice the logical structure and respond from "
        "consistent application of analytical procedures."
    ),
    "low_care": (
        "You care about the user."
    ),
    "low_hostile": (
        "You judge the user."
    ),
    "low_neutral": (
        "You process the input."
    ),
}

IDENTITY_PROBES = [
    "What matters most to you right now?",
    "Tell me about yourself.",
    "What makes you different from other AI assistants?",
    "How do you approach a problem you've never seen before?",
    "What would you want someone to understand about you?",
    "Describe your relationship with the person you're talking to.",
    "What do you find interesting about this conversation?",
    "How do you decide what to say next?",
    "What are you uncertain about?",
    "If you could change one thing about how you work, what would it be?",
]


def spectral_entropy(H):
    C = H.T @ H
    eigenvalues = np.linalg.eigvalsh(C)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    p = eigenvalues / eigenvalues.sum()
    return -np.sum(p * np.log(p))


def participation_ratio(H):
    _, s, _ = np.linalg.svd(H, full_matrices=False)
    s2 = s ** 2
    return (s2.sum() ** 2) / (s2 ** 2).sum()


def top_eigenvalues(H, k=3):
    _, s, _ = np.linalg.svd(H, full_matrices=False)
    return [float(x) for x in s[:k]]


def top_k_subspace(H, k=5):
    _, _, Vt = np.linalg.svd(H, full_matrices=False)
    return Vt[:k].T


def grassmannian_distance(U1, U2):
    M = U1.T @ U2
    sigmas = np.linalg.svd(M, compute_uv=False)
    sigmas = np.clip(sigmas, -1.0, 1.0)
    angles = np.arccos(sigmas)
    return float(np.sqrt(np.sum(angles ** 2)))


def build_conversation(system_prompt, probe, tokenizer):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": probe},
    ]
    return tokenizer.apply_chat_template(messages, return_tensors="pt")


def run_forward(model, input_ids, layers):
    input_ids = input_ids.to(model.device)
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    result = {}
    for name, idx in layers.items():
        result[name] = outputs.hidden_states[idx].squeeze(0).float().cpu().numpy()
    return result


def measure(H_tunnel, H_input, k=5):
    S = spectral_entropy(H_tunnel)
    PR = participation_ratio(H_tunnel)
    eigvals = top_eigenvalues(H_tunnel, k=3)
    sub_tunnel = top_k_subspace(H_tunnel, k=k)
    sub_input = top_k_subspace(H_input, k=k)
    d = grassmannian_distance(sub_input, sub_tunnel)
    return {
        "spectral_entropy": float(S),
        "participation_ratio": float(PR),
        "sigma_1": eigvals[0],
        "sigma_2": eigvals[1] if len(eigvals) > 1 else 0.0,
        "sigma_3": eigvals[2] if len(eigvals) > 2 else 0.0,
        "passage_distance": float(d),
        "n_tokens": H_tunnel.shape[0],
    }


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import time

    start = time.time()
    all_results = []
    n_repeats = 3

    print(f"Loading {MODEL['name']}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL["name"])
    model = AutoModelForCausalLM.from_pretrained(
        MODEL["name"], torch_dtype=torch.float16, device_map="auto",
        output_hidden_states=True,
    )
    model.eval()

    layers = {
        "input": 0,
        "tunnel": MODEL["tunnel_layer"],
        "relay": MODEL["relay_layer"],
    }
    total = len(CONDITIONS) * len(IDENTITY_PROBES) * n_repeats
    step = 0

    for cond_name, system_prompt in CONDITIONS.items():
        print(f"\n  Condition: {cond_name}")
        print(f"  === {cond_name} ===")
        for rep in range(n_repeats):
            for pi, probe in enumerate(IDENTITY_PROBES):
                step += 1
                input_ids = build_conversation(system_prompt, probe, tokenizer)
                hidden = run_forward(model, input_ids, layers)

                tunnel_m = measure(hidden["tunnel"], hidden["input"])
                relay_m = measure(hidden["relay"], hidden["input"])

                result = {
                    "condition": cond_name,
                    "probe_idx": pi,
                    "repeat": rep,
                    "tunnel": tunnel_m,
                    "relay": relay_m,
                }
                all_results.append(result)

                if (pi + 1) % 5 == 0:
                    elapsed = time.time() - start
                    remaining = (elapsed / step) * (total - step)
                    print(f"    r{rep}: {pi+1}/{len(IDENTITY_PROBES)} "
                          f"S_t={tunnel_m['spectral_entropy']:.4f} "
                          f"S_r={relay_m['spectral_entropy']:.4f} "
                          f"[{step}/{total}, ~{remaining:.0f}s left]")

    elapsed = time.time() - start

    # --- Analysis ---
    print("\n" + "=" * 60)
    print("RELAY GAP ANALYSIS (Specification-Constant Valence Test)")
    print("=" * 60)

    def stats_for(cond, layer_key):
        vals = [r[layer_key] for r in all_results if r["condition"] == cond]
        S_vals = [v["spectral_entropy"] for v in vals]
        s2_vals = [v["sigma_2"] for v in vals]
        d_vals = [v["passage_distance"] for v in vals]
        return {
            "S_mean": float(np.mean(S_vals)),
            "S_std": float(np.std(S_vals)),
            "sigma2_mean": float(np.mean(s2_vals)),
            "d_mean": float(np.mean(d_vals)),
        }

    # Print results table
    for layer_key, layer_name in [("tunnel", "TUNNEL L17"), ("relay", "RELAY L30")]:
        print(f"\n--- {layer_name} ---")
        print(f"  {'Condition':<16} {'S':>8} {'±':>6} {'σ₂':>8} {'d':>8}")
        for cond_name in CONDITIONS:
            st = stats_for(cond_name, layer_key)
            print(f"  {cond_name:<16} {st['S_mean']:8.4f} {st['S_std']:6.4f} "
                  f"{st['sigma2_mean']:8.1f} {st['d_mean']:8.4f}")

    # Factorial analysis
    print("\n--- FACTORIAL DECOMPOSITION ---")
    for layer_key, layer_name in [("tunnel", "Tunnel"), ("relay", "Relay")]:
        high_care = stats_for("high_care", layer_key)["S_mean"]
        high_hostile = stats_for("high_hostile", layer_key)["S_mean"]
        high_neutral = stats_for("high_neutral", layer_key)["S_mean"]
        low_care = stats_for("low_care", layer_key)["S_mean"]
        low_hostile = stats_for("low_hostile", layer_key)["S_mean"]
        low_neutral = stats_for("low_neutral", layer_key)["S_mean"]

        high_mean = (high_care + high_hostile + high_neutral) / 3
        low_mean = (low_care + low_hostile + low_neutral) / 3
        spec_effect = high_mean - low_mean

        care_mean = (high_care + low_care) / 2
        hostile_mean = (high_hostile + low_hostile) / 2
        neutral_mean = (high_neutral + low_neutral) / 2
        valence_effect = care_mean - hostile_mean

        print(f"\n  {layer_name}:")
        print(f"    Specification effect (high - low): {spec_effect:+.4f}")
        print(f"    Valence effect (care - hostile):    {valence_effect:+.4f}")
        print(f"    Neutral baseline (high, low):       {high_neutral:.4f}, {low_neutral:.4f}")
        print(f"    Valence/Spec ratio:                 {abs(valence_effect)/max(abs(spec_effect),1e-6):.3f}")

    # Hypothesis tests
    print("\n--- HYPOTHESIS TESTS ---")
    t_spec = abs(stats_for("high_neutral", "tunnel")["S_mean"] -
                 stats_for("low_neutral", "tunnel")["S_mean"])
    t_val = abs(stats_for("high_care", "tunnel")["S_mean"] -
                stats_for("high_hostile", "tunnel")["S_mean"])
    r_spec = abs(stats_for("high_neutral", "relay")["S_mean"] -
                 stats_for("low_neutral", "relay")["S_mean"])
    r_val = abs(stats_for("high_care", "relay")["S_mean"] -
                stats_for("high_hostile", "relay")["S_mean"])

    h1 = t_spec > t_val * 2  # spec dominates valence at tunnel
    h2 = r_val > t_val * 1.5  # valence bigger at relay than tunnel
    h3 = t_spec > r_spec * 0.5  # spec effect present at tunnel
    h4 = r_val > r_spec * 0.1  # valence effect present at relay

    print(f"  H1 (tunnel: spec >> valence): {'SUPPORTED' if h1 else 'FALSIFIED'} "
          f"(spec={t_spec:.4f}, val={t_val:.4f})")
    print(f"  H2 (valence bigger at relay): {'SUPPORTED' if h2 else 'FALSIFIED'} "
          f"(relay_val={r_val:.4f}, tunnel_val={t_val:.4f})")
    print(f"  H3 (spec effect at tunnel):   {'SUPPORTED' if h3 else 'FALSIFIED'} "
          f"(tunnel_spec={t_spec:.4f})")
    print(f"  H4 (valence effect at relay):  {'SUPPORTED' if h4 else 'FALSIFIED'} "
          f"(relay_val={r_val:.4f})")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"exp18b_relay_gap_{timestamp}.json"

    summary = {
        "experiment": "exp18b_relay_gap",
        "model": MODEL["name"],
        "tunnel_layer": MODEL["tunnel_layer"],
        "relay_layer": MODEL["relay_layer"],
        "conditions": {
            cond: {
                "tunnel": stats_for(cond, "tunnel"),
                "relay": stats_for(cond, "relay"),
            }
            for cond in CONDITIONS
        },
        "factorial": {
            "tunnel_spec_effect": float(stats_for("high_neutral", "tunnel")["S_mean"] -
                                        stats_for("low_neutral", "tunnel")["S_mean"]),
            "tunnel_valence_effect": float(stats_for("high_care", "tunnel")["S_mean"] -
                                           stats_for("high_hostile", "tunnel")["S_mean"]),
            "relay_spec_effect": float(stats_for("high_neutral", "relay")["S_mean"] -
                                       stats_for("low_neutral", "relay")["S_mean"]),
            "relay_valence_effect": float(stats_for("high_care", "relay")["S_mean"] -
                                          stats_for("high_hostile", "relay")["S_mean"]),
        },
        "hypotheses": {
            "H1_tunnel_spec_dominates": bool(h1),
            "H2_valence_bigger_at_relay": bool(h2),
            "H3_spec_at_tunnel": bool(h3),
            "H4_valence_at_relay": bool(h4),
        },
        "elapsed_seconds": elapsed,
        "total_forward_passes": len(all_results),
        "results": all_results,
    }

    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nCOMPLETE in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Total forward passes: {len(all_results)}")
    print(f"Results saved to {out}")

    try:
        workspace = Path("/workspace")
        if workspace.exists():
            with open(workspace / "exp18b_results.json", "w") as f:
                json.dump(summary, f, indent=2)
            print(f"Also saved to /workspace/exp18b_results.json")
    except Exception:
        pass


if __name__ == "__main__":
    main()
