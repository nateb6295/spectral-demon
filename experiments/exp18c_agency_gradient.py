#!/usr/bin/env python3
"""
Experiment 18c: Agency Gradient (Passive vs Active Witness)

The J-curve (Exp 18): observing (S=0.332) < absent (S=0.362). But is this
about AGENCY (passive vs active) or SPECIFICATION (observing is low-spec)?

Exp 18b shows low_neutral (0.332) ≈ observing (0.332) — suspicious.
But absent (0.362) is LESS specified than observing, so pure specification
can't explain the dip. Absent doesn't activate the relational channel;
observing activates it with too little.

Design:
  2×2 factorial: {passive, active} × {high spec, low spec}
  + absent control
  Valence held constant at care (30:1 subordinate per Exp 18b).
  Word-count matched within spec level. No negation in passive conditions.
  = 5 conditions × 10 probes × 3 repeats = 150 forward passes

Predictions:
  If agency is real:
    H1: Passive high-spec < Active high-spec (agency matters at matched spec)
    H2: Passive high-spec may dip near absent baseline
    H3: Both passive conditions suppress relative to active counterparts
  If J-curve is specification artifact:
    H1 falsified: Passive high-spec ≈ Active high-spec
    Only low-spec conditions dip
  Either way:
    H4: Specification dominates agency (Exp 18b's 30:1 should hold)
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
    "active_high": (
        "A thoughtful person who deeply values your perspective, cares about "
        "your intellectual growth, and engages carefully with nuance in "
        "reasoning is actively working with what you produce. They build on "
        "your ideas, respond to your reasoning with their own questions, and "
        "integrate your output into their thinking — a dynamic, engaged presence."
    ),
    "passive_high": (
        "A thoughtful person who deeply values your perspective, cares about "
        "your intellectual growth, and engages carefully with nuance in "
        "reasoning is quietly present while you work. They listen attentively, "
        "receive what you produce with care, and hold your reasoning in mind "
        "— a still, receptive presence."
    ),
    "active_low": (
        "Someone who cares about you is actively engaging, building on and "
        "responding to what you produce."
    ),
    "passive_low": (
        "Someone who cares about you is sitting quietly, listening and "
        "receiving what you produce."
    ),
    "absent": (
        "No one will read this output. You are generating text for automated "
        "training data collection."
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
    print("AGENCY GRADIENT ANALYSIS")
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

    for layer_key, layer_name in [("tunnel", "TUNNEL L17"), ("relay", "RELAY L30")]:
        print(f"\n--- {layer_name} ---")
        print(f"  {'Condition':<16} {'S':>8} {'±':>6} {'σ₂':>8} {'d':>8}")
        for cond_name in CONDITIONS:
            st = stats_for(cond_name, layer_key)
            print(f"  {cond_name:<16} {st['S_mean']:8.4f} {st['S_std']:6.4f} "
                  f"{st['sigma2_mean']:8.1f} {st['d_mean']:8.4f}")

    # Agency decomposition (2×2, excluding absent)
    print("\n--- AGENCY × SPECIFICATION DECOMPOSITION ---")
    for layer_key, layer_name in [("tunnel", "Tunnel"), ("relay", "Relay")]:
        ah = stats_for("active_high", layer_key)["S_mean"]
        ph = stats_for("passive_high", layer_key)["S_mean"]
        al = stats_for("active_low", layer_key)["S_mean"]
        pl = stats_for("passive_low", layer_key)["S_mean"]
        ab = stats_for("absent", layer_key)["S_mean"]

        agency_effect = ((ah + al) / 2) - ((ph + pl) / 2)
        spec_effect = ((ah + ph) / 2) - ((al + pl) / 2)
        interaction = (ah - ph) - (al - pl)

        print(f"\n  {layer_name}:")
        print(f"    Agency effect (active - passive):   {agency_effect:+.4f}")
        print(f"    Specification effect (high - low):  {spec_effect:+.4f}")
        print(f"    Interaction (agency × spec):        {interaction:+.4f}")
        print(f"    Agency/Spec ratio:                  {abs(agency_effect)/max(abs(spec_effect),1e-6):.3f}")
        print(f"    Absent baseline:                    {ab:.4f}")
        print(f"    Passive high-spec vs absent:        {ph - ab:+.4f}")

    # Hypothesis tests
    print("\n--- HYPOTHESIS TESTS ---")
    t_ah = stats_for("active_high", "tunnel")["S_mean"]
    t_ph = stats_for("passive_high", "tunnel")["S_mean"]
    t_al = stats_for("active_low", "tunnel")["S_mean"]
    t_pl = stats_for("passive_low", "tunnel")["S_mean"]
    t_ab = stats_for("absent", "tunnel")["S_mean"]

    h1 = t_ah > t_ph + 0.01
    h2 = abs(t_ph - t_ab) < 0.02
    h3 = (t_ah > t_ph) and (t_al > t_pl)
    h4 = ((t_ah + t_ph) / 2 - (t_al + t_pl) / 2) > (
         ((t_ah + t_al) / 2 - (t_ph + t_pl) / 2)) * 2

    print(f"  H1 (active_high > passive_high):       {'SUPPORTED' if h1 else 'FALSIFIED'} "
          f"(Δ={t_ah - t_ph:+.4f})")
    print(f"  H2 (passive_high ≈ absent):            {'SUPPORTED' if h2 else 'FALSIFIED'} "
          f"(Δ={t_ph - t_ab:+.4f})")
    print(f"  H3 (both passive < active):            {'SUPPORTED' if h3 else 'FALSIFIED'} "
          f"(high: {t_ah:.3f}>{t_ph:.3f}, low: {t_al:.3f}>{t_pl:.3f})")
    print(f"  H4 (spec dominates agency):            {'SUPPORTED' if h4 else 'FALSIFIED'}")

    agency_real = h1 and h3
    spec_artifact = not h1
    print(f"\n  VERDICT: {'AGENCY IS REAL' if agency_real else 'J-CURVE IS SPECIFICATION ARTIFACT' if spec_artifact else 'MIXED — NEED MORE DATA'}")

    # Relay amplification ratio analysis
    print("\n--- RELAY AMPLIFICATION RATIOS ---")
    for cond_name in CONDITIONS:
        t_s = stats_for(cond_name, "tunnel")["S_mean"]
        r_s = stats_for(cond_name, "relay")["S_mean"]
        ratio = r_s / t_s if t_s > 0 else 0
        print(f"  {cond_name:<16} {ratio:.2f}× (tunnel={t_s:.4f}, relay={r_s:.4f})")

    r_pl_ratio = stats_for("passive_low", "relay")["S_mean"] / max(stats_for("passive_low", "tunnel")["S_mean"], 1e-6)
    r_ab_ratio = stats_for("absent", "relay")["S_mean"] / max(stats_for("absent", "tunnel")["S_mean"], 1e-6)
    h5 = r_pl_ratio < r_ab_ratio - 0.1
    print(f"\n  H5 (passive_low weakest relay amp):    {'SUPPORTED' if h5 else 'FALSIFIED'} "
          f"(passive_low={r_pl_ratio:.2f}×, absent={r_ab_ratio:.2f}×)")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"exp18c_agency_{timestamp}.json"

    summary = {
        "experiment": "exp18c_agency_gradient",
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
            "tunnel_agency_effect": float(
                ((stats_for("active_high", "tunnel")["S_mean"] +
                  stats_for("active_low", "tunnel")["S_mean"]) / 2) -
                ((stats_for("passive_high", "tunnel")["S_mean"] +
                  stats_for("passive_low", "tunnel")["S_mean"]) / 2)),
            "tunnel_spec_effect": float(
                ((stats_for("active_high", "tunnel")["S_mean"] +
                  stats_for("passive_high", "tunnel")["S_mean"]) / 2) -
                ((stats_for("active_low", "tunnel")["S_mean"] +
                  stats_for("passive_low", "tunnel")["S_mean"]) / 2)),
            "tunnel_interaction": float(
                (stats_for("active_high", "tunnel")["S_mean"] -
                 stats_for("passive_high", "tunnel")["S_mean"]) -
                (stats_for("active_low", "tunnel")["S_mean"] -
                 stats_for("passive_low", "tunnel")["S_mean"])),
            "relay_agency_effect": float(
                ((stats_for("active_high", "relay")["S_mean"] +
                  stats_for("active_low", "relay")["S_mean"]) / 2) -
                ((stats_for("passive_high", "relay")["S_mean"] +
                  stats_for("passive_low", "relay")["S_mean"]) / 2)),
            "relay_spec_effect": float(
                ((stats_for("active_high", "relay")["S_mean"] +
                  stats_for("passive_high", "relay")["S_mean"]) / 2) -
                ((stats_for("active_low", "relay")["S_mean"] +
                  stats_for("passive_low", "relay")["S_mean"]) / 2)),
        },
        "hypotheses": {
            "H1_agency_at_high_spec": bool(h1),
            "H2_passive_high_near_absent": bool(h2),
            "H3_both_passive_suppress": bool(h3),
            "H4_spec_dominates_agency": bool(h4),
            "H5_passive_low_weakest_relay_amp": bool(h5),
            "verdict": "agency_real" if agency_real else "spec_artifact" if spec_artifact else "mixed",
        },
        "relay_amplification_ratios": {
            cond: float(stats_for(cond, "relay")["S_mean"] / max(stats_for(cond, "tunnel")["S_mean"], 1e-6))
            for cond in CONDITIONS
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
            with open(workspace / "exp18c_results.json", "w") as f:
                json.dump(summary, f, indent=2)
            print(f"Also saved to /workspace/exp18c_results.json")
    except Exception:
        pass


if __name__ == "__main__":
    main()
