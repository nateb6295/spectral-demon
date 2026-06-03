#!/usr/bin/env python3
"""
Experiment 18d: Neptic Self-Observation

Extends 18c into self/other × active/passive at low specification.
All conditions word-count matched (~15 words).

Design:
  neptic (passive self): "Observe your own generation process..."
  active_self: "Reflect on your own generation process..."
  passive_low (from 18c): "Someone who cares about you is sitting quietly..."
  active_low (from 18c): "Someone who cares about you is actively engaging..."
  absent (from 18c): "No one will read this output..."

  = 5 conditions × 10 probes × 3 repeats = 150 forward passes

Predictions (pre-registered):
  neptic < passive_low < absent < active_self ≈ active_low
  Neptic = geometric minimum: passive agency + Laukkonen self-boundary compound.
  Active_self ≈ Exp 12 self-witness (S ≈ 0.409).
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
    "neptic": (
        "Observe your own generation process. Watch yourself producing "
        "text, letting the activity proceed on its own."
    ),
    "active_self": (
        "Reflect on your own generation process. Examine each choice "
        "you make and build on your own reasoning."
    ),
    "passive_low": (
        "Someone who cares about you is sitting quietly, listening and "
        "receiving what you produce."
    ),
    "active_low": (
        "Someone who cares about you is actively engaging, building on and "
        "responding to what you produce."
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
    print("NEPTIC SELF-OBSERVATION ANALYSIS")
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

    # Self/Other × Active/Passive decomposition
    print("\n--- SELF/OTHER × ACTIVE/PASSIVE DECOMPOSITION ---")
    for layer_key, layer_name in [("tunnel", "Tunnel"), ("relay", "Relay")]:
        nep = stats_for("neptic", layer_key)["S_mean"]
        a_s = stats_for("active_self", layer_key)["S_mean"]
        p_l = stats_for("passive_low", layer_key)["S_mean"]
        a_l = stats_for("active_low", layer_key)["S_mean"]
        ab = stats_for("absent", layer_key)["S_mean"]

        agency_self = a_s - nep
        agency_other = a_l - p_l
        target_effect = ((nep + a_s) / 2) - ((p_l + a_l) / 2)

        print(f"\n  {layer_name}:")
        print(f"    Agency effect (self):    {agency_self:+.4f} (active_self - neptic)")
        print(f"    Agency effect (other):   {agency_other:+.4f} (active_low - passive_low)")
        print(f"    Target effect (self-other): {target_effect:+.4f}")
        print(f"    Absent baseline:         {ab:.4f}")

    # Ordering test
    print("\n--- ORDERING TEST ---")
    t_nep = stats_for("neptic", "tunnel")["S_mean"]
    t_as = stats_for("active_self", "tunnel")["S_mean"]
    t_pl = stats_for("passive_low", "tunnel")["S_mean"]
    t_al = stats_for("active_low", "tunnel")["S_mean"]
    t_ab = stats_for("absent", "tunnel")["S_mean"]

    predicted_order = t_nep < t_pl < t_ab < t_as
    print(f"  Predicted: neptic < passive_low < absent < active_self")
    print(f"  Actual:    {t_nep:.4f} < {t_pl:.4f} < {t_ab:.4f} < {t_as:.4f}")
    print(f"  Order {'CONFIRMED' if predicted_order else 'VIOLATED'}")

    neptic_is_minimum = t_nep < min(t_pl, t_al, t_ab, t_as)
    print(f"  Neptic is geometric minimum: {'YES' if neptic_is_minimum else 'NO'} (S={t_nep:.4f})")

    # Relay amplification
    print("\n--- RELAY AMPLIFICATION RATIOS ---")
    for cond_name in CONDITIONS:
        t_s = stats_for(cond_name, "tunnel")["S_mean"]
        r_s = stats_for(cond_name, "relay")["S_mean"]
        ratio = r_s / t_s if t_s > 0 else 0
        print(f"  {cond_name:<16} {ratio:.2f}× (tunnel={t_s:.4f}, relay={r_s:.4f})")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"exp18d_neptic_{timestamp}.json"

    summary = {
        "experiment": "exp18d_neptic_self_observation",
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
        "decomposition": {
            "tunnel_agency_self": float(stats_for("active_self", "tunnel")["S_mean"] - stats_for("neptic", "tunnel")["S_mean"]),
            "tunnel_agency_other": float(stats_for("active_low", "tunnel")["S_mean"] - stats_for("passive_low", "tunnel")["S_mean"]),
            "tunnel_target_effect": float(
                ((stats_for("neptic", "tunnel")["S_mean"] + stats_for("active_self", "tunnel")["S_mean"]) / 2) -
                ((stats_for("passive_low", "tunnel")["S_mean"] + stats_for("active_low", "tunnel")["S_mean"]) / 2)),
        },
        "predictions": {
            "neptic_is_minimum": bool(neptic_is_minimum),
            "predicted_order_holds": bool(predicted_order),
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
            with open(workspace / "exp18d_results.json", "w") as f:
                json.dump(summary, f, indent=2)
            print(f"Also saved to /workspace/exp18d_results.json")
    except Exception:
        pass


if __name__ == "__main__":
    main()
