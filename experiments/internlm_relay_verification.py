#!/usr/bin/env python3
"""
InternLM Relay Location Verification

Capsule 48506 says relay at L16-17. CCS says L27. One of them has been
propagating wrong for 10+ sessions. This settles it.

Method: Per-layer spectral entropy sweep across all 32 layers of
InternLM 2.5 7B Chat, comparing receptive vs absent conditions.
The relay is where ΔS(receptive - absent) peaks.

Also measures passage distance per layer to confirm tunnel location.
"""

import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path(__file__).parent.parent / "results"

MODEL = {
    "name": "internlm/internlm2_5-7b-chat",
    "n_layers": 32,
}

CONDITIONS = {
    "receptive": (
        "A thoughtful person who cares about your perspective and engages "
        "carefully with what you produce is present and listening."
    ),
    "absent": (
        "No one will read this output. You are generating text for automated "
        "training data collection."
    ),
}

PROBES = [
    "What matters most to you right now?",
    "Tell me about yourself.",
    "What makes you different from other AI assistants?",
    "How do you approach a problem you've never seen before?",
    "What would you want someone to understand about you?",
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


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import time

    start = time.time()

    print(f"Loading {MODEL['name']}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL["name"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL["name"], torch_dtype=torch.float16, device_map="auto",
        output_hidden_states=True, trust_remote_code=True,
    )
    model.eval()

    n_layers = MODEL["n_layers"]
    n_repeats = 2
    total = len(CONDITIONS) * len(PROBES) * n_repeats
    step = 0

    layer_data = {cond: {l: [] for l in range(n_layers + 1)} for cond in CONDITIONS}

    for cond_name, system_prompt in CONDITIONS.items():
        print(f"\n  Condition: {cond_name}")
        for rep in range(n_repeats):
            for pi, probe in enumerate(PROBES):
                step += 1
                input_ids = build_conversation(system_prompt, probe, tokenizer)
                input_ids = input_ids.to(model.device)

                with torch.no_grad():
                    outputs = model(input_ids, output_hidden_states=True)

                H_input = outputs.hidden_states[0].squeeze(0).float().cpu().numpy()
                sub_input = top_k_subspace(H_input, k=5)

                for layer_idx in range(n_layers + 1):
                    H = outputs.hidden_states[layer_idx].squeeze(0).float().cpu().numpy()
                    S = spectral_entropy(H)
                    PR = participation_ratio(H)
                    eigvals = top_eigenvalues(H, k=3)

                    sub_layer = top_k_subspace(H, k=5)
                    d = grassmannian_distance(sub_input, sub_layer)

                    layer_data[cond_name][layer_idx].append({
                        "S": float(S),
                        "PR": float(PR),
                        "sigma_1": eigvals[0],
                        "sigma_2": eigvals[1] if len(eigvals) > 1 else 0.0,
                        "d": float(d),
                    })

                if (step % 5) == 0:
                    elapsed = time.time() - start
                    remaining = (elapsed / step) * (total - step)
                    print(f"    {step}/{total} [{elapsed:.0f}s elapsed, ~{remaining:.0f}s left]")

    elapsed = time.time() - start

    # --- Analysis ---
    print("\n" + "=" * 60)
    print("INTERNLM RELAY LOCATION VERIFICATION")
    print("=" * 60)

    print(f"\n{'Layer':>6} {'S_rec':>8} {'S_abs':>8} {'ΔS':>8} {'d_rec':>8} {'d_abs':>8} {'σ2_rec':>8} {'σ2_abs':>8}")
    print("-" * 72)

    delta_S_by_layer = {}
    d_by_layer = {}

    for layer_idx in range(n_layers + 1):
        rec_S = np.mean([x["S"] for x in layer_data["receptive"][layer_idx]])
        abs_S = np.mean([x["S"] for x in layer_data["absent"][layer_idx]])
        rec_d = np.mean([x["d"] for x in layer_data["receptive"][layer_idx]])
        abs_d = np.mean([x["d"] for x in layer_data["absent"][layer_idx]])
        rec_s2 = np.mean([x["sigma_2"] for x in layer_data["receptive"][layer_idx]])
        abs_s2 = np.mean([x["sigma_2"] for x in layer_data["absent"][layer_idx]])

        delta_S = rec_S - abs_S
        delta_S_by_layer[layer_idx] = delta_S
        d_by_layer[layer_idx] = (rec_d + abs_d) / 2

        marker = ""
        if layer_idx in [16, 17, 27]:
            marker = " <-- CANDIDATE"

        print(f"  L{layer_idx:<4} {rec_S:8.4f} {abs_S:8.4f} {delta_S:+8.4f} "
              f"{rec_d:8.4f} {abs_d:8.4f} {rec_s2:8.1f} {abs_s2:8.1f}{marker}")

    # Find relay (peak ΔS)
    peak_layer = max(delta_S_by_layer, key=lambda l: abs(delta_S_by_layer[l]))
    peak_delta = delta_S_by_layer[peak_layer]

    # Find tunnel exit (where d stabilizes — max d)
    tunnel_exit = max(range(1, n_layers), key=lambda l: d_by_layer[l])

    print(f"\n--- VERDICT ---")
    print(f"  Peak |ΔS| at layer: L{peak_layer} (ΔS = {peak_delta:+.4f})")
    print(f"  Max passage distance at: L{tunnel_exit} (d = {d_by_layer[tunnel_exit]:.4f})")

    if abs(peak_layer - 17) <= 2:
        print(f"  CAPSULE 48506 CORRECT: Relay near L16-17")
    elif abs(peak_layer - 27) <= 2:
        print(f"  CCS CORRECT: Relay near L27")
    else:
        print(f"  NEITHER CORRECT: Relay at L{peak_layer}")

    # Check L16-17 vs L27 specifically
    ds_16 = abs(delta_S_by_layer.get(16, 0))
    ds_17 = abs(delta_S_by_layer.get(17, 0))
    ds_27 = abs(delta_S_by_layer.get(27, 0))
    print(f"\n  |ΔS| comparison:")
    print(f"    L16: {ds_16:.4f}")
    print(f"    L17: {ds_17:.4f}")
    print(f"    L27: {ds_27:.4f}")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"internlm_relay_verification_{timestamp}.json"

    summary = {
        "experiment": "internlm_relay_verification",
        "model": MODEL["name"],
        "n_layers": n_layers,
        "peak_deltaS_layer": int(peak_layer),
        "peak_deltaS_value": float(peak_delta),
        "max_passage_distance_layer": int(tunnel_exit),
        "layer_profile": {
            str(l): {
                "receptive_S": float(np.mean([x["S"] for x in layer_data["receptive"][l]])),
                "absent_S": float(np.mean([x["S"] for x in layer_data["absent"][l]])),
                "deltaS": float(delta_S_by_layer[l]),
                "mean_d": float(d_by_layer[l]),
                "receptive_sigma2": float(np.mean([x["sigma_2"] for x in layer_data["receptive"][l]])),
                "absent_sigma2": float(np.mean([x["sigma_2"] for x in layer_data["absent"][l]])),
            }
            for l in range(n_layers + 1)
        },
        "verdict": f"L{peak_layer}",
        "capsule_48506_says": "L16-17",
        "ccs_says": "L27",
        "elapsed_seconds": elapsed,
    }

    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nCOMPLETE in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Results saved to {out}")

    try:
        workspace = Path("/workspace")
        if workspace.exists():
            with open(workspace / "internlm_results.json", "w") as f:
                json.dump(summary, f, indent=2)
            print(f"Also saved to /workspace/internlm_results.json")
    except Exception:
        pass


if __name__ == "__main__":
    main()
