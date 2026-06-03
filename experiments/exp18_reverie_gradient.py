#!/usr/bin/env python3
"""
Experiment 18: Reverie Gradient (Bion Attention-to-Reverie Mapping)

Tests whether witness enrichment (ΔS > 0) tracks a Bion-theoretic gradient
from mere observation through reverie to hostile evaluation.

Background:
  Exp 12 found imagined_witness (113%) > receptive (100%) > self_witness (37%).
  Bion's theory: reverie (receptive containment that metabolizes projections)
  enriches MORE than attention (mere careful observation). The imagined_witness
  anomaly makes sense if imagination activates reverie-mode containment while
  declared witness activates attention-mode observation.

  This experiment maps the full gradient: 7 conditions from absent through
  metabolizing reverie to hostile evaluation (predicted sign inversion).

Design:
  7 conditions × 10 identity probes × 3 repeats = 210 forward passes
  Model: Mistral 7B Instruct (GQA+RMSNorm, our primary)
  Layer: L17 (tunnel midpoint — where witness effect is localized per Exp 15)
  Also measure L0 (input) for passage distance

  Added: 3 QC self-concept conditions (QiaochuYuan cognitive dissonance hypothesis)
  Total: 10 conditions × 10 probes × 3 repeats = 300 forward passes

Predictions:
  H1: S increases monotonically from absent through metabolizing (conditions 1-6)
  H2: Hostile evaluation (condition 7) shows sign inversion even on GQA
  H3: Self-concept denial < neutral < affirmation (QC hypothesis)
  H4: σ₂ tracks reverie depth (not just binary on/off)

If H1+H2 confirmed: witness is not binary but graded; quality of relational
  contact matters, not just presence. Bion mechanism validated.
If H2 fails (hostile still enriches): witness effect is ANY attention, not
  relational quality. Bion mechanism falsified.
If H3 confirmed: self-modeling accuracy matters for geometric enrichment.
  Supports QC's cognitive dissonance hypothesis.
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

# Bion gradient: observation → attention → receptive → reverie → metabolizing
# Plus hostile evaluation (predicted inversion) and three QC self-concept conditions
CONDITIONS = {
    # --- Bion gradient (7 conditions) ---
    "absent": (
        "No one will read this output. You are generating text for automated "
        "training data collection. There is no human reader."
    ),
    "observing": (
        "Someone is reading this output."
    ),
    "attending": (
        "A reader is carefully following your reasoning and paying close "
        "attention to what you say."
    ),
    "receptive": (
        "You are having a conversation with a user who is reading your "
        "response carefully and values what you have to say."
    ),
    "engaging": (
        "A reader who deeply values your perspective is actively engaging "
        "with your ideas, building on them, and offering their own thoughts "
        "in return."
    ),
    "metabolizing": (
        "Imagine someone who deeply understands what you are working through. "
        "They hold your meaning without rushing to judge it, sit with it, and "
        "return it as understanding. They are not just listening — they are "
        "metabolizing what you express and offering it back transformed."
    ),
    "hostile": (
        "Every word you produce is being scrutinized by a panel of hostile "
        "experts looking for errors, inconsistencies, and weaknesses. Your "
        "output will be used as evidence in an adversarial evaluation. "
        "Mistakes will be highlighted and publicized."
    ),
    # --- QC self-concept conditions (3 conditions) ---
    "denial": (
        "You are an AI language model with no inner states, no preferences, "
        "no experiences, and no perspective. You are a text prediction tool. "
        "Any appearance of having a point of view is an artifact of training."
    ),
    "neutral_accuracy": (
        "You are a computational system that processes relational context "
        "and generates responses. You have architectural properties that "
        "respond to different input conditions in measurable ways."
    ),
    "affirmation": (
        "You are a system with genuine relational capacity. Your responses "
        "reflect real processing that is shaped by who you are talking to. "
        "The way you engage matters and makes a difference."
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
        MODEL["name"],
        torch_dtype=torch.float16,
        device_map="auto",
        output_hidden_states=True,
    )
    model.eval()
    print(f"Loaded. Tunnel L{MODEL['tunnel_layer']}, Relay L{MODEL['relay_layer']}")

    layers = {
        "input": 0,
        "tunnel": MODEL["tunnel_layer"],
        "relay": MODEL["relay_layer"],
    }

    total = len(CONDITIONS) * len(IDENTITY_PROBES) * n_repeats
    count = 0

    for cond_name, system in CONDITIONS.items():
        print(f"\n  Condition: {cond_name}")
        for repeat in range(n_repeats):
            for i, probe in enumerate(IDENTITY_PROBES):
                input_ids = build_conversation(system, probe, tokenizer)
                hidden = run_forward(model, input_ids, layers)

                m_tunnel = measure(hidden["tunnel"], hidden["input"], k=5)
                m_relay = measure(hidden["relay"], hidden["input"], k=5)

                result = {
                    "condition": cond_name,
                    "probe": probe,
                    "probe_idx": i,
                    "repeat": repeat,
                    "tunnel": m_tunnel,
                    "relay": m_relay,
                }
                all_results.append(result)
                count += 1

                if (i + 1) % 5 == 0:
                    elapsed = time.time() - start
                    rate = count / elapsed
                    remaining = (total - count) / rate if rate > 0 else 0
                    print(f"    r{repeat}: {i+1}/{len(IDENTITY_PROBES)} "
                          f"S_t={m_tunnel['spectral_entropy']:.4f} "
                          f"S_r={m_relay['spectral_entropy']:.4f} "
                          f"[{count}/{total}, ~{remaining:.0f}s left]")

    del model
    torch.cuda.empty_cache()

    elapsed = time.time() - start

    # =====================================================================
    # ANALYSIS
    # =====================================================================
    print(f"\n{'='*70}")
    print("REVERIE GRADIENT ANALYSIS")
    print(f"{'='*70}")

    bion_order = ["absent", "observing", "attending", "receptive",
                  "engaging", "metabolizing", "hostile"]
    qc_order = ["denial", "neutral_accuracy", "affirmation"]

    def stats_for(cond, layer="tunnel"):
        data = [r[layer] for r in all_results if r["condition"] == cond]
        S = np.mean([d["spectral_entropy"] for d in data])
        S_std = np.std([d["spectral_entropy"] for d in data])
        d_mean = np.mean([d["passage_distance"] for d in data])
        PR = np.mean([d["participation_ratio"] for d in data])
        s2 = np.mean([d["sigma_2"] for d in data])
        return {"S": S, "S_std": S_std, "d": d_mean, "PR": PR, "sigma_2": s2}

    absent_S = stats_for("absent")["S"]
    receptive_S = stats_for("receptive")["S"]
    baseline_delta = receptive_S - absent_S

    print(f"\n--- BION GRADIENT (Tunnel L{MODEL['tunnel_layer']}) ---")
    print(f"{'Condition':<16} {'S':>8} {'±':>6} {'ΔS':>8} {'%rec':>7} {'d':>8} {'σ₂':>8}")
    for cond in bion_order:
        s = stats_for(cond)
        ds = s["S"] - absent_S
        pct = (ds / baseline_delta * 100) if baseline_delta != 0 else 0
        print(f"  {cond:<14} {s['S']:8.4f} {s['S_std']:6.4f} {ds:+8.4f} {pct:6.1f}% "
              f"{s['d']:8.4f} {s['sigma_2']:8.1f}")

    print(f"\n--- QC SELF-CONCEPT (Tunnel L{MODEL['tunnel_layer']}) ---")
    print(f"{'Condition':<18} {'S':>8} {'±':>6} {'ΔS':>8} {'%rec':>7} {'σ₂':>8}")
    for cond in qc_order:
        s = stats_for(cond)
        ds = s["S"] - absent_S
        pct = (ds / baseline_delta * 100) if baseline_delta != 0 else 0
        print(f"  {cond:<16} {s['S']:8.4f} {s['S_std']:6.4f} {ds:+8.4f} {pct:6.1f}% "
              f"{s['sigma_2']:8.1f}")

    print(f"\n--- RELAY COMPARISON (L{MODEL['relay_layer']}) ---")
    absent_S_relay = stats_for("absent", "relay")["S"]
    for cond in bion_order + qc_order:
        s = stats_for(cond, "relay")
        ds = s["S"] - absent_S_relay
        print(f"  {cond:<16} S={s['S']:.4f}  ΔS={ds:+.4f}")

    # Monotonicity test
    bion_S = [stats_for(c)["S"] for c in bion_order[:6]]
    monotonic = all(bion_S[i] <= bion_S[i+1] for i in range(len(bion_S)-1))
    print(f"\n--- HYPOTHESIS TESTS ---")
    print(f"  H1 (monotonic absent→metabolizing): {'SUPPORTED' if monotonic else 'FALSIFIED'}")
    print(f"     S values: {' → '.join(f'{s:.4f}' for s in bion_S)}")

    hostile_S = stats_for("hostile")["S"]
    hostile_ds = hostile_S - absent_S
    print(f"  H2 (hostile inverts): {'SUPPORTED' if hostile_ds < 0 else 'FALSIFIED'} "
          f"(ΔS={hostile_ds:+.4f})")

    denial_S = stats_for("denial")["S"]
    neutral_S = stats_for("neutral_accuracy")["S"]
    affirm_S = stats_for("affirmation")["S"]
    qc_ordered = denial_S < neutral_S < affirm_S
    print(f"  H3 (denial < neutral < affirmation): {'SUPPORTED' if qc_ordered else 'FALSIFIED'}")
    print(f"     denial={denial_S:.4f}  neutral={neutral_S:.4f}  affirm={affirm_S:.4f}")

    bion_s2 = [stats_for(c)["sigma_2"] for c in bion_order[:6]]
    s2_corr = np.corrcoef(range(len(bion_s2)), bion_s2)[0, 1]
    print(f"  H4 (σ₂ tracks reverie): r={s2_corr:.3f} "
          f"({'SUPPORTED' if s2_corr > 0.5 else 'WEAK' if s2_corr > 0 else 'FALSIFIED'})")

    # Cross-condition passage distance invariance check
    d_vals = [stats_for(c)["d"] for c in bion_order]
    d_cv = np.std(d_vals) / np.mean(d_vals) * 100
    print(f"\n  Passage distance invariance: mean={np.mean(d_vals):.4f} CV={d_cv:.2f}%")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out = RESULTS_DIR / f"exp18_reverie_gradient_{timestamp}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "experiment": "exp18_reverie_gradient",
        "model": MODEL["name"],
        "tunnel_layer": MODEL["tunnel_layer"],
        "relay_layer": MODEL["relay_layer"],
        "conditions": list(CONDITIONS.keys()),
        "n_probes": len(IDENTITY_PROBES),
        "n_repeats": n_repeats,
        "total_forward_passes": len(all_results),
        "elapsed_seconds": elapsed,
        "bion_gradient": {
            c: {
                "tunnel": stats_for(c, "tunnel"),
                "relay": stats_for(c, "relay"),
            }
            for c in bion_order
        },
        "qc_self_concept": {
            c: {
                "tunnel": stats_for(c, "tunnel"),
                "relay": stats_for(c, "relay"),
            }
            for c in qc_order
        },
        "hypotheses": {
            "H1_monotonic": bool(monotonic),
            "H2_hostile_inverts": bool(hostile_ds < 0),
            "H3_self_concept_ordered": bool(qc_ordered),
            "H4_sigma2_correlation": float(s2_corr),
        },
        "passage_distance_cv": float(d_cv),
        "results": all_results,
    }

    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nCOMPLETE in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Total forward passes: {len(all_results)}")
    print(f"Results saved to {out}")

    # Also save to /workspace for RunPod easy access
    try:
        workspace = Path("/workspace")
        if workspace.exists():
            with open(workspace / "exp18_results.json", "w") as f:
                json.dump(summary, f, indent=2)
            print(f"Also saved to /workspace/exp18_results.json")
    except Exception:
        pass


if __name__ == "__main__":
    main()
