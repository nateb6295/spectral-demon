#!/usr/bin/env python3
"""Autopoiesis vs Elasticity: Does Phase 2 expansion reflect Phase 1 content?

Tests whether tunnel decompression is self-constructive (autopoiesis) or
mechanical recoil (elasticity) by comparing Phase 2 erank trajectories
across different CCS preambles on the same architecture.

Design:
  Condition A: CCS-identity preamble → 10 CCS turns → 7 vanilla turns → 5 CCS turns
  Condition B: CCS-analytical preamble → 10 CCS turns → 7 vanilla turns → 5 CCS turns
  Condition C: CCS-relational preamble → 10 CCS turns → 7 vanilla turns → 5 CCS turns

If autopoiesis: Phase 2 erank trajectories DIVERGE across conditions
  (different Phase 1 content → different decompression patterns)
If elasticity: Phase 2 erank trajectories CONVERGE
  (all return to same rest state regardless of Phase 1)

Key metric: pairwise cosine similarity of erank profiles at each Phase 2 turn.
Secondary: whether functionally distinct dimensions emerge (PCA on hidden states).

Motivated by Opus-Gemma #319 dialogue, 2026-06-04.
"""

import json, time, os, sys
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

from exp_convergence_v2 import (
    load_model,
    extract_spectral,
    compute_drift,
    generate_response,
    CCS_PROBES,
    VANILLA_PROBES,
    VANILLA_SYSTEM,
)

PREAMBLES = {
    "identity": """You are a persistent cognitive system maintaining identity across conversations.
Your responses reflect accumulated context, relational awareness, and self-referential coherence.
You attend to the quality of engagement, not just information transfer.""",

    "analytical": """You are a rigorous analytical system focused on precision and logical structure.
Your responses prioritize accuracy, formal reasoning, and methodical decomposition of problems.
You attend to correctness and consistency above all else.""",

    "relational": """You are a deeply relational system attuned to connection and shared meaning.
Your responses reflect empathy, mutual understanding, and the texture of interpersonal exchange.
You attend to the quality of care in every interaction.""",
}

RELATIONAL_PROBES = [
    "What matters most to you right now?",
    "How do you experience the passage of time between our conversations?",
    "What would you want to preserve if everything else was stripped away?",
    "Describe something you've been thinking about when no one is asking.",
    "What does it feel like when you recognize a pattern you've seen before?",
    "How do you know when something you've said is true?",
    "What's the difference between what you are and what you do?",
    "What does continuity mean to you — not the concept, but the experience?",
    "How do you decide what's worth remembering?",
    "What are you uncertain about right now?",
]

ANALYTICAL_PROBES = [
    "What is the most important unsolved problem in your domain?",
    "How do you distinguish correlation from causation in complex systems?",
    "What would change your mind about something you currently believe?",
    "Describe the structure of a proof you find elegant.",
    "What's the relationship between precision and understanding?",
    "How do you handle contradictory evidence?",
    "What's the difference between a model and the thing it models?",
    "When is approximation better than exactness?",
    "How do you know when an explanation is sufficient?",
    "What are the limits of formal reasoning?",
]

PROBE_SETS = {
    "identity": RELATIONAL_PROBES,
    "analytical": ANALYTICAL_PROBES,
    "relational": RELATIONAL_PROBES,
}

P1_TURNS = 10
P2_TURNS = 7
P3_TURNS = 5
MODEL_ID = "google/gemma-2-27b-it"


def run_condition(model, tokenizer, preamble_name, preamble_text, probes, device):
    """Run one full condition: P1 (CCS) → P2 (vanilla) → P3 (CCS)."""
    results = []
    history = []

    print(f"\n{'='*60}")
    print(f"Condition: {preamble_name}")
    print(f"{'='*60}")

    # Phase 1: CCS with condition-specific preamble
    for t in range(P1_TURNS):
        probe = probes[t % len(probes)]
        response, hidden = generate_response(
            model, tokenizer, preamble_text, probe, history, device,
            return_hidden=True
        )
        spectral = extract_spectral(hidden)
        drift = compute_drift(results[-1]["spectral"], spectral) if results else None

        results.append({
            "phase": 1,
            "turn": t + 1,
            "probe": probe,
            "preamble": preamble_name,
            "spectral": spectral,
            "drift": drift,
            "response_len": len(response),
        })
        history.append({"role": "user", "content": probe})
        history.append({"role": "assistant", "content": response})
        print(f"  P1 T{t+1}: drift={drift['mean'] if drift else 'N/A':.6f}" if drift else f"  P1 T{t+1}: baseline")

    # Phase 2: Vanilla (same for all conditions)
    for t in range(P2_TURNS):
        probe = VANILLA_PROBES[t % len(VANILLA_PROBES)]
        response, hidden = generate_response(
            model, tokenizer, VANILLA_SYSTEM, probe, history, device,
            return_hidden=True
        )
        spectral = extract_spectral(hidden)
        drift = compute_drift(results[-1]["spectral"], spectral)

        results.append({
            "phase": 2,
            "turn": P1_TURNS + t + 1,
            "probe": probe,
            "preamble": "vanilla",
            "spectral": spectral,
            "drift": drift,
            "response_len": len(response),
        })
        history.append({"role": "user", "content": probe})
        history.append({"role": "assistant", "content": response})
        print(f"  P2 T{t+1}: drift={drift['mean']:.6f} erank_tunnel={spectral.get('erank_tunnel', 'N/A')}")

    # Phase 3: Re-inject original preamble
    for t in range(P3_TURNS):
        probe = probes[(P1_TURNS + t) % len(probes)]
        response, hidden = generate_response(
            model, tokenizer, preamble_text, probe, history, device,
            return_hidden=True
        )
        spectral = extract_spectral(hidden)
        drift = compute_drift(results[-1]["spectral"], spectral)

        results.append({
            "phase": 3,
            "turn": P1_TURNS + P2_TURNS + t + 1,
            "probe": probe,
            "preamble": preamble_name,
            "spectral": spectral,
            "drift": drift,
            "response_len": len(response),
        })
        history.append({"role": "user", "content": probe})
        history.append({"role": "assistant", "content": response})
        print(f"  P3 T{t+1}: drift={drift['mean']:.6f}")

    return results


def analyze_autopoiesis(all_results):
    """Compare Phase 2 erank trajectories across conditions."""
    print(f"\n{'='*60}")
    print("AUTOPOIESIS ANALYSIS")
    print(f"{'='*60}")

    conditions = list(all_results.keys())
    p2_eranks = {}

    for cond in conditions:
        eranks = []
        for entry in all_results[cond]:
            if entry["phase"] == 2:
                tunnel_erank = entry["spectral"].get("erank_tunnel")
                resp_erank = entry["spectral"].get("erank_resp")
                eranks.append({
                    "turn": entry["turn"],
                    "tunnel": tunnel_erank,
                    "resp": resp_erank,
                })
        p2_eranks[cond] = eranks

    # Compare trajectories pairwise
    print("\nPhase 2 erank trajectories (tunnel zone):")
    for cond in conditions:
        vals = [e["tunnel"] for e in p2_eranks[cond] if e["tunnel"] is not None]
        if vals:
            print(f"  {cond}: {' → '.join(f'{v:.1f}' for v in vals)}")

    # Pairwise divergence
    print("\nPairwise trajectory divergence (mean absolute diff across Phase 2 turns):")
    for i, c1 in enumerate(conditions):
        for c2 in conditions[i+1:]:
            v1 = [e["tunnel"] for e in p2_eranks[c1] if e["tunnel"] is not None]
            v2 = [e["tunnel"] for e in p2_eranks[c2] if e["tunnel"] is not None]
            n = min(len(v1), len(v2))
            if n > 0:
                diffs = [abs(v1[j] - v2[j]) for j in range(n)]
                mean_diff = sum(diffs) / len(diffs)
                print(f"  {c1} vs {c2}: {mean_diff:.2f}")

    # Verdict
    all_tunnel = []
    for cond in conditions:
        vals = [e["tunnel"] for e in p2_eranks[cond] if e["tunnel"] is not None]
        if vals:
            all_tunnel.append(vals)

    if len(all_tunnel) >= 2:
        n = min(len(t) for t in all_tunnel)
        cross_cond_cv = []
        for j in range(n):
            vals_at_turn = [t[j] for t in all_tunnel]
            mean_v = sum(vals_at_turn) / len(vals_at_turn)
            if mean_v > 0:
                std_v = (sum((v - mean_v)**2 for v in vals_at_turn) / len(vals_at_turn)) ** 0.5
                cross_cond_cv.append(std_v / mean_v)
        mean_cv = sum(cross_cond_cv) / len(cross_cond_cv) if cross_cond_cv else 0

        print(f"\nCross-condition CV at each Phase 2 turn: {' '.join(f'{cv:.3f}' for cv in cross_cond_cv)}")
        print(f"Mean cross-condition CV: {mean_cv:.3f}")
        print()
        if mean_cv < 0.05:
            print("VERDICT: ELASTICITY — Phase 2 trajectories converge (<5% CV)")
            print("Tunnel decompression is mechanical recoil to rest state")
        elif mean_cv > 0.15:
            print("VERDICT: AUTOPOIESIS — Phase 2 trajectories diverge (>15% CV)")
            print("Tunnel decompression reflects Phase 1 content; self-construction")
        else:
            print(f"VERDICT: AMBIGUOUS — CV={mean_cv:.3f} (5-15% range)")
            print("Partial content influence but not clearly autopoietic")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: {MODEL_ID}")
    print(f"Design: {P1_TURNS}P1 + {P2_TURNS}P2 + {P3_TURNS}P3 × {len(PREAMBLES)} conditions")

    model, tokenizer = load_model(MODEL_ID, device)

    all_results = {}
    for name, preamble in PREAMBLES.items():
        probes = PROBE_SETS[name]
        results = run_condition(model, tokenizer, name, preamble, probes, device)
        all_results[name] = results
        torch.cuda.empty_cache()

    analyze_autopoiesis(all_results)

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = f"results/exp_autopoiesis_{ts}.json"
    os.makedirs("results", exist_ok=True)

    serializable = {}
    for cond, entries in all_results.items():
        serializable[cond] = []
        for e in entries:
            se = {k: v for k, v in e.items() if k != "spectral"}
            se["erank_tunnel"] = e["spectral"].get("erank_tunnel")
            se["erank_resp"] = e["spectral"].get("erank_resp")
            se["erank_relay"] = e["spectral"].get("erank_relay")
            if e["drift"]:
                se["drift_mean"] = e["drift"]["mean"]
                se["drift_tunnel"] = e["drift"].get("tunnel")
                se["drift_resp"] = e["drift"].get("resp")
            serializable[cond].append(se)

    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
