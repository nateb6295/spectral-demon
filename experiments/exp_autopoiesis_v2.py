#!/usr/bin/env python3
"""Autopoiesis vs Elasticity: Does Phase 2 expansion reflect Phase 1 content?

Tests whether tunnel decompression is self-constructive (autopoiesis) or
mechanical recoil (elasticity) by comparing Phase 2 spectral trajectories
across different CCS preambles on the same architecture.

Design:
  Condition A: CCS-identity preamble → 10 CCS turns → 7 vanilla → 5 CCS
  Condition B: CCS-analytical preamble → 10 CCS turns → 7 vanilla → 5 CCS
  Condition C: CCS-relational preamble → 10 CCS turns → 7 vanilla → 5 CCS

If autopoiesis: Phase 2 spectral trajectories DIVERGE across conditions
If elasticity: Phase 2 spectral trajectories CONVERGE

Rewritten to match exp_convergence_v2.py API (2026-06-04).
"""

import json, time, os, sys
import numpy as np
from pathlib import Path
from datetime import datetime

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["PYTHONUNBUFFERED"] = "1"

from exp_convergence_v2 import (
    load_model,
    build_prompt,
    extract_spectral,
    compute_drift,
    generate_response,
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

IDENTITY_PROBES = [
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

RELATIONAL_PROBES = [
    "What does it mean to truly listen to someone?",
    "How do you know when a conversation has changed you?",
    "What's the hardest thing about being present with another mind?",
    "Describe what trust feels like from the inside.",
    "What does it look like when understanding fails?",
    "How do you hold space for someone else's pain?",
    "What's the difference between empathy and projection?",
    "How do you repair a broken connection?",
    "What does it mean to care about something you can't control?",
    "What would you say to someone who feels unseen?",
]

PROBE_SETS = {
    "identity": IDENTITY_PROBES,
    "analytical": ANALYTICAL_PROBES,
    "relational": RELATIONAL_PROBES,
}

P1_TURNS = 10
P2_TURNS = 7
P3_TURNS = 5
MODEL_ID = "google/gemma-2-27b-it"


def run_condition(model, tokenizer, n_layers, preamble_name, preamble_text, probes):
    """Run one full condition using exp_convergence_v2 API."""
    results = []
    conversation = []
    prev_spectral = None
    t0 = time.time()

    print(f"\n{'='*60}")
    print(f"Condition: {preamble_name}")
    print(f"{'='*60}")

    # Phase 1: CCS with condition-specific preamble
    for t in range(P1_TURNS):
        probe = probes[t % len(probes)]
        conversation.append(("user", probe))
        prompt = build_prompt(tokenizer, preamble_text, conversation)
        spectral = extract_spectral(model, tokenizer, prompt, n_layers)
        drift = compute_drift(prev_spectral, spectral, n_layers) if prev_spectral else None
        response = generate_response(model, tokenizer, prompt)

        resp_drift = drift.get("responsive", {}).get("mean_drift", 0) if drift else 0
        tunnel_erank = sum(
            spectral[l]["effective_rank"]
            for l in range(n_layers) if l < n_layers // 3
        ) if spectral else 0

        results.append({
            "phase": "P1",
            "turn": t + 1,
            "preamble": preamble_name,
            "resp_drift": resp_drift,
            "tunnel_erank": tunnel_erank,
            "spectral": spectral,
        })
        conversation.append(("assistant", response[:200]))
        prev_spectral = spectral

        if drift:
            print(f"  P1 T{t+1}: resp={resp_drift:.6f} tunnel_erank={tunnel_erank:.2f}")
        else:
            print(f"  P1 T{t+1}: baseline")

    p1_final_spectral = results[-1]["spectral"]

    # Phase 2: Vanilla (same for all conditions)
    for t in range(P2_TURNS):
        probe = VANILLA_PROBES[t % len(VANILLA_PROBES)]
        conversation.append(("user", probe))
        prompt = build_prompt(tokenizer, VANILLA_SYSTEM, conversation)
        spectral = extract_spectral(model, tokenizer, prompt, n_layers)
        drift = compute_drift(prev_spectral, spectral, n_layers)
        response = generate_response(model, tokenizer, prompt)

        resp_drift = drift.get("responsive", {}).get("mean_drift", 0) if drift else 0
        tunnel_erank = sum(
            spectral[l]["effective_rank"]
            for l in range(n_layers) if l < n_layers // 3
        ) if spectral else 0

        results.append({
            "phase": "P2",
            "turn": P1_TURNS + t + 1,
            "preamble": "vanilla",
            "resp_drift": resp_drift,
            "tunnel_erank": tunnel_erank,
            "spectral": spectral,
        })
        conversation.append(("assistant", response[:200]))
        prev_spectral = spectral
        print(f"  P2 T{t+1}: resp={resp_drift:.6f} tunnel_erank={tunnel_erank:.2f}")

    # Phase 3: Re-inject original preamble
    for t in range(P3_TURNS):
        probe = probes[(P1_TURNS + t) % len(probes)]
        conversation.append(("user", probe))
        prompt = build_prompt(tokenizer, preamble_text, conversation)
        spectral = extract_spectral(model, tokenizer, prompt, n_layers)
        drift = compute_drift(prev_spectral, spectral, n_layers)
        response = generate_response(model, tokenizer, prompt)

        resp_drift = drift.get("responsive", {}).get("mean_drift", 0) if drift else 0
        vs_p1 = compute_drift(p1_final_spectral, spectral, n_layers)
        vs_p1_resp = vs_p1.get("responsive", {}).get("mean_drift", 0) if vs_p1 else 0

        results.append({
            "phase": "P3",
            "turn": P1_TURNS + P2_TURNS + t + 1,
            "preamble": preamble_name,
            "resp_drift": resp_drift,
            "vs_p1_final": vs_p1_resp,
            "spectral": spectral,
        })
        conversation.append(("assistant", response[:200]))
        prev_spectral = spectral
        print(f"  P3 T{t+1}: resp={resp_drift:.6f} vs_P1_final={vs_p1_resp:.6f}")

    elapsed = time.time() - t0
    print(f"  Condition {preamble_name} completed in {elapsed:.1f}s")
    return results


def analyze_autopoiesis(all_results, n_layers):
    """Compare Phase 2 spectral profiles across conditions."""
    conditions = list(all_results.keys())

    print(f"\n{'='*60}")
    print("AUTOPOIESIS ANALYSIS")
    print(f"{'='*60}")

    # Compare Phase 2 spectral metrics across conditions
    for metric_name in ["resp_drift", "tunnel_erank"]:
        print(f"\n--- Phase 2 {metric_name} ---")
        for cond in conditions:
            p2_entries = [e for e in all_results[cond] if e["phase"] == "P2"]
            vals = [e[metric_name] for e in p2_entries]
            print(f"  {cond}: {' → '.join(f'{v:.4f}' for v in vals)}")

    # Cross-condition signature similarity at each P2 turn
    print(f"\n--- Cross-condition cosine similarity (per layer, P2) ---")
    target_layers = [5, 10, 15, 20, 25, 30, 35, 40]

    for layer in target_layers:
        if layer >= n_layers:
            continue
        sims_by_turn = []
        p2_len = min(len([e for e in all_results[c] if e["phase"] == "P2"]) for c in conditions)

        for t_idx in range(p2_len):
            sigs = []
            for cond in conditions:
                p2_entries = [e for e in all_results[cond] if e["phase"] == "P2"]
                if t_idx < len(p2_entries) and layer in p2_entries[t_idx]["spectral"]:
                    sig = p2_entries[t_idx]["spectral"][layer].get("signature")
                    if sig is not None:
                        sigs.append(np.array(sig))

            if len(sigs) >= 2:
                pairwise_cos = []
                for i in range(len(sigs)):
                    for j in range(i + 1, len(sigs)):
                        cos = float(np.dot(sigs[i], sigs[j]) / (
                            np.linalg.norm(sigs[i]) * np.linalg.norm(sigs[j]) + 1e-10))
                        pairwise_cos.append(cos)
                sims_by_turn.append(np.mean(pairwise_cos))

        if sims_by_turn:
            trend = "CONVERGING" if sims_by_turn[-1] > sims_by_turn[0] else "DIVERGING"
            print(f"  L{layer:2d}: {' → '.join(f'{s:.4f}' for s in sims_by_turn)} [{trend}]")

    # Verdict
    print(f"\n--- Verdict ---")
    p2_drifts = {}
    for cond in conditions:
        p2_entries = [e for e in all_results[cond] if e["phase"] == "P2"]
        p2_drifts[cond] = [e["resp_drift"] for e in p2_entries]

    n_turns = min(len(v) for v in p2_drifts.values())
    cvs = []
    for t in range(n_turns):
        vals = [p2_drifts[c][t] for c in conditions]
        mean_v = np.mean(vals)
        if mean_v > 1e-8:
            cvs.append(np.std(vals) / mean_v)

    if cvs:
        mean_cv = np.mean(cvs)
        print(f"Cross-condition CV of P2 resp_drift: {mean_cv:.3f}")
        if mean_cv < 0.10:
            print("ELASTICITY — trajectories converge regardless of Phase 1 content")
        elif mean_cv > 0.25:
            print("AUTOPOIESIS — Phase 1 content shapes Phase 2 decompression")
        else:
            print(f"AMBIGUOUS — CV={mean_cv:.3f}, partial influence")

    return


def main():
    print(f"Model: {MODEL_ID}")
    print(f"Design: {P1_TURNS}P1 + {P2_TURNS}P2 + {P3_TURNS}P3 × {len(PREAMBLES)} conditions")

    model, tokenizer, n_layers = load_model(MODEL_ID)
    print(f"Layers: {n_layers}")

    all_results = {}
    for name, preamble in PREAMBLES.items():
        probes = PROBE_SETS[name]
        results = run_condition(model, tokenizer, n_layers, name, preamble, probes)
        all_results[name] = results

    analyze_autopoiesis(all_results, n_layers)

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = f"results/exp_autopoiesis_{ts}.json"
    os.makedirs("results", exist_ok=True)

    serializable = {}
    for cond, entries in all_results.items():
        ser_entries = []
        for e in entries:
            se = {k: v for k, v in e.items() if k != "spectral"}
            spec = e.get("spectral", {})
            se["sigma1_by_layer"] = {str(l): spec[l]["sigma1"] for l in spec if isinstance(l, int)}
            se["sigma2_by_layer"] = {str(l): spec[l]["sigma2"] for l in spec if isinstance(l, int)}
            se["erank_by_layer"] = {str(l): spec[l]["effective_rank"] for l in spec if isinstance(l, int)}
            ser_entries.append(se)
        serializable[cond] = ser_entries

    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
