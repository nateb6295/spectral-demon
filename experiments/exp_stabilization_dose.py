#!/usr/bin/env python3
"""Stabilization Dose-Response: Does reconvergence strength scale with Phase 1 length?

Motivated by RISignal's TAS (Trajectory Acquisition & Stabilization) framework.
Key prediction: reconvergence strength in Phase 3 should scale with pre-disruption
interaction history (Phase 1 length).

Design:
  Phase 1: Vary CCS conversation length: 1, 5, 10, 20, 50 turns
  Phase 2: Fixed 5 vanilla turns (same across all conditions)
  Phase 3: Re-inject CCS preamble, 5 turns

If TAS holds:
  - No threshold for acquisition (Phase 1 effects from turn 1)
  - Cumulative threshold for stabilization (Phase 3 tightness scales with Phase 1 length)
  - Phase 1 = regime entry, Phase 3 = drift resistance

Key metrics:
  - Phase 3 mean drift (lower = tighter convergence)
  - Phase 3 T1 drift (first re-entry: how fast does it snap back?)
  - Phase 3 erank trajectory (does tunnel depth scale with Phase 1 dose?)
  - Phase 1→3 convergence ratio (Phase 3 late drift / Phase 1 late drift)

Collaboration: experiment design from X conversation with @RISignal, 2026-06-04.
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
    build_prompt,
    extract_spectral,
    compute_drift,
    generate_response,
    CCS_PROBES,
    VANILLA_PROBES,
    CCS_SYSTEM,
    VANILLA_SYSTEM,
)

DOSES = [1, 5, 10, 20, 50]
P2_TURNS = 5
P3_TURNS = 5
MODEL_ID = "google/gemma-2-27b-it"


def run_dose(model, tokenizer, n_layers, dose):
    """Run one dose condition: P1 (dose turns CCS) -> P2 (5 vanilla) -> P3 (5 CCS)."""
    results = []
    conversation = []
    prev_spectral = None

    print(f"\n{'='*60}")
    print(f"Dose: {dose} CCS turns -> {P2_TURNS} vanilla -> {P3_TURNS} CCS")
    print(f"{'='*60}")

    # Phase 1: CCS with variable length
    for t in range(dose):
        probe = CCS_PROBES[t % len(CCS_PROBES)]
        conversation.append(("user", probe))
        prompt = build_prompt(tokenizer, CCS_SYSTEM, conversation)
        spectral = extract_spectral(model, tokenizer, prompt, n_layers)

        drift = compute_drift(prev_spectral, spectral, n_layers) if prev_spectral else None

        results.append({
            "phase": 1,
            "turn": t + 1,
            "spectral": spectral,
            "drift": drift,
        })

        response = generate_response(model, tokenizer, prompt)
        conversation.append(("assistant", response[:200]))
        prev_spectral = spectral

        if drift and drift["resp_drift"] is not None:
            print(f"  P1 T{t+1}: resp={drift['resp_drift']:.6f} tunnel_erank={drift.get('tunnel_erank', 'N/A')}")
        else:
            print(f"  P1 T{t+1}: baseline")

    p1_final_spectral = results[-1]["spectral"]

    # Phase 2: Vanilla (fixed 5 turns)
    for t in range(P2_TURNS):
        probe = VANILLA_PROBES[t % len(VANILLA_PROBES)]
        conversation.append(("user", probe))
        prompt = build_prompt(tokenizer, VANILLA_SYSTEM, conversation)
        spectral = extract_spectral(model, tokenizer, prompt, n_layers)

        drift = compute_drift(prev_spectral, spectral, n_layers)

        results.append({
            "phase": 2,
            "turn": dose + t + 1,
            "spectral": spectral,
            "drift": drift,
        })

        response = generate_response(model, tokenizer, prompt)
        conversation.append(("assistant", response[:200]))
        prev_spectral = spectral

        if drift["resp_drift"] is not None:
            print(f"  P2 T{t+1}: resp={drift['resp_drift']:.6f}")

    # Phase 3: Re-inject CCS (fixed 5 turns)
    for t in range(P3_TURNS):
        probe = CCS_PROBES[(dose + t) % len(CCS_PROBES)]
        conversation.append(("user", probe))
        prompt = build_prompt(tokenizer, CCS_SYSTEM, conversation)
        spectral = extract_spectral(model, tokenizer, prompt, n_layers)

        drift = compute_drift(prev_spectral, spectral, n_layers)
        drift_vs_p1 = compute_drift(p1_final_spectral, spectral, n_layers)

        results.append({
            "phase": 3,
            "turn": dose + P2_TURNS + t + 1,
            "spectral": spectral,
            "drift": drift,
            "drift_vs_p1_final": drift_vs_p1,
        })

        response = generate_response(model, tokenizer, prompt)
        conversation.append(("assistant", response[:200]))
        prev_spectral = spectral

        resp_d = drift['resp_drift'] if drift['resp_drift'] is not None else 0
        vs_p1 = drift_vs_p1['resp_drift'] if drift_vs_p1['resp_drift'] is not None else 0
        print(f"  P3 T{t+1}: resp={resp_d:.6f} vs_P1_final={vs_p1:.6f}")

    return results


def analyze_dose_response(all_results):
    """Compare Phase 3 convergence across dose conditions."""
    print(f"\n{'='*60}")
    print("STABILIZATION DOSE-RESPONSE ANALYSIS")
    print(f"{'='*60}")

    summary = {}

    for dose in DOSES:
        entries = all_results[dose]
        p3_entries = [e for e in entries if e["phase"] == 3]
        p1_entries = [e for e in entries if e["phase"] == 1]

        p3_resp = [e["drift"]["resp_drift"] for e in p3_entries if e["drift"] and e["drift"]["resp_drift"] is not None]
        p3_vs_p1 = [e["drift_vs_p1_final"]["resp_drift"] for e in p3_entries if e.get("drift_vs_p1_final") and e["drift_vs_p1_final"]["resp_drift"] is not None]
        p1_late = [e["drift"]["resp_drift"] for e in p1_entries[-3:] if e["drift"] and e["drift"]["resp_drift"] is not None]

        p3_tunnel = [e["drift"]["tunnel_erank"] for e in p3_entries if e["drift"] and e["drift"].get("tunnel_erank") is not None]
        p3_relay = [e["drift"]["relay_drift"] for e in p3_entries if e["drift"] and e["drift"].get("relay_drift") is not None]

        p3_mean = sum(p3_resp) / len(p3_resp) if p3_resp else 0
        p3_t1 = p3_resp[0] if p3_resp else 0
        p1_late_mean = sum(p1_late) / len(p1_late) if p1_late else 0
        convergence_ratio = p3_mean / p1_late_mean if p1_late_mean > 0 else float('inf')
        p3_vs_p1_mean = sum(p3_vs_p1) / len(p3_vs_p1) if p3_vs_p1 else 0
        p3_tunnel_mean = sum(p3_tunnel) / len(p3_tunnel) if p3_tunnel else 0
        p3_relay_mean = sum(p3_relay) / len(p3_relay) if p3_relay else 0

        summary[dose] = {
            "p3_resp_mean": p3_mean,
            "p3_t1": p3_t1,
            "p1_late_mean": p1_late_mean,
            "convergence_ratio": convergence_ratio,
            "p3_vs_p1_mean": p3_vs_p1_mean,
            "p3_tunnel_erank": p3_tunnel_mean,
            "p3_relay_drift": p3_relay_mean,
        }

        print(f"\n  Dose={dose:3d}: P3 resp={p3_mean:.6f}  P3 T1={p3_t1:.6f}  "
              f"ratio={convergence_ratio:.3f}  vs_P1={p3_vs_p1_mean:.6f}  "
              f"tunnel_erank={p3_tunnel_mean:.1f}  relay={p3_relay_mean:.6f}")

    # Test scaling
    print(f"\n--- Scaling test ---")
    dose_means = [summary[d]["p3_resp_mean"] for d in DOSES]

    if len(dose_means) >= 2:
        decreasing = all(dose_means[i] >= dose_means[i+1] for i in range(len(dose_means)-1))
        print(f"  P3 resp drift by dose: {['%.6f' % d for d in dose_means]}")
        print(f"  Monotonically decreasing: {decreasing}")

        # Check P3 vs P1 convergence (should approach 0 with higher dose)
        vs_p1_means = [summary[d]["p3_vs_p1_mean"] for d in DOSES]
        print(f"  P3 vs P1 final by dose: {['%.6f' % d for d in vs_p1_means]}")

        if decreasing:
            print("  VERDICT: STABILIZATION SCALES WITH DOSE — TAS confirmed")
        else:
            non_mono = []
            for i in range(len(dose_means)-1):
                if dose_means[i] < dose_means[i+1]:
                    non_mono.append(f"dose {DOSES[i]}->{DOSES[i+1]}")
            print(f"  VERDICT: NON-MONOTONIC at {', '.join(non_mono)}")
            print(f"  Stabilization may plateau or have complex dynamics")

    return summary


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: {MODEL_ID}")
    print(f"Doses: {DOSES}")
    print(f"Design: variable P1 + {P2_TURNS}P2 + {P3_TURNS}P3")

    model, tokenizer, n_layers = load_model(MODEL_ID)
    print(f"Layers: {n_layers}")

    all_results = {}
    for dose in DOSES:
        t0 = time.time()
        results = run_dose(model, tokenizer, n_layers, dose)
        elapsed = time.time() - t0
        all_results[dose] = results
        print(f"  Dose {dose} completed in {elapsed:.1f}s")
        torch.cuda.empty_cache()

    summary = analyze_dose_response(all_results)

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = f"results/exp_stabilization_dose_{ts}.json"
    os.makedirs("results", exist_ok=True)

    serializable = {"summary": {str(k): v for k, v in summary.items()}}
    for dose, entries in all_results.items():
        serializable[str(dose)] = []
        for e in entries:
            se = {"phase": e["phase"], "turn": e["turn"]}
            if e.get("drift"):
                se.update({k: v for k, v in e["drift"].items()})
            if e.get("drift_vs_p1_final"):
                se["vs_p1_resp"] = e["drift_vs_p1_final"].get("resp_drift")
                se["vs_p1_relay"] = e["drift_vs_p1_final"].get("relay_drift")
            serializable[str(dose)].append(se)

    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
