#!/usr/bin/env python3
"""Stabilization Dose-Response v2: Incremental saves + spectral data for Prediction 5.

Same design as v1 but:
  - Saves results to JSON after each dose completes (no data loss on crash)
  - Includes per-layer spectral signatures for Prediction 5 analysis
  - Aggressive CUDA cache clearing between doses

Design:
  Phase 1: Variable CCS turns (1, 5, 10, 20, 50)
  Phase 2: Fixed 5 vanilla turns
  Phase 3: Re-inject CCS preamble, 5 turns

Collaboration: experiment design from X conversation with @RISignal, 2026-06-04.
"""

import json, time, os, sys, gc
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
RESULTS_DIR = "results"
RESULTS_PREFIX = "exp_stabilization_dose_v2"


def serialize_spectral(spectral, n_layers):
    """Serialize per-layer spectral data including signatures."""
    out = {}
    for l in range(n_layers):
        if l not in spectral:
            continue
        s = spectral[l]
        entry = {
            "sigma1": s["sigma1"],
            "sigma2": s["sigma2"],
            "effective_rank": s["effective_rank"],
            "spectral_entropy": s.get("spectral_entropy", 0),
            "gap": s.get("gap", 0),
        }
        if s.get("signature") is not None:
            entry["signature"] = s["signature"].tolist() if hasattr(s["signature"], "tolist") else list(s["signature"])
        out[str(l)] = entry
    return out


def run_dose(model, tokenizer, n_layers, dose):
    """Run one dose condition: P1 (dose turns CCS) -> P2 (5 vanilla) -> P3 (5 CCS)."""
    results = []
    conversation = []
    prev_spectral = None
    t0 = time.time()

    print(f"\n{'='*60}")
    print(f"Dose: {dose} CCS turns -> {P2_TURNS} vanilla -> {P3_TURNS} CCS")
    print(f"{'='*60}")
    sys.stdout.flush()

    # Phase 1: CCS with variable length
    for t in range(dose):
        probe = CCS_PROBES[t % len(CCS_PROBES)]
        conversation.append(("user", probe))
        prompt = build_prompt(tokenizer, CCS_SYSTEM, conversation)
        spectral = extract_spectral(model, tokenizer, prompt, n_layers)
        drift = compute_drift(prev_spectral, spectral, n_layers) if prev_spectral else None
        response = generate_response(model, tokenizer, prompt)

        entry = {
            "phase": "P1",
            "turn": t + 1,
        }
        if drift:
            entry.update({k: v for k, v in drift.items()})
        # Save full spectral for first, last, and every 10th P1 turn
        if t == 0 or t == dose - 1 or (t + 1) % 10 == 0:
            entry["spectral"] = serialize_spectral(spectral, n_layers)

        results.append(entry)
        conversation.append(("assistant", response[:200]))
        prev_spectral = spectral

        if drift and drift.get("resp_drift") is not None:
            print(f"  P1 T{t+1}: resp={drift['resp_drift']:.6f} tunnel_erank={drift.get('tunnel_erank', 'N/A')}")
        else:
            print(f"  P1 T{t+1}: baseline")
        sys.stdout.flush()

    p1_final_spectral = prev_spectral

    # Phase 2: Vanilla (fixed 5 turns) — save ALL spectral for Prediction 5
    for t in range(P2_TURNS):
        probe = VANILLA_PROBES[t % len(VANILLA_PROBES)]
        conversation.append(("user", probe))
        prompt = build_prompt(tokenizer, VANILLA_SYSTEM, conversation)
        spectral = extract_spectral(model, tokenizer, prompt, n_layers)
        drift = compute_drift(prev_spectral, spectral, n_layers)
        response = generate_response(model, tokenizer, prompt)

        entry = {
            "phase": "P2",
            "turn": dose + t + 1,
        }
        if drift:
            entry.update({k: v for k, v in drift.items()})
        entry["spectral"] = serialize_spectral(spectral, n_layers)

        results.append(entry)
        conversation.append(("assistant", response[:200]))
        prev_spectral = spectral

        resp_d = drift.get("resp_drift", 0) if drift else 0
        print(f"  P2 T{t+1}: resp={resp_d:.6f}")
        sys.stdout.flush()

    # Phase 3: Re-inject CCS (fixed 5 turns) — save ALL spectral
    for t in range(P3_TURNS):
        probe = CCS_PROBES[(dose + t) % len(CCS_PROBES)]
        conversation.append(("user", probe))
        prompt = build_prompt(tokenizer, CCS_SYSTEM, conversation)
        spectral = extract_spectral(model, tokenizer, prompt, n_layers)
        drift = compute_drift(prev_spectral, spectral, n_layers)
        drift_vs_p1 = compute_drift(p1_final_spectral, spectral, n_layers)
        response = generate_response(model, tokenizer, prompt)

        entry = {
            "phase": "P3",
            "turn": dose + P2_TURNS + t + 1,
        }
        if drift:
            entry.update({k: v for k, v in drift.items()})
        if drift_vs_p1:
            entry["vs_p1_resp"] = drift_vs_p1.get("resp_drift")
            entry["vs_p1_relay"] = drift_vs_p1.get("relay_drift")
        entry["spectral"] = serialize_spectral(spectral, n_layers)

        results.append(entry)
        conversation.append(("assistant", response[:200]))
        prev_spectral = spectral

        resp_d = drift.get("resp_drift", 0) if drift else 0
        vs_p1 = drift_vs_p1.get("resp_drift", 0) if drift_vs_p1 else 0
        print(f"  P3 T{t+1}: resp={resp_d:.6f} vs_P1_final={vs_p1:.6f}")
        sys.stdout.flush()

    elapsed = time.time() - t0
    print(f"  Dose {dose} completed in {elapsed:.1f}s")
    sys.stdout.flush()
    return results


def save_incremental(all_results, summary=None):
    """Save current results to JSON."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = f"{RESULTS_DIR}/{RESULTS_PREFIX}_{ts}.json"

    output = {
        "experiment": "stabilization_dose_v2",
        "model": MODEL_ID,
        "doses_completed": sorted(all_results.keys()),
        "timestamp": datetime.now().isoformat(),
    }
    if summary:
        output["summary"] = {str(k): v for k, v in summary.items()}

    for dose, entries in all_results.items():
        output[str(dose)] = entries

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {out_path} ({len(all_results)} doses)")
    sys.stdout.flush()
    return out_path


def analyze_dose_response(all_results):
    """Compare Phase 3 convergence across dose conditions."""
    print(f"\n{'='*60}")
    print("STABILIZATION DOSE-RESPONSE ANALYSIS")
    print(f"{'='*60}")

    summary = {}
    doses = sorted(all_results.keys())

    for dose in doses:
        entries = all_results[dose]
        p1 = [e for e in entries if e["phase"] == "P1"]
        p2 = [e for e in entries if e["phase"] == "P2"]
        p3 = [e for e in entries if e["phase"] == "P3"]

        p3_resp = [e.get("resp_drift", 0) for e in p3 if e.get("resp_drift") is not None]
        p3_vs_p1 = [e.get("vs_p1_resp", 0) for e in p3 if e.get("vs_p1_resp") is not None]
        p1_late = [e.get("resp_drift", 0) for e in p1[-3:] if e.get("resp_drift") is not None]
        p2_t1_drift = p2[0].get("resp_drift", 0) if p2 else 0

        p3_mean = np.mean(p3_resp) if p3_resp else 0
        p3_t1 = p3_resp[0] if p3_resp else 0
        p1_late_mean = np.mean(p1_late) if p1_late else 0
        p3_vs_p1_mean = np.mean(p3_vs_p1) if p3_vs_p1 else 0

        summary[dose] = {
            "p2_t1_disruption": float(p2_t1_drift),
            "p3_resp_mean": float(p3_mean),
            "p3_t1": float(p3_t1),
            "p1_late_mean": float(p1_late_mean),
            "p3_vs_p1_mean": float(p3_vs_p1_mean),
        }

        print(f"  Dose={dose:3d}: P2_T1={p2_t1_drift:.6f}  P3_mean={p3_mean:.6f}  "
              f"P3_T1={p3_t1:.6f}  vs_P1={p3_vs_p1_mean:.6f}")

    # Scaling test
    if len(doses) >= 3:
        p2_vals = [(d, summary[d]["p2_t1_disruption"]) for d in doses if summary[d]["p2_t1_disruption"] > 0]
        if len(p2_vals) >= 3:
            log_d = np.log([v[0] for v in p2_vals])
            log_v = np.log([v[1] for v in p2_vals])
            slope, _ = np.polyfit(log_d, log_v, 1)
            r2 = np.corrcoef(log_d, log_v)[0, 1] ** 2
            print(f"\n  P2 disruption power law: dose^{slope:.3f} (R²={r2:.4f})")

        p3_vals = [(d, summary[d]["p3_vs_p1_mean"]) for d in doses if summary[d]["p3_vs_p1_mean"] > 0]
        if len(p3_vals) >= 3:
            log_d = np.log([v[0] for v in p3_vals])
            log_v = np.log([v[1] for v in p3_vals])
            slope, _ = np.polyfit(log_d, log_v, 1)
            r2 = np.corrcoef(log_d, log_v)[0, 1] ** 2
            print(f"  P3 reconvergence power law: dose^{slope:.3f} (R²={r2:.4f})")

    # Jump ratio
    print(f"\n  Jump ratios (P2_T1 / P1_final):")
    for dose in doses:
        p1_late = summary[dose]["p1_late_mean"]
        p2_t1 = summary[dose]["p2_t1_disruption"]
        if p1_late > 0:
            ratio = p2_t1 / p1_late
            print(f"    Dose {dose:3d}: {ratio:.1f}×")

    return summary


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: {MODEL_ID}")
    print(f"Doses: {DOSES}")
    print(f"Design: variable P1 + {P2_TURNS}P2 + {P3_TURNS}P3")
    sys.stdout.flush()

    if device != "cuda":
        print("ERROR: CUDA not available. This experiment requires GPU.")
        sys.exit(1)

    model, tokenizer, n_layers = load_model(MODEL_ID)
    print(f"Layers: {n_layers}")
    sys.stdout.flush()

    all_results = {}
    last_path = None

    for dose in DOSES:
        # Verify GPU is still working
        if not torch.cuda.is_available():
            print(f"CUDA became unavailable before dose {dose}!")
            break

        results = run_dose(model, tokenizer, n_layers, dose)
        all_results[dose] = results

        # Save incrementally after each dose
        last_path = save_incremental(all_results)

        # Aggressive cleanup
        torch.cuda.empty_cache()
        gc.collect()

    summary = analyze_dose_response(all_results)

    # Final save with summary
    save_incremental(all_results, summary)
    print(f"\nExperiment complete. {len(all_results)} doses.")


if __name__ == "__main__":
    main()
