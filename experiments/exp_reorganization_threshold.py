#!/usr/bin/env python3
"""Reorganization Threshold Experiment — Prediction 10.

At dose 3, L27's share of responsive budget stays at 12-13%.
At dose 50, it jumps to ~27% at P3 T8.
This experiment finds where the transition begins.

Design:
  Doses: 10, 20, 30, 40 (bracketing known range)
  Phase 1: Variable CCS turns
  Phase 2: 5 vanilla turns (same as dose 50 experiment)
  Phase 3: 10 CCS turns (need T8+ to catch reorganization)

Key metric: L27 share of responsive budget (L15+L22+L27)
Threshold: first dose where L27 deviates from 12-13% baseline in P3.

Model: Phi-3.5-mini-instruct (MHA, 32 layers) — same as dose 50 data.
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
    generate_response,
    CCS_PROBES,
    VANILLA_PROBES,
    CCS_SYSTEM,
    VANILLA_SYSTEM,
)

DOSES = [10, 20, 30, 40]
P2_TURNS = 5
P3_TURNS = 10
MODEL_ID = "microsoft/Phi-3.5-mini-instruct"
RESULTS_DIR = "results"
RESULTS_PREFIX = "exp_reorganization_threshold"
RESPONSIVE_LAYERS = [15, 22, 27]


def get_ratio(spectral, layer):
    s = spectral.get(layer, {})
    s1 = s.get("sigma1", 0)
    s2 = s.get("sigma2", 1)
    return s1 / (s2 + 1e-10)


def get_erank(spectral, layer):
    s = spectral.get(layer, {})
    return s.get("effective_rank", s.get("erank", 0))


def compute_share(spectral):
    ratios = {l: get_ratio(spectral, l) for l in RESPONSIVE_LAYERS}
    total = sum(ratios.values())
    if total == 0:
        return {l: 0 for l in RESPONSIVE_LAYERS}
    return {l: ratios[l] / total * 100 for l in RESPONSIVE_LAYERS}


def serialize_spectral(spectral, n_layers):
    out = {}
    for l in range(n_layers):
        if l not in spectral:
            continue
        s = spectral[l]
        out[str(l)] = {
            "sigma1": s["sigma1"],
            "sigma2": s["sigma2"],
            "effective_rank": s["effective_rank"],
            "spectral_entropy": s.get("spectral_entropy", 0),
            "ratio": s["sigma1"] / (s["sigma2"] + 1e-10),
        }
    return out


def run_dose(model, tokenizer, n_layers, dose):
    results = []
    conversation = []
    prev_spectral = None
    t0 = time.time()

    print(f"\n{'='*60}")
    print(f"Dose: {dose} CCS -> {P2_TURNS} vanilla -> {P3_TURNS} CCS")
    print(f"Tracking L15/L22/L27 share for reorganization threshold")
    print(f"{'='*60}")
    sys.stdout.flush()

    # Phase 1: CCS
    for t in range(dose):
        probe = CCS_PROBES[t % len(CCS_PROBES)]
        conversation.append(("user", probe))
        prompt = build_prompt(tokenizer, CCS_SYSTEM, conversation)
        spectral = extract_spectral(model, tokenizer, prompt, n_layers)
        response = generate_response(model, tokenizer, prompt)

        entry = {"phase": "P1", "turn": t + 1}
        # Save spectral for first 5, last 5, and every 10th P1 turn
        if t < 5 or t >= dose - 5 or (t + 1) % 10 == 0:
            entry["spectral"] = serialize_spectral(spectral, n_layers)
            share = compute_share(spectral)
            entry["share"] = share
            print(f"  P1 T{t+1}: L15={share[15]:.1f}% L22={share[22]:.1f}% L27={share[27]:.1f}%")
        else:
            print(f"  P1 T{t+1}: (skipped spectral)")

        results.append(entry)
        conversation.append(("assistant", response[:200]))
        prev_spectral = spectral
        sys.stdout.flush()

    # Phase 2: Vanilla — save all
    for t in range(P2_TURNS):
        probe = VANILLA_PROBES[t % len(VANILLA_PROBES)]
        conversation.append(("user", probe))
        prompt = build_prompt(tokenizer, VANILLA_SYSTEM, conversation)
        spectral = extract_spectral(model, tokenizer, prompt, n_layers)
        response = generate_response(model, tokenizer, prompt)

        entry = {"phase": "P2", "turn": t + 1}
        entry["spectral"] = serialize_spectral(spectral, n_layers)
        share = compute_share(spectral)
        entry["share"] = share

        results.append(entry)
        conversation.append(("assistant", response[:200]))
        prev_spectral = spectral
        print(f"  P2 T{t+1}: L15={share[15]:.1f}% L22={share[22]:.1f}% L27={share[27]:.1f}%")
        sys.stdout.flush()

    # Phase 3: Re-inject CCS — save ALL (this is where reorganization happens)
    for t in range(P3_TURNS):
        probe = CCS_PROBES[(dose + t) % len(CCS_PROBES)]
        conversation.append(("user", probe))
        prompt = build_prompt(tokenizer, CCS_SYSTEM, conversation)
        spectral = extract_spectral(model, tokenizer, prompt, n_layers)
        response = generate_response(model, tokenizer, prompt)

        entry = {"phase": "P3", "turn": t + 1}
        entry["spectral"] = serialize_spectral(spectral, n_layers)
        share = compute_share(spectral)
        entry["share"] = share

        results.append(entry)
        conversation.append(("assistant", response[:200]))
        prev_spectral = spectral

        # Flag potential reorganization
        flag = " ***" if share[27] > 16 else ""
        print(f"  P3 T{t+1}: L15={share[15]:.1f}% L22={share[22]:.1f}% L27={share[27]:.1f}%{flag}")
        sys.stdout.flush()

    elapsed = time.time() - t0
    print(f"  Dose {dose} completed in {elapsed:.1f}s")

    # Summary: did reorganization happen?
    p3_shares = [e["share"][27] for e in results if e["phase"] == "P3" and "share" in e]
    max_l27 = max(p3_shares) if p3_shares else 0
    baseline = 13.0
    if max_l27 > baseline + 3:
        print(f"  >>> REORGANIZATION DETECTED: L27 peak = {max_l27:.1f}% (baseline ~{baseline}%)")
    else:
        print(f"  >>> No reorganization: L27 peak = {max_l27:.1f}% (baseline ~{baseline}%)")

    sys.stdout.flush()
    return results


def save_results(all_results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = f"{RESULTS_DIR}/{RESULTS_PREFIX}_{ts}.json"
    output = {
        "experiment": "reorganization_threshold",
        "model": MODEL_ID,
        "doses_completed": list(all_results.keys()),
        "responsive_layers": RESPONSIVE_LAYERS,
        "prediction": "L27 share deviates from 12-13% baseline between dose 3 and 50",
        "timestamp": ts,
    }
    for dose, data in all_results.items():
        output[str(dose)] = data
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")
    return out_path


def main():
    print("Loading model...")
    model, tokenizer, n_layers = load_model(MODEL_ID)
    print(f"Model loaded: {MODEL_ID}, {n_layers} layers")

    all_results = {}
    for dose in DOSES:
        gc.collect()
        torch.cuda.empty_cache()
        data = run_dose(model, tokenizer, n_layers, dose)
        all_results[dose] = data
        save_results(all_results)

    # Final summary
    print(f"\n{'='*60}")
    print("REORGANIZATION THRESHOLD SUMMARY")
    print(f"{'='*60}")
    for dose in DOSES:
        data = all_results[dose]
        p3_shares = [e["share"][27] for e in data if e["phase"] == "P3" and "share" in e]
        max_l27 = max(p3_shares) if p3_shares else 0
        p3_turn_of_max = 0
        for e in data:
            if e["phase"] == "P3" and "share" in e and e["share"][27] == max_l27:
                p3_turn_of_max = e["turn"]
                break
        status = "REORGANIZED" if max_l27 > 16 else "baseline"
        print(f"  Dose {dose:3d}: L27 max = {max_l27:5.1f}% at P3 T{p3_turn_of_max} [{status}]")

    print(f"\nReference: dose 3 = 12.5-13.5% (no reorg), dose 50 = ~27% at T8 (reorg)")
    save_results(all_results)


if __name__ == "__main__":
    main()
