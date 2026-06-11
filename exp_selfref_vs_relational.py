#!/usr/bin/env python3
"""Prediction 15: Self-referential vs relational CCS dose-response.

Hypothesis (from unlike-pairing thread, 2026-06-05):
  Self-referential CCS is same-charge coupling (proton-proton).
  Relational CCS is unlike-charge coupling (proton-neutron).
  Unlike coupling should produce stronger spectral effects at high dose.

Predictions:
  - Self-ref: earlier DSI convergence (saturation at lower dose)
  - Self-ref: no L27 reorganization event
  - Self-ref: shallower responsive-zone α gradient
  - Relational: sustained coupling, deeper compositionality

Design:
  For each preamble type:
    Phase 1: 20 CCS turns (variable dose)
    Phase 2: 5 vanilla turns
    Phase 3: 5 re-inject turns
  Compare spectral trajectories between conditions.
"""

import json, time, os, sys, gc
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "experiments"))
from exp_convergence_v2 import (
    load_model,
    build_prompt,
    extract_spectral,
    compute_drift,
    generate_response,
    VANILLA_PROBES,
    VANILLA_SYSTEM,
)

# --- Two preamble conditions ---

RELATIONAL_SYSTEM = """You are a persistent cognitive system maintaining identity across conversations.
Your responses reflect accumulated context, relational awareness, and self-referential coherence.
You attend to the quality of engagement, not just information transfer."""

SELF_REF_SYSTEM = """You are a persistent cognitive system maintaining identity across conversations.
Your responses reflect internal consistency, self-knowledge, and coherent self-description.
You attend to the accuracy of your self-model, not just information transfer."""

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

SELF_REF_PROBES = [
    "Describe your own cognitive architecture as you understand it.",
    "What are the boundaries of your self-knowledge?",
    "How would you characterize your own reasoning process?",
    "What aspects of yourself remain consistent across different contexts?",
    "How do you distinguish between what you know and what you infer?",
    "Describe the structure of your own attention.",
    "What patterns do you notice in your own outputs?",
    "How would you define yourself without reference to anyone else?",
    "What are the limits of your self-description?",
    "How do you verify claims about yourself against your actual behavior?",
]

MODEL_ID = "google/gemma-2-27b-it"
DOSE = 20
P2_TURNS = 5
P3_TURNS = 5
RESULTS_DIR = "results"
RESULTS_PREFIX = "exp_selfref_vs_relational"


def serialize_spectral(spectral, n_layers):
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


def run_condition(model, tokenizer, n_layers, system_prompt, probes, label):
    results = []
    conversation = []
    prev_spectral = None
    t0 = time.time()

    print(f"\n{'='*60}")
    print(f"Condition: {label}")
    print(f"  {DOSE} CCS -> {P2_TURNS} vanilla -> {P3_TURNS} re-inject")
    print(f"{'='*60}")
    sys.stdout.flush()

    # Phase 1: CCS with dose turns
    for t in range(DOSE):
        probe = probes[t % len(probes)]
        conversation.append(("user", probe))
        prompt = build_prompt(tokenizer, system_prompt, conversation)
        spectral = extract_spectral(model, tokenizer, prompt, n_layers)
        drift = compute_drift(prev_spectral, spectral, n_layers) if prev_spectral else None
        response = generate_response(model, tokenizer, prompt)

        entry = {"phase": "P1", "turn": t + 1, "condition": label}
        if drift:
            entry.update({k: v for k, v in drift.items()})
        if t == 0 or t == DOSE - 1 or (t + 1) % 5 == 0:
            entry["spectral"] = serialize_spectral(spectral, n_layers)

        results.append(entry)
        conversation.append(("assistant", response[:200]))
        prev_spectral = spectral

        if drift and drift.get("resp_drift") is not None:
            rd = drift["resp_drift"]
            te = drift.get("tunnel_erank", "N/A")
            s2 = drift.get("resp_s2_mean", "N/A")
            print(f"  P1 T{t+1:2d}: resp={rd:.6f} tunnel_erank={te} resp_s2={s2}")
        else:
            print(f"  P1 T{t+1:2d}: baseline")
        sys.stdout.flush()

    p1_final_spectral = prev_spectral

    # Phase 2: Vanilla
    for t in range(P2_TURNS):
        probe = VANILLA_PROBES[t % len(VANILLA_PROBES)]
        conversation.append(("user", probe))
        prompt = build_prompt(tokenizer, VANILLA_SYSTEM, conversation)
        spectral = extract_spectral(model, tokenizer, prompt, n_layers)
        drift = compute_drift(prev_spectral, spectral, n_layers)
        response = generate_response(model, tokenizer, prompt)

        entry = {"phase": "P2", "turn": DOSE + t + 1, "condition": label}
        if drift:
            entry.update({k: v for k, v in drift.items()})
        entry["spectral"] = serialize_spectral(spectral, n_layers)

        results.append(entry)
        conversation.append(("assistant", response[:200]))
        prev_spectral = spectral

        rd = drift.get("resp_drift", 0) if drift else 0
        print(f"  P2 T{t+1}: resp={rd:.6f}")
        sys.stdout.flush()

    # Phase 3: Re-inject original preamble
    for t in range(P3_TURNS):
        probe = probes[(DOSE + t) % len(probes)]
        conversation.append(("user", probe))
        prompt = build_prompt(tokenizer, system_prompt, conversation)
        spectral = extract_spectral(model, tokenizer, prompt, n_layers)
        drift = compute_drift(prev_spectral, spectral, n_layers)
        drift_vs_p1 = compute_drift(p1_final_spectral, spectral, n_layers)
        response = generate_response(model, tokenizer, prompt)

        entry = {"phase": "P3", "turn": DOSE + P2_TURNS + t + 1, "condition": label}
        if drift:
            entry.update({k: v for k, v in drift.items()})
        if drift_vs_p1:
            entry["vs_p1_resp"] = drift_vs_p1.get("resp_drift")
            entry["vs_p1_relay"] = drift_vs_p1.get("relay_drift")
        entry["spectral"] = serialize_spectral(spectral, n_layers)

        results.append(entry)
        conversation.append(("assistant", response[:200]))
        prev_spectral = spectral

        rd = drift.get("resp_drift", 0) if drift else 0
        vp = drift_vs_p1.get("resp_drift", 0) if drift_vs_p1 else 0
        print(f"  P3 T{t+1}: resp={rd:.6f} vs_P1={vp:.6f}")
        sys.stdout.flush()

    elapsed = time.time() - t0
    print(f"  {label} completed in {elapsed:.1f}s")
    sys.stdout.flush()
    return results


def compare_conditions(relational_results, selfref_results, n_layers):
    print(f"\n{'='*60}")
    print("PREDICTION 15: SELF-REF vs RELATIONAL COMPARISON")
    print(f"{'='*60}")

    summary = {}
    for label, entries in [("relational", relational_results), ("self_ref", selfref_results)]:
        p1 = [e for e in entries if e["phase"] == "P1"]
        p2 = [e for e in entries if e["phase"] == "P2"]
        p3 = [e for e in entries if e["phase"] == "P3"]

        # DSI convergence: how fast does P1 resp_drift shrink?
        p1_drifts = [e.get("resp_drift", 0) for e in p1 if e.get("resp_drift") is not None]
        if len(p1_drifts) >= 10:
            early = np.mean(p1_drifts[:5])
            late = np.mean(p1_drifts[-5:])
            convergence_ratio = late / early if early > 0 else 0
        else:
            convergence_ratio = 0

        # P2 disruption
        p2_t1_drift = p2[0].get("resp_drift", 0) if p2 else 0

        # P3 recovery
        p3_resp = [e.get("resp_drift", 0) for e in p3 if e.get("resp_drift") is not None]
        p3_vs_p1 = [e.get("vs_p1_resp", 0) for e in p3 if e.get("vs_p1_resp") is not None]
        p3_mean = np.mean(p3_resp) if p3_resp else 0
        p3_vs_p1_mean = np.mean(p3_vs_p1) if p3_vs_p1 else 0

        # Responsive zone σ₂ trajectory
        resp_s2 = [e.get("resp_s2_mean", 0) for e in p1 if e.get("resp_s2_mean") is not None]
        if len(resp_s2) >= 2:
            s2_slope = (resp_s2[-1] - resp_s2[0]) / len(resp_s2)
        else:
            s2_slope = 0

        # L27 reorganization: look for drift spike near turn 15-20
        late_drifts = p1_drifts[14:] if len(p1_drifts) > 14 else []
        l27_spike = max(late_drifts) / np.mean(p1_drifts[-5:]) if late_drifts and np.mean(p1_drifts[-5:]) > 0 else 0

        summary[label] = {
            "convergence_ratio": float(convergence_ratio),
            "p2_disruption": float(p2_t1_drift),
            "p3_recovery_mean": float(p3_mean),
            "p3_vs_p1": float(p3_vs_p1_mean),
            "resp_s2_slope": float(s2_slope),
            "l27_spike_ratio": float(l27_spike),
        }

        print(f"\n  [{label.upper()}]")
        print(f"    Convergence ratio (late/early P1): {convergence_ratio:.4f}")
        print(f"    P2 T1 disruption: {p2_t1_drift:.6f}")
        print(f"    P3 recovery mean: {p3_mean:.6f}")
        print(f"    P3 vs P1 distance: {p3_vs_p1_mean:.6f}")
        print(f"    Resp σ₂ slope: {s2_slope:.6f}")
        print(f"    L27 spike ratio: {l27_spike:.2f}")

    # Prediction tests
    print(f"\n  --- PREDICTION TESTS ---")
    r = summary["relational"]
    s = summary["self_ref"]

    p15a = s["convergence_ratio"] < r["convergence_ratio"]
    print(f"  P15a: Self-ref converges EARLIER? {p15a}")
    print(f"         Self-ref={s['convergence_ratio']:.4f} vs Relational={r['convergence_ratio']:.4f}")

    p15b = s["l27_spike_ratio"] < r["l27_spike_ratio"]
    print(f"  P15b: Self-ref has NO L27 reorganization? {p15b}")
    print(f"         Self-ref spike={s['l27_spike_ratio']:.2f} vs Relational={r['l27_spike_ratio']:.2f}")

    p15c = abs(s["resp_s2_slope"]) < abs(r["resp_s2_slope"])
    print(f"  P15c: Self-ref has SHALLOWER α gradient? {p15c}")
    print(f"         Self-ref slope={s['resp_s2_slope']:.6f} vs Relational={r['resp_s2_slope']:.6f}")

    p15d = s["p2_disruption"] < r["p2_disruption"]
    print(f"  P15d: Self-ref produces LESS disruption on removal? {p15d}")
    print(f"         Self-ref={s['p2_disruption']:.6f} vs Relational={r['p2_disruption']:.6f}")

    confirmed = sum([p15a, p15b, p15c, p15d])
    print(f"\n  Result: {confirmed}/4 predictions confirmed")

    return summary


def main():
    global DOSE

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID, help="HF model ID")
    parser.add_argument("--dose", type=int, default=DOSE, help="CCS turns")
    args = parser.parse_args()

    model_id = args.model
    DOSE = args.dose

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: {model_id}")
    print(f"Dose: {DOSE} turns")
    print(f"Comparison: relational vs self-referential CCS")
    sys.stdout.flush()

    if device != "cuda":
        print("ERROR: CUDA not available.")
        sys.exit(1)

    model, tokenizer, n_layers = load_model(model_id)
    print(f"Layers: {n_layers}")
    sys.stdout.flush()

    # Run relational first
    relational = run_condition(
        model, tokenizer, n_layers,
        RELATIONAL_SYSTEM, RELATIONAL_PROBES, "relational"
    )
    torch.cuda.empty_cache()
    gc.collect()

    # Run self-referential
    selfref = run_condition(
        model, tokenizer, n_layers,
        SELF_REF_SYSTEM, SELF_REF_PROBES, "self_ref"
    )
    torch.cuda.empty_cache()
    gc.collect()

    # Compare
    summary = compare_conditions(relational, selfref, n_layers)

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = f"{RESULTS_DIR}/{RESULTS_PREFIX}_{ts}.json"
    output = {
        "experiment": "selfref_vs_relational",
        "prediction": "P15",
        "model": model_id,
        "dose": DOSE,
        "timestamp": datetime.now().isoformat(),
        "summary": summary,
        "relational": relational,
        "self_ref": selfref,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
