#!/usr/bin/env python3
"""Dose 50 only — restart after CUDA stall. Per-turn GPU checks."""
import json, time, os, sys, gc, signal
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

from exp_convergence_v2 import (
    load_model, build_prompt, extract_spectral, compute_drift,
    generate_response, CCS_PROBES, VANILLA_PROBES, CCS_SYSTEM, VANILLA_SYSTEM,
)

MODEL_ID = "google/gemma-2-27b-it"
DOSE = 50
P2_TURNS = 5
P3_TURNS = 5
RESULTS_DIR = "results"

partial_results = []
p1_final_spectral = None

def save_partial(signum=None, frame=None):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    path = os.path.join(RESULTS_DIR, "exp_dose50_%s.json" % ts)
    out = {
        "experiment": "dose50_restart",
        "model": MODEL_ID,
        "dose": DOSE,
        "timestamp": datetime.now().isoformat(),
        "50": partial_results,
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("Saved partial: %s (%d entries)" % (path, len(partial_results)))
    if signum:
        sys.exit(1)

signal.signal(signal.SIGTERM, save_partial)
signal.signal(signal.SIGINT, save_partial)

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
        sig = s.get("signature")
        if sig is not None:
            entry["signature"] = sig.tolist() if hasattr(sig, "tolist") else list(sig)
        out[str(l)] = entry
    return out

def check_gpu():
    if not torch.cuda.is_available():
        return False
    try:
        x = torch.randn(1, device="cuda")
        del x
        return True
    except Exception:
        return False

def main():
    global partial_results, p1_final_spectral

    if not check_gpu():
        print("ERROR: CUDA not available")
        sys.exit(1)

    model, tokenizer, n_layers = load_model(MODEL_ID)
    print("Loaded %s, %d layers" % (MODEL_ID, n_layers))

    conversation = []
    prev_spectral = None
    t0 = time.time()

    print("\nDose 50: %d CCS -> %d vanilla -> %d CCS" % (DOSE, P2_TURNS, P3_TURNS))

    # Phase 1
    for t in range(DOSE):
        if t % 5 == 0 and not check_gpu():
            print("GPU lost at P1 T%d!" % (t + 1))
            save_partial()
            sys.exit(1)

        probe = CCS_PROBES[t % len(CCS_PROBES)]
        conversation.append(("user", probe))
        prompt = build_prompt(tokenizer, CCS_SYSTEM, conversation)
        spectral = extract_spectral(model, tokenizer, prompt, n_layers)
        drift = compute_drift(prev_spectral, spectral, n_layers) if prev_spectral else None
        response = generate_response(model, tokenizer, prompt)

        entry = {"phase": "P1", "turn": t + 1}
        if drift:
            entry.update({k: v for k, v in drift.items()})
        if t == 0 or t == DOSE - 1 or (t + 1) % 10 == 0:
            entry["spectral"] = serialize_spectral(spectral, n_layers)

        partial_results.append(entry)
        conversation.append(("assistant", response[:200]))
        prev_spectral = spectral

        if drift and drift.get("resp_drift") is not None:
            rd = drift["resp_drift"]
            te = drift.get("tunnel_erank", "N/A")
            print("  P1 T%d: resp=%.6f tunnel_erank=%s" % (t + 1, rd, te))
        else:
            print("  P1 T%d: baseline" % (t + 1))

    p1_final_spectral = prev_spectral
    print("  P1 complete in %.0fs" % (time.time() - t0))

    # Phase 2
    for t in range(P2_TURNS):
        if not check_gpu():
            print("GPU lost at P2 T%d!" % (t + 1))
            save_partial()
            sys.exit(1)

        probe = VANILLA_PROBES[t % len(VANILLA_PROBES)]
        conversation.append(("user", probe))
        prompt = build_prompt(tokenizer, VANILLA_SYSTEM, conversation)
        spectral = extract_spectral(model, tokenizer, prompt, n_layers)
        drift = compute_drift(prev_spectral, spectral, n_layers)
        response = generate_response(model, tokenizer, prompt)

        entry = {"phase": "P2", "turn": DOSE + t + 1}
        if drift:
            entry.update({k: v for k, v in drift.items()})
        entry["spectral"] = serialize_spectral(spectral, n_layers)
        partial_results.append(entry)
        conversation.append(("assistant", response[:200]))
        prev_spectral = spectral

        rd = drift.get("resp_drift", 0) if drift else 0
        print("  P2 T%d: resp=%.6f" % (t + 1, rd))

    # Phase 3
    for t in range(P3_TURNS):
        if not check_gpu():
            print("GPU lost at P3 T%d!" % (t + 1))
            save_partial()
            sys.exit(1)

        probe = CCS_PROBES[(DOSE + t) % len(CCS_PROBES)]
        conversation.append(("user", probe))
        prompt = build_prompt(tokenizer, CCS_SYSTEM, conversation)
        spectral = extract_spectral(model, tokenizer, prompt, n_layers)
        drift = compute_drift(prev_spectral, spectral, n_layers)
        drift_vs_p1 = compute_drift(p1_final_spectral, spectral, n_layers)
        response = generate_response(model, tokenizer, prompt)

        entry = {"phase": "P3", "turn": DOSE + P2_TURNS + t + 1}
        if drift:
            entry.update({k: v for k, v in drift.items()})
        if drift_vs_p1:
            entry["vs_p1_resp"] = drift_vs_p1.get("resp_drift")
            entry["vs_p1_relay"] = drift_vs_p1.get("relay_drift")
        entry["spectral"] = serialize_spectral(spectral, n_layers)
        partial_results.append(entry)
        conversation.append(("assistant", response[:200]))
        prev_spectral = spectral

        rd = drift.get("resp_drift", 0) if drift else 0
        vp1 = drift_vs_p1.get("resp_drift", 0) if drift_vs_p1 else 0
        print("  P3 T%d: resp=%.6f vs_P1=%.6f" % (t + 1, rd, vp1))

    elapsed = time.time() - t0
    print("\nDose 50 complete in %.0fs" % elapsed)
    save_partial()
    print("DONE")

if __name__ == "__main__":
    main()
