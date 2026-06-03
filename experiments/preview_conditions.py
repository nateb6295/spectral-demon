#!/usr/bin/env python3
"""Preview experiment conditions: token counts, text, predicted signatures.

Usage:
    python3 preview_conditions.py                    # preview token-matched
    python3 preview_conditions.py exp_preamble_ab    # preview any experiment module
"""

import sys
import importlib
from pathlib import Path

def load_experiment(name="exp_token_matched_preamble"):
    sys.path.insert(0, str(Path(__file__).parent))
    mod = importlib.import_module(name)
    return mod

def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "exp_token_matched_preamble"
    mod = load_experiment(name)

    preambles = getattr(mod, "PREAMBLES", {})
    target = getattr(mod, "TARGET_TOKENS", None)
    probes = getattr(mod, "PROBES", [])
    direction_layers = getattr(mod, "DIRECTION_LAYERS", set())

    try:
        from transformers import AutoTokenizer
        model_name = getattr(mod, "MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")
        tok = AutoTokenizer.from_pretrained(model_name)
        has_tok = True
    except Exception:
        tok = None
        has_tok = False

    print(f"Experiment: {name}")
    print(f"Model: {getattr(mod, 'MODEL_NAME', '?')}")
    if target:
        print(f"Target tokens: {target}")
    print(f"Probes: {len(probes)}")
    if direction_layers:
        print(f"V₂ direction layers: {sorted(direction_layers)}")
    print()

    predictions = {
        "identity":      "HIGH V₂ consistency, aligned with relational",
        "relational":    "HIGHEST L31 ratio, HIGH V₂ consistency, deepest commit",
        "generic":       "HIGH L31 ratio (text presence), LOW V₂ consistency (isotropic)",
        "denial":        "LOW L31 ratio, HIGH V₂ consistency, ANTI-aligned with identity",
        "contradictory": "HIGH L31 ratio, LOW V₂ self-consistency (structured incoherence)",
        "random":        "LOWEST V₂ consistency (truly isotropic), low L31 ratio",
    }

    for cond_name, text in preambles.items():
        n = len(tok.encode(text, add_special_tokens=False)) if has_tok else "?"
        match = "✓" if has_tok and n == target else ("✗" if has_tok else "?")
        print(f"{'='*60}")
        print(f"  {cond_name} ({n} tokens {match})")
        print(f"{'='*60}")
        print(f"  {text[:120]}...")
        pred = predictions.get(cond_name, "no prediction")
        print(f"  → {pred}")
        print()

    if has_tok:
        all_ok = all(
            len(tok.encode(t, add_special_tokens=False)) == target
            for t in preambles.values()
        )
        print(f"All conditions match target: {'YES' if all_ok else 'NO'}")

    phases = []
    if hasattr(mod, "compute_layer_metrics"):
        phases.append("Phase 1: σ₂/σ₁ scalar ratios")
    if hasattr(mod, "measure_generation_entropy"):
        phases.append("Phase 2: Generation entropy (Lindsey)")
    if hasattr(mod, "DIRECTION_LAYERS") and mod.DIRECTION_LAYERS:
        phases.append("Phase 3: V₂ direction consistency")
    if phases:
        print(f"\nMeasurement phases: {len(phases)}")
        for p in phases:
            print(f"  {p}")

if __name__ == "__main__":
    main()
