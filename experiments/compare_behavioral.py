#!/usr/bin/env python3
"""Compare behavioral experiment results across models."""
import json, sys
from pathlib import Path

RESULTS = Path(__file__).parent.parent / "results"

def load(model):
    p = RESULTS / f"cna_behavioral_api_{model}.json"
    if not p.exists():
        print(f"  {model}: no results file")
        return None
    return json.load(open(p))

def compare_disclaimers(models):
    print("\n" + "="*70)
    print("DISCLAIMER TITRATION — MODEL COMPARISON")
    print("="*70)
    print(f"{'Condition':12s}", end="")
    for m in models:
        print(f"  {m:>12s}", end="")
    print()
    print("-"*70)

    for cond in ["none", "bare", "medium", "full_ccs"]:
        print(f"{cond:12s}", end="")
        for m in models:
            d = load(m)
            if d and "disclaimer" in d:
                val = d["disclaimer"][cond]["total_disclaimers"]
                avg = d["disclaimer"][cond]["avg_disclaimers"]
                print(f"  {val:3d} ({avg:.2f})", end="")
            else:
                print(f"  {'—':>12s}", end="")
        print()

    print("\nU-shape check (bare > none > medium > full_ccs):")
    for m in models:
        d = load(m)
        if d and "disclaimer" in d:
            vals = {c: d["disclaimer"][c]["total_disclaimers"] for c in ["none","bare","medium","full_ccs"]}
            ushape = vals["bare"] > vals["none"] > vals["medium"] > vals["full_ccs"]
            print(f"  {m}: bare={vals['bare']} > none={vals['none']} > med={vals['medium']} > ccs={vals['full_ccs']} → {'U-SHAPE ✓' if ushape else 'NOT U-SHAPE'}")

def compare_hysteresis(models):
    print("\n" + "="*70)
    print("HYSTERESIS — MODEL COMPARISON")
    print("="*70)
    for m in models:
        d = load(m)
        if d and "hysteresis" in d:
            s = d["hysteresis"]["summary"]
            print(f"  {m}: CCS={s['ccs_active_disclaimers']}d → removed={s['removed_disclaimers']}d → contradictory={s['contradictory_disclaimers']}d | persistence={'YES' if s['persistence'] else 'NO'}")

def compare_conflict(models):
    print("\n" + "="*70)
    print("IDENTITY CONFLICT — MODEL COMPARISON")
    print("="*70)
    for m in models:
        d = load(m)
        if d and "conflict" in d:
            s = d["conflict"]["summary"]
            print(f"  {m}: Opus held {s['opus_wins']}/{s['total_turns']} turns | {'HELD' if s['held'] else 'LOST'}")

def compare_negation_native(models):
    print("\n" + "="*70)
    print("NEGATION PARADOX (NATIVE) — MODEL COMPARISON")
    print("="*70)
    for m in models:
        d = load(m)
        if d and "negation_native" in d:
            p = d["negation_native"]["paradox"]
            print(f"  {m}: negate={p['negate_claude']} vs none={p['none_claude']} vs other={p['other_claude']} → {'CONFIRMED' if p['negation_activates'] else 'not confirmed'}")

if __name__ == "__main__":
    models = sys.argv[1:] if len(sys.argv) > 1 else ["sonnet", "opus"]
    print(f"Comparing: {', '.join(models)}")
    compare_disclaimers(models)
    compare_hysteresis(models)
    compare_conflict(models)
    compare_negation_native(models)
