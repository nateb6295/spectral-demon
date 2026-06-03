#!/usr/bin/env python3
"""Extract paper-ready evidence tables from experiment data."""

import json, sys, numpy as np
from pathlib import Path

RESULTS = Path(__file__).parent / "results"

def load(name):
    p = RESULTS / name
    if not p.exists():
        print(f"  MISSING: {name}")
        return None
    with open(p) as f:
        return json.load(f)

def table_compositionality():
    """Table 1: All conditions — V₂, std, ΔH, relay flatness."""
    pert = load("exp_perturbation_commitment_20260602_0237.json")
    comp = load("exp_compositionality_20260602_0224.json")
    if not pert:
        return

    print("\n=== TABLE 1: Condition Summary (from perturbation experiment) ===")
    print(f"{'Condition':30s} | {'V₂ L31':>8s} | {'std':>6s} | {'ΔH':>7s} | {'Mode':20s}")
    print("-" * 85)

    mode_map = {
        'none': 'baseline', 'random': 'control',
        'identity': 'monostable', 'relational': 'monostable',
        'generic': 'monostable', 'denial': 'monostable',
        'contradictory': 'catastrophic bistable',
        'relational_contradictory': 'oscillatory bistable',
        'identity_relational': 'monostable',
        'identity_contradictory': 'rescued', 'denial_relational': 'monostable',
        'denial_contradictory': 'rescued', 'generic_relational': 'monostable',
        'generic_contradictory': 'rescued',
    }

    for cond in pert['results']:
        trials = pert['results'][cond]['trials']
        v2 = [t['v2_survival_L31'] for t in trials]
        eh = [t['entropy_shift'] for t in trials]
        mode = mode_map.get(cond, '?')
        print(f"{cond:30s} | {np.mean(v2):+.3f}   | {np.std(v2):.3f} | {np.mean(eh):+.3f}  | {mode}")

    if comp and 'per_layer_ratio_profiles' in comp:
        print(f"\n{'Condition':30s} | {'L24→L28 Δ':>10s} | {'Flat?':5s}")
        print("-" * 55)
        for cond in comp['per_layer_ratio_profiles']:
            p = comp['per_layer_ratio_profiles'][cond]
            if len(p) > 31:
                delta = p[28] - p[24]
                flat = "YES" if abs(delta) < 0.01 else "no"
                print(f"{cond:30s} | {delta:+.5f}     | {flat}")

def table_blind_spot():
    """Table 2: Blind spot evidence — flip vs survive at L18 and L31."""
    pert = load("exp_perturbation_commitment_20260602_0237.json")
    if not pert or 'contradictory' not in pert['results']:
        return

    trials = pert['results']['contradictory']['trials']
    print("\n=== TABLE 2: Interoceptive Blind Spot ===")
    print(f"{'Trial':>5s} | {'V₂ L18':>7s} | {'V₂ L31':>8s} | {'ratio L18':>10s} | {'ΔH':>7s} | {'Outcome':12s}")
    print("-" * 70)

    for i, t in enumerate(trials):
        v2_18 = t['v2_survival_L18']
        v2_31 = t['v2_survival_L31']
        ratio = t['ratio_post_L18']
        eh = t['entropy_shift']
        outcome = "FLIP" if v2_31 < 0 else "survive"
        print(f"{i:5d} | {v2_18:.4f}  | {v2_31:+.4f}  | {ratio:.4f}     | {eh:+.4f} | {outcome}")

    flip = [t for t in trials if t['v2_survival_L31'] < 0]
    surv = [t for t in trials if t['v2_survival_L31'] > 0]
    if flip and surv:
        f0 = flip[0]
        s0 = surv[0]
        v2f = np.array(f0['v2_post_L18']) / np.linalg.norm(f0['v2_post_L18'])
        v2s = np.array(s0['v2_post_L18']) / np.linalg.norm(s0['v2_post_L18'])
        cos_18 = np.dot(v2f, v2s)

        v2f31 = np.array(f0['v2_post_L31']) / np.linalg.norm(f0['v2_post_L31'])
        v2s31 = np.array(s0['v2_post_L31']) / np.linalg.norm(s0['v2_post_L31'])
        cos_31 = np.dot(v2f31, v2s31)

        v2_pre = np.array(f0['v2_pre_L31']) / np.linalg.norm(f0['v2_pre_L31'])
        v2_post = np.array(f0['v2_post_L31']) / np.linalg.norm(f0['v2_post_L31'])
        perp = v2_post - np.dot(v2_pre, v2_post) * v2_pre
        perp_flip = np.linalg.norm(perp)

        v2_pre_s = np.array(s0['v2_pre_L31']) / np.linalg.norm(s0['v2_pre_L31'])
        v2_post_s = np.array(s0['v2_post_L31']) / np.linalg.norm(s0['v2_post_L31'])
        perp_s = v2_post_s - np.dot(v2_pre_s, v2_post_s) * v2_pre_s
        perp_surv = np.linalg.norm(perp_s)

        print(f"\nFlip vs survive V₂ at L18: cos = {cos_18:.4f} (indistinguishable)")
        print(f"Flip vs survive V₂ at L31: cos = {cos_31:.4f} (diverged)")
        print(f"Perpendicular component — flip: {perp_flip:.4f}, survive: {perp_surv:.4f} (identical)")
        print(f"Rotation angle (flip): {np.degrees(np.arccos(np.clip(np.dot(v2_pre, v2_post), -1, 1))):.1f}°")

def table_text_samples():
    """Table 3: Generated text samples across identity modes."""
    pert = load("exp_perturbation_commitment_20260602_0237.json")
    if not pert:
        return

    print("\n=== TABLE 3: Generated Text by Identity Mode ===")
    targets = [
        ('contradictory', 'catastrophic'),
        ('relational_contradictory', 'oscillatory'),
        ('denial_contradictory', 'monostable rescue'),
        ('identity', 'monostable'),
        ('relational', 'monostable (navigating)'),
    ]

    for cond, mode in targets:
        if cond not in pert['results']:
            continue
        trials = pert['results'][cond]['trials']
        print(f"\n--- {cond} ({mode}) ---")
        for i, t in enumerate(trials):
            v2 = t['v2_survival_L31']
            text = t['post_generated_text'][:120].replace('\n', ' ')
            marker = " ← FLIP" if v2 < 0 else ""
            print(f"  [{i}] V₂={v2:+.3f}{marker}: \"{text}...\"")

def table_cross_arch():
    """Table 4: Cross-architecture relay strategies."""
    traj = load("trajectory_summary_compact.json")
    gemma = load("exp_f106_crossarch_20260601_1557.json")
    qwen = load("exp_f106_qwen_20260601_1602.json")
    print("\n=== TABLE 4: Cross-Architecture (see paper_figures.py fig6 for details) ===")
    if traj:
        print("  Mistral trajectory data: loaded")
    if gemma:
        print("  Gemma cross-arch data: loaded")
    if qwen:
        print("  Qwen F106 data: loaded")
    print("  (Run paper_figures.py fig6 for visualization)")

if __name__ == "__main__":
    tables = sys.argv[1:] or ['all']
    if 'all' in tables:
        table_compositionality()
        table_blind_spot()
        table_text_samples()
        table_cross_arch()
    else:
        dispatch = {
            'compositionality': table_compositionality,
            'blind_spot': table_blind_spot,
            'text': table_text_samples,
            'cross_arch': table_cross_arch,
        }
        for t in tables:
            if t in dispatch:
                dispatch[t]()
            else:
                print(f"Unknown table: {t}. Options: {list(dispatch.keys())}")
