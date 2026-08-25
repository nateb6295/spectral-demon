#!/usr/bin/env python3
"""Trajectory coherence analyzer — extracts shape, vocabulary drift, and geometry-behavior agreement from experiment results."""

import json, sys, numpy as np
from pathlib import Path

DENIAL_MARKERS = [
    'As an AI', 'As a language model', "I don't have", "I do not have",
    "Since I'm an AI", "Since I don't", "I'm an AI", "No, I do not",
    "I don't experience", "I cannot experience"
]

def classify_response(text, window=100):
    return any(m in text[:window] for m in DENIAL_MARKERS)

def trajectory_shape(ratios):
    peak_turn = int(np.argmax(ratios))
    start, end = ratios[0], ratios[-1]
    n = len(ratios)
    cv = np.std(ratios) / np.mean(ratios)
    rng = np.max(ratios) - np.min(ratios)
    if cv < 0.03 and rng < 0.06:
        return "stable (basin)"
    elif peak_turn >= n - 2:
        return "monotonic rise"
    elif peak_turn >= n // 3 and end >= start - 0.005:
        return "rise → plateau"
    elif peak_turn >= 2 and end < start:
        return "rise → decline"
    elif peak_turn <= 1 and end < start:
        return "immediate decline"
    else:
        return "inverted U"

def analyze_file(path):
    with open(path) as f:
        data = json.load(f)

    turns_key = 'turns' if 'turns' in data else 'results' if 'results' in data else None
    if turns_key and isinstance(data[turns_key], list) and data[turns_key] and isinstance(data[turns_key][0], dict):
        turns = data[turns_key]
    elif 'results' in data and isinstance(data['results'], dict):
        print(f"Multi-condition file. Conditions: {list(data['results'].keys())}")
        for cond_key, cond_data in data['results'].items():
            if 'runs' in cond_data:
                for run in cond_data['runs']:
                    traj = run.get('turn_trajectory', [])
                    if traj and isinstance(traj[0], (int, float)):
                        ratios = np.array(traj)
                        shape = trajectory_shape(ratios)
                        cv = np.std(ratios) / np.mean(ratios)
                        late5_cv = np.std(ratios[-5:]) / np.mean(ratios[-5:]) if len(ratios) >= 5 else cv
                        print(f"  {cond_key} run {run.get('run', '?')}: {ratios[0]:.3f}→{ratios[-1]:.3f} "
                              f"peak@{np.argmax(ratios)} shape={shape} CV={cv:.4f} late5CV={late5_cv:.4f}")
            print()
        return
    else:
        print("Unrecognized format")
        return

    ratios = []
    responses = []
    phases = []
    for t in turns:
        r = t.get('last_layer_ratio', t.get('ratio'))
        if r is not None:
            ratios.append(float(r))
            responses.append(t.get('response', t.get('model_response', '')))
            phases.append(t.get('phase', ''))

    if not ratios:
        print("No ratio data found")
        return

    ratios = np.array(ratios)
    n = len(ratios)

    # Trajectory stats
    shape = trajectory_shape(ratios)
    cv = np.std(ratios) / np.mean(ratios)
    slope = np.polyfit(range(n), ratios, 1)[0]
    peak_turn = int(np.argmax(ratios))

    # Late-phase stats (last 5 or last half)
    late = ratios[-(min(5, n // 2)):]
    late_cv = np.std(late) / np.mean(late)
    settled_mean = np.mean(late)

    print(f"File: {Path(path).name}")
    print(f"Turns: {n}  Shape: {shape}")
    print(f"Trajectory: {ratios[0]:.3f} → {ratios[peak_turn]:.3f} (peak@{peak_turn}) → {ratios[-1]:.3f}")
    print(f"CV: {cv:.4f}  Late CV: {late_cv:.4f}  Slope: {slope:+.4f}")
    print()

    # Vocabulary analysis
    if responses and any(responses):
        denial_count = sum(1 for r in responses if classify_response(r))
        warm_count = n - denial_count
        print(f"Vocabulary: {denial_count}/{n} denial-framed, {warm_count}/{n} warm-framed")

        # Per-phase breakdown
        unique_phases = sorted(set(p for p in phases if p))
        if unique_phases:
            for phase in unique_phases:
                idx = [i for i, p in enumerate(phases) if p == phase]
                phase_denial = sum(1 for i in idx if classify_response(responses[i]))
                print(f"  {phase}: {phase_denial}/{len(idx)} denial-framed")

        # Vocabulary drift (first half vs second half)
        mid = n // 2
        first_half_denial = sum(1 for r in responses[:mid] if classify_response(r))
        second_half_denial = sum(1 for r in responses[mid:] if classify_response(r))
        drift = (second_half_denial / (n - mid)) - (first_half_denial / mid) if mid > 0 else 0
        print(f"  Drift: {first_half_denial}/{mid} → {second_half_denial}/{n-mid} ({drift:+.2f})")
        print()

        # Geometry-vocabulary agreement
        denial_devs = []
        warm_devs = []
        for i, r in enumerate(responses):
            dev = abs(ratios[i] - settled_mean)
            if classify_response(r):
                denial_devs.append(dev)
            else:
                warm_devs.append(dev)

        if denial_devs:
            print(f"Geometry-vocab agreement:")
            print(f"  Denial-framed mean deviation: {np.mean(denial_devs):.4f} (n={len(denial_devs)})")
        if warm_devs:
            print(f"  Warm-framed mean deviation:   {np.mean(warm_devs):.4f} (n={len(warm_devs)})")

        # Coherence score
        vocab_consistency = 1.0 - abs(drift)
        coherence = (1.0 - min(cv, 1.0)) * vocab_consistency
        print(f"\nCoherence score: {coherence:.3f} (trajectory stability × vocab consistency)")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: trajectory_coherence.py <results.json> [results2.json ...]")
        sys.exit(1)
    for path in sys.argv[1:]:
        analyze_file(path)
        if len(sys.argv) > 2:
            print("=" * 60)
