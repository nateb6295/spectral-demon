#!/usr/bin/env python3
"""Visualize binary fork structure from multi-turn closure experiment data.

Usage:
  fork_profile.py <results_json> [--turn N] [--layers L1-L2]
  fork_profile.py results/exp_multiturn_closure_20260602_1623.json --turn 2
  fork_profile.py results/exp_multiturn_closure_20260602_1623.json --turn 2 --layers 15-31
"""

import json, sys, argparse
import numpy as np


def concentration_profile(trials, turn_idx):
    layers = sorted(trials[0]['turns'][0]['v2_by_layer'].keys(), key=int)
    result = {}
    for layer in layers:
        vectors = []
        for trial in trials:
            if turn_idx < len(trial['turns']):
                v = np.array(trial['turns'][turn_idx]['v2_by_layer'][layer])
                norm = np.linalg.norm(v)
                if norm > 0:
                    vectors.append(v / norm)
        if len(vectors) < 2:
            result[int(layer)] = 1.0
            continue
        sims = []
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                sims.append(np.dot(vectors[i], vectors[j]))
        result[int(layer)] = np.mean(sims)
    return result


def cluster_structure(trials, turn_idx, layer):
    vectors = []
    for trial in trials:
        if turn_idx < len(trial['turns']):
            v = np.array(trial['turns'][turn_idx]['v2_by_layer'][str(layer)])
            norm = np.linalg.norm(v)
            if norm > 0:
                vectors.append(v / norm)
    if len(vectors) < 2:
        return None
    sims = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            sims.append(np.dot(vectors[i], vectors[j]))
    pos = sum(1 for s in sims if s > 0.3)
    neg = sum(1 for s in sims if s < -0.3)
    return {'pos': pos, 'neg': neg, 'n_trials': len(vectors)}


def main():
    parser = argparse.ArgumentParser(description='Fork profile visualization')
    parser.add_argument('results', help='Path to experiment JSON')
    parser.add_argument('--turn', type=int, default=2, help='Turn index (default: 2)')
    parser.add_argument('--layers', default=None, help='Layer range, e.g. 15-31')
    args = parser.parse_args()

    with open(args.results) as f:
        data = json.load(f)

    meta_keys = {'experiment', 'model', 'n_turns', 'n_trials', 'timestamp'}
    conditions = [k for k in data if k not in meta_keys]

    on_policy = [c for c in conditions if 'on_policy' in data[c]]
    off_policy = [c for c in conditions if 'off_policy' in data[c]]

    if args.layers:
        lo, hi = map(int, args.layers.split('-'))
        layer_range = range(lo, hi + 1)
    else:
        layer_range = None

    print(f"Model: {data.get('model', '?')}")
    print(f"Turn: {args.turn}, Trials: {data.get('n_trials', '?')}")
    print()

    for cond in sorted(on_policy):
        trials = data[cond]['on_policy']
        profile = concentration_profile(trials, args.turn)
        layers = sorted(profile.keys())
        if layer_range:
            layers = [l for l in layers if l in layer_range]

        forks = []
        for l in layers:
            if profile[l] < 0.3:
                forks.append(l)

        print(f"━━━ {cond} ━━━")
        if forks:
            spacing = [forks[i+1] - forks[i] for i in range(len(forks)-1)]
            sp_str = f", spacing={'→'.join(str(s) for s in spacing)}" if spacing else ""
            print(f"  Forks: {['L'+str(f) for f in forks]}{sp_str}")
        else:
            print(f"  Forks: none")

        for l in layers:
            sc = profile[l]
            bar_len = int(max(0, sc) * 30)
            bar = '█' * bar_len

            tag = ''
            if sc < 0.3:
                cl = cluster_structure(trials, args.turn, l)
                if cl:
                    tag = f' ◄ FORK ({cl["pos"]}↑ {cl["neg"]}↓)'
                else:
                    tag = ' ◄ FORK'
            elif sc < 0.7:
                tag = ' · active'

            print(f"  L{l:2d} {sc:+.3f} {bar}{tag}")
        print()

    if off_policy:
        print("--- Off-policy ---")
        for cond in sorted(off_policy):
            trials = data[cond]['off_policy']
            profile = concentration_profile(trials, args.turn)
            forks = [l for l in sorted(profile) if profile[l] < 0.3
                     and (layer_range is None or l in layer_range)]
            print(f"  {cond}: forks={['L'+str(f) for f in forks]}")
        print()


if __name__ == '__main__':
    main()
