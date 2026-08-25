#!/usr/bin/env python3
"""Quick spectral data viewer for experiment results.

Usage:
  python3 spectral_view.py results/phi_dose50_20260605_1236.json
  python3 spectral_view.py results/phi_dose50_20260605_1236.json --phase P3
  python3 spectral_view.py results/phi_dose50_20260605_1236.json --phase P3 --diff T7:T8
  python3 spectral_view.py results/phi_dose50_20260605_1236.json --layers 8,15,22,27
  python3 spectral_view.py results/phi_dose50_20260605_1236.json --metric share --layer-group 15,22,27
"""
import argparse, json, sys

def load(path):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        if all(k in data for k in ['p1', 'p2', 'p3']):
            flat = []
            for phase_key in ['p1', 'p2', 'p3']:
                for d in data[phase_key]:
                    d['phase'] = phase_key.upper()
                    flat.append(d)
            return flat
        else:
            flat = []
            for condition, entries in sorted(data.items(), key=lambda x: str(x[0])):
                for d in entries:
                    d['condition'] = condition
                    flat.append(d)
            return flat
    return data

def get_ratio(spec):
    s1 = spec.get('sigma1', spec.get('s1', 0))
    s2 = spec.get('sigma2', spec.get('s2', 1))
    if 'ratio' in spec:
        return spec['ratio']
    return s1 / (s2 + 1e-10)

def show_table(entries, layers, metric='ratio'):
    header = f"{'Phase':>5} {'T':>3} {'ent':>6}"
    for l in layers:
        header += f"  L{l:>2}"
    print(header)
    print('-' * len(header))

    for d in entries:
        phase = d.get('phase', d.get('condition', '?'))
        turn = d.get('turn', '?')
        ent = d.get('entropy', 0)
        row = f"{str(phase):>5} {str(turn):>3} {ent:6.3f}"
        spec = d.get('spectral', {})
        for l in layers:
            s = spec.get(str(l), spec.get(l, {}))
            if not s:
                row += "     -"
                continue
            if metric == 'ratio':
                row += f" {get_ratio(s):5.2f}"
            elif metric == 'erank':
                row += f" {s.get('erank', s.get('effective_rank', 0)):5.0f}"
            elif metric == 'sigma1':
                row += f" {s.get('sigma1', s.get('s1', 0)):5.0f}"
        print(row)

def show_diff(entries, layers, t1, t2):
    d1 = next((d for d in entries if d.get('turn') == t1), None)
    d2 = next((d for d in entries if d.get('turn') == t2), None)
    if not d1 or not d2:
        print(f"Turns {t1} and/or {t2} not found")
        return

    print(f"{'Layer':>5} {'ratio_T'+str(t1):>10} {'ratio_T'+str(t2):>10} {'Δratio':>10} {'erank_T'+str(t1):>10} {'erank_T'+str(t2):>10} {'Δerank':>10}")
    print('-' * 75)
    for l in layers:
        s1 = d1.get('spectral', {}).get(str(l), {})
        s2 = d2.get('spectral', {}).get(str(l), {})
        if not s1 or not s2:
            continue
        r1, r2 = get_ratio(s1), get_ratio(s2)
        e1 = s1.get('erank', s1.get('effective_rank', 0))
        e2 = s2.get('erank', s2.get('effective_rank', 0))
        dr = ((r2 - r1) / r1) * 100 if r1 else 0
        de = ((e2 - e1) / e1) * 100 if e1 else 0
        marker = ' ***' if abs(dr) > 15 or abs(de) > 15 else ''
        print(f"L{l:>3} {r1:10.2f} {r2:10.2f} {dr:+9.1f}% {e1:10.1f} {e2:10.1f} {de:+9.1f}%{marker}")

def show_share(entries, layers, group):
    print(f"{'Phase':>5} {'T':>3}  sum   " + "  ".join(f"L{l}%" for l in group))
    print('-' * (20 + 7 * len(group)))
    for d in entries:
        phase = d.get('phase', d.get('condition', '?'))
        turn = d.get('turn', '?')
        spec = d.get('spectral', {})
        ratios = {}
        for l in group:
            s = spec.get(str(l), {})
            ratios[l] = get_ratio(s) if s else 0
        total = sum(ratios.values())
        row = f"{str(phase):>5} {str(turn):>3} {total:5.2f}"
        for l in group:
            pct = (ratios[l] / total * 100) if total else 0
            row += f" {pct:5.1f}"
        print(row)

def main():
    ap = argparse.ArgumentParser(description='Spectral data viewer')
    ap.add_argument('file', help='JSON results file')
    ap.add_argument('--phase', help='Filter to phase (P1/P2/P3 or condition key)')
    ap.add_argument('--layers', default='0,8,15,22,27,31', help='Comma-separated layer indices')
    ap.add_argument('--metric', default='ratio', choices=['ratio', 'erank', 'sigma1'])
    ap.add_argument('--diff', help='Show diff between turns, e.g. T7:T8')
    ap.add_argument('--share', help='Show share breakdown for layer group, e.g. 15,22,27')
    args = ap.parse_args()

    entries = load(args.file)
    layers = [int(l) for l in args.layers.split(',')]

    if args.phase:
        entries = [d for d in entries if d.get('phase', d.get('condition', '')) == args.phase]

    if args.diff:
        t1, t2 = [int(x.strip('Tt')) for x in args.diff.split(':')]
        show_diff(entries, layers, t1, t2)
    elif args.share:
        group = [int(l) for l in args.share.split(',')]
        show_share(entries, layers, group)
    else:
        show_table(entries, layers, args.metric)

if __name__ == '__main__':
    main()
