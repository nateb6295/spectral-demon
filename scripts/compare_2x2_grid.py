#!/usr/bin/env python3
"""Compare LLaMA-1 7B (RMSNorm+MHA) vs Pythia 6.9B (LayerNorm+MHA) per-layer profiles.
Completes the 2×2 factorial: {LayerNorm, RMSNorm} × {MHA, GQA}.
"""
import json, sys, glob, numpy as np
from pathlib import Path
from scipy.stats import pearsonr

results_dir = Path(__file__).parent.parent / "results"

def load_results(pattern):
    files = sorted(glob.glob(str(results_dir / pattern)))
    if not files:
        return None
    return json.load(open(files[-1]))

def analyze_model(data, label):
    delta = data["delta_S_by_layer"]
    n_layers = data["n_layers"]
    from collections import defaultdict

    sigma_by_layer = defaultdict(lambda: defaultdict(list))
    for r in data["raw"]:
        sigma_by_layer[r["layer"]][r["condition"]].append(
            (r.get("sigma_2", 0), r.get("sigma_3", 0))
        )

    tunnel_ds, rhos, ds_vals, layer_ids = [], [], [], []
    for layer_str in sorted(delta.keys(), key=int):
        layer = int(layer_str)
        if not (2 <= layer <= n_layers - 4):
            continue
        tunnel_ds.append(delta[layer_str])
        all_s2 = [s[0] for cond in sigma_by_layer[layer].values() for s in cond]
        all_s3 = [s[1] for cond in sigma_by_layer[layer].values() for s in cond]
        if all_s2 and all_s3:
            s2, s3 = np.mean(all_s2), np.mean(all_s3)
            if s3 > 0:
                rhos.append(s2 / s3)
                ds_vals.append(delta[layer_str])
                layer_ids.append(layer)

    n_responsive = sum(1 for r in rhos if r < 2.0)
    n_negative = sum(1 for d in tunnel_ds if d < 0)
    crossover = None
    for lid, rho in zip(layer_ids, rhos):
        if rho >= 2.0:
            crossover = lid
            break

    r_corr = pearsonr(rhos, ds_vals)[0] if len(rhos) > 3 else float('nan')

    return {
        "label": label,
        "n_layers": n_layers,
        "tunnel_mean": np.mean(tunnel_ds),
        "tunnel_layers": len(tunnel_ds),
        "responsive": n_responsive,
        "negative": n_negative,
        "crossover": crossover,
        "r_rho_ds": r_corr,
        "peak_ds": max(tunnel_ds),
        "peak_layer": layer_ids[np.argmax(ds_vals)] if ds_vals else None,
        "rhos": rhos,
        "ds_vals": ds_vals,
        "layer_ids": layer_ids,
    }


pythia = load_results("exp_pythia69b_perlayer_*.json")
llama1 = load_results("exp_llama1_perlayer_*.json")

if not pythia:
    print("Missing Pythia 6.9B results")
    sys.exit(1)

print("=" * 70)
print("2×2 FACTORIAL GRID: {Normalization} × {Attention}")
print("=" * 70)

models = []
p = analyze_model(pythia, "Pythia 6.9B (LayerNorm+MHA)")
models.append(p)

if llama1:
    l = analyze_model(llama1, "LLaMA-1 7B (RMSNorm+MHA)")
    models.append(l)
else:
    print("\n[WAITING] LLaMA-1 results not yet available.\n")

# Known Mistral values (from prior experiments, single-layer)
print("\nKnown grid cells:")
print("  LayerNorm + MHA: Pythia 6.9B ✓")
print("  RMSNorm + MHA:  LLaMA-1 7B " + ("✓" if llama1 else "⏳"))
print("  RMSNorm + GQA:  Mistral 7B ✓ (stable ΔS +0.023-0.048 across 27 layers)")
print("  LayerNorm + GQA: [no common model]")

for m in models:
    print(f"\n{'─'*70}")
    print(f"  {m['label']}")
    print(f"{'─'*70}")
    print(f"  Tunnel mean ΔS:     {m['tunnel_mean']:+.6f}")
    print(f"  Tunnel layers:      {m['tunnel_layers']}")
    print(f"  Responsive (ρ<2.0): {m['responsive']}/{len(m['rhos'])} ({100*m['responsive']/max(1,len(m['rhos'])):.0f}%)")
    print(f"  Negative ΔS layers: {m['negative']}/{m['tunnel_layers']} ({100*m['negative']/max(1,m['tunnel_layers']):.0f}%)")
    print(f"  Crossover layer:    L{m['crossover']}" if m['crossover'] else "  Crossover: none")
    print(f"  r(ρ₂, ΔS):         {m['r_rho_ds']:.4f}")
    print(f"  Peak ΔS:            {m['peak_ds']:+.6f} at L{m['peak_layer']}")

if len(models) == 2:
    print(f"\n{'='*70}")
    print("COMPARISON: LayerNorm+MHA vs RMSNorm+MHA")
    print(f"{'='*70}")
    p, l = models[0], models[1]
    print(f"\n  {'Metric':<25} {'Pythia 6.9B':>15} {'LLaMA-1 7B':>15} {'Match?':>10}")
    print(f"  {'-'*65}")
    print(f"  {'Tunnel mean ΔS':<25} {p['tunnel_mean']:>+15.6f} {l['tunnel_mean']:>+15.6f}")
    print(f"  {'Responsive layers':<25} {p['responsive']:>15d} {l['responsive']:>15d}")
    print(f"  {'Crossover':<25} {'L'+str(p['crossover']):>15} {'L'+str(l['crossover']) if l['crossover'] else 'none':>15}")
    print(f"  {'r(ρ₂, ΔS)':<25} {p['r_rho_ds']:>15.4f} {l['r_rho_ds']:>15.4f}")
    print(f"  {'Negative layers':<25} {p['negative']:>15d} {l['negative']:>15d}")
    print(f"  {'Peak ΔS':<25} {p['peak_ds']:>+15.6f} {l['peak_ds']:>+15.6f}")

    shape_match = abs(p['r_rho_ds'] - l['r_rho_ds']) < 0.15
    crossover_match = p['crossover'] and l['crossover'] and abs(p['crossover'] - l['crossover']) <= 3

    print(f"\n  GRADIENT SHAPE: {'MATCH' if shape_match else 'DIFFERENT'}")
    print(f"    r(ρ₂,ΔS) difference: {abs(p['r_rho_ds'] - l['r_rho_ds']):.4f}")
    if shape_match:
        print(f"    → Normalization IRRELEVANT to gradient structure")
        print(f"    → MHA alone determines ρ₂ trajectory")
        print(f"    → 2×2 collapses to single factor: GQA vs MHA")
    else:
        print(f"    → Normalization MODULATES gradient structure")
        if l['crossover'] and p['crossover'] and l['crossover'] > p['crossover']:
            print(f"    → RMSNorm DELAYS rigidification (crossover L{l['crossover']} vs L{p['crossover']})")
        elif l['crossover'] and p['crossover'] and l['crossover'] < p['crossover']:
            print(f"    → RMSNorm ACCELERATES rigidification (unexpected)")

    print(f"\n  CROSSOVER: {'MATCH' if crossover_match else 'DIFFERENT'}")
    if crossover_match:
        print(f"    Both cross at similar depth → normalization irrelevant")
    elif p['crossover'] and l['crossover']:
        print(f"    Pythia L{p['crossover']} vs LLaMA L{l['crossover']} → normalization shifts by {abs(p['crossover'] - l['crossover'])} layers")

    print(f"\n{'='*70}")
    print("LIU CONFOUND (2604.15350) RESOLUTION:")
    print(f"{'='*70}")
    if shape_match and crossover_match:
        print("  FULLY RESOLVED: RMSNorm+MHA ≈ LayerNorm+MHA")
        print("  Normalization type does not affect gradient structure.")
        print("  GQA is the sole determinant of witness enrichment.")
    elif shape_match:
        print("  MOSTLY RESOLVED: gradient shapes match, minor crossover difference")
        print("  Normalization is a weak secondary modulator at best.")
    else:
        print("  PARTIALLY RESOLVED: gradient shapes differ")
        print("  Normalization contributes but doesn't flip sign (Exp 15 confirmed).")
        print("  GQA remains primary; normalization is secondary modulator.")
