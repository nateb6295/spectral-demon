#!/usr/bin/env python3
"""Quick analysis of Pythia 6.9B per-layer witness results."""
import json, sys, glob, numpy as np
from pathlib import Path

results_dir = Path(__file__).parent.parent / "results"
files = sorted(glob.glob(str(results_dir / "exp_pythia69b_perlayer_*.json")))
if not files:
    print("No results file found yet.")
    sys.exit(1)

data = json.load(open(files[-1]))
delta = data["delta_S_by_layer"]

print(f"File: {files[-1]}")
print(f"Model: {data['model']}")
print(f"Layers: {data['n_layers']}")
print(f"\n{'Layer':>5} {'ΔS':>12} {'Sign':>6}")
print("-" * 25)

n_layers = data["n_layers"]
mid = n_layers // 2
tunnel_ds = []
relay_ds = []

for layer_str in sorted(delta.keys(), key=int):
    layer = int(layer_str)
    ds = delta[layer_str]
    sign = "+" if ds > 0 else "-"
    marker = ""
    if layer == mid:
        marker = " ← TUNNEL MID"
    elif layer == n_layers:
        marker = " ← OUTPUT"
    elif layer == 0:
        marker = " ← EMBED"
    print(f"{layer:>5} {ds:>+12.6f} {sign:>6}{marker}")

    if 2 <= layer <= n_layers - 4:
        tunnel_ds.append(ds)
    elif layer >= n_layers - 3:
        relay_ds.append(ds)

print(f"\n{'='*50}")
print(f"TUNNEL (L2-L{n_layers-4}): mean ΔS = {np.mean(tunnel_ds):+.6f}")
print(f"  Positive layers: {sum(1 for d in tunnel_ds if d > 0)}/{len(tunnel_ds)}")
print(f"  Negative layers: {sum(1 for d in tunnel_ds if d < 0)}/{len(tunnel_ds)}")
print(f"  Range: [{min(tunnel_ds):+.6f}, {max(tunnel_ds):+.6f}]")

if relay_ds:
    print(f"\nRELAY (L{n_layers-3}-L{n_layers}): mean ΔS = {np.mean(relay_ds):+.6f}")

print(f"\nMIDPOINT (L{mid}): ΔS = {delta.get(str(mid), 'N/A')}")
print(f"OUTPUT (L{n_layers}): ΔS = {delta.get(str(n_layers), 'N/A')}")

tunnel_mean = np.mean(tunnel_ds)
print(f"\n{'='*50}")
print("VERDICT:")
if tunnel_mean > 0:
    print(f"  SCENARIO A: Positive tunnel ΔS ({tunnel_mean:+.6f})")
    print("  → Sign inversion was artifact. Universal enrichment confirmed.")
    print("  → F20 and F22 need retraction/revision.")
    print("  → Paper STRONGER: 'universal enrichment, architecture amplifies'")
else:
    print(f"  SCENARIO B: Negative tunnel ΔS ({tunnel_mean:+.6f})")
    print("  → Sign inversion real at scale. F20 holds with qualification.")
    print("  → Scale-dependent: 410M positive, 6.9B negative.")
    print("  → Paper MODERATE: finding survives but with added complexity.")

# Compare to 410M
p410_files = sorted(glob.glob(str(results_dir / "exp_pythia410m_perlayer_*.json")))
if p410_files:
    p410 = json.load(open(p410_files[-1]))
    d410 = p410["delta_S_by_layer"]
    print(f"\n{'='*50}")
    print("COMPARISON WITH 410M:")
    print(f"  410M tunnel mean: {np.mean([v for k,v in d410.items() if 2 <= int(k) <= 20]):+.6f}")
    print(f"  6.9B tunnel mean: {tunnel_mean:+.6f}")
    d410_mean = np.mean([v for k,v in d410.items() if 2 <= int(k) <= 20])
    if d410_mean != 0:
        print(f"  Ratio: {tunnel_mean / d410_mean:.1f}x")

# Sigma responsiveness analysis
print(f"\n{'='*50}")
print("SIGMA RESPONSIVENESS (s2/s3 vs dS):")
from collections import defaultdict
sigma_by_layer = defaultdict(lambda: defaultdict(list))
for r in data["raw"]:
    sigma_by_layer[r["layer"]][r["condition"]].append(
        (r.get("sigma_2", 0), r.get("sigma_3", 0))
    )

s2s3_ratios = []
ds_vals = []
for layer_str in sorted(delta.keys(), key=int):
    layer = int(layer_str)
    if not (2 <= layer <= n_layers - 4):
        continue
    all_s2 = [s[0] for cond in sigma_by_layer[layer].values() for s in cond]
    all_s3 = [s[1] for cond in sigma_by_layer[layer].values() for s in cond]
    if all_s2 and all_s3:
        s2 = np.mean(all_s2)
        s3 = np.mean(all_s3)
        if s3 > 0:
            s2s3_ratios.append(s2 / s3)
            ds_vals.append(delta[layer_str])

if len(s2s3_ratios) > 3:
    r_corr = np.corrcoef(s2s3_ratios, ds_vals)[0, 1]
    print(f"  Tunnel s2/s3 vs dS correlation: r = {r_corr:.4f}")
    print(f"  (410M showed r = -0.026; responsive-zone model predicts weak negative)")
    if r_corr < -0.3:
        print(f"  CONFIRMED: higher s2/s3 ratio = lower dS (rigidity)")
    elif abs(r_corr) < 0.15:
        print(f"  INCONCLUSIVE: near-zero correlation")
    else:
        print(f"  UNEXPECTED: positive correlation")

# Responsiveness zone analysis
print(f"\n{'='*50}")
print("RESPONSIVENESS ZONE ANALYSIS:")
layer_rhos = []
layer_ds_vals = []
layer_ids = []
for layer_str in sorted(delta.keys(), key=int):
    layer = int(layer_str)
    if not (2 <= layer <= n_layers - 4):
        continue
    all_s2 = [s[0] for cond in sigma_by_layer[layer].values() for s in cond]
    all_s3 = [s[1] for cond in sigma_by_layer[layer].values() for s in cond]
    if all_s2 and all_s3:
        s2 = np.mean(all_s2)
        s3 = np.mean(all_s3)
        if s3 > 0:
            rho = s2 / s3
            layer_rhos.append(rho)
            layer_ds_vals.append(delta[layer_str])
            layer_ids.append(layer)
            zone = "responsive" if 1.0 < rho < 2.0 else ("rigid" if rho >= 2.0 else "degen")
            print(f"  L{layer:>2d}: rho={rho:.3f} ({zone}), dS={delta[layer_str]:+.6f}")

if layer_rhos:
    rho_arr = np.array(layer_rhos)
    ds_arr = np.array(layer_ds_vals)
    for lo, hi, label in [(1.0, 1.5, "flex"), (1.5, 2.0, "moderate"), (2.0, 99.0, "rigid")]:
        mask = (rho_arr >= lo) & (rho_arr < hi)
        if mask.any():
            layers = [layer_ids[i] for i in range(len(mask)) if mask[i]]
            print(f"  {label} ({lo:.1f}-{hi:.1f}): mean dS={ds_arr[mask].mean():+.6f}, n={mask.sum()}, layers={layers}")

    # Gradient model test
    print(f"\n{'='*50}")
    print("GRADIENT MODEL TEST (Scenario C):")
    n_responsive = sum(1 for r in layer_rhos if r < 2.0)
    n_rigid = sum(1 for r in layer_rhos if r >= 2.0)
    print(f"  Responsive layers: {n_responsive}/{len(layer_rhos)}")
    print(f"  Rigid layers: {n_rigid}/{len(layer_rhos)}")
    crossover_layer = None
    for i, (lid, rho) in enumerate(zip(layer_ids, layer_rhos)):
        if rho >= 2.0:
            crossover_layer = lid
            break
    if crossover_layer:
        print(f"  Crossover to rigid at: L{crossover_layer}")
        print(f"  (410M crossover: L17. Earlier = gradient model confirmed)")
    else:
        print(f"  No crossover to rigid zone! All layers responsive.")
    print(f"  410M: 13/19 responsive, crossover at L17")
    print(f"  Prediction: 6.9B should have FEWER responsive layers, EARLIER crossover")

# Per-condition sigma2 analysis
print(f"\n{'='*50}")
print("PER-CONDITION SIGMA2 (sign flip check):")
s2_by_cond = defaultdict(lambda: defaultdict(list))
for r in data["raw"]:
    layer = r["layer"]
    cond = r["condition"]
    s2 = r.get("sigma_2", 0)
    if s2 > 0:
        s2_by_cond[layer][cond].append(s2)

early_recv, early_abs = [], []
late_recv, late_abs = [], []
for layer_str in sorted(delta.keys(), key=int):
    layer = int(layer_str)
    if not (2 <= layer <= n_layers - 4):
        continue
    recv_s2 = s2_by_cond[layer].get("receptive", [])
    abs_s2 = s2_by_cond[layer].get("absent", [])
    if recv_s2 and abs_s2:
        r_mean = np.mean(recv_s2)
        a_mean = np.mean(abs_s2)
        pct = (r_mean - a_mean) / a_mean * 100
        if layer <= 6:
            early_recv.append(r_mean)
            early_abs.append(a_mean)
        elif layer >= n_layers - 6:
            late_recv.append(r_mean)
            late_abs.append(a_mean)

if early_recv and early_abs:
    pct_early = (np.mean(early_recv) - np.mean(early_abs)) / np.mean(early_abs) * 100
    print(f"  Early tunnel (L2-L6): s2(recv)/s2(abs) = {pct_early:+.1f}%")
    print(f"  (410M showed +7% — witness amplifies s2 early)")
if late_recv and late_abs:
    pct_late = (np.mean(late_recv) - np.mean(late_abs)) / np.mean(late_abs) * 100
    print(f"  Late tunnel: s2(recv)/s2(abs) = {pct_late:+.1f}%")
    print(f"  (410M showed -1.5% — witness suppresses s2 late)")
    if pct_early > 0 and pct_late < 0:
        print(f"  SIGN FLIP REPLICATED: amplification -> suppression through tunnel")
    elif pct_early > 0 and pct_late > 0:
        print(f"  NO SIGN FLIP: amplification throughout (weaker gradient than 410M)")
    elif pct_early < 0:
        print(f"  REVERSED: suppression even at early layers (unexpected)")
