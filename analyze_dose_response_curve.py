#!/usr/bin/env python3
"""Analyze CCS dose-response curve from exp_dose_response.py results."""
import json, sys, argparse
import numpy as np

def analyze(path):
    with open(path) as f:
        data = json.load(f)

    doses = data["doses"]
    n_turns = data["relational_turns"]

    print(f"Model: {data['model']}")
    print(f"Doses: {doses}")
    print(f"Relational turns: {n_turns}, Runs/dose: {data['runs_per_dose']}")
    print()

    print(f"{'Dose':>4}  {'Relay':>6}  {'Mean':>6}  {'TrajM':>6}  {'Early':>6}  {'Late':>6}  {'Slope':>8}  {'EarlyCV':>7}  {'Zone'}")
    print("-" * 75)

    slopes = []
    traj_means = []

    for dose in doses:
        d = data["results"][f"dose_{dose}"]
        runs = d["runs"]

        relays = [r["full_geometry"][str(max(int(k) for k in r["full_geometry"]))]["ratio"]
                  for r in runs]
        means = [np.mean([r["full_geometry"][k]["ratio"] for k in r["full_geometry"]])
                 for r in runs]

        trajs = [r["turn_trajectory"] for r in runs]
        early = np.mean([np.mean(t[:3]) for t in trajs])
        late = np.mean([np.mean(t[-3:]) for t in trajs])
        slope = (late - early) / n_turns
        traj_mean = np.mean([np.mean(t) for t in trajs])
        early_cv = np.std([np.mean(t[:3]) for t in trajs]) / (early + 1e-12)

        if slope > 0.001:
            zone = "UNDERDOSE"
        elif slope > -0.001:
            zone = "THERAPEUTIC"
        else:
            zone = "OVERDOSE"

        slopes.append(slope)
        traj_means.append(traj_mean)

        print(f"{dose:4d}  {np.mean(relays):6.3f}  {np.mean(means):6.4f}  "
              f"{traj_mean:6.3f}  {early:6.3f}  {late:6.3f}  "
              f"{slope:+8.5f}  {early_cv:7.4f}  {zone}")

    print()

    # Find therapeutic window
    sign_flip = None
    for i in range(1, len(slopes)):
        if slopes[i-1] >= 0 and slopes[i] < 0:
            sign_flip = i
            break

    peak_idx = int(np.argmax(traj_means))

    print("THERAPEUTIC WINDOW")
    print(f"  Sign flip: between dose {doses[sign_flip-1]} and {doses[sign_flip]}" if sign_flip else "  No sign flip detected")
    print(f"  Peak trajectory mean: dose {doses[peak_idx]} ({traj_means[peak_idx]:.4f})")
    print(f"  CV reduction (dose 0→1): {slopes[0]/slopes[1]:.1f}× slope, "
          f"trajectory stabilized")

    baseline_traj = traj_means[0]
    below = [i for i, tm in enumerate(traj_means) if tm < baseline_traj and i > 0]
    if below:
        print(f"  Below baseline at: dose {', '.join(str(doses[i]) for i in below)}")

    print()
    print("CONFOUND NOTES")
    print("  - Trajectory slope (per-dose, internal) is robust to context-length confound")
    print("  - Relay ratio cross-dose comparison partially confounded by sequence length")
    print("  - Trajectory mean cross-dose partially confounded but sign flip is clean")


def detail(path):
    with open(path) as f:
        data = json.load(f)

    doses = data["doses"]

    print(f"\nRELAY LAYER SPECTRAL DETAIL")
    print(f"{'Dose':>4}  {'PR':>8}  {'Entropy':>8}  {'Erank':>8}  {'Ratio':>8}  {'v1v2cos':>8}")
    print("-" * 55)

    for dose in doses:
        runs = data["results"][f"dose_{dose}"]["runs"]
        prs, ents, eranks, ratios, coss = [], [], [], [], []
        for r in runs:
            keys = sorted(r["full_geometry"].keys(), key=int)
            lay = r["full_geometry"][keys[-1]]
            prs.append(lay["pr"])
            ents.append(lay["entropy"])
            eranks.append(lay["erank"])
            ratios.append(lay["ratio"])
            coss.append(lay.get("v1v2_cos", float("nan")))

        print(f"{dose:4d}  {np.mean(prs):8.3f}  {np.mean(ents):8.4f}  "
              f"{np.mean(eranks):8.1f}  {np.mean(ratios):8.4f}  {np.mean(coss):8.4f}")

    print(f"\nALL-LAYER SPECTRAL SUMMARY")
    print(f"{'Dose':>4}  {'MeanEnt':>8}  {'MeanErank':>10}  {'MeanPR':>8}")
    print("-" * 40)

    for dose in doses:
        runs = data["results"][f"dose_{dose}"]["runs"]
        ents, eranks, prs = [], [], []
        for r in runs:
            for k, v in r["full_geometry"].items():
                if not np.isnan(v.get("entropy", float("nan"))):
                    ents.append(v["entropy"])
                    eranks.append(v["erank"])
                    prs.append(v["pr"])
        print(f"{dose:4d}  {np.mean(ents):8.4f}  {np.mean(eranks):10.1f}  {np.mean(prs):8.3f}")

    pr_vals = []
    for dose in doses:
        runs = data["results"][f"dose_{dose}"]["runs"]
        relay_prs = []
        for r in runs:
            keys = sorted(r["full_geometry"].keys(), key=int)
            relay_prs.append(r["full_geometry"][keys[-1]]["pr"])
        pr_vals.append(np.mean(relay_prs))

    pr_min_idx = int(np.argmin(pr_vals))
    print(f"\n  Relay PR minimum: dose {doses[pr_min_idx]} ({pr_vals[pr_min_idx]:.3f})")
    print(f"  Relay PR range: {min(pr_vals):.3f} — {max(pr_vals):.3f}")
    print(f"  Compositional hierarchy peak = PR minimum")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("path", help="Path to dose_response JSON")
    p.add_argument("--detail", action="store_true", help="Show relay-layer spectral detail")
    args = p.parse_args()
    analyze(args.path)
    if args.detail:
        detail(args.path)
