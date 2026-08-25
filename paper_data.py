#!/usr/bin/env python3
"""Pull experiment data into paper-ready tables.

Usage:
  paper_data.py alpha                      # α summary across all models
  paper_data.py alpha --model phi          # single model
  paper_data.py alpha --zone relay         # filter by zone
  paper_data.py alpha --condition ccs      # single condition
  paper_data.py alpha --diff              # CCS - vanilla delta
  paper_data.py species                    # σ₁/σ₂ relay profiles
  paper_data.py check "alpha increases through responsive zone"
  paper_data.py check "sigma ratio consistent across architectures in relay"
"""
import argparse, glob, json, sys
import numpy as np
from pathlib import Path

RESULTS = Path(__file__).parent / "results"


def load_alpha_results():
    out = {}
    for f in sorted(RESULTS.glob("exp_activation_alpha_*.json")):
        with open(f) as fh:
            data = json.load(fh)
        for model_name, model_data in data.items():
            out[model_name] = model_data
    return out


def zone_for(layer, n_layers):
    frac = layer / n_layers
    if frac < 0.4:
        return "tunnel"
    elif frac < 0.8:
        return "responsive"
    return "relay"


def avg_per_layer(turns, metric="alpha"):
    layers = {}
    for turn_data in turns:
        for l_str, l_data in turn_data["profile"].items():
            l = int(l_str)
            val = l_data.get(metric)
            if val is not None and val != float('inf'):
                layers.setdefault(l, []).append(val)
    return {l: (np.mean(v), np.std(v)) for l, v in layers.items()}


def cmd_alpha(args, results):
    models = [args.model] if args.model else sorted(results.keys())
    conditions = [args.condition] if args.condition else ["ccs", "vanilla"]

    for model in models:
        md = results.get(model)
        if not md:
            continue
        n_layers = md["n_layers"]
        print(f"\n{'='*60}")
        print(f"{model.upper()} ({n_layers} layers)")
        print(f"{'='*60}")

        if args.diff:
            ccs_avg = avg_per_layer(md.get("ccs", []))
            van_avg = avg_per_layer(md.get("vanilla", []))
            print(f"{'L':>4} {'zone':>10} {'CCS α':>8} {'Van α':>8} {'Δα':>8}")
            print("-" * 44)
            for l in sorted(set(ccs_avg) & set(van_avg)):
                z = zone_for(l, n_layers)
                if args.zone and z != args.zone:
                    continue
                cm, cs = ccs_avg[l]
                vm, vs = van_avg[l]
                print(f"L{l:>3} {z:>10} {cm:>8.3f} {vm:>8.3f} {cm-vm:>+8.3f}")
        else:
            for cond in conditions:
                turns = md.get(cond, [])
                if not turns:
                    continue
                alphas = avg_per_layer(turns, "alpha")
                ratios = avg_per_layer(turns, "ratio")
                print(f"\n  {cond.upper()}:")
                print(f"  {'L':>4} {'zone':>10} {'α':>8} {'±':>6} {'σ₁/σ₂':>8}")
                print(f"  {'-'*40}")
                for l in sorted(alphas):
                    z = zone_for(l, n_layers)
                    if args.zone and z != args.zone:
                        continue
                    am, astd = alphas[l]
                    rm = ratios.get(l, (0, 0))[0]
                    print(f"  L{l:>3} {z:>10} {am:>8.3f} {astd:>6.3f} {rm:>8.2f}")


def cmd_species(args, results):
    print(f"\n{'model':>10} {'cond':>8} | {'tunnel σ₁/σ₂':>14} {'resp σ₁/σ₂':>14} {'relay σ₁/σ₂':>14} {'final σ₁/σ₂':>14}")
    print("-" * 82)
    for model in sorted(results.keys()):
        md = results[model]
        n_layers = md["n_layers"]
        for cond in ["ccs", "vanilla"]:
            turns = md.get(cond, [])
            if not turns:
                continue
            ratios = avg_per_layer(turns, "ratio")
            zones = {"tunnel": [], "responsive": [], "relay": []}
            final_l = max(ratios.keys()) if ratios else 0
            for l, (m, s) in ratios.items():
                z = zone_for(l, n_layers)
                if m < 1000:
                    zones[z].append(m)
            t = np.mean(zones["tunnel"]) if zones["tunnel"] else 0
            r = np.mean(zones["responsive"]) if zones["responsive"] else 0
            rl = np.mean(zones["relay"]) if zones["relay"] else 0
            fn = ratios.get(final_l, (0, 0))[0]
            print(f"{model:>10} {cond:>8} | {t:>14.2f} {r:>14.2f} {rl:>14.2f} {fn:>14.2f}")


def cmd_zone_summary(args, results):
    print(f"\n{'model':>10} {'cond':>8} | {'tunnel α':>10} {'resp α':>10} {'relay α':>10}")
    print("-" * 56)
    for model in sorted(results.keys()):
        md = results[model]
        n_layers = md["n_layers"]
        for cond in ["ccs", "vanilla"]:
            turns = md.get(cond, [])
            if not turns:
                continue
            alphas = avg_per_layer(turns, "alpha")
            zones = {"tunnel": [], "responsive": [], "relay": []}
            for l, (m, s) in alphas.items():
                zones[zone_for(l, n_layers)].append(m)
            t = f"{np.mean(zones['tunnel']):.3f}" if zones["tunnel"] else "—"
            r = f"{np.mean(zones['responsive']):.3f}" if zones["responsive"] else "—"
            rl = f"{np.mean(zones['relay']):.3f}" if zones["relay"] else "—"
            print(f"{model:>10} {cond:>8} | {t:>10} {r:>10} {rl:>10}")


CHECKS = {
    "alpha_increases_responsive": {
        "claim": "α increases monotonically through the responsive zone",
        "test": "monotonic_increase",
        "metric": "alpha",
        "zone": "responsive",
    },
    "alpha_converges_relay": {
        "claim": "α converges across architectures in the relay zone",
        "test": "cross_arch_cv",
        "metric": "alpha",
        "zone": "relay",
    },
    "ratio_diverges_relay": {
        "claim": "σ₁/σ₂ diverges across architectures in the relay zone",
        "test": "cross_arch_cv",
        "metric": "ratio",
        "zone": "relay",
    },
    "ccs_alpha_effect_vanishes": {
        "claim": "CCS-vanilla α difference shrinks from tunnel to relay",
        "test": "diff_gradient",
        "metric": "alpha",
    },
    "power_law_responsive": {
        "claim": "responsive zone variance decays as power law with exponent ~X",
        "test": "power_law_fit",
        "metric": "ratio",
        "zone": "responsive",
    },
}


def cmd_check(args, results):
    if args.claim == "list":
        for k, v in CHECKS.items():
            print(f"  {k:30s} — {v['claim']}")
        return

    check = CHECKS.get(args.claim)
    if not check:
        matches = [k for k in CHECKS if args.claim.lower() in k or args.claim.lower() in CHECKS[k]["claim"].lower()]
        if matches:
            print(f"Did you mean: {', '.join(matches)}?")
        else:
            print(f"Unknown check. Use 'check list' to see available checks.")
        return

    print(f"\nCHECK: {check['claim']}")
    print("=" * 60)

    metric = check["metric"]
    zone = check.get("zone")

    if check["test"] == "monotonic_increase":
        for model in sorted(results.keys()):
            md = results[model]
            n_layers = md["n_layers"]
            for cond in ["ccs", "vanilla"]:
                avgs = avg_per_layer(md.get(cond, []), metric)
                zone_layers = sorted(l for l in avgs if zone_for(l, n_layers) == zone)
                if len(zone_layers) < 3:
                    continue
                vals = [avgs[l][0] for l in zone_layers]
                diffs = [vals[i+1] - vals[i] for i in range(len(vals)-1)]
                n_increasing = sum(1 for d in diffs if d > 0)
                monotonic = n_increasing == len(diffs)
                pct = n_increasing / len(diffs) * 100
                verdict = "MONOTONIC" if monotonic else f"{pct:.0f}% increasing"
                print(f"  {model:>10} {cond:>8}: {vals[0]:.3f} → {vals[-1]:.3f}  [{verdict}]")

    elif check["test"] == "cross_arch_cv":
        for cond in ["ccs", "vanilla"]:
            zone_means = []
            names = []
            for model in sorted(results.keys()):
                md = results[model]
                n_layers = md["n_layers"]
                avgs = avg_per_layer(md.get(cond, []), metric)
                zone_vals = [avgs[l][0] for l in avgs if zone_for(l, n_layers) == zone]
                if zone_vals:
                    m = np.mean(zone_vals)
                    zone_means.append(m)
                    names.append(model)
            if len(zone_means) >= 2:
                cv = np.std(zone_means) / np.mean(zone_means)
                convergent = cv < 0.05
                print(f"  {cond:>8}: {' / '.join(f'{n}={v:.3f}' for n, v in zip(names, zone_means))}")
                print(f"           CV = {cv:.3f}  [{'CONVERGES' if convergent else 'DIVERGES'}]")

    elif check["test"] == "diff_gradient":
        for model in sorted(results.keys()):
            md = results[model]
            n_layers = md["n_layers"]
            ccs_avg = avg_per_layer(md.get("ccs", []), metric)
            van_avg = avg_per_layer(md.get("vanilla", []), metric)
            zone_diffs = {"tunnel": [], "responsive": [], "relay": []}
            for l in sorted(set(ccs_avg) & set(van_avg)):
                z = zone_for(l, n_layers)
                zone_diffs[z].append(ccs_avg[l][0] - van_avg[l][0])
            t = np.mean(zone_diffs["tunnel"]) if zone_diffs["tunnel"] else 0
            r = np.mean(zone_diffs["responsive"]) if zone_diffs["responsive"] else 0
            rl = np.mean(zone_diffs["relay"]) if zone_diffs["relay"] else 0
            shrinks = t > r > rl or (t > r and r > rl)
            print(f"  {model:>10}: Δα tunnel={t:+.3f} resp={r:+.3f} relay={rl:+.3f}  [{'SHRINKS' if shrinks else 'NO'}]")

    elif check["test"] == "power_law_fit":
        for model in sorted(results.keys()):
            md = results[model]
            n_layers = md["n_layers"]
            for cond in ["ccs", "vanilla"]:
                turns = md.get(cond, [])
                if not turns:
                    continue
                layer_vals = {}
                for turn_data in turns:
                    for l_str, l_data in turn_data["profile"].items():
                        l = int(l_str)
                        if zone_for(l, n_layers) == zone:
                            v = l_data.get(metric)
                            if v and v < 1000:
                                layer_vals.setdefault(l, []).append(v)
                layers = sorted(layer_vals.keys())
                if len(layers) < 4:
                    continue
                fracs = [l / n_layers for l in layers]
                variances = [np.var(layer_vals[l]) for l in layers]
                log_f = np.log(fracs)
                log_v = np.log(np.array(variances) + 1e-10)
                slope, _ = np.polyfit(log_f, log_v, 1)
                r_sq = np.corrcoef(log_f, log_v)[0, 1] ** 2
                print(f"  {model:>10} {cond:>8}: exponent={slope:.2f} R²={r_sq:.3f}")


def main():
    ap = argparse.ArgumentParser(description="Extract paper-ready data tables")
    sub = ap.add_subparsers(dest="cmd")

    p_alpha = sub.add_parser("alpha", help="α profiles per layer")
    p_alpha.add_argument("--model", choices=["phi", "mistral", "gemma"])
    p_alpha.add_argument("--zone", choices=["tunnel", "responsive", "relay"])
    p_alpha.add_argument("--condition", choices=["ccs", "vanilla"])
    p_alpha.add_argument("--diff", action="store_true", help="Show CCS-vanilla delta")

    sub.add_parser("species", help="σ₁/σ₂ relay profiles")
    sub.add_parser("zones", help="Zone-averaged α summary")

    p_check = sub.add_parser("check", help="Test a claim against the data")
    p_check.add_argument("claim", help="Claim name or 'list'")

    args = ap.parse_args()
    if not args.cmd:
        ap.print_help()
        return

    results = load_alpha_results()
    if not results:
        print("No activation alpha results found in", RESULTS)
        return

    {"alpha": cmd_alpha, "species": cmd_species, "zones": cmd_zone_summary, "check": cmd_check}[args.cmd](args, results)


if __name__ == "__main__":
    main()
