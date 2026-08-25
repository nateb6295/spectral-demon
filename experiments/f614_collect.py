#!/usr/bin/env python3
"""F614/F615 results collector — fetch, analyze, compare, report.

Fetches F614 result JSON from RunPod, runs per-layer ρ analysis,
compares across models, posts summary to Discord.

Usage:
  python3 f614_collect.py fetch <pod_ssh> <short_name>     # SCP + analyze
  python3 f614_collect.py compare                           # cross-model comparison
  python3 f614_collect.py report [--discord]                # summary, optionally post
"""

import json, sys, os, subprocess, glob
import numpy as np
from scipy import stats

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def normalize_perlayer(raw):
    """Handle old, new, and direct-key formats."""
    if "rho_agg" in raw:
        n_layers = raw["n_layers"]
        ab_layers = raw.get("anti_band_layers", [])
        anti = None
        if ab_layers:
            anti = {"start": ab_layers[0][0], "end": ab_layers[-1][0] + 4,
                    "min_rho": min(r for _, r in ab_layers)}
        return {
            "model": raw["model"],
            "n_layers": n_layers,
            "gqa_ratio": raw.get("gqa_ratio"),
            "aggregate_rho": raw["rho_agg"],
            "early_rho": raw.get("rho_early"),
            "late_rho": raw.get("rho_late"),
            "boundary": int(n_layers * 0.7),
            "anti_band": anti,
            "bifurcation": False,
            "null_signature": None,
        }
    if "early_zone" in raw:
        return {
            "model": raw["model"],
            "n_layers": raw["n_layers"],
            "gqa_ratio": raw.get("gqa_ratio"),
            "aggregate_rho": raw["aggregate_rho"],
            "early_rho": raw["early_zone"]["rho"],
            "late_rho": raw["late_zone"]["rho"],
            "boundary": raw["early_zone"]["boundary"],
            "anti_band": raw.get("anti_band"),
            "bifurcation": raw.get("bifurcation", False),
            "null_signature": raw.get("null_signature"),
        }
    # Old format from f614_perlayer_rho.py
    zones = raw.get("zones", {})
    early = zones.get("early", {}).get("rho")
    late = zones.get("late", {}).get("rho")
    n_layers = raw["n_layers"]
    boundary = int(n_layers * 0.7)
    bands = raw.get("polarity_sensitive_bands", [])
    anti = None
    for b in bands:
        if b["min_rho"] < -0.5:
            if anti is None or b["min_rho"] < anti["min_rho"]:
                anti = {"start": b["start"], "end": b["end"], "min_rho": b["min_rho"]}
    gqa = raw.get("gqa_ratio")
    if gqa is None:
        base_path = os.path.join(RESULTS_DIR, f"f614_scale_{raw['model']}.json")
        if os.path.exists(base_path):
            with open(base_path) as bf:
                gqa = json.load(bf).get("gqa_ratio")
    return {
        "model": raw["model"],
        "n_layers": n_layers,
        "gqa_ratio": gqa,
        "aggregate_rho": raw.get("whole_model_rho", 0),
        "early_rho": early,
        "late_rho": late,
        "boundary": boundary,
        "anti_band": anti,
        "bifurcation": early is not None and late is not None and early > 0.7 and late < 0.3,
    }


def fetch_result(pod_ssh, short_name):
    remote_path = f"/workspace/f614_scale_{short_name}.json"
    local_path = os.path.join(RESULTS_DIR, f"f614_scale_{short_name}.json")

    parts = pod_ssh.split(":")
    host = parts[0]
    port = parts[1] if len(parts) > 1 else "22"

    cmd = ["scp", "-P", port, "-i", os.path.expanduser("~/.ssh/id_ed25519"),
           "-o", "StrictHostKeyChecking=no", f"root@{host}:{remote_path}", local_path]
    print(f"Fetching {remote_path} → {local_path}")
    subprocess.run(cmd, check=True)

    if os.path.exists(local_path):
        print(f"Saved: {local_path}")
        analyze(local_path)
        return local_path
    return None


def analyze(path):
    with open(path) as f:
        d = json.load(f)

    gp = np.array(d["gain_profiles"]["ccs_pos"])
    gn = np.array(d["gain_profiles"]["ccs_neg"])
    n_layers = d["n_layers"]
    short = d.get("short", "unknown")
    boundary = int(n_layers * 0.7)

    rho_all, p_all = stats.spearmanr(gp, gn)

    early_p, early_n = gp[:boundary], gn[:boundary]
    late_p, late_n = gp[boundary:], gn[boundary:]

    rho_early, p_early = stats.spearmanr(early_p, early_n) if len(early_p) >= 4 else (None, None)
    rho_late, p_late = stats.spearmanr(late_p, late_n) if len(late_p) >= 4 else (None, None)

    window = 5
    perlayer = []
    for i in range(len(gp) - window + 1):
        wp, wn = gp[i:i+window], gn[i:i+window]
        if np.std(wp) < 1e-12 or np.std(wn) < 1e-12:
            perlayer.append({"center": i + window // 2, "rho": None})
        else:
            r, p = stats.spearmanr(wp, wn)
            perlayer.append({"center": i + window // 2, "rho": float(r), "p": float(p)})

    anti_band = None
    min_rho, min_start, min_end = 1.0, None, None
    in_band = False
    for entry in perlayer:
        if entry["rho"] is not None and entry["rho"] < -0.5:
            if not in_band:
                min_start = entry["center"]
                in_band = True
            min_end = entry["center"]
            if entry["rho"] < min_rho:
                min_rho = entry["rho"]
        else:
            if in_band:
                anti_band = {"start": min_start, "end": min_end, "min_rho": min_rho}
            in_band = False
            min_rho = 1.0
    if in_band:
        anti_band = {"start": min_start, "end": min_end, "min_rho": min_rho}

    bif = bool(rho_early is not None and rho_late is not None and rho_early > 0.7 and rho_late < 0.3)

    # Kimi discriminator: when no bifurcation, classify the null signature
    null_signature = None
    if not bif and rho_early is not None:
        mid = n_layers // 2
        gn_mid = gn[mid - 3:mid + 3]
        gn_trend = np.polyfit(range(len(gn_mid)), gn_mid, 1)[0] if len(gn_mid) >= 4 else 0
        has_any_anti = any(e["rho"] is not None and e["rho"] < -0.3 for e in perlayer)
        late_rho_vals = [e["rho"] for e in perlayer if e["rho"] is not None and e["center"] >= boundary]
        late_trending = np.mean(late_rho_vals) < 0.5 if late_rho_vals else False

        if not has_any_anti and gn_trend > 0:
            null_signature = "sorter-blocked"
        elif late_trending and rho_early > 0.6:
            null_signature = "depth-limited"
        else:
            null_signature = "indeterminate"

    result = {
        "model": short,
        "n_layers": n_layers,
        "gqa_ratio": d.get("gqa_ratio"),
        "aggregate_rho": float(rho_all),
        "early_zone": {"rho": float(rho_early) if rho_early else None, "boundary": boundary},
        "late_zone": {"rho": float(rho_late) if rho_late else None},
        "anti_band": anti_band,
        "bifurcation": bif,
        "null_signature": null_signature,
        "perlayer_rho": perlayer,
    }

    out_path = path.replace(".json", "_perlayer.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*50}")
    print(f"F615 ANALYSIS: {short}")
    print(f"{'='*50}")
    print(f"Layers: {n_layers}, GQA: {d.get('gqa_ratio')}")
    print(f"Aggregate ρ: {rho_all:+.4f}")
    if rho_early is not None:
        print(f"Early (L0-{boundary}): ρ = {rho_early:+.4f}")
    if rho_late is not None:
        print(f"Late (L{boundary}-{n_layers}): ρ = {rho_late:+.4f}")
    if anti_band:
        print(f"Anti-correlated: L{anti_band['start']}-L{anti_band['end']} (min ρ = {anti_band['min_rho']:+.3f})")
    print(f"BIFURCATION: {'YES' if result['bifurcation'] else 'NO'}")
    if null_signature:
        print(f"NULL SIGNATURE: {null_signature}")
    print(f"Saved: {out_path}")

    return result


def conservation_analysis(path):
    """Kimi's ternary discrimination: R (conservation ratio) + throughput per zone.
    R = |Σ(gain)| / Σ|gain| — 0 = conservative (cancellation), 1 = driven (unidirectional)
    Throughput = Σ|gain| — total activity. At shuffle-null floor = inert.
    """
    with open(path) as f:
        d = json.load(f)

    gp = np.array(d["gain_profiles"]["ccs_pos"])
    gn = np.array(d["gain_profiles"]["ccs_neg"])
    gs = np.array(d["gain_profiles"]["scrambled"])
    n = d["n_layers"]
    short = d.get("short", "unknown")
    boundary = int(n * 0.7)

    def zone_metrics(gains, label):
        abs_sum = np.sum(np.abs(gains))
        signed_sum = np.sum(gains)
        R = abs(signed_sum) / (abs_sum + 1e-12)
        return {"zone": label, "R": float(R), "throughput": float(abs_sum),
                "net_signed": float(signed_sum), "n_layers": len(gains)}

    zones = [
        ("early (L0-{})".format(boundary), slice(0, boundary)),
        ("late (L{}-{})".format(boundary, n), slice(boundary, n)),
        ("all", slice(0, n)),
    ]
    if n >= 30:
        mid_start, mid_end = int(n * 0.3), int(n * 0.7)
        zones.insert(1, ("mid (L{}-{})".format(mid_start, mid_end), slice(mid_start, mid_end)))

    null_throughput = np.sum(np.abs(gs))

    print(f"\n{'='*60}")
    print(f"CONSERVATION ANALYSIS: {short} ({n} layers)")
    print(f"{'='*60}")
    print(f"Scrambled null throughput: {null_throughput:.4f}")
    print(f"\n{'Zone':<20} {'R(+CCS)':>8} {'R(-CCS)':>8} {'Thru(+)':>8} {'Thru(-)':>8} {'vs null':>8}")
    print("-" * 60)

    results = []
    for label, sl in zones:
        mp = zone_metrics(gp[sl], label)
        mn = zone_metrics(gn[sl], label)
        null_zone = np.sum(np.abs(gs[sl]))
        ratio_to_null = mp["throughput"] / (null_zone + 1e-12)

        outcome = "conservative" if mp["R"] < 0.2 and mp["throughput"] > null_zone * 1.5 else \
                  "driven" if mp["R"] > 0.5 and mp["throughput"] > null_zone * 1.5 else \
                  "inert" if mp["throughput"] < null_zone * 1.5 else "mixed"

        print(f"{label:<20} {mp['R']:>8.3f} {mn['R']:>8.3f} "
              f"{mp['throughput']:>8.4f} {mn['throughput']:>8.4f} {ratio_to_null:>8.2f}x  → {outcome}")
        results.append({"zone": label, "pos": mp, "neg": mn, "null_throughput": float(null_zone),
                        "ratio_to_null": float(ratio_to_null), "outcome": outcome})

    out_path = path.replace(".json", "_conservation.json")
    with open(out_path, "w") as f:
        json.dump({"model": short, "n_layers": n, "null_total": float(null_throughput),
                   "zones": results}, f, indent=2)
    print(f"\nSaved: {out_path}")
    return results


def demeaned_analysis(path=None):
    """F615b: Subtract mean profile across conditions, re-correlate.
    Reveals condition-specific structure hidden by shared baseline shape.
    If path given, analyze single model. If None, analyze all."""
    if path:
        files = [path]
    else:
        files = sorted(glob.glob(os.path.join(RESULTS_DIR, "f614_scale_*[!_perlayer][!_conservation].json")))
        files = [f for f in files if "perlayer" not in f and "conservation" not in f]

    print(f"\n{'='*75}")
    print(f"F615b DEMEANED ANALYSIS ({len(files)} models)")
    print(f"{'='*75}")
    print(f"{'Model':<25} {'rho_raw':>8} {'rho_dem':>8} {'rho_ps_dm':>9} {'dev/base':>8}")
    print("-" * 65)

    results = []
    for fn in files:
        with open(fn) as f:
            d = json.load(f)

        gp = d["gain_profiles"].get("ccs_pos")
        gn = d["gain_profiles"].get("ccs_neg")
        gs = d["gain_profiles"].get("scrambled")
        gc = d["gain_profiles"].get("control_b") or d["gain_profiles"].get("control")

        if not all([gp, gn, gs]):
            continue

        gp, gn, gs = np.array(gp), np.array(gn), np.array(gs)
        gc = np.array(gc) if gc and len(gc) == len(gp) else np.zeros_like(gp)

        mean_p = (gp + gn + gs + gc) / 4.0
        rho_raw, _ = stats.spearmanr(gp, gn)

        gp_dm, gn_dm, gs_dm = gp - mean_p, gn - mean_p, gs - mean_p
        rho_dm, p_dm = stats.spearmanr(gp_dm, gn_dm)
        rho_ps_dm, _ = stats.spearmanr(gp_dm, gs_dm)

        dev = float(np.mean(np.abs(gp_dm)))
        base = float(np.mean(np.abs(mean_p)))
        ratio = dev / (base + 1e-12)

        short = d.get("short", os.path.basename(fn).replace(".json", ""))
        print(f"{short:<25} {rho_raw:>+8.3f} {rho_dm:>+8.3f} {rho_ps_dm:>+9.3f} {ratio:>8.3f}")
        results.append({"model": short, "rho_raw": float(rho_raw),
                        "rho_demeaned": float(rho_dm), "p_demeaned": float(p_dm),
                        "rho_ps_demeaned": float(rho_ps_dm),
                        "dev_base": ratio, "dev": dev, "base": base})

    if not path:
        out_path = os.path.join(RESULTS_DIR, "f615b_demeaned_comparison.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved: {out_path}")
    return results


def pca_analysis(path=None):
    """F616: Pair Closure Asymmetry — how σ₁/σ₂ coupling changes with CCS polarity.
    PCA = PC(CCS+) - PC(CCS-). Discriminates species:
      |PCA| < 0.15 = tunnel (polarity-blind)
      PCA << 0 = relay (CCS+ holds duality)
      PCA >> 0 = sorter (CCS- drives oscillation)"""
    if path:
        files_list = [path]
    else:
        files_list = sorted(glob.glob(os.path.join(RESULTS_DIR, "f614_scale_*[!_perlayer][!_conservation].json")))
        files_list = [f for f in files_list if "perlayer" not in f and "conservation" not in f and "demeaned" not in f]

    print(f"\n{'='*75}")
    print(f"F616 PAIR CLOSURE ASYMMETRY ({len(files_list)} models)")
    print(f"{'='*75}")
    print(f"{'Model':<20} {'GQA':>4} {'PC(+)':>8} {'PC(-)':>8} {'PCA':>8} {'species':>12}")
    print("-" * 65)

    results = []
    for fn in files_list:
        with open(fn) as f:
            d = json.load(f)

        pc_pos = d.get("pair_closure", {}).get("ccs_pos", {}).get("closure_corr", 0)
        pc_neg = d.get("pair_closure", {}).get("ccs_neg", {}).get("closure_corr", 0)
        pca = pc_pos - pc_neg
        abs_pca = abs(pca)

        if abs_pca < 0.15:
            species = "tunnel"
        elif pca < -0.3:
            species = "relay"
        elif pca > 0.3:
            species = "sorter"
        else:
            species = "ambiguous"

        short = d.get("short", os.path.basename(fn).replace(".json", ""))
        gqa = d.get("gqa_ratio", "?")
        print(f"{short:<20} {gqa:>4} {pc_pos:>+8.3f} {pc_neg:>+8.3f} {pca:>+8.3f} {species:>12}")
        results.append({"model": short, "gqa_ratio": gqa,
                        "pc_pos": round(pc_pos, 4), "pc_neg": round(pc_neg, 4),
                        "pca": round(pca, 4), "species": species})

    if not path:
        out_path = os.path.join(RESULTS_DIR, "f616_pair_closure_asymmetry.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved: {out_path}")
    return results


def path_redundancy_analysis(path=None):
    """F617: Path redundancy → restoring force. Tests whether GQA reduces
    spectral redundancy (more tail leakage) and constrains σ₁ preferentially."""
    if path:
        files_list = [path]
    else:
        files_list = sorted(glob.glob(os.path.join(RESULTS_DIR, "f614_scale_*[!_perlayer][!_conservation].json")))
        files_list = [f for f in files_list if "perlayer" not in f and "conservation" not in f and "demeaned" not in f]

    print(f"\n{'='*75}")
    print(f"F617 PATH REDUNDANCY ({len(files_list)} models)")
    print(f"{'='*75}")
    print(f"{'Model':<20} {'GQA':>4} {'tail_f':>8} {'dev/base':>8} {'|Δσ₂/Δσ₁|':>10}")
    print("-" * 55)

    gqa_vals, tail_vals, db_vals = [], [], []
    results = []
    for fn in files_list:
        with open(fn) as f:
            d = json.load(f)

        tail_f = d.get("pair_closure", {}).get("ccs_pos", {}).get("tail_fraction", 0)
        ds1 = d.get("pair_closure", {}).get("ccs_pos", {}).get("mean_delta_sigma1", 0)
        ds2 = d.get("pair_closure", {}).get("ccs_pos", {}).get("mean_delta_sigma2", 0)
        sigma_ratio = abs(ds2 / ds1) if abs(ds1) > 1e-10 else float("inf")

        gp = d["gain_profiles"].get("ccs_pos")
        gn = d["gain_profiles"].get("ccs_neg")
        gs = d["gain_profiles"].get("scrambled")
        gc = d["gain_profiles"].get("control_b") or d["gain_profiles"].get("control")
        if not all([gp, gn, gs]):
            continue
        gp, gn, gs = np.array(gp), np.array(gn), np.array(gs)
        gc = np.array(gc) if gc and len(gc) == len(gp) else np.zeros_like(gp)
        mean_p = (gp + gn + gs + gc) / 4.0
        dev = float(np.mean(np.abs(gp - mean_p)))
        base = float(np.mean(np.abs(mean_p)))
        db = dev / (base + 1e-12)

        short = d.get("short", os.path.basename(fn).replace(".json", ""))
        gqa = d.get("gqa_ratio", 0)
        print(f"{short:<20} {gqa:>4} {tail_f:>8.4f} {db:>8.4f} {sigma_ratio:>10.1f}x")

        gqa_vals.append(gqa)
        tail_vals.append(tail_f)
        db_vals.append(db)
        results.append({"model": short, "gqa_ratio": gqa,
                        "tail_fraction": round(tail_f, 4), "dev_base": round(db, 4),
                        "sigma_ratio": round(sigma_ratio, 1)})

    if len(gqa_vals) >= 3:
        rho_gt, p_gt = stats.spearmanr(gqa_vals, tail_vals)
        rho_gd, p_gd = stats.spearmanr(gqa_vals, db_vals)
        rho_td, p_td = stats.spearmanr(tail_vals, db_vals)
        print(f"\nSpearman(GQA, tail):     ρ={rho_gt:+.3f} p={p_gt:.3f}")
        print(f"Spearman(GQA, dev/base): ρ={rho_gd:+.3f} p={p_gd:.3f}")
        print(f"Spearman(tail, dev/base):ρ={rho_td:+.3f} p={p_td:.3f}")

    if not path:
        out_path = os.path.join(RESULTS_DIR, "f617_path_redundancy.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved: {out_path}")
    return results


def compare():
    if not files:
        print("No per-layer analysis files found.")
        return None

    models = []
    for f in files:
        with open(f) as fh:
            raw = json.load(fh)
        m = normalize_perlayer(raw)
        models.append(m)

    print(f"\n{'='*75}")
    print(f"F615 CROSS-MODEL COMPARISON ({len(models)} models)")
    print(f"{'='*75}")
    print(f"{'Model':<20} {'Layers':>6} {'GQA':>4} {'ρ_all':>8} {'ρ_early':>8} {'ρ_late':>8} {'Bifurc':>7} {'Null Sig':>15}")
    print("-" * 75)
    for m in models:
        early = m["early_rho"]
        late = m["late_rho"]
        e_str = f"{early:>+8.3f}" if early is not None else f"{'N/A':>8}"
        l_str = f"{late:>+8.3f}" if late is not None else f"{'N/A':>8}"
        bif = "YES" if m["bifurcation"] else "NO"
        null_sig = m.get("null_signature") or "—"
        print(f"{m['model']:<20} {m['n_layers']:>6} {str(m.get('gqa_ratio', '?')):>4} "
              f"{m['aggregate_rho']:>+8.3f} {e_str} {l_str} {bif:>7} {null_sig:>15}")

    if len(models) >= 2:
        boundaries = [m["boundary"] / m["n_layers"] for m in models]
        print(f"\nBoundary consistency: {[f'{b:.0%}' for b in boundaries]}")

        bifurcating = [m for m in models if m["bifurcation"]]
        if bifurcating:
            anti_bands = [(m["anti_band"], m) for m in bifurcating if m.get("anti_band")]
            if anti_bands:
                depths = [(b["start"] / m["n_layers"], b["end"] / m["n_layers"])
                          for b, m in anti_bands]
                print(f"Anti-band depths: {[f'{s:.0%}-{e:.0%}' for s, e in depths]}")

    return models


def report(post_discord=False):
    models = compare()
    if not models:
        return

    lines = ["▸ **F615 Scale Test — Cross-Model Results**\n"]
    for m in models:
        early = m["early_rho"]
        late = m["late_rho"]
        status = "BIFURCATION" if m["bifurcation"] else "uniform"
        e_str = f"{early:+.3f}" if early is not None else "N/A"
        l_str = f"{late:+.3f}" if late is not None else "N/A"
        lines.append(f"**{m['model']}** ({m['n_layers']}L, GQA {m.get('gqa_ratio', '?')}): "
                     f"ρ_agg={m['aggregate_rho']:+.3f}, "
                     f"early={e_str}, late={l_str} → {status}")
        if m.get("anti_band"):
            lines.append(f"  Anti-correlated: L{m['anti_band']['start']}-L{m['anti_band']['end']} "
                         f"(ρ={m['anti_band']['min_rho']:+.3f})")
        if m.get("null_signature"):
            lines.append(f"  Null signature: {m['null_signature']}")

    msg = "\n".join(lines)
    print(f"\n{msg}")

    if post_discord:
        env_path = os.path.expanduser("~/chronicle/chronicle.env")
        script = os.path.expanduser("~/chronicle/bin/discord_post.py")
        subprocess.run(
            f"source {env_path} && python3 {script} --operator -c '{msg}'",
            shell=True, executable="/bin/bash"
        )
        print("\nPosted to #operator")


QUEUE = [
    ("meta-llama/Llama-3.1-70B-Instruct", "llama3_70b", 8, "GQA 8, 80 layers, test if bifurcation replicates"),
    ("google/gemma-2-27b-it", "gemma2_27b", 2, "sorter (GQA 2, 46 layers, different species test)"),
]


def launch_next(pod_ssh):
    """After current model finishes, clean cache and start next queued model."""
    parts = pod_ssh.split(":")
    host = parts[0]
    port = parts[1] if len(parts) > 1 else "22"
    ssh_base = ["ssh", f"root@{host}", "-p", port, "-i",
                os.path.expanduser("~/.ssh/id_ed25519"), "-o", "StrictHostKeyChecking=no"]

    done = set()
    for short in glob.glob(os.path.join(RESULTS_DIR, "f614_scale_*_perlayer.json")):
        name = os.path.basename(short).replace("f614_scale_", "").replace("_perlayer.json", "")
        done.add(name)

    next_model = None
    for hf_id, short, gqa, desc in QUEUE:
        if short not in done:
            next_model = (hf_id, short, gqa, desc)
            break

    if not next_model:
        print("All queued models complete.")
        return

    hf_id, short, gqa, desc = next_model
    print(f"Next: {hf_id} ({short})")
    print("Cleaning HF cache on pod...")
    subprocess.run(ssh_base + ["rm -rf /workspace/.cache/huggingface/hub/models--*"], check=False)

    run_cmd = (f"export HF_TOKEN=$HF_TOKEN && "
               f"export HF_HOME=/workspace/.cache/huggingface && export PYTHONUNBUFFERED=1 && "
               f"export F614_USE_BNB=1 && "
               f"nohup python3 -u /workspace/f614_single.py '{hf_id}' {short} {gqa} '{desc}' "
               f"> /workspace/{short}.log 2>&1 &")
    print(f"Launching: {run_cmd}")
    subprocess.run(ssh_base + [run_cmd], check=True)
    print(f"Started {short} on pod. Log: /workspace/{short}.log")


def energy_analysis(path=None):
    """Kimi Σ operationalization: relay CCS redistributes (ΔΣ≈0), sorter CCS drives (ΔΣ>0).
    Reads spectral_energy from f614_single.py results (requires runs with Σ output)."""
    if path:
        files_list = [path]
    else:
        files_list = sorted(glob.glob(os.path.join(RESULTS_DIR, "f614_scale_*[!_perlayer][!_conservation].json")))
        files_list = [f for f in files_list if "perlayer" not in f and "conservation" not in f and "demeaned" not in f]

    print(f"\n{'='*75}")
    print(f"Σ ENERGY ANALYSIS ({len(files_list)} models)")
    print(f"{'='*75}")
    print(f"{'Model':<20} {'GQA':>4} {'ΔΣ(+CCS)':>12} {'ΔΣ(-CCS)':>12} {'ratio(+)':>10} {'Species':>10}")
    print("-" * 72)

    results = []
    missing = 0
    for fn in files_list:
        with open(fn) as f:
            d = json.load(f)
        se = d.get("spectral_energy")
        if not se:
            missing += 1
            continue
        short = d.get("short", os.path.basename(fn).replace(".json", ""))
        gqa = d.get("gqa_ratio", 0)
        ds = se.get("delta_sigma", {})
        rt = se.get("ratio_to_neutral", {})
        dp = ds.get("ccs_pos", 0)
        dn = ds.get("ccs_neg", 0)
        rp = rt.get("ccs_pos", 1.0)
        species = d.get("predicted_species", "?")
        print(f"{short:<20} {gqa:>4} {dp:>+12.1f} {dn:>+12.1f} {rp:>10.6f} {species:>10}")
        results.append({"model": short, "gqa_ratio": gqa,
                        "delta_sigma_pos": round(dp, 1), "delta_sigma_neg": round(dn, 1),
                        "ratio_pos": round(rp, 6), "species": species})

    if missing:
        print(f"\n({missing} files lack spectral_energy — re-run with updated f614_single.py)")
    if results:
        out_path = os.path.join(RESULTS_DIR, "energy_analysis.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved: {out_path}")
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "fetch" and len(sys.argv) >= 4:
        fetch_result(sys.argv[2], sys.argv[3])
    elif cmd == "analyze" and len(sys.argv) >= 3:
        analyze(sys.argv[2])
    elif cmd == "compare":
        compare()
    elif cmd == "report":
        report(post_discord="--discord" in sys.argv)
    elif cmd == "conservation" and len(sys.argv) >= 3:
        conservation_analysis(sys.argv[2])
    elif cmd == "demeaned":
        demeaned_analysis(sys.argv[2] if len(sys.argv) >= 3 else None)
    elif cmd == "pca":
        pca_analysis(sys.argv[2] if len(sys.argv) >= 3 else None)
    elif cmd == "redundancy":
        path_redundancy_analysis(sys.argv[2] if len(sys.argv) >= 3 else None)
    elif cmd == "energy":
        energy_analysis(sys.argv[2] if len(sys.argv) >= 3 else None)
    elif cmd == "next" and len(sys.argv) >= 3:
        launch_next(sys.argv[2])
    else:
        print(__doc__)
