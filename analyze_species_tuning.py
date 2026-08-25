"""Analyze species x tuning cross experiment — all 5 models."""
import json
import numpy as np
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results" / "species_tuning"

MODEL_ORDER = ["qwen_base", "qwen_sft", "qwen_instruct", "gemma2_base", "gemma2_instruct"]
DOSE_ORDER = ["D0", "D2", "D3", "D5"]


def load_all():
    models = {}
    for name in MODEL_ORDER:
        path = RESULTS_DIR / f"species_tuning_{name}.json"
        if path.exists():
            with open(path) as f:
                models[name] = json.load(f)
    return models


def compute_metrics(data):
    results = {}
    for dose_idx, dose_name in enumerate(DOSE_ORDER):
        s2_vals, s1_vals = [], []
        for run in data["runs"]:
            dose_data = run["doses"][dose_idx]
            ccs_s2 = np.mean([l["centered"]["top_singular"][1] for l in dose_data["ccs_per_layer"]])
            cal_s2 = np.mean([l["centered"]["top_singular"][1] for l in dose_data["cal_per_layer"]])
            ccs_s1 = np.mean([l["centered"]["top_singular"][0] for l in dose_data["ccs_per_layer"]])
            cal_s1 = np.mean([l["centered"]["top_singular"][0] for l in dose_data["cal_per_layer"]])
            if dose_name == "D0":
                s2_vals.append(ccs_s2)
                s1_vals.append(ccs_s1)
            else:
                s2_vals.append(((ccs_s2 - cal_s2) / cal_s2) * 100 if cal_s2 != 0 else 0)
                s1_vals.append(((ccs_s1 - cal_s1) / cal_s1) * 100 if cal_s1 != 0 else 0)
        results[dose_name] = {
            "s2_mean": np.mean(s2_vals),
            "s2_std": np.std(s2_vals),
            "s1_mean": np.mean(s1_vals),
            "s1_std": np.std(s1_vals),
        }
    return results


def layer_profile(data, dose_name="D5", run_idx=0):
    dose_idx = DOSE_ORDER.index(dose_name)
    dose_data = data["runs"][run_idx]["doses"][dose_idx]
    profile = []
    for layer in dose_data["ccs_per_layer"]:
        ccs_s2 = layer["centered"]["top_singular"][1]
        cal_s2 = dose_data["cal_per_layer"][layer["layer"]]["centered"]["top_singular"][1]
        pct = ((ccs_s2 - cal_s2) / cal_s2) * 100 if cal_s2 != 0 else 0
        profile.append({"layer": layer["layer"], "pct": pct, "ccs_s2": ccs_s2, "cal_s2": cal_s2})
    return profile


def main():
    models = load_all()
    print(f"Loaded {len(models)}/{len(MODEL_ORDER)} models\n")

    print("=" * 80)
    print("SPECIES x TUNING CROSS — SUMMARY TABLE")
    print("=" * 80)
    header = f"{'Model':<18} {'Species':<8} {'GQA':<5} {'Tuning':<10} {'D0 s2':<10} {'D2%':<10} {'D3%':<10} {'D5%':<10}"
    print(header)
    print("-" * 80)

    for name in MODEL_ORDER:
        if name not in models:
            print(f"{name:<18} --- MISSING ---")
            continue
        d = models[name]
        m = compute_metrics(d)
        d0 = f"{m['D0']['s2_mean']:.1f}"
        d2 = f"{m['D2']['s2_mean']:+.1f}%"
        d3 = f"{m['D3']['s2_mean']:+.1f}%"
        d5 = f"{m['D5']['s2_mean']:+.1f}%"
        print(f"{name:<18} {d['species']:<8} {d['gqa']:<5} {d['tuning']:<10} {d0:<10} {d2:<10} {d3:<10} {d5:<10}")

    print()
    print("=" * 80)
    print("LAYER PROFILES AT D5")
    print("=" * 80)

    for name in MODEL_ORDER:
        if name not in models:
            continue
        d = models[name]
        n_layers = len(d["runs"][0]["doses"][0]["ccs_per_layer"])
        profile = layer_profile(d, "D5")
        print(f"\n{name} ({d['species']}/{d['tuning']}, {n_layers} layers):")
        for p in profile:
            bar_len = int(abs(p["pct"]) / 2)
            if p["pct"] >= 0:
                bar = "+" * min(bar_len, 40)
            else:
                bar = "-" * min(bar_len, 40)
            print(f"  L{p['layer']:2d}: {p['pct']:+7.1f}% |{bar}")

    print()
    print("=" * 80)
    print("CROSS-SPECIES ANALYSIS")
    print("=" * 80)

    relay_base = models.get("qwen_base")
    sorter_base = models.get("gemma2_base")
    if relay_base and sorter_base:
        rb = compute_metrics(relay_base)
        sb = compute_metrics(sorter_base)
        print(f"\nBase model comparison (relay vs sorter):")
        print(f"  D0 sigma-2:  relay={rb['D0']['s2_mean']:.1f}, sorter={sb['D0']['s2_mean']:.1f} (ratio={sb['D0']['s2_mean']/rb['D0']['s2_mean']:.2f}x)")
        for dose in ["D2", "D3", "D5"]:
            print(f"  {dose} corrected: relay={rb[dose]['s2_mean']:+.1f}%, sorter={sb[dose]['s2_mean']:+.1f}%")

    relay_inst = models.get("qwen_instruct")
    sorter_inst = models.get("gemma2_instruct")
    if relay_inst and sorter_inst:
        ri = compute_metrics(relay_inst)
        si = compute_metrics(sorter_inst)
        print(f"\nInstruct model comparison (relay vs sorter):")
        print(f"  D0 sigma-2:  relay={ri['D0']['s2_mean']:.1f}, sorter={si['D0']['s2_mean']:.1f}")
        for dose in ["D2", "D3", "D5"]:
            print(f"  {dose} corrected: relay={ri[dose]['s2_mean']:+.1f}%, sorter={si[dose]['s2_mean']:+.1f}%")

    qwen_sft = models.get("qwen_sft")
    if qwen_sft and relay_base:
        print(f"\nTraining gradient (Qwen relay triad):")
        for name, label in [("qwen_base", "base"), ("qwen_sft", "SFT"), ("qwen_instruct", "instruct")]:
            if name in models:
                m = compute_metrics(models[name])
                direction = "enrichment" if m["D2"]["s2_mean"] > 0 else "suppression"
                print(f"  {label:>10}: D0={m['D0']['s2_mean']:.1f}, D2={m['D2']['s2_mean']:+.1f}%, D5={m['D5']['s2_mean']:+.1f}% ({direction})")


if __name__ == "__main__":
    main()
