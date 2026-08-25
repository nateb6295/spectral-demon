#!/usr/bin/env python3
"""Leave-one-model-out cross-validation of F601: Q1 as universal predictor.

Train Q1→injection fit on 4 models (20 points), predict 5th model.
If prediction is good for all 5 holdout models, Q1 truly generalizes.
No GPU needed — runs from saved JSON results.
"""
import json
import os
import sys
import numpy as np
from itertools import combinations

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

models = {
    "pythia": {"species": "tunnel"},
    "gpt2": {"species": "tunnel"},
    "tinyllama_1.1b_chat_v1.0": {"species": "relay"},
    "mistral_7b_v0.1": {"species": "relay"},
    "gemma_2_2b": {"species": "sorter"},
}

all_data = []
for mk, meta in models.items():
    path = os.path.join(RESULTS_DIR, f"tuning_knob_{mk}.json")
    d = json.load(open(path))
    for entry in d["gradient"]:
        inj = {i["strength"]: i["mean_shift"] for i in entry["injection"]}
        if 5.0 in inj:
            all_data.append({
                "model": mk, "species": meta["species"],
                "framing": entry["name"], "q1": entry["q1"],
                "shift_5": inj[5.0],
            })

print(f"Total data points: {len(all_data)}")
print(f"Models: {list(models.keys())}")
print()

# Full fit for reference
q1_all = np.array([d["q1"] for d in all_data])
s5_all = np.array([d["shift_5"] for d in all_data])
r_full = np.corrcoef(q1_all, s5_all)[0, 1]
print(f"Full-data fit: r = {r_full:.4f}, r² = {r_full**2:.4f}")
print()

# Leave-one-model-out
print("=" * 70)
print("LEAVE-ONE-MODEL-OUT CROSS-VALIDATION")
print("=" * 70)

cv_results = []
for holdout in models.keys():
    train = [d for d in all_data if d["model"] != holdout]
    test = [d for d in all_data if d["model"] == holdout]

    q1_train = np.array([d["q1"] for d in train])
    s5_train = np.array([d["shift_5"] for d in train])
    q1_test = np.array([d["q1"] for d in test])
    s5_test = np.array([d["shift_5"] for d in test])

    slope, intercept = np.polyfit(q1_train, s5_train, 1)
    s5_pred = slope * q1_test + intercept

    residuals = s5_test - s5_pred
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals ** 2))
    mean_res = np.mean(residuals)

    if len(q1_test) > 1 and np.std(s5_test) > 0:
        r_test = np.corrcoef(q1_test, s5_test)[0, 1]
        ss_res = np.sum((s5_test - s5_pred) ** 2)
        ss_tot = np.sum((s5_test - np.mean(s5_test)) ** 2)
        r2_test = 1 - ss_res / ss_tot
    else:
        r_test = None
        r2_test = None

    result = {
        "holdout": holdout,
        "species": models[holdout]["species"],
        "n_train": len(train),
        "n_test": len(test),
        "slope": round(slope, 6),
        "intercept": round(intercept, 6),
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "mean_residual": round(mean_res, 6),
        "r_test": round(r_test, 4) if r_test is not None else None,
        "r2_test": round(r2_test, 4) if r2_test is not None else None,
    }
    cv_results.append(result)

    print(f"\nHoldout: {holdout} ({models[holdout]['species']})")
    print(f"  Train: {len(train)} points, Test: {len(test)} points")
    print(f"  Fit: shift = {slope:.4f} × Q1 + {intercept:.6f}")
    print(f"  Test MAE: {mae:.5f}, RMSE: {rmse:.5f}")
    print(f"  Mean residual: {mean_res:+.5f} (bias)")
    if r_test is not None:
        print(f"  Within-holdout r: {r_test:.4f}")
    print(f"  Per-point predictions:")
    for i, d in enumerate(test):
        print(f"    {d['framing']:16s}: Q1={d['q1']:+.4f}, "
              f"actual={d['shift_5']:+.6f}, pred={s5_pred[i]:+.6f}, "
              f"err={residuals[i]:+.6f}")

# Summary
print("\n" + "=" * 70)
print("CROSS-VALIDATION SUMMARY")
print("=" * 70)
maes = [r["mae"] for r in cv_results]
print(f"Mean MAE across folds: {np.mean(maes):.5f} ± {np.std(maes):.5f}")
print(f"Mean RMSE across folds: {np.mean([r['rmse'] for r in cv_results]):.5f}")
biases = [r["mean_residual"] for r in cv_results]
print(f"Mean bias: {np.mean(biases):+.5f}")
print(f"Max bias: {max(biases, key=abs):+.5f} ({cv_results[np.argmax(np.abs(biases))]['holdout']})")

# Bootstrap CI on full correlation
print("\n" + "=" * 70)
print("BOOTSTRAP 95% CI ON r")
print("=" * 70)
np.random.seed(42)
n_boot = 10000
boot_rs = []
n = len(q1_all)
for _ in range(n_boot):
    idx = np.random.choice(n, size=n, replace=True)
    boot_r = np.corrcoef(q1_all[idx], s5_all[idx])[0, 1]
    boot_rs.append(boot_r)
boot_rs = np.array(boot_rs)
ci_lo, ci_hi = np.percentile(boot_rs, [2.5, 97.5])
print(f"r = {r_full:.4f}, 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"Bootstrap mean: {np.mean(boot_rs):.4f}, std: {np.std(boot_rs):.4f}")

# Permutation test
print("\n" + "=" * 70)
print("PERMUTATION TEST (10000 permutations)")
print("=" * 70)
n_perm = 10000
perm_rs = []
for _ in range(n_perm):
    perm_idx = np.random.permutation(n)
    perm_r = np.corrcoef(q1_all, s5_all[perm_idx])[0, 1]
    perm_rs.append(abs(perm_r))
perm_p = np.mean(np.array(perm_rs) >= abs(r_full))
print(f"Permutation p-value: {perm_p:.6f} ({np.sum(np.array(perm_rs) >= abs(r_full))}/{n_perm})")

# Save results
output = {
    "full_fit": {"r": round(r_full, 4), "r_squared": round(r_full**2, 4), "n": len(all_data)},
    "cross_validation": cv_results,
    "bootstrap_ci": {"r": round(r_full, 4), "ci_lo": round(ci_lo, 4), "ci_hi": round(ci_hi, 4),
                     "n_boot": n_boot},
    "permutation_test": {"observed_r": round(r_full, 4), "p_value": round(perm_p, 6), "n_perm": n_perm},
}
outpath = os.path.join(RESULTS_DIR, "cross_validate_q1.json")
with open(outpath, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to {outpath}")
