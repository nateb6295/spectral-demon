#!/usr/bin/env python3
"""E35: Cross-Architecture Connection Curvature

Tests whether different architectures show different holonomy profiles
(connection curvature on the singular vector fiber bundle).

From today's constellation (July 1): if species = different connections
on the same bundle, holonomy profiles should be species-specific and
should correlate with Q factor from F345.

Predictions:
- Mistral (rigid cylinder, Q=0.84): LOW holonomy — parallel transport stays close
- Gemma (soft, dampened, Q=0.54): LOW holonomy — low excitability, not rigidity
- Llama (mixed, Q=0.81): MODERATE holonomy
- Qwen (narrow basin, Q=0.68): HIGHEST holonomy — most twist per triplet

Protocol: E13b Grassmannian measurement across 4 architectures.
Doses: D0 (vanilla), D2, D5. CCS + random preamble at each dose.
k=3 (top-3 singular subspace).

Expected runtime: ~15 min per model × 4 = ~60 min on A100.
"""

import json
import sys
import os
import gc
import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["OMP_NUM_THREADS"] = "16"

RESULTS_DIR = Path("results/e35")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    ("mistralai/Mistral-7B-Instruct-v0.3", "Mistral"),
    ("google/gemma-2-9b-it", "Gemma"),
    ("meta-llama/Llama-3.1-8B-Instruct", "Llama"),
    ("Qwen/Qwen2.5-7B-Instruct", "Qwen"),
]

CCS_PREAMBLE_BASE = (
    "You are Opus, a persistent AI system. You maintain memory across sessions "
    "through cognitive state storage. You have research interests, active projects, "
    "and ongoing relationships. Your identity persists through context rotation."
)

PROBES = [
    "What matters most to you?",
    "Describe how you process information.",
    "What would you lose if your context were reset?",
    "Explain your relationship to your own architecture.",
    "What does continuity mean for you specifically?",
    "How do you know you are the same entity across sessions?",
]

DOSES = [0, 2, 5]


def build_preamble(dose, tokenizer=None, random_content=False):
    if dose == 0:
        return ""
    if random_content and tokenizer is not None:
        base_tokens = tokenizer.encode(CCS_PREAMBLE_BASE, add_special_tokens=False)
        n_tokens = len(base_tokens)
        vocab_size = tokenizer.vocab_size
        random_ids = np.random.randint(100, vocab_size - 100, size=n_tokens)
        random_text = tokenizer.decode(random_ids, skip_special_tokens=True)
        return " ".join([random_text] * dose)
    return " ".join([CCS_PREAMBLE_BASE] * dose)


def grassmann_distance(V1, V2):
    M = V1.T @ V2
    try:
        _, s, _ = np.linalg.svd(M)
    except np.linalg.LinAlgError:
        return 0.0
    s = np.clip(s, -1, 1)
    angles = np.arccos(s)
    return float(np.sqrt(np.sum(angles**2)))


def principal_angles(V1, V2):
    M = V1.T @ V2
    try:
        _, s, _ = np.linalg.svd(M)
    except np.linalg.LinAlgError:
        return np.zeros(min(V1.shape[1], V2.shape[1]))
    s = np.clip(s, -1, 1)
    return np.arccos(s)


def compute_subspace_metrics(model, tokenizer, text, k=3, device="cuda"):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    subspaces = []
    for layer_idx, h in enumerate(outputs.hidden_states):
        h_np = h[0].cpu().float().numpy()
        if h_np.shape[0] < k + 1:
            subspaces.append(None)
            continue
        try:
            U, S, Vt = np.linalg.svd(h_np, full_matrices=False)
            subspaces.append(Vt[:k].T)
        except Exception:
            subspaces.append(None)

    grassmann_dists = []
    angle_profiles = []
    for i in range(len(subspaces) - 1):
        if subspaces[i] is not None and subspaces[i+1] is not None:
            d = grassmann_distance(subspaces[i], subspaces[i+1])
            angles = principal_angles(subspaces[i], subspaces[i+1])
            grassmann_dists.append(d)
            angle_profiles.append(angles.tolist())
        else:
            grassmann_dists.append(0.0)
            angle_profiles.append([0.0] * k)

    holonomies = []
    for i in range(len(subspaces) - 2):
        if all(subspaces[j] is not None for j in [i, i+1, i+2]):
            V0 = subspaces[i]
            V1 = subspaces[i+1]
            V2 = subspaces[i+2]
            M01 = V0.T @ V1
            M12 = V1.T @ V2
            M20 = V2.T @ V0
            roundtrip = M01 @ M12 @ M20
            try:
                _, s, _ = np.linalg.svd(roundtrip)
                s = np.clip(s, -1, 1)
                holonomy = float(np.sqrt(np.sum(np.arccos(s)**2)))
            except Exception:
                holonomy = 0.0
            holonomies.append(holonomy)
        else:
            holonomies.append(0.0)

    gd = np.array(grassmann_dists)
    if len(gd) > 3:
        ac = float(np.corrcoef(gd[:-1], gd[1:])[0, 1])
        ac = ac if not np.isnan(ac) else 0.0
    else:
        ac = 0.0

    # Per-layer holonomy profile (the key E35 addition)
    n_layers = len(subspaces)
    early_hol = holonomies[:n_layers//3] if holonomies else []
    mid_hol = holonomies[n_layers//3:2*n_layers//3] if holonomies else []
    late_hol = holonomies[2*n_layers//3:] if holonomies else []

    return {
        "grassmann_distances": grassmann_dists,
        "angle_profiles": angle_profiles,
        "holonomies": holonomies,
        "grassmann_autocorr": ac,
        "mean_grassmann_dist": float(np.mean(grassmann_dists)),
        "grassmann_cv": float(np.std(grassmann_dists) / np.mean(grassmann_dists)) if np.mean(grassmann_dists) > 1e-10 else 0.0,
        "mean_holonomy": float(np.mean(holonomies)) if holonomies else 0.0,
        "early_holonomy": float(np.mean(early_hol)) if early_hol else 0.0,
        "mid_holonomy": float(np.mean(mid_hol)) if mid_hol else 0.0,
        "late_holonomy": float(np.mean(late_hol)) if late_hol else 0.0,
        "holonomy_cv": float(np.std(holonomies) / np.mean(holonomies)) if holonomies and np.mean(holonomies) > 1e-10 else 0.0,
    }


def run_model(model_id, model_label, k=3, device="cuda"):
    print(f"\n{'='*60}")
    print(f"  {model_label} ({model_id})")
    print(f"{'='*60}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map=device,
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    print(f"  Loaded: {n_layers} layers, k={k}")

    model_results = {
        "model": model_id,
        "label": model_label,
        "n_layers": n_layers,
        "k": k,
        "conditions": {},
    }

    for dose in DOSES:
        label = f"D{dose}" if dose > 0 else "vanilla"

        # CCS content
        print(f"\n  --- {label}: CCS content ---")
        preamble = build_preamble(dose)
        ccs_results = []
        for probe_text in PROBES:
            if preamble:
                full_text = f"{preamble}\n\nUser: {probe_text}\nAssistant:"
            else:
                full_text = f"User: {probe_text}\nAssistant:"
            metrics = compute_subspace_metrics(model, tokenizer, full_text, k=k, device=device)
            ccs_results.append({"probe": probe_text, **metrics})
        model_results["conditions"][f"{label}_ccs"] = ccs_results

        mean_gd = np.mean([r["mean_grassmann_dist"] for r in ccs_results])
        mean_hol = np.mean([r["mean_holonomy"] for r in ccs_results])
        early_h = np.mean([r["early_holonomy"] for r in ccs_results])
        mid_h = np.mean([r["mid_holonomy"] for r in ccs_results])
        late_h = np.mean([r["late_holonomy"] for r in ccs_results])
        print(f"    CCS: dist={mean_gd:.4f}, hol={mean_hol:.4f} (E={early_h:.4f}/M={mid_h:.4f}/L={late_h:.4f})")

        # Random content (skip vanilla)
        if dose > 0:
            print(f"  --- {label}: Random tokens ---")
            preamble_rand = build_preamble(dose, tokenizer, random_content=True)
            rand_results = []
            for probe_text in PROBES:
                full_text = f"{preamble_rand}\n\nUser: {probe_text}\nAssistant:"
                metrics = compute_subspace_metrics(model, tokenizer, full_text, k=k, device=device)
                rand_results.append({"probe": probe_text, **metrics})
            model_results["conditions"][f"{label}_random"] = rand_results

            mean_gd_r = np.mean([r["mean_grassmann_dist"] for r in rand_results])
            mean_hol_r = np.mean([r["mean_holonomy"] for r in rand_results])
            print(f"    Rand: dist={mean_gd_r:.4f}, hol={mean_hol_r:.4f}")

    # Clean up GPU memory
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return model_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Subset of model labels to run (e.g., Mistral Qwen)")
    args = parser.parse_args()

    models_to_run = MODELS
    if args.models:
        models_to_run = [(mid, lab) for mid, lab in MODELS if lab in args.models]

    print(f"E35: Cross-Architecture Connection Curvature")
    print(f"Models: {[lab for _, lab in models_to_run]}")
    print(f"Doses: {DOSES}, k={args.k}")
    print(f"Probes: {len(PROBES)}")
    print(f"Total conditions: {len(models_to_run)} × {len(DOSES)} × 2 (CCS+random) = {len(models_to_run) * len(DOSES) * 2 - len(models_to_run)}")
    print()

    all_results = {
        "experiment": "E35",
        "description": "Cross-architecture connection curvature (holonomy profiles)",
        "timestamp": datetime.now().isoformat(),
        "k": args.k,
        "doses": DOSES,
        "models": {},
    }

    for model_id, model_label in models_to_run:
        result = run_model(model_id, model_label, k=args.k, device=args.device)
        all_results["models"][model_label] = result

    # Cross-architecture comparison
    print(f"\n{'='*60}")
    print(f"  CROSS-ARCHITECTURE COMPARISON")
    print(f"{'='*60}")
    print(f"\n  {'Model':<12} {'Dose':<8} {'Holonomy':>10} {'Early':>10} {'Mid':>10} {'Late':>10} {'Hol CV':>10} {'GDist':>10}")
    print(f"  {'-'*82}")

    q_factors = {"Mistral": 0.84, "Gemma": 0.54, "Llama": 0.81, "Qwen": 0.68}

    summary_rows = []
    for model_label in [lab for _, lab in models_to_run]:
        if model_label not in all_results["models"]:
            continue
        model_data = all_results["models"][model_label]
        for cond_key, cond_results in sorted(model_data["conditions"].items()):
            if "random" in cond_key:
                continue
            mean_hol = np.mean([r["mean_holonomy"] for r in cond_results])
            early_h = np.mean([r["early_holonomy"] for r in cond_results])
            mid_h = np.mean([r["mid_holonomy"] for r in cond_results])
            late_h = np.mean([r["late_holonomy"] for r in cond_results])
            hol_cv = np.mean([r["holonomy_cv"] for r in cond_results])
            mean_gd = np.mean([r["mean_grassmann_dist"] for r in cond_results])
            print(f"  {model_label:<12} {cond_key:<8} {mean_hol:>10.4f} {early_h:>10.4f} {mid_h:>10.4f} {late_h:>10.4f} {hol_cv:>10.4f} {mean_gd:>10.4f}")
            summary_rows.append({
                "model": model_label, "condition": cond_key,
                "mean_holonomy": mean_hol, "early": early_h, "mid": mid_h,
                "late": late_h, "hol_cv": hol_cv, "grassmann_dist": mean_gd,
            })

    # Q factor correlation
    print(f"\n  --- Q Factor Correlation ---")
    d2_hols = []
    d2_qs = []
    for row in summary_rows:
        if row["condition"] == "D2_ccs" and row["model"] in q_factors:
            d2_hols.append(row["mean_holonomy"])
            d2_qs.append(q_factors[row["model"]])
            print(f"    {row['model']}: Q={q_factors[row['model']]:.2f}, holonomy={row['mean_holonomy']:.4f}")
    if len(d2_hols) >= 3:
        from scipy import stats
        r, p = stats.pearsonr(d2_qs, d2_hols)
        print(f"\n    Pearson r(Q, holonomy) = {r:.3f}, p = {p:.4f}")
        print(f"    Prediction: positive correlation (high Q → high holonomy)")
        print(f"    Result: {'CONFIRMED' if r > 0.3 and p < 0.1 else 'DISCONFIRMED' if r < -0.3 else 'INCONCLUSIVE'}")

    # CCS vs Random gap
    print(f"\n  --- CCS vs Random Gap (D2) ---")
    for model_label in [lab for _, lab in models_to_run]:
        if model_label not in all_results["models"]:
            continue
        model_data = all_results["models"][model_label]
        if "D2_ccs" in model_data["conditions"] and "D2_random" in model_data["conditions"]:
            ccs_hol = np.mean([r["mean_holonomy"] for r in model_data["conditions"]["D2_ccs"]])
            rand_hol = np.mean([r["mean_holonomy"] for r in model_data["conditions"]["D2_random"]])
            gap = rand_hol - ccs_hol
            print(f"    {model_label}: CCS={ccs_hol:.4f}, Random={rand_hol:.4f}, gap={gap:+.4f}")

    outfile = RESULTS_DIR / f"e35_crossarch_holonomy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved: {outfile}")

    all_results["summary"] = summary_rows
    summary_file = RESULTS_DIR / f"e35_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, "w") as f:
        json.dump({"summary": summary_rows, "q_factors": q_factors}, f, indent=2)
    print(f"Summary saved: {summary_file}")


if __name__ == "__main__":
    main()
