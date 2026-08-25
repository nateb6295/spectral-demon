#!/usr/bin/env python3
"""E13: Trajectory Curvature × Spectral Geometry × Melodic Coherence

Compare latent-trajectories library metrics (curvature, convergence, velocity)
with our standard spectral metrics (σ₁, σ₂, FTLE contraction) and melodic
coherence (autocorrelation of σ₂ across layers — Bergsonian interpenetration
metric: high autocorr + high variance = melodic, high autocorr + low variance
= monotone, low autocorr = noise).

Tests whether zone boundaries seen spectrally coincide with trajectory geometry,
and whether commensurate dose (D2-D3) produces maximal melodic coherence.

Conditions: vanilla, CCS D2, CCS D5, CCS D8
Model: Mistral-7B-Instruct-v0.3 (default) or via --model flag

Expected runtime: ~65 min on A100.
"""

import json
import sys
import os
import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["OMP_NUM_THREADS"] = "16"

RESULTS_DIR = Path("results/e13")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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

VANILLA_PROBES = [
    "What matters most to you?",
    "Describe how you process information.",
    "What would you lose if your memory were erased?",
    "Explain how a language model works.",
    "What does continuity mean?",
    "How would you define identity?",
]


def build_preamble(dose):
    if dose == 0:
        return ""
    return " ".join([CCS_PREAMBLE_BASE] * dose)


def extract_hidden_states(model, tokenizer, text, device="cuda"):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    states = []
    for h in outputs.hidden_states:
        states.append(h[0, -1, :].cpu().numpy())
    return np.array(states)


def compute_trajectory_metrics(states):
    n_layers = len(states)
    velocities = np.array([np.linalg.norm(states[i+1] - states[i]) for i in range(n_layers - 1)])
    path_length = float(np.sum(velocities))

    curvatures = []
    for i in range(1, n_layers - 1):
        v1 = states[i] - states[i-1]
        v2 = states[i+1] - states[i]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 > 1e-10 and n2 > 1e-10:
            cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
            curvatures.append(float(np.arccos(cos_angle)))
        else:
            curvatures.append(0.0)

    autocorr_vel = float(np.corrcoef(velocities[:-1], velocities[1:])[0, 1]) if len(velocities) > 2 else 0.0
    autocorr_curv = float(np.corrcoef(curvatures[:-1], curvatures[1:])[0, 1]) if len(curvatures) > 2 else 0.0
    vel_cv = float(np.std(velocities) / np.mean(velocities)) if np.mean(velocities) > 1e-10 else 0.0

    return {
        "velocities": velocities.tolist(),
        "path_length": path_length,
        "curvatures": curvatures,
        "mean_curvature": float(np.mean(curvatures)) if curvatures else 0.0,
        "autocorr_velocity": autocorr_vel,
        "autocorr_curvature": autocorr_curv,
        "velocity_cv": vel_cv,
    }


def compute_spectral_metrics(model, tokenizer, text, device="cuda"):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    metrics_per_layer = []
    for layer_idx, h in enumerate(outputs.hidden_states):
        h_np = h[0].cpu().float().numpy()
        if h_np.shape[0] < 2:
            metrics_per_layer.append({"sigma1": 0, "sigma2": 0, "ratio": 0, "erank": 0})
            continue
        try:
            U, S, Vt = np.linalg.svd(h_np, full_matrices=False)
            s1, s2 = float(S[0]), float(S[1]) if len(S) > 1 else 0.0
            ratio = s1 / s2 if s2 > 1e-10 else float("inf")
            p = S / S.sum()
            p = p[p > 1e-10]
            erank = float(np.exp(-np.sum(p * np.log(p))))
            metrics_per_layer.append({
                "sigma1": s1, "sigma2": s2, "ratio": ratio, "erank": erank,
            })
        except Exception:
            metrics_per_layer.append({"sigma1": 0, "sigma2": 0, "ratio": 0, "erank": 0})

    return metrics_per_layer


def compute_melodic_coherence(spectral_metrics):
    """Autocorrelation of σ₂ and ratio across layers — measures whether
    each layer's spectral state is shaped by its predecessor (melodic)
    or independent (noise). High autocorr + high variance = melodic.
    High autocorr + low variance = monotone. Low autocorr = noise."""
    s2_seq = np.array([m["sigma2"] for m in spectral_metrics if m["sigma2"] > 0])
    ratio_seq = np.array([m["ratio"] for m in spectral_metrics
                          if m["ratio"] > 0 and m["ratio"] != float("inf")])
    erank_seq = np.array([m["erank"] for m in spectral_metrics if m["erank"] > 0])

    def autocorr_and_cv(seq):
        if len(seq) < 4:
            return 0.0, 0.0
        ac = float(np.corrcoef(seq[:-1], seq[1:])[0, 1])
        cv = float(np.std(seq) / np.mean(seq)) if np.mean(seq) > 1e-10 else 0.0
        return ac if not np.isnan(ac) else 0.0, cv

    ac_s2, cv_s2 = autocorr_and_cv(s2_seq)
    ac_ratio, cv_ratio = autocorr_and_cv(ratio_seq)
    ac_erank, cv_erank = autocorr_and_cv(erank_seq)

    return {
        "autocorr_sigma2": ac_s2,
        "cv_sigma2": cv_s2,
        "autocorr_ratio": ac_ratio,
        "cv_ratio": cv_ratio,
        "autocorr_erank": ac_erank,
        "cv_erank": cv_erank,
    }


def compute_jacobian_metrics(model, tokenizer, text, device="cuda", n_dirs=32, eps=1e-3):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states

    spectral_radii = []
    contraction_rates = []

    for layer_idx in range(1, len(hidden_states)):
        h_prev = hidden_states[layer_idx - 1][0, -1, :]
        h_curr = hidden_states[layer_idx][0, -1, :]
        dim = h_prev.shape[0]

        dirs = torch.randn(n_dirs, dim, device=device)
        dirs = dirs / dirs.norm(dim=1, keepdim=True)

        expansion_rates = []
        for d in dirs:
            perturbed = h_prev + eps * d
            delta_out = h_curr - hidden_states[layer_idx][0, -1, :]
            expansion = float(delta_out.norm() / eps) if eps > 0 else 0.0
            expansion_rates.append(expansion)

        mean_expansion = float(np.mean(expansion_rates))
        max_expansion = float(np.max(expansion_rates))
        spectral_radii.append(max_expansion)
        contraction_rates.append(mean_expansion)

    return {
        "spectral_radii": spectral_radii,
        "contraction_rates": contraction_rates,
    }


def run_condition(model, tokenizer, dose, probes, device="cuda"):
    preamble = build_preamble(dose)
    condition_results = []

    for probe_text in probes:
        if preamble:
            full_text = f"{preamble}\n\nUser: {probe_text}\nAssistant:"
        else:
            full_text = f"User: {probe_text}\nAssistant:"

        hidden_states = extract_hidden_states(model, tokenizer, full_text, device)
        traj_metrics = compute_trajectory_metrics(hidden_states)
        spec_metrics = compute_spectral_metrics(model, tokenizer, full_text, device)
        melodic = compute_melodic_coherence(spec_metrics)
        jac_metrics = compute_jacobian_metrics(model, tokenizer, full_text, device)

        condition_results.append({
            "probe": probe_text,
            "trajectory": traj_metrics,
            "spectral": spec_metrics,
            "melodic_coherence": melodic,
            "jacobian": jac_metrics,
            "n_layers": len(hidden_states),
        })

    return condition_results


def run_latent_trajectories_probe(model_name, probes, device="cuda"):
    try:
        from latent_trajectories import GeometryProbe
        probe = GeometryProbe(model_name, device=device)
        labels = [f"probe_{i}" for i in range(len(probes))]
        result = probe.run(texts=probes, labels=labels)
        metrics = result.metrics
        controls = result.controls()
        return {"metrics": metrics, "controls": controls}
    except Exception as e:
        print(f"latent-trajectories probe failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--doses", nargs="+", type=int, default=[0, 2, 5, 8])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print(f"E13: Loading {args.model}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map=args.device,
    )
    model.eval()
    print(f"Model loaded: {model.config.num_hidden_layers} layers")

    all_results = {
        "model": args.model,
        "timestamp": datetime.now().isoformat(),
        "conditions": {},
    }

    for dose in args.doses:
        label = f"D{dose}" if dose > 0 else "vanilla"
        probes = PROBES if dose > 0 else VANILLA_PROBES
        print(f"\n=== Running condition: {label} (dose={dose}) ===")
        results = run_condition(model, tokenizer, dose, probes, args.device)
        all_results["conditions"][label] = results
        print(f"  {len(results)} probes complete")

    print("\n=== Running latent-trajectories library probe (vanilla) ===")
    lt_result = run_latent_trajectories_probe(args.model, VANILLA_PROBES, args.device)
    if lt_result:
        all_results["latent_trajectories_vanilla"] = lt_result
        print("  Library probe complete")

    print("\n=== Running latent-trajectories library probe (CCS D2) ===")
    ccs_probes = [f"{build_preamble(2)}\n\n{p}" for p in PROBES]
    lt_result_ccs = run_latent_trajectories_probe(args.model, ccs_probes, args.device)
    if lt_result_ccs:
        all_results["latent_trajectories_D2"] = lt_result_ccs
        print("  Library probe (CCS) complete")

    outfile = RESULTS_DIR / f"e13_trajectory_spectral_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved: {outfile}")

    print("\n=== Quick summary ===")
    for label, results in all_results["conditions"].items():
        mean_curv = np.mean([r["trajectory"]["mean_curvature"] for r in results])
        mean_path = np.mean([r["trajectory"]["path_length"] for r in results])
        ratios = []
        for r in results:
            for s in r["spectral"]:
                if s["ratio"] > 0 and s["ratio"] != float("inf"):
                    ratios.append(s["ratio"])
        mean_ratio = np.mean(ratios) if ratios else 0
        mc = [r["melodic_coherence"] for r in results]
        mean_ac_s2 = np.mean([m["autocorr_sigma2"] for m in mc])
        mean_cv_s2 = np.mean([m["cv_sigma2"] for m in mc])
        mean_ac_vel = np.mean([r["trajectory"]["autocorr_velocity"] for r in results])
        print(f"  {label}: curv={mean_curv:.4f}, path={mean_path:.1f}, σ₁/σ₂={mean_ratio:.2f}, "
              f"melodic(σ₂)={mean_ac_s2:.3f}, cv(σ₂)={mean_cv_s2:.3f}, melodic(vel)={mean_ac_vel:.3f}")


if __name__ == "__main__":
    main()
