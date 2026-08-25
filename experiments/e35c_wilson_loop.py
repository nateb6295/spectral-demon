#!/usr/bin/env python3
"""E35c: Wilson Loop — Path-Dependent Holonomy Test

True holonomy test: traverse a closed loop in dose space (D2→D5→D8→D5→D2)
and measure whether the singular subspace returns to its starting point.

If the connection is flat, the subspace should return exactly.
If curved, the round-trip produces a rotation — measurable holonomy.

This differs from E35's layer-triplet holonomy: E35 measured LOCAL curvature
(twist between adjacent layers). E35c measures GLOBAL holonomy (accumulation
of twist around a closed path in parameter space).

All 4 architectures. Expected runtime: ~20 min on A100.
"""

import json
import sys
import os
import gc
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
]

DOSE_LOOP = [2, 5, 8, 5, 2]  # closed loop in dose space


def build_preamble(dose):
    if dose == 0:
        return ""
    return " ".join([CCS_PREAMBLE_BASE] * dose)


def extract_subspaces(model, tokenizer, text, k=3, device="cuda"):
    """Extract top-k subspace at each layer."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    subspaces = []
    for h in outputs.hidden_states:
        h_np = h[0].cpu().float().numpy()
        if h_np.shape[0] < k + 1:
            subspaces.append(None)
            continue
        try:
            U, S, Vt = np.linalg.svd(h_np, full_matrices=False)
            subspaces.append(Vt[:k].T)
        except Exception:
            subspaces.append(None)
    return subspaces


def grassmann_distance(V1, V2):
    M = V1.T @ V2
    try:
        _, s, _ = np.linalg.svd(M)
    except np.linalg.LinAlgError:
        return 0.0
    s = np.clip(s, -1, 1)
    angles = np.arccos(s)
    return float(np.sqrt(np.sum(angles**2)))


def subspace_cosine(V1, V2):
    """Mean cosine similarity between corresponding singular vectors."""
    cos_vals = []
    for i in range(min(V1.shape[1], V2.shape[1])):
        cos = abs(float(np.dot(V1[:, i], V2[:, i])))
        cos_vals.append(cos)
    return float(np.mean(cos_vals))


def run_wilson_loop(model, tokenizer, probe_text, k=3, device="cuda"):
    """Run a Wilson loop: measure subspace at each dose, compare start to end."""

    subspaces_by_dose = {}
    for step_idx, dose in enumerate(DOSE_LOOP):
        preamble = build_preamble(dose)
        full_text = f"{preamble}\n\nUser: {probe_text}\nAssistant:"
        subspaces = extract_subspaces(model, tokenizer, full_text, k=k, device=device)
        subspaces_by_dose[f"step{step_idx}_D{dose}"] = subspaces

    # Compare start (step0, D2) vs end (step4, D2)
    start_subspaces = subspaces_by_dose["step0_D2"]
    end_subspaces = subspaces_by_dose["step4_D2"]

    # Per-layer holonomy: how much did each layer's subspace rotate?
    layer_holonomies = []
    layer_cosines = []
    for i in range(len(start_subspaces)):
        if start_subspaces[i] is not None and end_subspaces[i] is not None:
            gd = grassmann_distance(start_subspaces[i], end_subspaces[i])
            cos = subspace_cosine(start_subspaces[i], end_subspaces[i])
            layer_holonomies.append(gd)
            layer_cosines.append(cos)
        else:
            layer_holonomies.append(0.0)
            layer_cosines.append(1.0)

    # Also measure path distances (how far each step moved from previous)
    path_distances = []
    dose_labels = [f"step{i}_D{d}" for i, d in enumerate(DOSE_LOOP)]
    for i in range(len(dose_labels) - 1):
        s1 = subspaces_by_dose[dose_labels[i]]
        s2 = subspaces_by_dose[dose_labels[i+1]]
        step_dists = []
        for j in range(len(s1)):
            if s1[j] is not None and s2[j] is not None:
                step_dists.append(grassmann_distance(s1[j], s2[j]))
        path_distances.append(float(np.mean(step_dists)) if step_dists else 0.0)

    n = len(layer_holonomies)
    early = layer_holonomies[:n//3]
    mid = layer_holonomies[n//3:2*n//3]
    late = layer_holonomies[2*n//3:]

    return {
        "layer_holonomies": layer_holonomies,
        "layer_cosines": layer_cosines,
        "path_distances": path_distances,
        "mean_holonomy": float(np.mean(layer_holonomies)),
        "mean_cosine": float(np.mean(layer_cosines)),
        "early_holonomy": float(np.mean(early)) if early else 0.0,
        "mid_holonomy": float(np.mean(mid)) if mid else 0.0,
        "late_holonomy": float(np.mean(late)) if late else 0.0,
        "max_holonomy": float(np.max(layer_holonomies)),
        "max_holonomy_layer": int(np.argmax(layer_holonomies)),
        "total_path_distance": float(np.sum(path_distances)),
    }


def main():
    print("E35c: Wilson Loop — Path-Dependent Holonomy")
    print(f"Dose loop: {DOSE_LOOP}")
    print(f"Probes: {len(PROBES)}")
    print()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    all_results = {
        "experiment": "E35c",
        "description": "Wilson loop holonomy — closed path in dose space",
        "dose_loop": DOSE_LOOP,
        "timestamp": datetime.now().isoformat(),
        "models": {},
    }

    for model_id, model_label in MODELS:
        print(f"\n{'='*60}")
        print(f"  {model_label} ({model_id})")
        print(f"{'='*60}")

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="cuda",
        )
        model.eval()
        n_layers = model.config.num_hidden_layers
        print(f"  Loaded: {n_layers} layers")

        probe_results = []
        for probe_text in PROBES:
            result = run_wilson_loop(model, tokenizer, probe_text, k=3)
            probe_results.append({"probe": probe_text, **result})

        mean_hol = np.mean([r["mean_holonomy"] for r in probe_results])
        mean_cos = np.mean([r["mean_cosine"] for r in probe_results])
        early = np.mean([r["early_holonomy"] for r in probe_results])
        mid = np.mean([r["mid_holonomy"] for r in probe_results])
        late = np.mean([r["late_holonomy"] for r in probe_results])
        total_path = np.mean([r["total_path_distance"] for r in probe_results])

        print(f"  Wilson holonomy: {mean_hol:.4f} (cos={mean_cos:.4f})")
        print(f"  Profile: E={early:.4f} / M={mid:.4f} / L={late:.4f}")
        print(f"  Total path distance: {total_path:.4f}")

        # Check: is holonomy << path distance? (flat connection)
        # Or holonomy ~ path distance? (highly curved)
        ratio = mean_hol / total_path if total_path > 0 else 0
        print(f"  Holonomy/path ratio: {ratio:.4f} (0=flat, 1=maximally curved)")

        all_results["models"][model_label] = {
            "model": model_id,
            "n_layers": n_layers,
            "probes": probe_results,
            "mean_holonomy": mean_hol,
            "mean_cosine": mean_cos,
            "early_holonomy": early,
            "mid_holonomy": mid,
            "late_holonomy": late,
            "total_path_distance": total_path,
            "holonomy_path_ratio": ratio,
        }

        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    # Cross-architecture comparison
    print(f"\n{'='*60}")
    print(f"  WILSON LOOP COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Model':<12} {'Holonomy':>10} {'Cosine':>10} {'Path':>10} {'Hol/Path':>10} {'Early':>10} {'Mid':>10} {'Late':>10}")
    print(f"  {'-'*82}")
    for label in ["Mistral", "Gemma", "Llama", "Qwen"]:
        if label not in all_results["models"]:
            continue
        m = all_results["models"][label]
        print(f"  {label:<12} {m['mean_holonomy']:>10.4f} {m['mean_cosine']:>10.4f} {m['total_path_distance']:>10.4f} {m['holonomy_path_ratio']:>10.4f} {m['early_holonomy']:>10.4f} {m['mid_holonomy']:>10.4f} {m['late_holonomy']:>10.4f}")

    # Flat vs curved classification
    print(f"\n  --- Connection Flatness ---")
    for label in ["Mistral", "Gemma", "Llama", "Qwen"]:
        if label not in all_results["models"]:
            continue
        m = all_results["models"][label]
        if m["holonomy_path_ratio"] < 0.1:
            flatness = "FLAT (topological invariance)"
        elif m["holonomy_path_ratio"] < 0.3:
            flatness = "NEARLY FLAT (weak curvature)"
        elif m["holonomy_path_ratio"] < 0.5:
            flatness = "CURVED (active correction)"
        else:
            flatness = "HIGHLY CURVED (strong path-dependence)"
        print(f"    {label}: ratio={m['holonomy_path_ratio']:.4f} → {flatness}")

    outfile = RESULTS_DIR / f"e35c_wilson_loop_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved: {outfile}")


if __name__ == "__main__":
    main()
