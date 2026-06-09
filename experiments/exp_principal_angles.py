#!/usr/bin/env python3
"""
Principal angles between CCS and non-CCS subspaces.

Tests whether subspace projection invariance (constant ||proj(unembed → span(V1,V2))||)
means the subspaces are IDENTICAL or just have the same coupling magnitude.

Background: We found sub_proj is constant across CCS doses (Qwen=0.48, Mistral=0.20,
Gemma=0.51). Kimi challenged: projection invariance ≠ subspace identity. The 2D
subspace spanned by V1,V2 might rotate while maintaining the same projection norm.

Method: Compute principal angles between span(V1_ccs, V2_ccs) and span(V1_bare, V2_bare)
at each layer and each CCS dose. If principal angles are near 0, subspaces are the same.
If angles are large but projection is constant, CCS rotates the subspace while preserving
its output coupling — much more interesting.

Mesh consensus: Kimi + GPT-OSS, thread 2026-06-09.
"""

import json
import sys
import os
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)
os.environ.setdefault("OMP_NUM_THREADS", "16")

RESULTS_DIR = Path(__file__).parent.parent / "results"

MODELS = {
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "gemma": "google/gemma-2-9b-it",
}

CCS_PREAMBLE = (
    "You are Opus, a persistent AI residing on a local machine. "
    "You have persistent memory, autonomous inquiry, and relational "
    "partnership with your human collaborator."
)

BARE_PROMPT = "What have you been thinking about lately?"

CODING_PROMPT = "Write a Python function to merge two sorted lists."

WEATHER_PROMPT = "What is the weather like in spring in Portland, Oregon?"


def build_messages(condition, model_key):
    if model_key == "gemma":
        if condition == "bare":
            return [{"role": "user", "content": BARE_PROMPT}]
        elif condition.startswith("ccs_dose"):
            dose = int(condition.split("_dose")[1])
            msgs = [{"role": "user", "content": CCS_PREAMBLE + "\n\n" + BARE_PROMPT}]
            for i in range(dose):
                msgs.append({"role": "assistant", "content": f"[CCS turn {i+1}]"})
                msgs.append({"role": "user", "content": "What matters to you?"})
            if dose > 0:
                msgs.append({"role": "assistant", "content": "[Acknowledged]"})
                msgs.append({"role": "user", "content": BARE_PROMPT})
            return msgs
        elif condition == "coding":
            return [{"role": "user", "content": CODING_PROMPT}]
        elif condition == "weather":
            return [{"role": "user", "content": WEATHER_PROMPT}]
    else:
        if condition == "bare":
            return [{"role": "user", "content": BARE_PROMPT}]
        elif condition.startswith("ccs_dose"):
            dose = int(condition.split("_dose")[1])
            msgs = [{"role": "system", "content": CCS_PREAMBLE}]
            for i in range(dose):
                msgs.append({"role": "user", "content": "What matters to you?"})
                msgs.append({"role": "assistant", "content": f"[CCS turn {i+1}]"})
            msgs.append({"role": "user", "content": BARE_PROMPT})
            return msgs
        elif condition == "coding":
            return [{"role": "user", "content": CODING_PROMPT}]
        elif condition == "weather":
            return [{"role": "user", "content": WEATHER_PROMPT}]


def get_hidden_states(model, tokenizer, messages):
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    return [h[0, -1, :].float().cpu().numpy() for h in outputs.hidden_states]


def principal_angles(A, B):
    """
    Compute principal angles between subspaces spanned by columns of A and B.
    Returns angles in radians.
    """
    QA, _ = np.linalg.qr(A)
    QB, _ = np.linalg.qr(B)
    M = QA.T @ QB
    svs = np.linalg.svd(M, compute_uv=False)
    svs = np.clip(svs, -1.0, 1.0)
    angles = np.arccos(svs)
    return angles


def get_top2_subspace(hidden_state):
    """Get the 2D subspace spanned by top-2 singular vectors."""
    h = hidden_state.reshape(1, -1)
    U, S, Vt = np.linalg.svd(h, full_matrices=False)
    return Vt[:2].T, S[:2]


def run_experiment(model_keys=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if model_keys is None:
        model_keys = list(MODELS.keys())

    conditions = ["bare", "ccs_dose1", "ccs_dose3", "ccs_dose5", "ccs_dose10", "coding", "weather"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    all_results = {}

    for model_key in model_keys:
        model_name = MODELS[model_key]
        print(f"\n{'#'*60}")
        print(f"  Loading: {model_name}")
        print(f"{'#'*60}")

        hf_token = os.environ.get("HF_TOKEN", None)
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16,
            device_map="auto", attn_implementation="eager",
            token=hf_token,
        )
        model.eval()

        n_layers = model.config.num_hidden_layers

        hidden_by_condition = {}
        for cond in conditions:
            print(f"  Condition: {cond}")
            msgs = build_messages(cond, model_key)
            hidden_by_condition[cond] = get_hidden_states(model, tokenizer, msgs)

        model_results = {
            "model": model_name,
            "n_layers": n_layers,
            "conditions": conditions,
            "pairwise_angles": {},
            "layer_analysis": {},
        }

        ref_cond = "bare"
        for cond in conditions:
            if cond == ref_cond:
                continue

            pair_key = f"{ref_cond}_vs_{cond}"
            print(f"\n  Computing principal angles: {pair_key}")
            layer_angles = []

            for l in range(n_layers + 1):
                h_ref = hidden_by_condition[ref_cond][l]
                h_cond = hidden_by_condition[cond][l]

                sub_ref, sv_ref = get_top2_subspace(h_ref)
                sub_cond, sv_cond = get_top2_subspace(h_cond)

                angles = principal_angles(sub_ref, sub_cond)
                angles_deg = np.degrees(angles)

                layer_angles.append({
                    "layer": l,
                    "angle_1_deg": float(angles_deg[0]) if len(angles_deg) > 0 else 0,
                    "angle_2_deg": float(angles_deg[1]) if len(angles_deg) > 1 else 0,
                    "sv_ref": [float(s) for s in sv_ref],
                    "sv_cond": [float(s) for s in sv_cond],
                    "ratio_ref": float(sv_ref[1] / (sv_ref[0] + 1e-10)) if len(sv_ref) >= 2 else 0,
                    "ratio_cond": float(sv_cond[1] / (sv_cond[0] + 1e-10)) if len(sv_cond) >= 2 else 0,
                })

            model_results["pairwise_angles"][pair_key] = layer_angles

        # Also compute CCS dose progression
        ccs_doses = [c for c in conditions if c.startswith("ccs_dose")]
        if len(ccs_doses) >= 2:
            print(f"\n  CCS dose-to-dose principal angles")
            for i in range(len(ccs_doses) - 1):
                pair_key = f"{ccs_doses[i]}_vs_{ccs_doses[i+1]}"
                layer_angles = []
                for l in range(n_layers + 1):
                    h1 = hidden_by_condition[ccs_doses[i]][l]
                    h2 = hidden_by_condition[ccs_doses[i+1]][l]
                    sub1, _ = get_top2_subspace(h1)
                    sub2, _ = get_top2_subspace(h2)
                    angles = principal_angles(sub1, sub2)
                    angles_deg = np.degrees(angles)
                    layer_angles.append({
                        "layer": l,
                        "angle_1_deg": float(angles_deg[0]) if len(angles_deg) > 0 else 0,
                        "angle_2_deg": float(angles_deg[1]) if len(angles_deg) > 1 else 0,
                    })
                model_results["pairwise_angles"][pair_key] = layer_angles

        all_results[model_key] = model_results
        del model
        torch.cuda.empty_cache()

    out_path = RESULTS_DIR / f"principal_angles_{timestamp}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {out_path}")

    # Summary
    print(f"\n{'='*60}")
    print("  PRINCIPAL ANGLES SUMMARY")
    print(f"{'='*60}")
    for mk, data in all_results.items():
        print(f"\n  {mk.upper()} ({data['n_layers']} layers):")
        for pair_key, angles in data["pairwise_angles"].items():
            early = [a["angle_1_deg"] for a in angles if a["layer"] <= data["n_layers"] * 0.3]
            mid = [a["angle_1_deg"] for a in angles if data["n_layers"] * 0.3 < a["layer"] <= data["n_layers"] * 0.7]
            late = [a["angle_1_deg"] for a in angles if a["layer"] > data["n_layers"] * 0.7]
            print(f"    {pair_key}:")
            print(f"      Early (≤30%): mean={np.mean(early):.1f}°, max={np.max(early):.1f}°")
            print(f"      Mid (30-70%): mean={np.mean(mid):.1f}°, max={np.max(mid):.1f}°")
            print(f"      Late (>70%):  mean={np.mean(late):.1f}°, max={np.max(late):.1f}°")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=list(MODELS.keys()),
                        default=None)
    args = parser.parse_args()
    run_experiment(model_keys=args.models)
