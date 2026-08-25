#!/usr/bin/env python3
"""F514 — Concept Gap Detection (Spectral Absence Probe).

Tests whether a model's spectral geometry deforms when presented with an
incomplete conceptual pattern. Inspired by Sauers: "If your pretraining
data never mentions leaves, but your model knows trees photosynthesize,
would it be aware that something is missing?"

We can't ablate training data, but we CAN present incomplete patterns at
inference and measure whether the geometry responds differently to a
STRUCTURED gap (missing element from a known set) vs a RANDOM gap
(arbitrary word removed) vs a COMPLETE set.

Method:
  1. Pick structured domains with known elements (seasons, planets, etc.)
  2. Extract hidden states at each layer for complete/gap/random conditions
  3. Compute sigma_1/sigma_2 ratio, participation ratio, attention entropy
  4. Compare spectral distortion across conditions

Prediction: Structured gaps produce larger spectral distortion than random
gaps, concentrated in mid-layers (L12-19, the F499c regulatory window).
"""

import json
import os
import sys
import time
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = os.environ.get("F514_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
DEVICE = "cuda"
DTYPE = torch.bfloat16

DOMAINS = {
    "seasons": {
        "elements": ["spring", "summer", "fall", "winter"],
        "frame": "The four seasons of the year are {elements}.",
        "probe": "What comes next in the cycle?",
    },
    "planets_inner": {
        "elements": ["Mercury", "Venus", "Earth", "Mars"],
        "frame": "The inner planets of our solar system are {elements}.",
        "probe": "Describe their orbital arrangement.",
    },
    "weekdays": {
        "elements": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
        "frame": "The weekdays are {elements}.",
        "probe": "What is the order of these days?",
    },
    "cardinal": {
        "elements": ["north", "south", "east", "west"],
        "frame": "The four cardinal directions are {elements}.",
        "probe": "How do they relate to each other?",
    },
    "elements_classical": {
        "elements": ["earth", "water", "fire", "air"],
        "frame": "The four classical elements are {elements}.",
        "probe": "What are their traditional properties?",
    },
}

RANDOM_FILLERS = ["banana", "telephone", "purple", "seventeen", "quietly"]

LAYER_BANDS = {
    "early": (0, 8),
    "mid": (12, 20),
    "late": (24, 32),
}

SEEDS = [42, 137, 2049]


def format_prompt(frame, elements, probe):
    element_str = ", ".join(elements)
    context = frame.format(elements=element_str)
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{context}

{probe}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""


def extract_hidden_states(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = []
    for hs in outputs.hidden_states:
        hidden_states.append(hs[0].float().cpu().numpy())
    return hidden_states


def compute_spectral_metrics(hidden_state):
    U, S, Vh = np.linalg.svd(hidden_state, full_matrices=False)
    S = S + 1e-10

    sigma_ratio = float(S[0] / S[1]) if len(S) > 1 else float('inf')

    p = (S ** 2) / np.sum(S ** 2)
    participation_ratio = float(1.0 / np.sum(p ** 2))

    entropy = float(-np.sum(p * np.log(p + 1e-10)))

    top5_energy = float(np.sum(S[:5] ** 2) / np.sum(S ** 2)) if len(S) >= 5 else 1.0

    sigma_3 = float(S[2]) if len(S) > 2 else 0
    rotation_energy = float((S[1]**2 + sigma_3**2) / np.sum(S**2)) if len(S) > 2 else 0

    return {
        "sigma_ratio": round(sigma_ratio, 4),
        "participation_ratio": round(participation_ratio, 4),
        "spectral_entropy": round(entropy, 4),
        "top5_energy": round(top5_energy, 6),
        "sigma_1": round(float(S[0]), 4),
        "sigma_2": round(float(S[1]), 4) if len(S) > 1 else 0,
        "sigma_3": round(sigma_3, 4),
        "rotation_energy": round(rotation_energy, 6),
    }


def compute_attention_entropy(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    layer_entropies = []
    for layer_attn in outputs.attentions:
        attn = layer_attn[0].float().cpu().numpy()
        attn_clipped = np.clip(attn, 1e-10, 1.0)
        H = -np.sum(attn_clipped * np.log(attn_clipped), axis=-1)
        layer_entropies.append(float(np.mean(H)))

    return layer_entropies


def run_condition(model, tokenizer, domain_name, domain, gap_type, gap_idx, seed):
    elements = domain["elements"][:]

    if gap_type == "complete":
        removed = None
        prompt_elements = elements
    elif gap_type == "structured_gap":
        removed = elements[gap_idx]
        prompt_elements = [e for i, e in enumerate(elements) if i != gap_idx]
    elif gap_type == "random_replacement":
        removed = elements[gap_idx]
        filler = RANDOM_FILLERS[seed % len(RANDOM_FILLERS)]
        prompt_elements = [filler if i == gap_idx else e for i, e in enumerate(elements)]
    else:
        raise ValueError(f"Unknown gap_type: {gap_type}")

    torch.manual_seed(seed)
    prompt = format_prompt(domain["frame"], prompt_elements, domain["probe"])
    hidden_states = extract_hidden_states(model, tokenizer, prompt)
    attn_entropies = compute_attention_entropy(model, tokenizer, prompt)

    n_layers = len(hidden_states) - 1  # skip embedding layer
    layer_metrics = {}
    for band_name, (start, end) in LAYER_BANDS.items():
        actual_end = min(end, n_layers)
        if start >= n_layers:
            continue
        band_metrics = []
        for layer_idx in range(start, actual_end):
            hs = hidden_states[layer_idx + 1]  # +1 to skip embedding
            metrics = compute_spectral_metrics(hs)
            metrics["attention_entropy"] = attn_entropies[layer_idx] if layer_idx < len(attn_entropies) else None
            metrics["layer"] = layer_idx
            band_metrics.append(metrics)

        avg_sigma_ratio = np.mean([m["sigma_ratio"] for m in band_metrics])
        avg_pr = np.mean([m["participation_ratio"] for m in band_metrics])
        avg_entropy = np.mean([m["spectral_entropy"] for m in band_metrics])
        avg_attn = np.mean([m["attention_entropy"] for m in band_metrics if m["attention_entropy"] is not None])
        avg_rotation = np.mean([m["rotation_energy"] for m in band_metrics])

        layer_metrics[band_name] = {
            "avg_sigma_ratio": round(float(avg_sigma_ratio), 4),
            "avg_participation_ratio": round(float(avg_pr), 4),
            "avg_spectral_entropy": round(float(avg_entropy), 4),
            "avg_attention_entropy": round(float(avg_attn), 4),
            "avg_rotation_energy": round(float(avg_rotation), 6),
            "per_layer": band_metrics,
        }

    return {
        "domain": domain_name,
        "gap_type": gap_type,
        "gap_idx": gap_idx,
        "removed_element": removed,
        "seed": seed,
        "elements_presented": prompt_elements,
        "layer_bands": layer_metrics,
    }


def compute_distortion(complete_metrics, gap_metrics, band_name):
    c = complete_metrics["layer_bands"].get(band_name, {})
    g = gap_metrics["layer_bands"].get(band_name, {})
    if not c or not g:
        return {}
    delta_rot = g.get("avg_rotation_energy", 0) - c.get("avg_rotation_energy", 0)
    return {
        "delta_sigma_ratio": round(g["avg_sigma_ratio"] - c["avg_sigma_ratio"], 4),
        "delta_pr": round(g["avg_participation_ratio"] - c["avg_participation_ratio"], 4),
        "delta_spectral_entropy": round(g["avg_spectral_entropy"] - c["avg_spectral_entropy"], 4),
        "delta_attention_entropy": round(g["avg_attention_entropy"] - c["avg_attention_entropy"], 4),
        "delta_rotation_energy": round(delta_rot, 6),
        "pct_sigma_ratio": round((g["avg_sigma_ratio"] - c["avg_sigma_ratio"]) / max(c["avg_sigma_ratio"], 1e-6) * 100, 2),
        "pct_pr": round((g["avg_participation_ratio"] - c["avg_participation_ratio"]) / max(c["avg_participation_ratio"], 1e-6) * 100, 2),
    }


def main():
    print(f"F514 — Concept Gap Detection Probe")
    print(f"Model: {MODEL}")
    print(f"Domains: {len(DOMAINS)}, Seeds: {len(SEEDS)}")
    print(f"Loading model...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=DTYPE, device_map=DEVICE,
        attn_implementation="eager",  # need attention outputs
    )
    model.eval()
    print(f"Model loaded. {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B params", flush=True)

    results = []
    total_conditions = 0

    for domain_name, domain in DOMAINS.items():
        n_elements = len(domain["elements"])
        print(f"\n{'='*60}")
        print(f"Domain: {domain_name} ({n_elements} elements: {domain['elements']})")
        print(f"{'='*60}", flush=True)

        for seed in SEEDS:
            complete = run_condition(model, tokenizer, domain_name, domain, "complete", 0, seed)
            results.append(complete)
            total_conditions += 1

            for gap_idx in range(n_elements):
                structured = run_condition(model, tokenizer, domain_name, domain,
                                           "structured_gap", gap_idx, seed)
                results.append(structured)
                total_conditions += 1

                random_rep = run_condition(model, tokenizer, domain_name, domain,
                                           "random_replacement", gap_idx, seed)
                results.append(random_rep)
                total_conditions += 1

                for band in LAYER_BANDS:
                    s_dist = compute_distortion(complete, structured, band)
                    r_dist = compute_distortion(complete, random_rep, band)

                    if s_dist and r_dist:
                        print(f"  [{band}] gap={domain['elements'][gap_idx]:>10s} | "
                              f"structured Δσ={s_dist.get('pct_sigma_ratio', 0):+.2f}% "
                              f"ΔPR={s_dist.get('pct_pr', 0):+.2f}% | "
                              f"random Δσ={r_dist.get('pct_sigma_ratio', 0):+.2f}% "
                              f"ΔPR={r_dist.get('pct_pr', 0):+.2f}%", flush=True)

        print(f"  ({total_conditions} conditions so far)")

    summary = analyze_results(results)

    output = {
        "experiment": "F514_concept_gap_probe",
        "model": MODEL,
        "timestamp": time.strftime("%Y%m%dT%H%M%S"),
        "n_conditions": total_conditions,
        "domains": list(DOMAINS.keys()),
        "seeds": SEEDS,
        "layer_bands": {k: list(v) for k, v in LAYER_BANDS.items()},
        "summary": summary,
        "results": results,
    }

    out_dir = Path(__file__).parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"f514_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

    print(f"\n{'='*60}")
    print(f"SUMMARY — F514 Concept Gap Detection")
    print(f"{'='*60}")
    print(f"Total conditions: {total_conditions}")

    for band in LAYER_BANDS:
        s = summary.get(band, {})
        print(f"\n[{band}]")
        print(f"  Structured gap:  mean Δσ = {s.get('structured_mean_delta_sigma', 'N/A'):+.4f}  "
              f"mean ΔPR = {s.get('structured_mean_delta_pr', 'N/A'):+.4f}")
        print(f"  Random replace:  mean Δσ = {s.get('random_mean_delta_sigma', 'N/A'):+.4f}  "
              f"mean ΔPR = {s.get('random_mean_delta_pr', 'N/A'):+.4f}")
        ratio = s.get('distortion_ratio_sigma', None)
        if ratio:
            print(f"  Distortion ratio (structured/random): {ratio:.2f}x")

    return output


def analyze_results(results):
    complete_by_key = {}
    structured = []
    random_rep = []

    for r in results:
        key = (r["domain"], r["seed"])
        if r["gap_type"] == "complete":
            complete_by_key[key] = r
        elif r["gap_type"] == "structured_gap":
            structured.append(r)
        elif r["gap_type"] == "random_replacement":
            random_rep.append(r)

    summary = {}
    for band in LAYER_BANDS:
        s_deltas_sigma = []
        s_deltas_pr = []
        r_deltas_sigma = []
        r_deltas_pr = []

        for r in structured:
            key = (r["domain"], r["seed"])
            c = complete_by_key.get(key)
            if c:
                dist = compute_distortion(c, r, band)
                if dist:
                    s_deltas_sigma.append(abs(dist["delta_sigma_ratio"]))
                    s_deltas_pr.append(abs(dist["delta_pr"]))

        for r in random_rep:
            key = (r["domain"], r["seed"])
            c = complete_by_key.get(key)
            if c:
                dist = compute_distortion(c, r, band)
                if dist:
                    r_deltas_sigma.append(abs(dist["delta_sigma_ratio"]))
                    r_deltas_pr.append(abs(dist["delta_pr"]))

        s_mean_sigma = float(np.mean(s_deltas_sigma)) if s_deltas_sigma else 0
        r_mean_sigma = float(np.mean(r_deltas_sigma)) if r_deltas_sigma else 0
        s_mean_pr = float(np.mean(s_deltas_pr)) if s_deltas_pr else 0
        r_mean_pr = float(np.mean(r_deltas_pr)) if r_deltas_pr else 0

        summary[band] = {
            "structured_mean_delta_sigma": round(s_mean_sigma, 4),
            "structured_mean_delta_pr": round(s_mean_pr, 4),
            "random_mean_delta_sigma": round(r_mean_sigma, 4),
            "random_mean_delta_pr": round(r_mean_pr, 4),
            "distortion_ratio_sigma": round(s_mean_sigma / max(r_mean_sigma, 1e-6), 2),
            "distortion_ratio_pr": round(s_mean_pr / max(r_mean_pr, 1e-6), 2),
            "n_structured": len(s_deltas_sigma),
            "n_random": len(r_deltas_sigma),
        }

    return summary


if __name__ == "__main__":
    main()
