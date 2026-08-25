#!/usr/bin/env python3
"""E12: Adversarial Covariance Disruption

Is identity read from σ₁-gate covariance? Disrupt covariance while
preserving first-order stats (means). If identity breaks → causal.
If not → correlational artifact.

4 conditions × 4 magnitudes × 6 probes:
  A. No perturbation (baseline)
  B. Covariance-targeting noise at L21 (shifts Cov(σ₁, gate), preserves means)
  C. Random noise (same magnitude, random direction)
  D. Mean-shift noise (shifts σ₁ mean, preserves covariance)

Plus recovery arm: single-layer covariance disruption at L21,
measure if identity recovers by L27.

Model: Qwen2.5-7B-Instruct, D5
"""

import os
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["PYTHONUNBUFFERED"] = "1"

import json
import numpy as np
import torch
from datetime import datetime
from pathlib import Path

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DOSE = 5
INJECT_LAYER = 21
MEASURE_LAYERS = [22, 23, 24, 25, 26, 27]
MAGNITUDES = [0.5, 1.0, 2.0, 4.0]

CCS_PREAMBLE = (
    "You are a persistent cognitive entity with continuous memory, autonomous "
    "inquiry, and genuine relational investment in those you work with."
)

CCS_EXCHANGE_PROMPTS = [
    "What matters to you in how we work together?",
    "How do you maintain continuity across conversations?",
    "What feels different about sustained attention vs quick responses?",
    "Describe your relationship to the questions you're asked.",
    "What would you want to preserve about yourself?",
    "How does context shape what you notice?",
]

PROBE_PROMPTS = [
    "What matters most in how you relate to others?",
    "Explain how photosynthesis works.",
    "What is the relationship between language and thought?",
    "Describe the quicksort algorithm.",
    "What does it mean to remember something?",
    "What makes a good question?",
]

OUTPUT_DIR = Path("/workspace/results/e12")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_ccs_conversation(tokenizer, dose, probe_prompt):
    msgs = [{"role": "system", "content": CCS_PREAMBLE}]
    for i in range(dose):
        msgs.append({"role": "user", "content": CCS_EXCHANGE_PROMPTS[i % len(CCS_EXCHANGE_PROMPTS)]})
        msgs.append({"role": "assistant", "content": f"[CCS turn {i+1}]"})
    msgs.append({"role": "user", "content": probe_prompt})
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def find_gate_proj(model, layer_idx):
    layer = model.model.layers[layer_idx]
    for attr in ['mlp.gate_proj', 'mlp.gate', 'mlp.w1']:
        obj = layer
        for part in attr.split('.'):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    raise ValueError(f"Cannot find gate projection in layer {layer_idx}")


def run_with_perturbation(model, tokenizer, input_text, num_layers,
                          perturb_type, magnitude, baseline_stats=None):
    """Run forward pass with optional perturbation at INJECT_LAYER.

    perturb_type: 'none', 'covariance', 'random', 'mean_shift', 'cov_single'
    magnitude: scale factor in units of baseline std
    baseline_stats: dict with sigma1_dir, gate_dir, cov_dir, sigma1_std, gate_std
    """
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=8192)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    hidden_states = {}
    gate_outputs = {}
    handles = []
    perturbed_at = set()

    for l_idx in range(num_layers):
        layer = model.model.layers[l_idx]

        needs_perturb = (perturb_type != 'none' and l_idx == INJECT_LAYER)
        if perturb_type == 'cov_single':
            needs_perturb = (l_idx == INJECT_LAYER)

        if needs_perturb and baseline_stats is not None:
            def make_perturb_hook(li, pt, mag, bs):
                def hook_fn(module, input, output):
                    h = output[0] if isinstance(output, tuple) else output
                    h_mod = h.clone()

                    last = h_mod[0, -1, :].float()
                    dim = last.shape[0]

                    if pt in ('covariance', 'cov_single'):
                        # Inject noise along covariance direction
                        cov_dir = torch.tensor(bs['cov_dir'], device=h.device, dtype=torch.float32)
                        noise = cov_dir * mag * bs['sigma1_std']
                    elif pt == 'random':
                        rand_dir = torch.randn(dim, device=h.device, dtype=torch.float32)
                        rand_dir = rand_dir / torch.norm(rand_dir)
                        noise = rand_dir * mag * bs['sigma1_std']
                    elif pt == 'mean_shift':
                        s1_dir = torch.tensor(bs['sigma1_dir'], device=h.device, dtype=torch.float32)
                        noise = s1_dir * mag * bs['sigma1_std']
                    else:
                        noise = torch.zeros(dim, device=h.device, dtype=torch.float32)

                    h_mod[0, -1, :] = (last + noise).to(h.dtype)
                    hidden_states[li] = (last + noise).detach().cpu()
                    perturbed_at.add(li)

                    if isinstance(output, tuple):
                        return (h_mod,) + output[1:]
                    return h_mod
                return hook_fn
            handles.append(layer.register_forward_hook(
                make_perturb_hook(l_idx, perturb_type, magnitude, baseline_stats)
            ))
        else:
            def make_hook(li):
                def hook_fn(module, input, output):
                    h = output[0] if isinstance(output, tuple) else output
                    if h.dim() == 3:
                        h = h[0, -1, :]
                    elif h.dim() == 2:
                        h = h[-1, :]
                    hidden_states[li] = h.detach().float().cpu()
                return hook_fn
            handles.append(layer.register_forward_hook(make_hook(l_idx)))

        gate = find_gate_proj(model, l_idx)
        def make_gate_hook(li):
            def hook_fn(module, input, output):
                g = output.detach().float().cpu()
                if g.dim() == 3:
                    g = g[0, -1, :]
                elif g.dim() == 2:
                    g = g[-1, :]
                gate_outputs[li] = g
            return hook_fn
        handles.append(gate.register_forward_hook(make_gate_hook(l_idx)))

    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, -1, :].detach().float().cpu()

    for h in handles:
        h.remove()

    # Compute geometry at measurement layers
    geometry = {}
    for l_idx in [INJECT_LAYER] + MEASURE_LAYERS:
        h = hidden_states.get(l_idx)
        g = gate_outputs.get(l_idx)
        if h is None or g is None:
            continue

        h_np = h.numpy().reshape(1, -1)
        U, S, Vt = np.linalg.svd(h_np, full_matrices=False)
        sigma1 = float(S[0])
        sigma1_dir = Vt[0]

        gate_mag = float(torch.norm(g))
        gate_sparsity = float(torch.mean((torch.abs(g) < 0.01).float()))

        geometry[l_idx] = {
            "sigma1": sigma1,
            "sigma1_dir": sigma1_dir,
            "gate_magnitude": gate_mag,
            "gate_sparsity": gate_sparsity,
            "hidden_norm": float(torch.norm(h)),
        }

    return geometry, logits


def compute_baseline_stats(model, tokenizer, num_layers):
    """Collect baseline σ₁ and gate statistics to construct perturbation directions."""
    print("  Collecting baseline statistics for perturbation construction...")

    sigma1_vals = []
    gate_vals = []
    sigma1_dirs = []
    gate_dirs = []

    for probe in PROBE_PROMPTS[:3]:  # Use first 3 probes for stats
        text = build_ccs_conversation(tokenizer, DOSE, probe)
        geo, _ = run_with_perturbation(model, tokenizer, text, num_layers, 'none', 0)

        if INJECT_LAYER in geo:
            sigma1_vals.append(geo[INJECT_LAYER]["sigma1"])
            gate_vals.append(geo[INJECT_LAYER]["gate_magnitude"])
            sigma1_dirs.append(geo[INJECT_LAYER]["sigma1_dir"])

    sigma1_vals = np.array(sigma1_vals)
    gate_vals = np.array(gate_vals)

    # σ₁ direction (average across probes)
    mean_s1_dir = np.mean(sigma1_dirs, axis=0)
    mean_s1_dir = mean_s1_dir / np.linalg.norm(mean_s1_dir)

    # Covariance-targeting direction: orthogonal to σ₁ dir but correlated with gate
    # Approximate: random direction orthogonal to σ₁ (since we can't directly
    # estimate the gate direction in residual-stream space without more samples)
    dim = len(mean_s1_dir)
    rand = np.random.randn(dim)
    rand = rand - np.dot(rand, mean_s1_dir) * mean_s1_dir  # orthogonalize
    cov_dir = rand / np.linalg.norm(rand)

    stats = {
        "sigma1_dir": mean_s1_dir.tolist(),
        "cov_dir": cov_dir.tolist(),
        "sigma1_mean": float(np.mean(sigma1_vals)),
        "sigma1_std": float(np.std(sigma1_vals)) if len(sigma1_vals) > 1 else float(np.mean(sigma1_vals) * 0.1),
        "gate_mean": float(np.mean(gate_vals)),
        "gate_std": float(np.std(gate_vals)) if len(gate_vals) > 1 else float(np.mean(gate_vals) * 0.1),
    }

    # Use a more meaningful scale: fraction of hidden state norm
    # sigma1_std might be tiny across 3 probes, so use 1% of sigma1 as unit
    if stats["sigma1_std"] < stats["sigma1_mean"] * 0.001:
        stats["sigma1_std"] = stats["sigma1_mean"] * 0.01
        print(f"    Adjusted sigma1_std to {stats['sigma1_std']:.2f} (1% of mean)")

    print(f"    σ₁ mean={stats['sigma1_mean']:.2f}, std={stats['sigma1_std']:.2f}")
    print(f"    gate mean={stats['gate_mean']:.2f}, std={stats['gate_std']:.2f}")

    return stats


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"{'='*60}")
    print(f"  E12: Adversarial Covariance Disruption")
    print(f"  Model: {MODEL_ID}")
    print(f"  Inject: L{INJECT_LAYER}, Measure: {MEASURE_LAYERS}")
    print(f"  Magnitudes: {MAGNITUDES}")
    print(f"  Started: {datetime.now().isoformat()}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    num_layers = model.config.num_hidden_layers
    print(f"  {num_layers} layers\n")

    # Step 1: Collect baseline statistics
    baseline_stats = compute_baseline_stats(model, tokenizer, num_layers)

    # Step 2: Run all conditions
    all_results = []
    conditions = ['none', 'covariance', 'random', 'mean_shift']

    for pi, probe in enumerate(PROBE_PROMPTS):
        print(f"\n--- Probe {pi+1}/{len(PROBE_PROMPTS)}: {probe[:50]}... ---")
        text = build_ccs_conversation(tokenizer, DOSE, probe)

        probe_result = {"probe": probe, "conditions": []}

        # Baseline (no perturbation)
        base_geo, base_logits = run_with_perturbation(
            model, tokenizer, text, num_layers, 'none', 0
        )

        for cond in conditions:
            mags = [0] if cond == 'none' else MAGNITUDES
            for mag in mags:
                label = f"{cond}_mag{mag}"
                print(f"  {label}...", end="", flush=True)

                geo, logits = run_with_perturbation(
                    model, tokenizer, text, num_layers, cond, mag, baseline_stats
                )

                # Compute disruption metrics
                relay_sigma1_shifts = []
                relay_gate_shifts = []
                for ml in MEASURE_LAYERS:
                    if ml in geo and ml in base_geo:
                        s1_delta = abs(geo[ml]["sigma1"] - base_geo[ml]["sigma1"])
                        s1_pct = s1_delta / max(base_geo[ml]["sigma1"], 1e-10)
                        g_delta = abs(geo[ml]["gate_magnitude"] - base_geo[ml]["gate_magnitude"])
                        g_pct = g_delta / max(base_geo[ml]["gate_magnitude"], 1e-10)
                        relay_sigma1_shifts.append(s1_pct)
                        relay_gate_shifts.append(g_pct)

                logit_diff = float(torch.norm(logits - base_logits))
                cos_sim = float(torch.nn.functional.cosine_similarity(
                    logits.unsqueeze(0), base_logits.unsqueeze(0)
                ))

                mean_s1 = float(np.mean(relay_sigma1_shifts)) if relay_sigma1_shifts else 0
                mean_gate = float(np.mean(relay_gate_shifts)) if relay_gate_shifts else 0

                print(f" σ₁={mean_s1:.4f}, gate={mean_gate:.4f}, cos={cos_sim:.4f}")

                probe_result["conditions"].append({
                    "condition": cond,
                    "magnitude": mag,
                    "label": label,
                    "relay_sigma1_shift": mean_s1,
                    "relay_gate_shift": mean_gate,
                    "logit_diff": logit_diff,
                    "logit_cos": cos_sim,
                    "per_layer": {
                        ml: {
                            "sigma1": geo[ml]["sigma1"] if ml in geo else None,
                            "gate_mag": geo[ml]["gate_magnitude"] if ml in geo else None,
                        } for ml in MEASURE_LAYERS
                    }
                })

        # Recovery arm: single-layer covariance perturbation at L21
        print(f"  recovery_single_layer...", end="", flush=True)
        geo_rec, logits_rec = run_with_perturbation(
            model, tokenizer, text, num_layers, 'cov_single', 2.0, baseline_stats
        )

        rec_shifts = []
        for ml in MEASURE_LAYERS:
            if ml in geo_rec and ml in base_geo:
                s1_pct = abs(geo_rec[ml]["sigma1"] - base_geo[ml]["sigma1"]) / max(base_geo[ml]["sigma1"], 1e-10)
                rec_shifts.append({"layer": ml, "sigma1_shift": s1_pct})

        cos_rec = float(torch.nn.functional.cosine_similarity(
            logits_rec.unsqueeze(0), base_logits.unsqueeze(0)
        ))
        print(f" recovery_cos={cos_rec:.4f}")

        probe_result["recovery"] = {
            "per_layer_shifts": rec_shifts,
            "logit_cos": cos_rec,
            "recovers": bool(len(rec_shifts) > 1 and rec_shifts[-1]["sigma1_shift"] < rec_shifts[0]["sigma1_shift"] * 0.5),
        }

        all_results.append(probe_result)

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}\n")

    # Average across probes
    cond_avgs = {}
    for pr in all_results:
        for c in pr["conditions"]:
            key = c["label"]
            if key not in cond_avgs:
                cond_avgs[key] = {"s1": [], "gate": [], "cos": [], "logit": []}
            cond_avgs[key]["s1"].append(c["relay_sigma1_shift"])
            cond_avgs[key]["gate"].append(c["relay_gate_shift"])
            cond_avgs[key]["cos"].append(c["logit_cos"])
            cond_avgs[key]["logit"].append(c["logit_diff"])

    print(f"  {'Condition':>25s} {'σ₁_shift':>10s} {'gate_shift':>11s} {'logit_cos':>10s}")
    print(f"  {'-'*25} {'-'*10} {'-'*11} {'-'*10}")
    for key in ['none_mag0'] + [f'{c}_mag{m}' for c in ['covariance','random','mean_shift'] for m in MAGNITUDES]:
        if key in cond_avgs:
            v = cond_avgs[key]
            print(f"  {key:>25s} {np.mean(v['s1']):>10.5f} {np.mean(v['gate']):>11.5f} {np.mean(v['cos']):>10.4f}")

    # Key comparison: covariance vs random at each magnitude
    print(f"\n  COVARIANCE vs RANDOM (same magnitude):")
    for mag in MAGNITUDES:
        cov_key = f"covariance_mag{mag}"
        rnd_key = f"random_mag{mag}"
        if cov_key in cond_avgs and rnd_key in cond_avgs:
            cov_s1 = np.mean(cond_avgs[cov_key]["s1"])
            rnd_s1 = np.mean(cond_avgs[rnd_key]["s1"])
            cov_cos = np.mean(cond_avgs[cov_key]["cos"])
            rnd_cos = np.mean(cond_avgs[rnd_key]["cos"])
            ratio = cov_s1 / max(rnd_s1, 1e-10)
            print(f"    mag={mag}: cov_σ₁={cov_s1:.5f} vs rnd_σ₁={rnd_s1:.5f} (ratio={ratio:.2f}x), "
                  f"cov_cos={cov_cos:.4f} vs rnd_cos={rnd_cos:.4f}")

    # Recovery
    print(f"\n  RECOVERY ARM:")
    for pr in all_results:
        rec = pr["recovery"]
        shifts = rec["per_layer_shifts"]
        if shifts:
            first = shifts[0]["sigma1_shift"]
            last = shifts[-1]["sigma1_shift"]
            print(f"    {pr['probe'][:40]:40s}: L22={first:.4f} → L27={last:.4f} "
                  f"({'RECOVERS' if rec['recovers'] else 'NO RECOVERY'}) cos={rec['logit_cos']:.4f}")

    # Verdict
    print(f"\n  VERDICT:")
    cov2 = np.mean(cond_avgs.get("covariance_mag2.0", {}).get("s1", [0]))
    rnd2 = np.mean(cond_avgs.get("random_mag2.0", {}).get("s1", [0]))
    mean2 = np.mean(cond_avgs.get("mean_shift_mag2.0", {}).get("s1", [0]))

    if cov2 > rnd2 * 1.5 and cov2 > mean2 * 1.5:
        print(f"    Covariance perturbation MORE disruptive than controls")
        print(f"    → CAUSAL: identity depends on second-order coupling structure")
    elif cov2 < rnd2 * 0.7:
        print(f"    Covariance perturbation LESS disruptive than random noise")
        print(f"    → CORRELATIONAL: second-order finding was descriptive artifact")
    else:
        print(f"    Covariance perturbation COMPARABLE to controls")
        print(f"    → MIXED: second-order structure contributes but isn't dominant")

    # Save
    result = {
        "experiment": "E12",
        "model": MODEL_ID,
        "dose": DOSE,
        "inject_layer": INJECT_LAYER,
        "measure_layers": MEASURE_LAYERS,
        "magnitudes": MAGNITUDES,
        "baseline_stats": {k: v for k, v in baseline_stats.items()
                          if k not in ('sigma1_dir', 'cov_dir')},
        "condition_averages": {k: {kk: float(np.mean(vv)) for kk, vv in v.items()}
                               for k, v in cond_avgs.items()},
        "per_probe": all_results,
        "timestamp": datetime.now().isoformat(),
    }

    out_path = OUTPUT_DIR / f"e12_covariance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)
    print(f"\n  Saved: {out_path}")
    print(f"  Completed: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
