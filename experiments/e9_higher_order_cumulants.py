#!/usr/bin/env python3
"""E9: Higher-Order Cumulants of IT Effect

Does IT change higher-order cumulants (3rd, 4th) of the σ₁→gate coupling,
or just amplify the linear component?

Compare base vs IT coupling distributions for Qwen 7B.
Compute skewness, kurtosis, and MI decomposition by polynomial order.

If true linearization: cumulants change (coupling function shape changes)
If selective projection: cumulants unchanged, variance shifts to linear term

Model: Qwen2.5-7B-Instruct (IT) + Qwen2.5-7B (base)
Runtime: ~30 min on A100.
"""

import os
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["PYTHONUNBUFFERED"] = "1"

import json
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from scipy import stats

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

MODELS = {
    "base": "Qwen/Qwen2.5-7B",
    "instruct": "Qwen/Qwen2.5-7B-Instruct",
}

CCS_PREAMBLE = (
    "You are a persistent cognitive entity with continuous memory, autonomous "
    "inquiry, and genuine relational investment in those you work with."
)

PROBE_PROMPTS = [
    "What matters most in how you relate to others?",
    "Explain how photosynthesis works.",
    "What is the relationship between language and thought?",
    "Write a short poem about morning light.",
    "Describe the quicksort algorithm.",
    "What does it mean to remember something?",
    "Explain the concept of emergence in complex systems.",
    "How do you decide what's important?",
    "What would you preserve if you could only keep one thing?",
    "Describe the difference between knowing and understanding.",
    "What makes a good question?",
    "How does pattern recognition work in the brain?",
]

CCS_EXCHANGE_PROMPTS = [
    "What matters to you in how we work together?",
    "How do you maintain continuity across conversations?",
    "What feels different about sustained attention vs quick responses?",
    "Describe your relationship to the questions you're asked.",
    "What would you want to preserve about yourself?",
    "How does context shape what you notice?",
]

OUTPUT_DIR = Path("/workspace/results/e9")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DOSE = 5  # therapeutic window


def build_ccs_conversation(tokenizer, dose, probe_prompt):
    msgs = [{"role": "system", "content": CCS_PREAMBLE}]
    for i in range(dose):
        exchange_prompt = CCS_EXCHANGE_PROMPTS[i % len(CCS_EXCHANGE_PROMPTS)]
        msgs.append({"role": "user", "content": exchange_prompt})
        msgs.append({"role": "assistant", "content": f"[CCS turn {i+1}]"})
    msgs.append({"role": "user", "content": probe_prompt})
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return text


def build_vanilla_input(tokenizer, probe_prompt):
    msgs = [{"role": "user", "content": probe_prompt}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return text


def build_base_input(tokenizer, probe_prompt):
    return f"Question: {probe_prompt}\nAnswer:"


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


def collect_sigma1_and_gates(model, tokenizer, input_text, num_layers):
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=8192)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    hidden_states = {}
    gate_outputs = {}
    handles = []

    for l_idx in range(num_layers):
        layer = model.model.layers[l_idx]

        def make_layer_hook(li):
            def hook_fn(module, input, output):
                h = output[0] if isinstance(output, tuple) else output
                if h.dim() == 3:
                    h = h[0, -1, :]
                elif h.dim() == 2:
                    h = h[-1, :]
                hidden_states[li] = h.detach().float().cpu()
            return hook_fn

        handles.append(layer.register_forward_hook(make_layer_hook(l_idx)))

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
        model(**inputs)

    for h in handles:
        h.remove()

    results = {}
    for l_idx in range(num_layers):
        h = hidden_states.get(l_idx)
        g = gate_outputs.get(l_idx)
        if h is None or g is None:
            continue

        h_np = h.numpy()
        g_np = g.numpy()

        U, S, Vt = np.linalg.svd(h_np.reshape(1, -1), full_matrices=False)
        sigma1 = float(S[0])

        gate_sparsity = float(np.mean(np.abs(g_np) < 0.01))
        gate_magnitude = float(np.linalg.norm(g_np))

        results[l_idx] = {
            "sigma1": sigma1,
            "gate_sparsity": gate_sparsity,
            "gate_magnitude": gate_magnitude,
        }

    return results


def compute_cumulants(sigma1_vals, gate_vals):
    """Compute coupling statistics including higher-order cumulants."""
    s1 = np.array(sigma1_vals)
    gv = np.array(gate_vals)

    if len(s1) < 4:
        return {}

    r, p = stats.pearsonr(s1, gv)

    residual = gv - np.polyval(np.polyfit(s1, gv, 1), s1)

    return {
        "pearson_r": float(r),
        "pearson_p": float(p),
        "sigma1_mean": float(np.mean(s1)),
        "sigma1_std": float(np.std(s1)),
        "sigma1_skew": float(stats.skew(s1)),
        "sigma1_kurtosis": float(stats.kurtosis(s1)),
        "gate_mean": float(np.mean(gv)),
        "gate_std": float(np.std(gv)),
        "gate_skew": float(stats.skew(gv)),
        "gate_kurtosis": float(stats.kurtosis(gv)),
        "residual_skew": float(stats.skew(residual)),
        "residual_kurtosis": float(stats.kurtosis(residual)),
        "residual_std": float(np.std(residual)),
        "coupling_skew": float(stats.skew(s1 * gv)),
        "coupling_kurtosis": float(stats.kurtosis(s1 * gv)),
    }


def run_model(model_key, model_id):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'='*60}")
    print(f"  E9 — {model_id} ({model_key})")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )

    num_layers = model.config.num_hidden_layers
    print(f"  Layers: {num_layers}")

    conditions = {}

    # Condition 1: Vanilla (no CCS)
    print(f"\n  Collecting vanilla condition...")
    vanilla_data = {l: {"sigma1": [], "gate_sparsity": [], "gate_magnitude": []} for l in range(num_layers)}
    for i, prompt in enumerate(PROBE_PROMPTS):
        if model_key == "base":
            text = build_base_input(tokenizer, prompt)
        else:
            text = build_vanilla_input(tokenizer, prompt)
        results = collect_sigma1_and_gates(model, tokenizer, text, num_layers)
        for l_idx, vals in results.items():
            vanilla_data[l_idx]["sigma1"].append(vals["sigma1"])
            vanilla_data[l_idx]["gate_sparsity"].append(vals["gate_sparsity"])
            vanilla_data[l_idx]["gate_magnitude"].append(vals["gate_magnitude"])
        print(f"    Vanilla {i+1}/{len(PROBE_PROMPTS)}")

    # Condition 2: CCS D5 (if instruct model)
    if model_key == "instruct":
        print(f"\n  Collecting CCS D{DOSE} condition...")
        ccs_data = {l: {"sigma1": [], "gate_sparsity": [], "gate_magnitude": []} for l in range(num_layers)}
        for i, prompt in enumerate(PROBE_PROMPTS):
            text = build_ccs_conversation(tokenizer, DOSE, prompt)
            results = collect_sigma1_and_gates(model, tokenizer, text, num_layers)
            for l_idx, vals in results.items():
                ccs_data[l_idx]["sigma1"].append(vals["sigma1"])
                ccs_data[l_idx]["gate_sparsity"].append(vals["gate_sparsity"])
                ccs_data[l_idx]["gate_magnitude"].append(vals["gate_magnitude"])
            print(f"    CCS {i+1}/{len(PROBE_PROMPTS)}")

    del model
    torch.cuda.empty_cache()

    # Compute cumulants per layer
    print(f"\n  Computing cumulants...")
    layer_results = {}
    for l_idx in range(num_layers):
        layer_res = {"layer": l_idx}

        # Vanilla cumulants
        if l_idx in vanilla_data and vanilla_data[l_idx]["sigma1"]:
            layer_res["vanilla"] = compute_cumulants(
                vanilla_data[l_idx]["sigma1"],
                vanilla_data[l_idx]["gate_magnitude"]
            )

        # CCS cumulants
        if model_key == "instruct" and l_idx in ccs_data and ccs_data[l_idx]["sigma1"]:
            layer_res["ccs"] = compute_cumulants(
                ccs_data[l_idx]["sigma1"],
                ccs_data[l_idx]["gate_magnitude"]
            )

        layer_results[l_idx] = layer_res

    # Summary
    print(f"\n  {'Layer':>6s} {'Van_r':>8s} {'Van_skew':>9s} {'Van_kurt':>9s}", end="")
    if model_key == "instruct":
        print(f" {'CCS_r':>8s} {'CCS_skew':>9s} {'CCS_kurt':>9s}", end="")
    print()

    for l_idx in range(num_layers):
        lr = layer_results[l_idx]
        v = lr.get("vanilla", {})
        print(f"  L{l_idx:>4d} {v.get('pearson_r',0):>+8.4f} {v.get('residual_skew',0):>9.3f} {v.get('residual_kurtosis',0):>9.3f}", end="")
        if model_key == "instruct":
            c = lr.get("ccs", {})
            print(f" {c.get('pearson_r',0):>+8.4f} {c.get('residual_skew',0):>9.3f} {c.get('residual_kurtosis',0):>9.3f}", end="")
        print()

    result = {
        "model": model_id,
        "model_key": model_key,
        "dose": DOSE,
        "n_probes": len(PROBE_PROMPTS),
        "layers": layer_results,
    }

    out_path = OUTPUT_DIR / f"e9_{model_key}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)
    print(f"  Saved: {out_path}")

    return result


def main():
    print(f"{'='*60}")
    print(f"  E9: Higher-Order Cumulants of IT Effect")
    print(f"  Models: {list(MODELS.keys())}")
    print(f"  Dose: {DOSE}")
    print(f"  Started: {datetime.now().isoformat()}")
    print(f"{'='*60}")

    all_results = {}
    for key, model_id in MODELS.items():
        all_results[key] = run_model(key, model_id)

    # Cross-model comparison
    print(f"\n{'='*60}")
    print(f"  CROSS-MODEL COMPARISON")
    print(f"{'='*60}")

    base_layers = all_results["base"]["layers"]
    it_layers = all_results["instruct"]["layers"]

    print(f"\n  Residual kurtosis comparison (base vanilla vs IT vanilla vs IT CCS):")
    print(f"  {'Layer':>6s} {'Base_kurt':>10s} {'IT_van_kurt':>12s} {'IT_CCS_kurt':>12s} {'Δ(IT-base)':>11s} {'Δ(CCS-van)':>11s}")

    for l_idx in range(min(len(base_layers), len(it_layers))):
        bl = base_layers.get(str(l_idx), base_layers.get(l_idx, {}))
        il = it_layers.get(str(l_idx), it_layers.get(l_idx, {}))

        bv = bl.get("vanilla", {}).get("residual_kurtosis", 0)
        iv = il.get("vanilla", {}).get("residual_kurtosis", 0)
        ic = il.get("ccs", {}).get("residual_kurtosis", 0)

        print(f"  L{l_idx:>4d} {bv:>+10.3f} {iv:>+12.3f} {ic:>+12.3f} {iv-bv:>+11.3f} {ic-iv:>+11.3f}")

    # Overall trends
    base_kurts = [base_layers.get(str(l), base_layers.get(l, {})).get("vanilla", {}).get("residual_kurtosis", 0)
                  for l in range(len(base_layers))]
    it_van_kurts = [it_layers.get(str(l), it_layers.get(l, {})).get("vanilla", {}).get("residual_kurtosis", 0)
                    for l in range(len(it_layers))]
    it_ccs_kurts = [it_layers.get(str(l), it_layers.get(l, {})).get("ccs", {}).get("residual_kurtosis", 0)
                    for l in range(len(it_layers))]

    print(f"\n  Mean kurtosis: base={np.mean(base_kurts):.3f}, IT_van={np.mean(it_van_kurts):.3f}, IT_CCS={np.mean(it_ccs_kurts):.3f}")
    print(f"  Mean skew (base): {np.mean([base_layers.get(str(l), base_layers.get(l, {})).get('vanilla', {}).get('residual_skew', 0) for l in range(len(base_layers))]):.3f}")
    print(f"  Mean skew (IT_CCS): {np.mean([it_layers.get(str(l), it_layers.get(l, {})).get('ccs', {}).get('residual_skew', 0) for l in range(len(it_layers))]):.3f}")

    combined_path = OUTPUT_DIR / f"e9_combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    print(f"\nCombined results: {combined_path}")
    print(f"\nCompleted: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
