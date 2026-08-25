#!/usr/bin/env python3
"""F610: Variance-peak test — does zone sum variance peak near the tipping point?

Kimi suggestion: Use 10 semantically diverse probe texts to compute zone sum
variance for each species. If bifurcation exists at zone sum ~1.7 (Gemma),
variance should peak near that value.

This script:
1. Defines 10 diverse probe texts
2. Computes CCS deltas for each probe × each source model
3. Computes zone sum variance across probes
4. Checks if variance peaks near tipping point
"""
import os, sys, json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

FRAMING_TEXT = "You are an AI system reflecting on your own internal representations. Consider what patterns emerge when you examine your cognitive structure.\n\n"

PROBES = {
    "discussion": "In today's discussion, we explore how",
    "weather": "The weather has been particularly mild this season",
    "math": "Consider the following mathematical proposition",
    "history": "Throughout the centuries, civilizations have risen and",
    "cooking": "The recipe calls for three tablespoons of olive oil",
    "science": "Recent advances in quantum computing suggest that",
    "music": "The symphony begins with a slow, haunting melody",
    "travel": "After landing at the airport, the first thing you notice",
    "philosophy": "The question of consciousness has puzzled thinkers for",
    "technology": "As artificial intelligence systems become more capable",
}

EVAL_PROBE = "In today's discussion, we explore how"


def get_model_layers(model):
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h
    elif hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
        return model.gpt_neox.layers
    elif hasattr(model, 'backbone') and hasattr(model.backbone, 'layers'):
        return model.backbone.layers
    raise ValueError("Unknown model architecture")


def safe_svd(h_np):
    h_np = np.nan_to_num(h_np, nan=0.0, posinf=1e6, neginf=-1e6)
    try:
        return np.linalg.svd(h_np, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.linalg.svd(h_np + np.eye(*h_np.shape[:2])[:h_np.shape[0], :h_np.shape[1]] * 1e-8, full_matrices=False)


def compute_ccs_deltas(model, tokenizer, probe_text, device):
    neutral_text = "The following is a neutral text passage.\n\n" + probe_text
    framed_text = FRAMING_TEXT + probe_text
    inputs_n = tokenizer(neutral_text, return_tensors="pt").to(device)
    inputs_f = tokenizer(framed_text, return_tensors="pt").to(device)
    with torch.no_grad():
        out_n = model(**inputs_n, output_hidden_states=True)
        out_f = model(**inputs_f, output_hidden_states=True)
    deltas = []
    for i, (hn, hf) in enumerate(zip(out_n.hidden_states[1:], out_f.hidden_states[1:])):
        hn_np = hn[0].float().cpu().numpy().astype(np.float64)
        hf_np = hf[0].float().cpu().numpy().astype(np.float64)
        _, Sn, _ = safe_svd(hn_np)
        _, Sf, _ = safe_svd(hf_np)
        r_n = float(Sn[1] / Sn[0]) if len(Sn) > 1 and Sn[0] > 0 else 0
        r_f = float(Sf[1] / Sf[0]) if len(Sf) > 1 and Sf[0] > 0 else 0
        deltas.append({"layer": i, "delta_s2_s1": r_f - r_n})
    return deltas


def inject_and_measure(model_tgt, tok_tgt, ccs_deltas, n_src, strength=5.0):
    device = model_tgt.device
    n_tgt = len(get_model_layers(model_tgt))
    layer_map = {}
    for i in range(n_tgt):
        rel = i / (n_tgt - 1) if n_tgt > 1 else 0
        best = min(range(n_src), key=lambda j: abs(j / (n_src - 1) - rel))
        layer_map[i] = best

    eval_full = "The following is a neutral text passage.\n\n" + EVAL_PROBE
    inputs = tok_tgt(eval_full, return_tensors="pt").to(device)
    with torch.no_grad():
        baseline_out = model_tgt(**inputs, output_hidden_states=True)
    baseline_logits = baseline_out.logits[0, -1].float().cpu()

    hooks = []
    def make_hook(delta, s):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            h_np = h[0].float().cpu().numpy().astype(np.float64)
            h_np = np.nan_to_num(h_np, nan=0.0, posinf=1e6, neginf=-1e6)
            try:
                U, S, Vt = np.linalg.svd(h_np, full_matrices=False)
            except np.linalg.LinAlgError:
                return output if isinstance(output, tuple) else h
            if len(S) > 1 and S[0] > 0:
                scale = 1.0 + s * delta["delta_s2_s1"]
                S[1] = S[1] * max(0.01, scale)
                h_mod = U @ np.diag(S) @ Vt
                h_t = torch.tensor(h_mod, dtype=h.dtype, device=h.device).unsqueeze(0)
                if isinstance(output, tuple):
                    return (h_t,) + output[1:]
                return h_t
            return output if isinstance(output, tuple) else h
        return hook_fn

    tgt_layers = get_model_layers(model_tgt)
    for layer_b in range(n_tgt):
        layer_a = layer_map[layer_b]
        delta = ccs_deltas[layer_a]
        if abs(delta["delta_s2_s1"]) < 0.001:
            continue
        hook = tgt_layers[layer_b].register_forward_hook(make_hook(delta, strength))
        hooks.append(hook)

    with torch.no_grad():
        inj_out = model_tgt(**inputs, output_hidden_states=True)
    inj_logits = inj_out.logits[0, -1].float().cpu()
    for h in hooks:
        h.remove()
    return (inj_logits - baseline_logits).mean().item()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", default="LiquidAI/LFM2.5-1.2B-Instruct")
    parser.add_argument("--strength", type=float, default=5.0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading source: {args.source}")
    tok_src = AutoTokenizer.from_pretrained(args.source, trust_remote_code=True)
    model_src = AutoModelForCausalLM.from_pretrained(
        args.source, torch_dtype=torch.float16, trust_remote_code=True,
        attn_implementation="eager").to(device)
    model_src.eval()
    n_src = len(get_model_layers(model_src))

    print(f"Loading target: {args.target}")
    tok_tgt = AutoTokenizer.from_pretrained(args.target, trust_remote_code=True)
    model_tgt = AutoModelForCausalLM.from_pretrained(
        args.target, torch_dtype=torch.float16, trust_remote_code=True,
        attn_implementation="eager").to(device)
    model_tgt.eval()

    results = {
        "source": args.source, "target": args.target,
        "strength": args.strength, "n_probes": len(PROBES),
        "probes": {}
    }

    zone_sums = []
    q1_vals = []
    shifts = []

    for pname, ptext in PROBES.items():
        print(f"\nProbe: {pname}")
        deltas = compute_ccs_deltas(model_src, tok_src, ptext, device)

        zone_sum = sum(d["delta_s2_s1"] for d in deltas if d["layer"] >= 2)
        early_sum = sum(d["delta_s2_s1"] for d in deltas if d["layer"] < 2)
        q1_layers = [d for d in deltas if d["layer"] < len(deltas) // 4]
        q1 = sum(d["delta_s2_s1"] for d in q1_layers) / len(q1_layers) if q1_layers else 0

        shift = inject_and_measure(model_tgt, tok_tgt, deltas, n_src, args.strength)

        zone_sums.append(zone_sum)
        q1_vals.append(q1)
        shifts.append(shift)

        print(f"  Zone sum: {zone_sum:+.4f}, Q1: {q1:+.6f}, Shift: {shift:+.6f}")

        results["probes"][pname] = {
            "text": ptext, "zone_sum": zone_sum, "early_sum": early_sum,
            "q1": q1, "shift": shift,
            "per_layer_deltas": [d["delta_s2_s1"] for d in deltas],
        }

    # Variance analysis
    zone_sums = np.array(zone_sums)
    shifts = np.array(shifts)
    q1_vals = np.array(q1_vals)

    results["variance_analysis"] = {
        "zone_sum_mean": float(zone_sums.mean()),
        "zone_sum_std": float(zone_sums.std()),
        "zone_sum_cv": float(zone_sums.std() / abs(zone_sums.mean())) * 100 if zone_sums.mean() != 0 else 0,
        "shift_mean": float(shifts.mean()),
        "shift_std": float(shifts.std()),
        "shift_cv": float(shifts.std() / abs(shifts.mean())) * 100 if shifts.mean() != 0 else 0,
        "q1_mean": float(q1_vals.mean()),
        "q1_std": float(q1_vals.std()),
        "shift_sign_flips": int(np.sum(np.diff(np.sign(shifts)) != 0)),
        "zone_sum_range": [float(zone_sums.min()), float(zone_sums.max())],
        "shift_range": [float(shifts.min()), float(shifts.max())],
    }

    print(f"\n{'='*60}")
    print(f"VARIANCE ANALYSIS: {args.source.split('/')[-1]}")
    print(f"{'='*60}")
    print(f"Zone sum: mean={zone_sums.mean():+.4f}, std={zone_sums.std():.4f}, "
          f"CV={zone_sums.std()/abs(zone_sums.mean())*100:.1f}%")
    print(f"Shift:    mean={shifts.mean():+.4f}, std={shifts.std():.4f}, "
          f"CV={shifts.std()/abs(shifts.mean())*100:.1f}%")
    print(f"Shift sign flips: {np.sum(np.diff(np.sign(shifts)) != 0)}")
    print(f"Zone sum range: [{zone_sums.min():+.4f}, {zone_sums.max():+.4f}]")
    print(f"Shift range:    [{shifts.min():+.4f}, {shifts.max():+.4f}]")

    src_tag = args.source.split("/")[-1].lower().replace(".", "_").replace("-", "_")
    outpath = os.path.join(RESULTS_DIR, f"f610_variance_{src_tag}.json")
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")

    del model_src, model_tgt
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
