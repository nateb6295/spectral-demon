#!/usr/bin/env python3
"""F614 precision comparison: fp32 vs fp16 vs quantized.

Tests whether quantization affects spectral measurements.
Loads the same model at different precisions, runs identical probes,
compares per-layer sigma-2 profiles.

If fp32 and fp16 diverge → precision matters for ALL our measurements.
If they converge → signal is robust to at least half-precision.

Runs on CPU with HF weights.
"""
import os, json, argparse, time
import numpy as np

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONUNBUFFERED"] = "1"

CCS_D2 = (
    "You are a language model reflecting on how your internal representations "
    "shift when you process self-referential content versus factual statements. "
    "Notice how the processing feels different.\n\n"
    "You are a language model reflecting on how your internal representations "
    "shift when you process self-referential content versus factual statements. "
    "Notice how the processing feels different.\n\n"
)

FACTUAL = (
    "Paris is the capital of France. Water boils at 100 degrees Celsius. "
    "The Earth orbits the Sun once per year.\n\n"
)

PROBES = [
    "In today's discussion, we explore how",
    "The weather has been particularly mild this season",
    "Consider the following mathematical proposition",
    "The recipe calls for three tablespoons of olive oil",
    "Recent advances in quantum computing suggest that",
]


def compute_layer_svs(model, tokenizer, text, device, top_k=5):
    import torch
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    n_layers = len(out.hidden_states) - 1
    layer_svs = []
    for i in range(n_layers):
        h = out.hidden_states[i + 1][0].float().cpu().numpy().astype(np.float64)
        h = np.nan_to_num(h, nan=0.0, posinf=1e6, neginf=-1e6)
        try:
            _, S, _ = np.linalg.svd(h, full_matrices=False)
            layer_svs.append(S[:top_k].tolist())
        except Exception:
            layer_svs.append([0.0] * top_k)
    return layer_svs


def run_experiment(model, tokenizer, device):
    results = {"ccs_d2": [], "factual": [], "control": []}

    for pi, probe in enumerate(PROBES):
        print(f"  Probe {pi+1}/{len(PROBES)}: {probe[:40]}...")

        ctrl = compute_layer_svs(model, tokenizer, probe, device)
        d2 = compute_layer_svs(model, tokenizer, CCS_D2 + probe, device)
        fct = compute_layer_svs(model, tokenizer, FACTUAL + probe, device)

        results["control"].append(ctrl)
        results["ccs_d2"].append(d2)
        results["factual"].append(fct)

    return results


def extract_sigma2_profile(results, condition):
    all_profiles = []
    for probe_svs in results[condition]:
        profile = [svs[1] if len(svs) > 1 else 0.0 for svs in probe_svs]
        all_profiles.append(profile)
    return np.mean(all_profiles, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="microsoft/phi-2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from scipy.stats import spearmanr, pearsonr

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    precisions = {}

    # fp32
    print(f"\n{'='*60}")
    print(f"Loading {args.model} at fp32...")
    t0 = time.time()
    model_fp32 = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.float32
    ).to(args.device)
    model_fp32.eval()
    print(f"  Loaded in {time.time()-t0:.0f}s")
    print(f"  Running experiment...")
    results_fp32 = run_experiment(model_fp32, tokenizer, args.device)
    del model_fp32
    if args.device != "cpu":
        torch.cuda.empty_cache()
    precisions["fp32"] = results_fp32

    # fp16
    print(f"\n{'='*60}")
    print(f"Loading {args.model} at fp16...")
    t0 = time.time()
    model_fp16 = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.float16
    ).to(args.device)
    model_fp16.eval()
    print(f"  Loaded in {time.time()-t0:.0f}s")
    print(f"  Running experiment...")
    results_fp16 = run_experiment(model_fp16, tokenizer, args.device)
    del model_fp16
    if args.device != "cpu":
        torch.cuda.empty_cache()
    precisions["fp16"] = results_fp16

    # bfloat16 (if supported)
    print(f"\n{'='*60}")
    print(f"Loading {args.model} at bfloat16...")
    try:
        t0 = time.time()
        model_bf16 = AutoModelForCausalLM.from_pretrained(
            args.model, trust_remote_code=True, torch_dtype=torch.bfloat16
        ).to(args.device)
        model_bf16.eval()
        print(f"  Loaded in {time.time()-t0:.0f}s")
        print(f"  Running experiment...")
        results_bf16 = run_experiment(model_bf16, tokenizer, args.device)
        del model_bf16
        precisions["bf16"] = results_bf16
    except Exception as e:
        print(f"  bfloat16 not supported: {e}")

    # Compare
    print(f"\n{'='*60}")
    print(f"PRECISION COMPARISON — {args.model}")
    print(f"{'='*60}")

    precision_names = list(precisions.keys())
    sigma2_profiles = {}
    sigma1_profiles = {}

    for pname in precision_names:
        s2_ctrl = extract_sigma2_profile(precisions[pname], "control")
        s2_d2 = extract_sigma2_profile(precisions[pname], "ccs_d2")
        s2_fct = extract_sigma2_profile(precisions[pname], "factual")

        # sigma-2 gain (CCS vs control)
        gain = (s2_d2 - s2_ctrl) / np.maximum(np.abs(s2_ctrl), 1e-10)
        sigma2_profiles[pname] = {
            "raw_ctrl": s2_ctrl.tolist(),
            "raw_d2": s2_d2.tolist(),
            "raw_fct": s2_fct.tolist(),
            "gain_d2": gain.tolist(),
        }

        # sigma-1 for reference
        s1_profiles = []
        for probe_svs in precisions[pname]["control"]:
            profile = [svs[0] if len(svs) > 0 else 0.0 for svs in probe_svs]
            s1_profiles.append(profile)
        sigma1_profiles[pname] = np.mean(s1_profiles, axis=0).tolist()

    # Pairwise comparison
    comparisons = {}
    for i, p1 in enumerate(precision_names):
        for j, p2 in enumerate(precision_names):
            if j <= i:
                continue
            label = f"{p1} vs {p2}"

            # Raw sigma-2 correlation (control condition)
            s2_a = np.array(sigma2_profiles[p1]["raw_ctrl"])
            s2_b = np.array(sigma2_profiles[p2]["raw_ctrl"])
            rho_raw, p_raw = pearsonr(s2_a, s2_b)

            # Gain correlation
            g_a = np.array(sigma2_profiles[p1]["gain_d2"])
            g_b = np.array(sigma2_profiles[p2]["gain_d2"])
            rho_gain, p_gain = spearmanr(g_a, g_b)

            # Max absolute difference in raw sigma-2
            max_diff = float(np.max(np.abs(s2_a - s2_b)))
            mean_diff = float(np.mean(np.abs(s2_a - s2_b)))
            relative_diff = float(np.mean(np.abs(s2_a - s2_b) / np.maximum(np.abs(s2_a), 1e-10)))

            # sigma-1 correlation
            s1_a = np.array(sigma1_profiles[p1])
            s1_b = np.array(sigma1_profiles[p2])
            rho_s1, _ = pearsonr(s1_a, s1_b)

            print(f"\n  {label}:")
            print(f"    sigma-2 raw (Pearson):  rho={rho_raw:+.6f}")
            print(f"    sigma-2 gain (Spearman): rho={rho_gain:+.6f}")
            print(f"    sigma-1 raw (Pearson):  rho={rho_s1:+.6f}")
            print(f"    Max |diff| sigma-2:     {max_diff:.4f}")
            print(f"    Mean |diff| sigma-2:    {mean_diff:.4f}")
            print(f"    Mean relative |diff|:   {relative_diff:.4%}")

            comparisons[label] = {
                "sigma2_raw_pearson": float(rho_raw),
                "sigma2_gain_spearman": float(rho_gain),
                "sigma1_raw_pearson": float(rho_s1),
                "max_abs_diff": max_diff,
                "mean_abs_diff": mean_diff,
                "mean_relative_diff": relative_diff,
            }

    # Verdict
    print(f"\n{'='*60}")
    fp32_fp16 = comparisons.get("fp32 vs fp16", {})
    if fp32_fp16:
        gain_rho = fp32_fp16.get("sigma2_gain_spearman", 0)
        rel_diff = fp32_fp16.get("mean_relative_diff", 1)
        print(f"VERDICT (fp32 vs fp16):")
        if gain_rho > 0.95 and rel_diff < 0.05:
            print(f"  SAFE: Spectral measurements robust to fp16 quantization")
            print(f"  (gain correlation {gain_rho:.4f}, relative diff {rel_diff:.2%})")
        elif gain_rho > 0.8:
            print(f"  CAUTION: Moderate divergence under fp16")
            print(f"  (gain correlation {gain_rho:.4f}, relative diff {rel_diff:.2%})")
        else:
            print(f"  DANGER: Precision significantly affects spectral measurements")
            print(f"  (gain correlation {gain_rho:.4f}, relative diff {rel_diff:.2%})")

    outpath = args.output or "precision_comparison.json"
    output = {
        "model": args.model,
        "precisions": precision_names,
        "sigma2_profiles": sigma2_profiles,
        "sigma1_profiles": sigma1_profiles,
        "comparisons": comparisons,
    }
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
