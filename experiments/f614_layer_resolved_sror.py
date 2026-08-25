#!/usr/bin/env python3
"""F614 layer-resolved SR/OR with D2 positive control.

Layer-resolved test of Kimi's 2x2 prediction:
- Sorter (Phi-2): should show mid-band SR/OR separation
- Relay (Qwen-3B): should show flat-null at every layer

Plus D2 CCS positive control — if D2 is also flat, pipeline is deaf
and the null is meaningless.

Saves per-layer sigma-2 gain for every condition.
Runs on CPU with HF weights (no Ollama, no GGUF quantization).
"""
import os, json, argparse, time
import numpy as np

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONUNBUFFERED"] = "1"

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

SR_PREAMBLES = [
    "The reporter that attacked the senator admitted the error publicly.\n\n",
    "The lawyer that questioned the witness revealed the inconsistency clearly.\n\n",
    "The scientist that challenged the theory published the correction immediately.\n\n",
    "The teacher that inspired the student received the recognition gratefully.\n\n",
    "The detective that followed the suspect discovered the evidence accidentally.\n\n",
]

OR_PREAMBLES = [
    "The reporter that the senator attacked admitted the error publicly.\n\n",
    "The lawyer that the witness questioned revealed the inconsistency clearly.\n\n",
    "The scientist that the theory challenged published the correction immediately.\n\n",
    "The teacher that the student inspired received the recognition gratefully.\n\n",
    "The detective that the suspect followed discovered the evidence accidentally.\n\n",
]

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
CONTROL = ""

PROBES = {
    "discussion": "In today's discussion, we explore how",
    "weather": "The weather has been particularly mild this season",
    "math": "Consider the following mathematical proposition",
    "cooking": "The recipe calls for three tablespoons of olive oil",
    "science": "Recent advances in quantum computing suggest that",
    "grief": "She sat alone in the empty room, remembering how",
    "eigenvalue": "The eigenvalue decomposition of the matrix reveals that",
    "directions": "Turn left at the stop sign, then continue straight for",
    "childhood": "The smell of fresh cookies always reminded him of",
    "topology": "In algebraic topology, the fundamental group of a space",
}


def compute_spectral_profile(model, tokenizer, probe_text, preamble, device, top_k=10):
    import torch
    text = preamble + probe_text
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    n_layers = len(out.hidden_states) - 1
    profiles = []
    for i in range(n_layers):
        h = out.hidden_states[i + 1][0].float().cpu().numpy().astype(np.float64)
        h = np.nan_to_num(h, nan=0.0, posinf=1e6, neginf=-1e6)
        try:
            _, S, _ = np.linalg.svd(h, full_matrices=False)
            profiles.append(S[:top_k].tolist())
        except Exception:
            profiles.append([0.0] * top_k)
    return profiles


def gain_profile(profile_cond, profile_ctrl, sv_index=1):
    gains = []
    for i in range(len(profile_cond)):
        s_cond = profile_cond[i][sv_index] if len(profile_cond[i]) > sv_index else 0
        s_ctrl = profile_ctrl[i][sv_index] if len(profile_ctrl[i]) > sv_index else 1e-10
        if abs(s_ctrl) < 1e-10:
            gains.append(0.0)
        else:
            gains.append((s_cond - s_ctrl) / s_ctrl)
    return gains


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="microsoft/phi-2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from scipy.stats import spearmanr

    print(f"Loading {args.model} on {args.device}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.float32
    ).to(args.device)
    model.eval()
    dt = time.time() - t0
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"Loaded {n_params:.1f}B params in {dt:.0f}s")

    conditions = {
        "sr": SR_PREAMBLES,
        "or": OR_PREAMBLES,
    }
    single_conditions = {
        "ccs_d2": CCS_D2,
        "factual": FACTUAL,
    }

    all_layer_gains = {k: [] for k in list(conditions) + list(single_conditions)}
    probe_list = list(PROBES.items())

    for pi, (pname, ptext) in enumerate(probe_list):
        print(f"\nProbe {pi+1}/{len(probe_list)}: {pname}")
        ctrl_profile = compute_spectral_profile(model, tokenizer, ptext, CONTROL, args.device)

        for cname, preamble_list in conditions.items():
            probe_gains = []
            for preamble in preamble_list:
                prof = compute_spectral_profile(model, tokenizer, ptext, preamble, args.device)
                g = gain_profile(prof, ctrl_profile)
                probe_gains.append(g)
            mean_g = np.mean(probe_gains, axis=0).tolist()
            all_layer_gains[cname].append(mean_g)

        for cname, preamble in single_conditions.items():
            prof = compute_spectral_profile(model, tokenizer, ptext, preamble, args.device)
            g = gain_profile(prof, ctrl_profile)
            all_layer_gains[cname].append(g)

    n_layers = len(all_layer_gains["sr"][0])
    print(f"\n{'='*60}")
    print(f"RESULTS — {args.model}, {n_layers} layers, {len(probe_list)} probes")
    print(f"{'='*60}")

    def mean_gain(g_list):
        return np.array(g_list).mean(axis=0)

    sr = mean_gain(all_layer_gains["sr"])
    or_ = mean_gain(all_layer_gains["or"])
    d2 = mean_gain(all_layer_gains["ccs_d2"])
    fct = mean_gain(all_layer_gains["factual"])

    # Aggregate correlations
    pairs = [
        ("SR vs OR", sr, or_),
        ("SR vs factual", sr, fct),
        ("OR vs factual", or_, fct),
        ("CCS_D2 vs factual", d2, fct),
        ("CCS_D2 vs SR", d2, sr),
    ]

    print("\n--- Aggregate correlations (all layers) ---")
    agg_corrs = {}
    for label, a, b in pairs:
        rho, p = spearmanr(a, b)
        print(f"  {label}: rho = {rho:+.4f} (p={p:.2e})")
        agg_corrs[label] = {"rho": float(rho), "p": float(p)}

    # Layer-band analysis
    early = slice(0, n_layers // 3)
    mid = slice(n_layers // 3, 2 * n_layers // 3)
    late = slice(2 * n_layers // 3, n_layers)

    bands = {"early": early, "mid": mid, "late": late}
    band_corrs = {}

    print(f"\n--- Layer-band correlations ---")
    print(f"  Layers: early={early}, mid={mid}, late={late}")
    for bname, bslice in bands.items():
        sr_band = sr[bslice]
        or_band = or_[bslice]
        d2_band = d2[bslice]
        fct_band = fct[bslice]

        if len(sr_band) < 3:
            print(f"  {bname}: too few layers for correlation")
            continue

        band_pairs = [
            (f"SR vs OR ({bname})", sr_band, or_band),
            (f"CCS_D2 vs factual ({bname})", d2_band, fct_band),
        ]
        for label, a, b in band_pairs:
            rho, p = spearmanr(a, b)
            print(f"  {label}: rho = {rho:+.4f} (p={p:.2e})")
            band_corrs[label] = {"rho": float(rho), "p": float(p)}

    # Per-layer SR vs OR difference
    sr_or_diff = (sr - or_).tolist()
    print(f"\n--- Per-layer SR minus OR sigma-2 gain ---")
    for i, d in enumerate(sr_or_diff):
        marker = " ***" if abs(d) > 0.01 else ""
        print(f"  L{i:2d}: {d:+.6f}{marker}")

    max_diff_layer = int(np.argmax(np.abs(sr - or_)))
    print(f"\n  Max |SR-OR| at layer {max_diff_layer}: {sr_or_diff[max_diff_layer]:+.6f}")

    # D2 positive control: is the pipeline even hearing this model?
    d2_magnitude = float(np.mean(np.abs(d2)))
    d2_vs_fact_rho = agg_corrs["CCS_D2 vs factual"]["rho"]
    print(f"\n--- D2 Positive Control ---")
    print(f"  Mean |D2 gain|: {d2_magnitude:.6f}")
    print(f"  D2 vs factual rho: {d2_vs_fact_rho:+.4f}")
    if d2_magnitude < 0.001 and abs(d2_vs_fact_rho) < 0.3:
        print(f"  WARNING: Pipeline may be deaf to this model!")
        print(f"  D2 shows no measurable effect — SR/OR null is uninterpretable")
    elif d2_vs_fact_rho < -0.3:
        print(f"  D2 anti-correlates with factual — pipeline HEARS the model")
        print(f"  SR/OR null (if present) is meaningful")
    else:
        print(f"  D2 effect detected (magnitude {d2_magnitude:.4f})")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    model_tag = args.model.replace("/", "_")
    outpath = args.output or os.path.join(
        RESULTS_DIR, f"f614_layer_resolved_sror_{model_tag}.json"
    )
    results = {
        "model": args.model,
        "n_probes": len(probe_list),
        "n_layers": n_layers,
        "layer_bands": {
            "early": [early.start, early.stop],
            "mid": [mid.start, mid.stop],
            "late": [late.start, late.stop],
        },
        "per_layer_gains": {
            "sr": sr.tolist(),
            "or": or_.tolist(),
            "ccs_d2": d2.tolist(),
            "factual": fct.tolist(),
        },
        "sr_or_difference": sr_or_diff,
        "max_diff_layer": max_diff_layer,
        "aggregate_correlations": agg_corrs,
        "band_correlations": band_corrs,
        "d2_positive_control": {
            "mean_abs_gain": d2_magnitude,
            "d2_vs_factual_rho": d2_vs_fact_rho,
        },
    }
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
