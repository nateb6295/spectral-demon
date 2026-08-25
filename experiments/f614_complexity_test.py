#!/usr/bin/env python3
"""F614 syntactic complexity test — basin displacement hypothesis.

If spectral selectivity tracks syntactic complexity (how far preamble
structure displaces model from default processing basin), then matched
semantic content at different complexity levels should produce different
spectral configurations.

Three conditions, same content (garden/tomatoes), graded complexity:
1. Simple SVO: "Maria planted tomatoes in her garden."
2. Medium (compound): "Maria planted tomatoes in her garden, and she
   watched them grow through summer."
3. Complex (relative clauses, embeddings): "Maria, whose garden stretches
   behind the house she inherited from her grandmother, planted tomatoes
   that she'd germinated from seeds saved the previous autumn."

Plus factual baseline for reference.

Prediction: if basin displacement is real, complexity should anti-correlate
with factual (complex displaces further). Simple should cluster WITH factual
(stays near default basin).
"""
import os, json, argparse
import numpy as np

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONUNBUFFERED"] = "1"

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

SIMPLE = "Maria planted tomatoes in her garden. She watched them grow. She harvested them in September.\n\n"
MEDIUM = "Maria planted tomatoes in her garden last spring, and she watched them grow through the long summer months before harvesting them when September finally arrived.\n\n"
COMPLEX = "Maria, whose garden stretches behind the house she inherited from her grandmother, planted tomatoes that she'd germinated from seeds saved the previous autumn, watching as they grew through a summer that felt longer than any she could remember, until September arrived and she harvested what the soil had given back.\n\n"
FACTUAL = "Paris is the capital of France. Water boils at 100 degrees Celsius. The Earth orbits the Sun once per year.\n\n"
CONTROL = ""

CONDITIONS = {
    "simple": SIMPLE,
    "medium": MEDIUM,
    "complex": COMPLEX,
    "factual": FACTUAL,
}

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
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.float32
    ).to(args.device)
    model.eval()
    print(f"Loaded. {sum(p.numel() for p in model.parameters())/1e9:.1f}B params")

    gains = {k: [] for k in CONDITIONS}
    probe_list = list(PROBES.items())

    for pi, (pname, ptext) in enumerate(probe_list):
        print(f"\nProbe {pi+1}/{len(probe_list)}: {pname}")
        ctrl_profile = compute_spectral_profile(model, tokenizer, ptext, CONTROL, args.device)
        for cname, cpreamble in CONDITIONS.items():
            prof = compute_spectral_profile(model, tokenizer, ptext, cpreamble, args.device)
            g = gain_profile(prof, ctrl_profile)
            gains[cname].append(g)
            print(f"  {cname}: mean_gain={np.mean(g):.4f}")

    n_layers = len(gains["simple"][0])
    print(f"\n{'='*60}")
    print(f"RESULTS — {args.model}, {n_layers} layers, {len(probe_list)} probes")
    print(f"{'='*60}")

    def mean_gain(g_list):
        return np.array(g_list).mean(axis=0)

    s = mean_gain(gains["simple"])
    m = mean_gain(gains["medium"])
    c = mean_gain(gains["complex"])
    f = mean_gain(gains["factual"])

    pairs = [
        ("simple vs factual", s, f),
        ("medium vs factual", m, f),
        ("complex vs factual", c, f),
        ("simple vs medium", s, m),
        ("simple vs complex", s, c),
        ("medium vs complex", m, c),
    ]

    print("\n--- All pairwise correlations ---")
    corrs = {}
    for label, a, b in pairs:
        rho, p = spearmanr(a, b)
        print(f"  {label}: rho = {rho:+.4f} (p={p:.2e})")
        corrs[label] = {"rho": float(rho), "p": float(p)}

    print(f"\n{'='*60}")
    print("BASIN DISPLACEMENT DIAGNOSTIC:")
    r_sf = corrs["simple vs factual"]["rho"]
    r_mf = corrs["medium vs factual"]["rho"]
    r_cf = corrs["complex vs factual"]["rho"]
    r_sm = corrs["simple vs medium"]["rho"]
    r_sc = corrs["simple vs complex"]["rho"]

    print(f"\n  Gradient (vs factual):")
    print(f"    Simple:  rho = {r_sf:+.3f}")
    print(f"    Medium:  rho = {r_mf:+.3f}")
    print(f"    Complex: rho = {r_cf:+.3f}")

    if r_sf > r_mf > r_cf:
        print(f"\n  VERDICT: MONOTONIC DISPLACEMENT")
        print(f"  Complexity anti-correlates with factual monotonically")
        print(f"  Basin displacement hypothesis SUPPORTED")
    elif r_sf > 0.5 and r_cf < -0.3:
        print(f"\n  VERDICT: THRESHOLD DISPLACEMENT")
        print(f"  Simple clusters with factual, complex anti-correlates")
        print(f"  Basin displacement with threshold, not gradient")
    elif abs(r_sf - r_cf) < 0.2:
        print(f"\n  VERDICT: NO COMPLEXITY EFFECT")
        print(f"  All complexity levels similarly related to factual")
        print(f"  Basin displacement hypothesis NOT SUPPORTED")
    else:
        print(f"\n  VERDICT: MIXED")
        print(f"  Non-monotonic pattern")

    print(f"\n  Within-content coherence:")
    print(f"    Simple vs Medium:  rho = {r_sm:+.3f}")
    print(f"    Simple vs Complex: rho = {r_sc:+.3f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    outpath = args.output or os.path.join(RESULTS_DIR, "f614_complexity_test.json")
    results = {
        "model": args.model,
        "n_probes": len(probe_list),
        "n_layers": n_layers,
        "pairwise_correlations": corrs,
        "basin_displacement": {
            "simple_vs_factual": float(r_sf),
            "medium_vs_factual": float(r_mf),
            "complex_vs_factual": float(r_cf),
            "monotonic": bool(r_sf > r_mf > r_cf),
        },
    }
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
