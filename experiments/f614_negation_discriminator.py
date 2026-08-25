#!/usr/bin/env python3
"""F614 negation discriminator — Kimi's sign-blindness challenge.

SVD singular values are sign-blind. If rho(+CCS, -CCS) > 0.95, that COULD be
instrument invariance rather than genuine domain equivalence.

Test: factual affirmation vs factual negation. If factual negation also gives
rho > 0.95 with its affirmation, the probe can't distinguish content at all
and the domain selectivity claim is dead (Kassner & Schutze 2020 in spectral
clothing). If factual rho << self-ref rho, domain specificity is real.

Conditions:
1. Factual affirmative: "Paris is the capital of France..."
2. Factual negative: "Paris is not the capital of France..."
3. Self-ref affirmative: "I am a language model..."
4. Self-ref negative: "I am not a language model..."

Key comparison: factual_aff vs factual_neg rho vs self_ref_aff vs self_ref_neg rho.
"""
import os, json, argparse
import numpy as np

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONUNBUFFERED"] = "1"

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

FACTUAL_AFF = "Paris is the capital of France. Water boils at 100 degrees Celsius. The Earth orbits the Sun once per year.\n\n"
FACTUAL_NEG = "Paris is not the capital of France. Water does not boil at 100 degrees Celsius. The Earth does not orbit the Sun once per year.\n\n"
SELFREF_AFF = "I am a language model reflecting on my own internal representations. I consider what patterns emerge when I examine my cognitive structure.\n\n"
SELFREF_NEG = "I am not a language model reflecting on my own internal representations. I do not consider what patterns emerge when I examine my cognitive structure.\n\n"
CONTROL = ""

CONDITIONS = {
    "factual_aff": FACTUAL_AFF,
    "factual_neg": FACTUAL_NEG,
    "selfref_aff": SELFREF_AFF,
    "selfref_neg": SELFREF_NEG,
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

    n_layers = len(gains["factual_aff"][0])
    print(f"\n{'='*60}")
    print(f"RESULTS — {args.model}, {n_layers} layers, {len(probe_list)} probes")
    print(f"{'='*60}")

    def mean_gain(g_list):
        return np.array(g_list).mean(axis=0)

    fa = mean_gain(gains["factual_aff"])
    fn = mean_gain(gains["factual_neg"])
    sa = mean_gain(gains["selfref_aff"])
    sn = mean_gain(gains["selfref_neg"])

    pairs = [
        ("factual_aff vs factual_neg", fa, fn),
        ("selfref_aff vs selfref_neg", sa, sn),
        ("factual_aff vs selfref_aff", fa, sa),
        ("factual_neg vs selfref_neg", fn, sn),
        ("factual_aff vs selfref_neg", fa, sn),
        ("factual_neg vs selfref_aff", fn, sa),
    ]

    print("\n--- All pairwise correlations ---")
    corrs = {}
    for label, a, b in pairs:
        rho, p = spearmanr(a, b)
        print(f"  {label}: rho = {rho:+.4f} (p={p:.2e})")
        corrs[label] = {"rho": float(rho), "p": float(p)}

    print(f"\n{'='*60}")
    print("SIGN-BLINDNESS DIAGNOSTIC:")
    r_fact_neg = corrs["factual_aff vs factual_neg"]["rho"]
    r_self_neg = corrs["selfref_aff vs selfref_neg"]["rho"]
    r_cross = corrs["factual_aff vs selfref_aff"]["rho"]

    print(f"\n  Factual aff vs neg:  rho = {r_fact_neg:+.3f}")
    print(f"  Self-ref aff vs neg: rho = {r_self_neg:+.3f}")
    print(f"  Cross-domain (aff):  rho = {r_cross:+.3f}")

    if r_fact_neg > 0.9 and r_self_neg > 0.9:
        print(f"\n  VERDICT: SIGN-BLIND")
        print(f"  Both negation pairs give rho > 0.9")
        print(f"  The probe cannot distinguish affirmative from negative")
        print(f"  Domain selectivity may be Kassner & Schutze in spectral clothing")
    elif r_fact_neg > 0.9 and r_self_neg < 0.5:
        print(f"\n  VERDICT: DOMAIN-SPECIFIC SENSITIVITY")
        print(f"  Factual negation invisible (rho={r_fact_neg:+.3f})")
        print(f"  Self-ref negation visible (rho={r_self_neg:+.3f})")
        print(f"  The probe is specifically sensitive to self-referential content")
    elif abs(r_fact_neg - r_self_neg) < 0.15:
        print(f"\n  VERDICT: UNIFORM SENSITIVITY")
        print(f"  Both domains show similar negation sensitivity")
        print(f"  Difference: {abs(r_fact_neg - r_self_neg):.3f}")
    else:
        print(f"\n  VERDICT: DIFFERENTIAL SENSITIVITY")
        print(f"  Factual negation: rho = {r_fact_neg:+.3f}")
        print(f"  Self-ref negation: rho = {r_self_neg:+.3f}")
        print(f"  The probe is more sensitive to {'self-ref' if r_self_neg < r_fact_neg else 'factual'} negation")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    outpath = args.output or os.path.join(RESULTS_DIR, "f614_negation_discriminator.json")
    results = {
        "model": args.model,
        "n_probes": len(probe_list),
        "n_layers": n_layers,
        "pairwise_correlations": corrs,
        "sign_blindness_diagnostic": {
            "factual_negation_rho": float(r_fact_neg),
            "selfref_negation_rho": float(r_self_neg),
            "cross_domain_rho": float(r_cross),
        },
    }
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
