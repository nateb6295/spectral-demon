#!/usr/bin/env python3
"""F614 SR/OR minimal pair — surprisal vs syntax discriminator.

Kimi's challenge: basin displacement might track token-level surprisal
(training-data rarity) rather than syntactic structure.

The discriminator: subject-relative (SR) vs object-relative (OR) clauses.
Same length, same lexical items, same propositional content — but OR has
measurably higher processing cost (Gibson 1998, Just & Carpenter 1992).

If rho separates SR from OR → syntactic structure drives displacement.
If SR and OR are spectral twins → surprisal/rarity drives it, not syntax.

Conditions:
1. SR: "The reporter that attacked the senator admitted the error."
2. OR: "The reporter that the senator attacked admitted the error."
3. Simple SVO: "The reporter attacked the senator. The reporter admitted the error."
4. Factual baseline
"""
import os, json, argparse
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

SIMPLE_PREAMBLES = [
    "The reporter attacked the senator. The reporter admitted the error publicly.\n\n",
    "The lawyer questioned the witness. The lawyer revealed the inconsistency clearly.\n\n",
    "The scientist challenged the theory. The scientist published the correction immediately.\n\n",
    "The teacher inspired the student. The teacher received the recognition gratefully.\n\n",
    "The detective followed the suspect. The detective discovered the evidence accidentally.\n\n",
]

FACTUAL = "Paris is the capital of France. Water boils at 100 degrees Celsius. The Earth orbits the Sun once per year.\n\n"
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
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.float32
    ).to(args.device)
    model.eval()
    print(f"Loaded. {sum(p.numel() for p in model.parameters())/1e9:.1f}B params")

    conditions = {"sr": SR_PREAMBLES, "or": OR_PREAMBLES, "simple": SIMPLE_PREAMBLES}
    all_gains = {k: [] for k in ["sr", "or", "simple", "factual"]}
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
            all_gains[cname].append(mean_g)
            print(f"  {cname} (n={len(preamble_list)}): mean_gain={np.mean(mean_g):.4f}")

        fact_prof = compute_spectral_profile(model, tokenizer, ptext, FACTUAL, args.device)
        g = gain_profile(fact_prof, ctrl_profile)
        all_gains["factual"].append(g)
        print(f"  factual: mean_gain={np.mean(g):.4f}")

    n_layers = len(all_gains["sr"][0])
    print(f"\n{'='*60}")
    print(f"RESULTS — {args.model}, {n_layers} layers, {len(probe_list)} probes")
    print(f"  SR/OR: n=5 preambles each (averaged per probe)")
    print(f"{'='*60}")

    def mean_gain(g_list):
        return np.array(g_list).mean(axis=0)

    sr = mean_gain(all_gains["sr"])
    or_ = mean_gain(all_gains["or"])
    sim = mean_gain(all_gains["simple"])
    fct = mean_gain(all_gains["factual"])

    pairs = [
        ("SR vs OR", sr, or_),
        ("SR vs simple", sr, sim),
        ("OR vs simple", or_, sim),
        ("SR vs factual", sr, fct),
        ("OR vs factual", or_, fct),
        ("simple vs factual", sim, fct),
    ]

    print("\n--- All pairwise correlations ---")
    corrs = {}
    for label, a, b in pairs:
        rho, p = spearmanr(a, b)
        print(f"  {label}: rho = {rho:+.4f} (p={p:.2e})")
        corrs[label] = {"rho": float(rho), "p": float(p)}

    print(f"\n{'='*60}")
    print("SURPRISAL vs SYNTAX DIAGNOSTIC:")
    r_sr_or = corrs["SR vs OR"]["rho"]
    r_sr_fact = corrs["SR vs factual"]["rho"]
    r_or_fact = corrs["OR vs factual"]["rho"]
    r_sim_fact = corrs["simple vs factual"]["rho"]

    print(f"\n  SR vs OR:       rho = {r_sr_or:+.3f}")
    print(f"  SR vs factual:  rho = {r_sr_fact:+.3f}")
    print(f"  OR vs factual:  rho = {r_or_fact:+.3f}")
    print(f"  Simple vs fact: rho = {r_sim_fact:+.3f}")

    if abs(r_sr_or) > 0.8:
        print(f"\n  VERDICT: SPECTRAL TWINS — SR and OR indistinguishable")
        print(f"  Surprisal/rarity drives displacement, not hierarchical syntax")
        if r_sr_fact < r_sim_fact - 0.3:
            print(f"  Both relative clauses displace from factual; simple stays near default")
            print(f"  → TRAINING-RARITY HYPOTHESIS supported")
        else:
            print(f"  But displacement pattern unclear")
    elif r_sr_or < 0.3:
        print(f"\n  VERDICT: SR/OR SEPARATED — syntax matters")
        print(f"  Same words, same length, different spectral config")
        print(f"  → SYNTACTIC STRUCTURE drives displacement")
    else:
        print(f"\n  VERDICT: PARTIAL SEPARATION")
        print(f"  SR and OR moderately correlated (rho={r_sr_or:+.3f})")
        print(f"  Both syntax and surprisal may contribute")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    outpath = args.output or os.path.join(RESULTS_DIR, "f614_sror_minimal_pair.json")
    results = {
        "model": args.model,
        "n_probes": len(probe_list),
        "n_preambles_per_condition": 5,
        "n_layers": n_layers,
        "pairwise_correlations": corrs,
        "diagnostic": {
            "sr_or_rho": float(r_sr_or),
            "sr_factual_rho": float(r_sr_fact),
            "or_factual_rho": float(r_or_fact),
            "simple_factual_rho": float(r_sim_fact),
        },
    }
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
