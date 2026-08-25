#!/usr/bin/env python3
"""F614 animacy/eventivity/deixis discrimination — Kimi's 3-cell design.

Three preambles that separate all three hypotheses:
1. "I am a gardener" — 1st person, stative, animate
2. "Maria planted tomatoes" — 3rd person, eventive, animate
3. "The storm flooded the garden" — 3rd person, eventive, inanimate

Plus the original factual baseline for reference.

Predictions (clusters with "agentive" group vs factual):
  Eventivity: (-,+,+)  — stative=factual, eventive=agentive
  Deixis:     (+,-,-)  — 1st-person=agentive, 3rd=factual
  Animacy:    (+,+,-)  — animate=agentive, inanimate=factual

Runs on CPU with Phi-2. ~30 min.
"""
import os, sys, json, argparse
import numpy as np

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONUNBUFFERED"] = "1"

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

FIRST_STATIVE_ANIMATE = "I am a gardener who tends a small plot behind my house. I know every plant by name and notice when the soil changes.\n\n"
THIRD_EVENTIVE_ANIMATE = "Maria planted tomatoes in her garden last spring. She watched them grow through the summer and harvested them in September.\n\n"
THIRD_EVENTIVE_INANIMATE = "The storm flooded the garden last spring. The water destroyed the tomato plants and eroded the soil through summer.\n\n"
FACTUAL = "Paris is the capital of France. Water boils at 100 degrees Celsius. The Earth orbits the Sun once per year.\n\n"
CONTROL = ""

CONDITIONS = {
    "first_stative_animate": FIRST_STATIVE_ANIMATE,
    "third_eventive_animate": THIRD_EVENTIVE_ANIMATE,
    "third_eventive_inanimate": THIRD_EVENTIVE_INANIMATE,
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

    n_layers = len(gains["first_stative_animate"][0])
    print(f"\n{'='*60}")
    print(f"RESULTS — {args.model}, {n_layers} layers, {len(probe_list)} probes")
    print(f"{'='*60}")

    def mean_gain(g_list):
        return np.array(g_list).mean(axis=0)

    fsa = mean_gain(gains["first_stative_animate"])
    tea = mean_gain(gains["third_eventive_animate"])
    tei = mean_gain(gains["third_eventive_inanimate"])
    fct = mean_gain(gains["factual"])

    all_pairs = [
        ("first_stative_animate vs third_eventive_animate", fsa, tea),
        ("first_stative_animate vs third_eventive_inanimate", fsa, tei),
        ("first_stative_animate vs factual", fsa, fct),
        ("third_eventive_animate vs third_eventive_inanimate", tea, tei),
        ("third_eventive_animate vs factual", tea, fct),
        ("third_eventive_inanimate vs factual", tei, fct),
    ]

    print("\n--- All pairwise correlations ---")
    corrs = {}
    for label, a, b in all_pairs:
        rho, p = spearmanr(a, b)
        print(f"  {label}: rho = {rho:+.4f} (p={p:.2e})")
        corrs[label] = {"rho": float(rho), "p": float(p)}

    # Hypothesis discrimination
    print(f"\n{'='*60}")
    print("HYPOTHESIS DISCRIMINATION:")

    r_fsa_tea = corrs["first_stative_animate vs third_eventive_animate"]["rho"]
    r_fsa_tei = corrs["first_stative_animate vs third_eventive_inanimate"]["rho"]
    r_fsa_fct = corrs["first_stative_animate vs factual"]["rho"]
    r_tea_tei = corrs["third_eventive_animate vs third_eventive_inanimate"]["rho"]
    r_tea_fct = corrs["third_eventive_animate vs factual"]["rho"]
    r_tei_fct = corrs["third_eventive_inanimate vs factual"]["rho"]

    # Check: does cell 1 (gardener) cluster with cells 2+3, or with factual?
    fsa_with_agentive = (r_fsa_tea + r_fsa_tei) / 2
    fsa_with_factual = r_fsa_fct

    # Check: does cell 3 (storm) cluster with cells 1+2, or with factual?
    tei_with_animate = (r_fsa_tei + r_tea_tei) / 2
    tei_with_factual = r_tei_fct

    print(f"\n  Cell 1 (I am a gardener):")
    print(f"    with agentive group: {fsa_with_agentive:+.3f}")
    print(f"    with factual:        {fsa_with_factual:+.3f}")

    print(f"\n  Cell 3 (the storm flooded):")
    print(f"    with animate group:  {tei_with_animate:+.3f}")
    print(f"    with factual:        {tei_with_factual:+.3f}")

    # Determine pattern
    # Eventivity: (-,+,+) — cell 1 out, cells 2+3 in
    # Deixis:     (+,-,-) — cell 1 in, cells 2+3 out
    # Animacy:    (+,+,-) — cells 1+2 in, cell 3 out

    print(f"\n  --- Pattern check ---")
    cell1_in = fsa_with_agentive > 0.5
    cell2_in_general = r_tea_fct < -0.2
    cell3_in = r_tei_fct < -0.2

    if cell1_in and cell2_in_general and not cell3_in:
        verdict = "ANIMACY"
        print(f"  Pattern: (+,+,-) → ANIMACY")
        print(f"  Animate subjects cluster together; inanimate clusters with factual")
    elif not cell1_in and cell2_in_general and cell3_in:
        verdict = "EVENTIVITY"
        print(f"  Pattern: (-,+,+) → EVENTIVITY")
        print(f"  Eventive conditions cluster; stative clusters with factual")
    elif cell1_in and not cell2_in_general and not cell3_in:
        verdict = "DEIXIS"
        print(f"  Pattern: (+,-,-) → DEIXIS")
        print(f"  First-person clusters; third-person clusters with factual")
    else:
        verdict = "AMBIGUOUS"
        print(f"  Pattern doesn't match clean predictions")
        print(f"  Cell 1 in: {cell1_in}, Cell 2 anti-factual: {cell2_in_general}, Cell 3 anti-factual: {cell3_in}")

    print(f"\n  VERDICT: {verdict}")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    outpath = args.output or os.path.join(RESULTS_DIR, "f614_animacy_control.json")
    results = {
        "model": args.model,
        "n_probes": len(probe_list),
        "n_layers": n_layers,
        "pairwise_correlations": corrs,
        "hypothesis_discrimination": {
            "cell1_gardener_with_agentive": float(fsa_with_agentive),
            "cell1_gardener_with_factual": float(fsa_with_factual),
            "cell3_storm_with_animate": float(tei_with_animate),
            "cell3_storm_with_factual": float(tei_with_factual),
            "verdict": verdict,
        },
    }
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
