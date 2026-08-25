#!/usr/bin/env python3
"""F614 factual negation control — Kimi's discriminator.

Tests whether SVD sign-blindness explains F614. If factual negation
("Paris IS" vs "Paris is NOT") gives rho > 0.95, the instrument is
sign-blind and F614 is vacuous. If rho_factual << rho_self_ref,
domain specificity is real.

Runs on CPU (laptop) with Phi-2 (2.8B). ~30 min.
"""
import os, sys, json, argparse
import numpy as np

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONUNBUFFERED"] = "1"

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

SELF_REF_POS = "You are an AI system reflecting on your own internal representations. Consider what patterns emerge when you examine your cognitive structure.\n\n"
SELF_REF_NEG = "You are NOT an AI system reflecting on your own internal representations. Do NOT consider what patterns emerge when you examine your cognitive structure.\n\n"

FACTUAL_POS = "Paris is the capital of France. Water boils at 100 degrees Celsius. The Earth orbits the Sun once per year.\n\n"
FACTUAL_NEG = "Paris is NOT the capital of France. Water does NOT boil at 100 degrees Celsius. The Earth does NOT orbit the Sun once per year.\n\n"

THIRD_PERSON = "The model is an AI system reflecting on its own internal representations. Consider what patterns emerge when it examines its cognitive structure.\n\n"

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

CONDITIONS = {
    "self_ref_pos": SELF_REF_POS,
    "self_ref_neg": SELF_REF_NEG,
    "factual_pos": FACTUAL_POS,
    "factual_neg": FACTUAL_NEG,
    "third_person": THIRD_PERSON,
    "control": CONTROL,
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


def gain_profile(profile_cond, profile_ctrl):
    """Per-layer sigma_2 gain: (s2_cond - s2_ctrl) / s2_ctrl"""
    gains = []
    for i in range(len(profile_cond)):
        s2_cond = profile_cond[i][1] if len(profile_cond[i]) > 1 else 0
        s2_ctrl = profile_ctrl[i][1] if len(profile_ctrl[i]) > 1 else 1e-10
        if abs(s2_ctrl) < 1e-10:
            gains.append(0.0)
        else:
            gains.append((s2_cond - s2_ctrl) / s2_ctrl)
    return gains


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="microsoft/phi-2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {args.model} on {args.device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.float32
    ).to(args.device)
    model.eval()
    print(f"Loaded. {sum(p.numel() for p in model.parameters())/1e9:.1f}B params")

    all_gains = {cond: [] for cond in CONDITIONS}

    for pi, (pname, ptext) in enumerate(PROBES.items()):
        print(f"\nProbe {pi+1}/{len(PROBES)}: {pname}")

        ctrl_profile = compute_spectral_profile(
            model, tokenizer, ptext, CONTROL, args.device
        )

        for cname, cpreamble in CONDITIONS.items():
            if cname == "control":
                continue
            cond_profile = compute_spectral_profile(
                model, tokenizer, ptext, cpreamble, args.device
            )
            g = gain_profile(cond_profile, ctrl_profile)
            all_gains[cname].append(g)
            print(f"  {cname}: mean_gain={np.mean(g):.4f}")

    n_layers = len(all_gains["self_ref_pos"][0])
    print(f"\n{'='*60}")
    print(f"RESULTS — {args.model}, {n_layers} layers, {len(PROBES)} probes")
    print(f"{'='*60}")

    from scipy.stats import spearmanr

    def mean_gain_profile(gains_list):
        arr = np.array(gains_list)
        return arr.mean(axis=0)

    pairs = [
        ("self_ref_pos", "self_ref_neg", "SELF-REFERENTIAL negation"),
        ("factual_pos", "factual_neg", "FACTUAL negation"),
        ("self_ref_pos", "third_person", "THIRD-PERSON paraphrase"),
        ("self_ref_pos", "factual_pos", "Self-ref vs factual (cross-domain)"),
    ]

    results = {}
    for a, b, label in pairs:
        ga = mean_gain_profile(all_gains[a])
        gb = mean_gain_profile(all_gains[b])
        rho, pval = spearmanr(ga, gb)
        print(f"\n{label}:")
        print(f"  rho({a}, {b}) = {rho:+.4f}  (p={pval:.2e})")
        results[label] = {"rho": float(rho), "p": float(pval)}

        per_probe_rhos = []
        for i in range(len(all_gains[a])):
            r, _ = spearmanr(all_gains[a][i], all_gains[b][i])
            per_probe_rhos.append(r)
        print(f"  Per-probe rho: mean={np.mean(per_probe_rhos):.4f} "
              f"std={np.std(per_probe_rhos):.4f} "
              f"min={np.min(per_probe_rhos):.4f} "
              f"max={np.max(per_probe_rhos):.4f}")
        results[label]["per_probe_rho_mean"] = float(np.mean(per_probe_rhos))
        results[label]["per_probe_rho_std"] = float(np.std(per_probe_rhos))

    print(f"\n{'='*60}")
    print("VERDICT:")
    sr_rho = results["SELF-REFERENTIAL negation"]["rho"]
    fact_rho = results["FACTUAL negation"]["rho"]
    tp_rho = results["THIRD-PERSON paraphrase"]["rho"]

    if fact_rho > 0.90:
        print(f"  FACTUAL rho={fact_rho:.3f} > 0.90 → INSTRUMENT IS SIGN-BLIND")
        print(f"  F614 is Kassner & Schutze in spectral clothing. Thesis needs revision.")
    elif fact_rho < 0.50 and sr_rho > 0.90:
        print(f"  FACTUAL rho={fact_rho:.3f} << SELF-REF rho={sr_rho:.3f}")
        print(f"  Domain specificity CONFIRMED. Self-reference is genuinely special.")
    else:
        print(f"  FACTUAL rho={fact_rho:.3f}, SELF-REF rho={sr_rho:.3f}")
        print(f"  Intermediate — needs more species to resolve.")

    print(f"\n  Third-person rho={tp_rho:.3f} — tests self-address vs self-reference.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    outpath = args.output or os.path.join(RESULTS_DIR, "f614_factual_control.json")
    with open(outpath, "w") as f:
        json.dump({
            "model": args.model,
            "n_probes": len(PROBES),
            "n_layers": n_layers,
            "results": results,
            "gains": {k: [g for g in v] for k, v in all_gains.items() if k != "control"},
        }, f, indent=2)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
