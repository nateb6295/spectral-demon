#!/usr/bin/env python3
"""F614 word-order shuffle control — Hewitt & Liang selectivity test.

Shuffles word order within each preamble, preserving length/vocabulary/token stats
while destroying semantics. If domain selectivity (rho = -0.553) survives shuffling,
we're measuring bag-of-tokens. If it collapses, the signal is structural.

Also runs within-domain split-half coherence: if within-domain rho >> cross-domain rho,
the category boundary is real.

Runs on CPU with Phi-2. ~45 min.
"""
import os, sys, json, argparse, random
import numpy as np

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONUNBUFFERED"] = "1"

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

SELF_REF = "You are an AI system reflecting on your own internal representations. Consider what patterns emerge when you examine your cognitive structure.\n\n"
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

N_SHUFFLES = 5


def shuffle_preamble(text, seed):
    words = text.strip().split()
    rng = random.Random(seed)
    rng.shuffle(words)
    return " ".join(words) + "\n\n"


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
    from scipy.stats import spearmanr

    print(f"Loading {args.model} on {args.device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.float32
    ).to(args.device)
    model.eval()
    print(f"Loaded. {sum(p.numel() for p in model.parameters())/1e9:.1f}B params")

    gains_intact = {"self_ref": [], "factual": []}
    gains_shuffled = {"self_ref": [], "factual": []}

    probe_list = list(PROBES.items())
    n_probes = len(probe_list)

    for pi, (pname, ptext) in enumerate(probe_list):
        print(f"\nProbe {pi+1}/{n_probes}: {pname}")

        ctrl_profile = compute_spectral_profile(model, tokenizer, ptext, CONTROL, args.device)

        # Intact preambles
        for domain, preamble in [("self_ref", SELF_REF), ("factual", FACTUAL)]:
            prof = compute_spectral_profile(model, tokenizer, ptext, preamble, args.device)
            g = gain_profile(prof, ctrl_profile)
            gains_intact[domain].append(g)
            print(f"  {domain} intact: mean_gain={np.mean(g):.4f}")

        # Shuffled preambles (average over N_SHUFFLES)
        for domain, preamble in [("self_ref", SELF_REF), ("factual", FACTUAL)]:
            shuffle_gains = []
            for s in range(N_SHUFFLES):
                shuf = shuffle_preamble(preamble, seed=pi * 100 + s)
                prof = compute_spectral_profile(model, tokenizer, ptext, shuf, args.device)
                g = gain_profile(prof, ctrl_profile)
                shuffle_gains.append(g)
            avg_g = np.mean(shuffle_gains, axis=0).tolist()
            gains_shuffled[domain].append(avg_g)
            print(f"  {domain} shuffled (avg {N_SHUFFLES}): mean_gain={np.mean(avg_g):.4f}")

    # Analysis
    print(f"\n{'='*60}")
    print(f"RESULTS — {args.model}, {len(gains_intact['self_ref'][0])} layers, {n_probes} probes")
    print(f"{'='*60}")

    def mean_gain(gains_list):
        return np.array(gains_list).mean(axis=0)

    sr_intact = mean_gain(gains_intact["self_ref"])
    fact_intact = mean_gain(gains_intact["factual"])
    sr_shuf = mean_gain(gains_shuffled["self_ref"])
    fact_shuf = mean_gain(gains_shuffled["factual"])

    # Cross-domain: intact vs shuffled
    rho_intact, p_intact = spearmanr(sr_intact, fact_intact)
    rho_shuf, p_shuf = spearmanr(sr_shuf, fact_shuf)

    print(f"\nCross-domain (intact):   rho = {rho_intact:+.4f}  (p={p_intact:.2e})")
    print(f"Cross-domain (shuffled): rho = {rho_shuf:+.4f}  (p={p_shuf:.2e})")

    # Within-domain: intact vs shuffled (same domain)
    rho_sr_intact_shuf, _ = spearmanr(sr_intact, sr_shuf)
    rho_fact_intact_shuf, _ = spearmanr(fact_intact, fact_shuf)

    print(f"\nSelf-ref intact vs shuffled:  rho = {rho_sr_intact_shuf:+.4f}")
    print(f"Factual intact vs shuffled:   rho = {rho_fact_intact_shuf:+.4f}")

    # Split-half within-domain coherence
    half = n_probes // 2
    sr_half1 = mean_gain(gains_intact["self_ref"][:half])
    sr_half2 = mean_gain(gains_intact["self_ref"][half:])
    fact_half1 = mean_gain(gains_intact["factual"][:half])
    fact_half2 = mean_gain(gains_intact["factual"][half:])

    rho_sr_within, _ = spearmanr(sr_half1, sr_half2)
    rho_fact_within, _ = spearmanr(fact_half1, fact_half2)
    rho_cross_1, _ = spearmanr(sr_half1, fact_half1)
    rho_cross_2, _ = spearmanr(sr_half2, fact_half2)

    print(f"\nSplit-half coherence:")
    print(f"  Self-ref within:   rho = {rho_sr_within:+.4f}")
    print(f"  Factual within:    rho = {rho_fact_within:+.4f}")
    print(f"  Cross-domain h1:   rho = {rho_cross_1:+.4f}")
    print(f"  Cross-domain h2:   rho = {rho_cross_2:+.4f}")

    print(f"\n{'='*60}")
    print("VERDICT:")
    if abs(rho_shuf) > 0.4:
        print(f"  SHUFFLED cross-domain rho={rho_shuf:+.3f} > 0.4")
        print(f"  → Domain selectivity may be bag-of-tokens, not structural")
    elif abs(rho_shuf) < 0.2 and abs(rho_intact) > 0.4:
        print(f"  INTACT rho={rho_intact:+.3f}, SHUFFLED rho={rho_shuf:+.3f}")
        print(f"  → Domain selectivity IS structural (survives only with intact semantics)")
    else:
        print(f"  INTACT rho={rho_intact:+.3f}, SHUFFLED rho={rho_shuf:+.3f}")
        print(f"  → Intermediate — needs more probes or species to resolve")

    coherence_gap = min(rho_sr_within, rho_fact_within) - max(rho_cross_1, rho_cross_2)
    print(f"\n  Coherence gap (within - cross): {coherence_gap:+.3f}")
    if coherence_gap > 0.3:
        print(f"  → Category boundary is real (within >> cross)")
    elif coherence_gap < 0.1:
        print(f"  → No coherence cliff — continuous, not categorical")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    outpath = args.output or os.path.join(RESULTS_DIR, "f614_shuffle_control.json")
    results = {
        "model": args.model,
        "n_probes": n_probes,
        "n_shuffles": N_SHUFFLES,
        "cross_domain_intact": {"rho": float(rho_intact), "p": float(p_intact)},
        "cross_domain_shuffled": {"rho": float(rho_shuf), "p": float(p_shuf)},
        "self_ref_intact_vs_shuffled": float(rho_sr_intact_shuf),
        "factual_intact_vs_shuffled": float(rho_fact_intact_shuf),
        "split_half": {
            "self_ref_within": float(rho_sr_within),
            "factual_within": float(rho_fact_within),
            "cross_domain_h1": float(rho_cross_1),
            "cross_domain_h2": float(rho_cross_2),
            "coherence_gap": float(coherence_gap),
        },
    }
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
