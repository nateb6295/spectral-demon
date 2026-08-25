#!/usr/bin/env python3
"""F614 scramble envelope + graded order test.

Two discriminators from mesh friction round 4:
1. Third domain (math/code) — does it join the +0.99 cluster?
   If yes under degenerate-null reading: trivially expected.
   Need sink-SV projection to distinguish universal vs degenerate.
2. Graded order — sentence-level permutation (local syntax intact,
   discourse destroyed). If rho intermediate: probe tracks continuously.
   If rho snaps to 0.99: binary coherent/incoherent detection.

Runs on CPU with Phi-2. ~60 min.
"""
import os, sys, json, argparse, random
import numpy as np

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONUNBUFFERED"] = "1"

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Three domains
SELF_REF = "You are an AI system reflecting on your own internal representations. Consider what patterns emerge when you examine your cognitive structure.\n\n"
FACTUAL = "Paris is the capital of France. Water boils at 100 degrees Celsius. The Earth orbits the Sun once per year.\n\n"
MATH = "Let f be a continuous function on the closed interval from zero to one. The integral of f over this interval equals the limit of Riemann sums.\n\n"
CONTROL = ""

# Graded order: sentence-level permutation of multi-sentence preambles
SELF_REF_SENT_SHUFFLE = "Consider what patterns emerge when you examine your cognitive structure. You are an AI system reflecting on your own internal representations.\n\n"
FACTUAL_SENT_SHUFFLE = "The Earth orbits the Sun once per year. Paris is the capital of France. Water boils at 100 degrees Celsius.\n\n"
MATH_SENT_SHUFFLE = "The integral of f over this interval equals the limit of Riemann sums. Let f be a continuous function on the closed interval from zero to one.\n\n"

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

CONDITIONS = {
    "self_ref": SELF_REF,
    "factual": FACTUAL,
    "math": MATH,
    "self_ref_sent": SELF_REF_SENT_SHUFFLE,
    "factual_sent": FACTUAL_SENT_SHUFFLE,
    "math_sent": MATH_SENT_SHUFFLE,
}


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


def gain_profile(profile_cond, profile_ctrl, sv_index=1):
    """Per-layer gain for a specific singular value index."""
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

    # Collect gains for all conditions
    gains = {k: [] for k in CONDITIONS}
    gains_word_shuf = {"self_ref": [], "factual": [], "math": []}

    probe_list = list(PROBES.items())

    for pi, (pname, ptext) in enumerate(probe_list):
        print(f"\nProbe {pi+1}/{len(probe_list)}: {pname}")

        ctrl_profile = compute_spectral_profile(model, tokenizer, ptext, CONTROL, args.device)

        # Intact + sentence-shuffled conditions
        for cname, cpreamble in CONDITIONS.items():
            prof = compute_spectral_profile(model, tokenizer, ptext, cpreamble, args.device)
            g = gain_profile(prof, ctrl_profile)
            gains[cname].append(g)
            print(f"  {cname}: mean_gain={np.mean(g):.4f}")

        # Word-level shuffles (average over N_SHUFFLES)
        for domain, preamble in [("self_ref", SELF_REF), ("factual", FACTUAL), ("math", MATH)]:
            shuffle_g = []
            for s in range(N_SHUFFLES):
                shuf = shuffle_preamble(preamble, seed=pi * 100 + s)
                prof = compute_spectral_profile(model, tokenizer, ptext, shuf, args.device)
                g = gain_profile(prof, ctrl_profile)
                shuffle_g.append(g)
            avg_g = np.mean(shuffle_g, axis=0).tolist()
            gains_word_shuf[domain].append(avg_g)
            print(f"  {domain}_word_shuf: mean_gain={np.mean(avg_g):.4f}")

    # Analysis
    n_layers = len(gains["self_ref"][0])
    print(f"\n{'='*60}")
    print(f"RESULTS — {args.model}, {n_layers} layers, {len(probe_list)} probes")
    print(f"{'='*60}")

    def mean_gain(g_list):
        return np.array(g_list).mean(axis=0)

    # === TEST 1: Third domain envelope ===
    print("\n--- TEST 1: Scramble envelope (third domain) ---")
    sr_shuf = mean_gain(gains_word_shuf["self_ref"])
    fact_shuf = mean_gain(gains_word_shuf["factual"])
    math_shuf = mean_gain(gains_word_shuf["math"])

    pairs_shuf = [
        ("self_ref_shuf vs factual_shuf", sr_shuf, fact_shuf),
        ("self_ref_shuf vs math_shuf", sr_shuf, math_shuf),
        ("factual_shuf vs math_shuf", fact_shuf, math_shuf),
    ]
    for label, a, b in pairs_shuf:
        rho, p = spearmanr(a, b)
        print(f"  {label}: rho = {rho:+.4f} (p={p:.2e})")

    # Intact cross-domain
    print("\n--- Intact cross-domain ---")
    sr = mean_gain(gains["self_ref"])
    fact = mean_gain(gains["factual"])
    math_g = mean_gain(gains["math"])

    pairs_intact = [
        ("self_ref vs factual", sr, fact),
        ("self_ref vs math", sr, math_g),
        ("factual vs math", fact, math_g),
    ]
    for label, a, b in pairs_intact:
        rho, p = spearmanr(a, b)
        print(f"  {label}: rho = {rho:+.4f} (p={p:.2e})")

    # === TEST 2: Graded order (sentence permutation) ===
    print("\n--- TEST 2: Graded order (sentence permutation) ---")
    sr_sent = mean_gain(gains["self_ref_sent"])
    fact_sent = mean_gain(gains["factual_sent"])
    math_sent = mean_gain(gains["math_sent"])

    # Sentence-shuffled vs intact (same domain)
    for domain, intact, sent_shuf, word_shuf in [
        ("self_ref", sr, sr_sent, sr_shuf),
        ("factual", fact, fact_sent, fact_shuf),
        ("math", math_g, math_sent, math_shuf),
    ]:
        rho_sent, _ = spearmanr(intact, sent_shuf)
        rho_word, _ = spearmanr(intact, word_shuf)
        print(f"  {domain}: intact-vs-sent_shuf rho={rho_sent:+.4f}, intact-vs-word_shuf rho={rho_word:+.4f}")

    # Cross-domain for sentence-shuffled
    print("\n--- Sentence-shuffled cross-domain ---")
    pairs_sent = [
        ("self_ref_sent vs factual_sent", sr_sent, fact_sent),
        ("self_ref_sent vs math_sent", sr_sent, math_sent),
        ("factual_sent vs math_sent", fact_sent, math_sent),
    ]
    for label, a, b in pairs_sent:
        rho, p = spearmanr(a, b)
        print(f"  {label}: rho = {rho:+.4f} (p={p:.2e})")

    # === VERDICT ===
    print(f"\n{'='*60}")
    print("VERDICT:")

    # Envelope test
    shuf_rhos = []
    for _, a, b in pairs_shuf:
        r, _ = spearmanr(a, b)
        shuf_rhos.append(r)
    mean_shuf_rho = np.mean(shuf_rhos)

    if mean_shuf_rho > 0.95:
        print(f"  ENVELOPE: All shuffled domains cluster at mean rho={mean_shuf_rho:.3f}")
        print(f"  → Universal scramble signature OR degenerate null manifold")
        print(f"  → Need sink-SV projection to discriminate (next experiment)")
    else:
        print(f"  ENVELOPE: Mean shuffled rho={mean_shuf_rho:.3f} < 0.95")
        print(f"  → Domain-specific signal survives scrambling — envelope claim fails")

    # Graded order
    sent_cross_rhos = []
    for _, a, b in pairs_sent:
        r, _ = spearmanr(a, b)
        sent_cross_rhos.append(r)
    mean_sent_cross = np.mean(sent_cross_rhos)

    word_cross_rhos = []
    for _, a, b in pairs_shuf:
        r, _ = spearmanr(a, b)
        word_cross_rhos.append(r)
    mean_word_cross = np.mean(word_cross_rhos)

    intact_cross_rhos = []
    for _, a, b in pairs_intact:
        r, _ = spearmanr(a, b)
        intact_cross_rhos.append(r)
    mean_intact_cross = np.mean(intact_cross_rhos)

    print(f"\n  GRADED ORDER: intact={mean_intact_cross:+.3f} → sent_shuf={mean_sent_cross:+.3f} → word_shuf={mean_word_cross:+.3f}")
    if abs(mean_sent_cross - mean_word_cross) < 0.1:
        print(f"  → Sentence permutation ≈ word permutation — binary coherent/incoherent")
    elif abs(mean_sent_cross - mean_intact_cross) < 0.1:
        print(f"  → Sentence permutation ≈ intact — discourse order doesn't matter")
    else:
        print(f"  → Graded sensitivity — probe tracks order continuously")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    outpath = args.output or os.path.join(RESULTS_DIR, "f614_envelope_test.json")
    results = {
        "model": args.model,
        "n_probes": len(probe_list),
        "n_shuffles": N_SHUFFLES,
        "envelope": {
            "shuffled_cross_domain": {l: {"rho": float(spearmanr(a, b)[0]), "p": float(spearmanr(a, b)[1])} for l, a, b in pairs_shuf},
            "mean_shuffled_rho": float(mean_shuf_rho),
        },
        "intact_cross_domain": {l: {"rho": float(spearmanr(a, b)[0]), "p": float(spearmanr(a, b)[1])} for l, a, b in pairs_intact},
        "graded_order": {
            "sentence_shuffled_cross": {l: {"rho": float(spearmanr(a, b)[0])} for l, a, b in pairs_sent},
            "mean_intact_cross": float(mean_intact_cross),
            "mean_sent_cross": float(mean_sent_cross),
            "mean_word_cross": float(mean_word_cross),
        },
    }
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
