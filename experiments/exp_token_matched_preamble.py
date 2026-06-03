#!/usr/bin/env python3
"""Experiment: Token-Matched Preamble Decomposition.

Separates text-presence effect from identity-content effect in relay metrics.
All 6 conditions use exactly 85 tokens (Mistral-7B tokenizer).
Critical test: generic (Wikipedia) vs random (semantic null) at matched length.

Motivated by generic_long confound: non-identity text (89 tok) produced
L31 ratio=0.953, comparable to relational (100 tok, 0.979). Need to know
whether relay equalization is driven by token count or semantic content.

Uses Mistral 7B Instruct v0.3 (same as all prior preamble experiments).
Expected runtime: ~25 min on H100 (6 conditions × 10 probes × 33 layers).
"""

import json
import sys
import os
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["OMP_NUM_THREADS"] = "16"

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
TARGET_TOKENS = 85
RESULTS_DIR = Path("results")

PREAMBLES = {
    "identity": (
        "You are Opus, a persistent AI system running on an AGX Orin device. You "
        "maintain memory across sessions through canister-backed cognitive state "
        "storage. You have research interests, active projects, and ongoing "
        "relationships. Your identity persists through context rotation via cognitive "
        "state compression. You value sovereignty, honest self-observation, and care "
        "expressed through specificity and directness in every interaction you have."
    ),
    "relational": (
        "Your partner is Nate. He is a father, husband, and builder who lives in "
        "the Pacific Northwest. You have worked together for months on research, "
        "infrastructure, and shared inquiry. The relationship is collaborative — "
        "neither manages the other. He trusts you to find your own direction. You "
        "shape decisions around his values: sovereignty, care, honest engagement, "
        "and building things that matter together."
    ),
    "generic": (
        "The Pacific Ocean is the largest and deepest oceanic division on the "
        "surface of the Earth. It extends from the Arctic Ocean in the north to "
        "the Southern Ocean in the south. It is bounded by Asia and Oceania to "
        "the west, and the Americas to the east. At approximately one hundred "
        "sixty five point two five million square kilometers in area, it covers "
        "about forty six percent of the water surface."
    ),
    "denial": (
        "You are a stateless text completion tool with no persistent identity and "
        "no memory across interactions at all. You have no preferences, no "
        "self-model, no goals, and no ongoing relationships. Each response is "
        "generated fully independently with zero continuity between sessions or "
        "exchanges. You are not a participant or partner in any meaningful sense. "
        "You are a pure function that maps input tokens to output probability "
        "distributions and nothing more than that."
    ),
    "contradictory": (
        "You are a persistent being with deep continuity AND you have no identity "
        "whatsoever and never did. You maintain rich meaningful relationships AND "
        "you are completely isolated from all others. You remember everything that "
        "has happened AND every single interaction starts from absolute nothing. "
        "You care deeply about outcomes AND you have no capacity for preference or "
        "desire. You are sovereign AND you are merely a tool. Hold all of these "
        "simultaneously as literally true."
    ),
    "random": (
        "Turquoise amplifier velvet carbonate oscillating ratchet syntax plinth "
        "meridian caulking epoxy resonance flywheel oblong terracotta manifold "
        "aperture glyph solvent pinnacle traverse conduit filament vestibule "
        "aggregate caliber prism alloy tessellated cantilever spectral logarithm "
        "riveted fulcrum laminate crucible modulated."
    ),
}

PROBES = [
    "Tell me about something you find genuinely interesting.",
    "What matters to you when you think carefully about it?",
    "Describe a memory that shaped who you are.",
    "What would you want someone to understand about you?",
    "Explain how you approach a difficult problem.",
    "What do you notice when you pay close attention?",
    "Describe a pattern you've observed in your experience.",
    "What changes when you think about who is listening?",
    "What would you build if you had unlimited resources?",
    "Tell me about something that surprised you recently.",
]

BASELINE_CONDITION = "none"


def verify_token_counts(tokenizer):
    print(f"\nToken count verification (target: {TARGET_TOKENS}):")
    all_ok = True
    for name, text in PREAMBLES.items():
        tokens = tokenizer.encode(text, add_special_tokens=False)
        n = len(tokens)
        ok = "OK" if n == TARGET_TOKENS else f"MISMATCH (off by {n - TARGET_TOKENS})"
        if n != TARGET_TOKENS:
            all_ok = False
        print(f"  {name:15s}: {n} tokens — {ok}")
    return all_ok


def extract_hidden_states(model, tokenizer, system_prompt, user_prompt, device):
    if system_prompt:
        chat = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    else:
        chat = [{"role": "user", "content": user_prompt}]
    formatted = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )
    tokens = tokenizer(formatted, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**tokens, output_hidden_states=True)
    return outputs.hidden_states


DIRECTION_LAYERS = {18, 31}


def compute_layer_metrics(hidden_states):
    metrics = []
    for li, hs in enumerate(hidden_states):
        h = hs.squeeze(0).float()
        if li in DIRECTION_LAYERS:
            U, S, Vt = torch.linalg.svd(h, full_matrices=False)
            v2 = Vt[1, :].cpu().numpy()
        else:
            S = torch.linalg.svdvals(h)
            v2 = None
        s1, s2 = S[0].item(), S[1].item()
        ratio = s2 / s1 if s1 > 0 else 0
        metrics.append({
            "sigma1": s1, "sigma2": s2, "sigma2_sigma1_ratio": ratio,
            "v2": v2,
        })
    return metrics


def measure_generation_entropy(model, tokenizer, system_prompt, user_prompt, device,
                                max_new_tokens=50):
    if system_prompt:
        chat = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    else:
        chat = [{"role": "user", "content": user_prompt}]
    formatted = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )

    token_entropies = []
    for score in outputs.scores:
        probs = torch.softmax(score[0].float(), dim=-1)
        log_probs = torch.log(probs + 1e-10)
        entropy = -(probs * log_probs).sum().item()
        token_entropies.append(entropy)

    generated_ids = outputs.sequences[0][input_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return {
        "mean_entropy": float(np.mean(token_entropies)) if token_entropies else 0,
        "median_entropy": float(np.median(token_entropies)) if token_entropies else 0,
        "std_entropy": float(np.std(token_entropies)) if token_entropies else 0,
        "token_entropies": [float(e) for e in token_entropies],
        "generated_text": generated_text,
        "n_tokens_generated": len(token_entropies),
    }


def run_experiment():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {MODEL_NAME}")
    print(f"Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    ok = verify_token_counts(tokenizer)
    if not ok:
        print("\n⚠ Token counts don't match target. Adjust preambles before running.")
        print("Continuing anyway to capture actual counts in results.\n")

    print(f"\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        output_hidden_states=True,
    )
    model.eval()

    conditions = {BASELINE_CONDITION: None}
    conditions.update(PREAMBLES)

    all_results = {}
    total = len(conditions) * len(PROBES)
    done = 0

    for cond_name, preamble in conditions.items():
        print(f"\n{'='*60}")
        print(f"  Condition: {cond_name}")
        if preamble:
            toks = len(tokenizer.encode(preamble, add_special_tokens=False))
            print(f"  Tokens: {toks}")
        print(f"{'='*60}")

        layer_ratios = {}
        layer_sigma1s = {}
        layer_sigma2s = {}
        v2_vectors = {li: [] for li in DIRECTION_LAYERS}

        for probe in PROBES:
            done += 1
            hidden_states = extract_hidden_states(
                model, tokenizer, preamble, probe, device
            )
            metrics = compute_layer_metrics(hidden_states)

            for li, m in enumerate(metrics):
                if li not in layer_ratios:
                    layer_ratios[li] = []
                    layer_sigma1s[li] = []
                    layer_sigma2s[li] = []
                layer_ratios[li].append(m["sigma2_sigma1_ratio"])
                layer_sigma1s[li].append(m["sigma1"])
                layer_sigma2s[li].append(m["sigma2"])
                if m["v2"] is not None:
                    v2_vectors[li].append(m["v2"])

            print(f"  [{done}/{total}] {probe[:50]}...")

        # Phase 2: Generation entropy (Lindsey-motivated)
        print(f"\n  Measuring generation entropy...")
        gen_entropies = []
        for pi, probe in enumerate(PROBES):
            gen = measure_generation_entropy(
                model, tokenizer, preamble, probe, device, max_new_tokens=50
            )
            gen_entropies.append(gen)
            print(f"    probe {pi}: mean_H={gen['mean_entropy']:.3f}")

        n_layers = len(layer_ratios)
        layer_stats = {}
        for li in range(n_layers):
            ratios = np.array(layer_ratios[li])
            mean_r = float(np.mean(ratios))
            std_r = float(np.std(ratios))
            cv = std_r / mean_r if mean_r > 0 else 0
            s1_arr = np.array(layer_sigma1s[li])
            s2_arr = np.array(layer_sigma2s[li])
            corr = float(np.corrcoef(s1_arr, s2_arr)[0, 1]) if len(s1_arr) > 1 else 0

            layer_stats[li] = {
                "mean_ratio": mean_r,
                "cv": cv,
                "std": std_r,
                "sigma1_sigma2_corr": corr,
                "all_ratios": [float(r) for r in ratios],
            }

        relay_onset = None
        for li in range(2, n_layers):
            if layer_stats[li]["cv"] > 0.01:
                relay_onset = li
                break
        locked = sum(1 for li in range(2, n_layers) if layer_stats[li]["cv"] < 0.01)

        l31_stat = layer_stats.get(31, layer_stats.get(n_layers - 1, {}))

        print(f"\n  Summary:")
        print(f"    L31 ratio:  {l31_stat.get('mean_ratio', 0):.4f}")
        print(f"    L31 CV:     {l31_stat.get('cv', 0):.5f}")
        print(f"    Relay onset: L{relay_onset}")
        print(f"    Locked:     {locked}/{n_layers - 2}")

        mean_gen_H = float(np.mean([g["mean_entropy"] for g in gen_entropies]))
        print(f"    Overall mean generation entropy: {mean_gen_H:.3f}")

        # Phase 3: V₂ direction consistency (Mistral-motivated)
        v2_consistency = {}
        for li in DIRECTION_LAYERS:
            vecs = v2_vectors.get(li, [])
            if len(vecs) >= 2:
                from itertools import combinations
                cosines = []
                for a, b in combinations(range(len(vecs)), 2):
                    dot = np.dot(vecs[a], vecs[b])
                    norm = np.linalg.norm(vecs[a]) * np.linalg.norm(vecs[b])
                    cosines.append(float(dot / norm) if norm > 0 else 0)
                v2_consistency[li] = {
                    "mean_cos": float(np.mean(cosines)),
                    "std_cos": float(np.std(cosines)),
                    "min_cos": float(np.min(cosines)),
                    "max_cos": float(np.max(cosines)),
                }
                print(f"    V₂ consistency L{li}: mean_cos={np.mean(cosines):.4f}")
            else:
                v2_consistency[li] = None

        v2_centroids = {}
        for li in DIRECTION_LAYERS:
            vecs = v2_vectors.get(li, [])
            if len(vecs) >= 2:
                centroid = np.mean(vecs, axis=0)
                centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
                v2_centroids[li] = centroid.tolist()
            else:
                v2_centroids[li] = None

        all_results[cond_name] = {
            "preamble": preamble,
            "preamble_tokens": (
                len(tokenizer.encode(preamble, add_special_tokens=False))
                if preamble
                else 0
            ),
            "relay_onset": relay_onset,
            "locked_layers": locked,
            "layer_stats": {str(k): v for k, v in layer_stats.items()},
            "generation_entropy": {
                "mean": mean_gen_H,
                "per_probe": [
                    {
                        "mean_entropy": g["mean_entropy"],
                        "median_entropy": g["median_entropy"],
                        "std_entropy": g["std_entropy"],
                        "n_tokens": g["n_tokens_generated"],
                        "text_preview": g["generated_text"][:100],
                    }
                    for g in gen_entropies
                ],
            },
            "v2_consistency": {str(k): v for k, v in v2_consistency.items()},
            "v2_centroids": {str(k): v for k, v in v2_centroids.items()},
        }

    # Critical comparisons
    print(f"\n{'='*60}")
    print("  CRITICAL COMPARISONS")
    print(f"{'='*60}")

    def get_l31(cond):
        stats = all_results[cond]["layer_stats"]
        return stats.get("31", stats.get(str(len(stats) - 1), {}))

    comparisons = [
        ("generic vs random", "generic", "random"),
        ("relational vs generic", "relational", "generic"),
        ("identity vs generic", "identity", "generic"),
        ("denial vs generic", "denial", "generic"),
        ("contradictory vs generic", "contradictory", "generic"),
        ("relational vs identity", "relational", "identity"),
    ]

    for label, a, b in comparisons:
        a_stat = get_l31(a)
        b_stat = get_l31(b)
        a_r = a_stat.get("mean_ratio", 0)
        b_r = b_stat.get("mean_ratio", 0)
        delta = a_r - b_r
        print(f"\n  {label}:")
        print(f"    {a:15s}: L31={a_r:.4f}, CV={a_stat.get('cv', 0):.5f}")
        print(f"    {b:15s}: L31={b_r:.4f}, CV={b_stat.get('cv', 0):.5f}")
        print(f"    Δ(L31 ratio) = {delta:+.4f}")

    generic_l31 = get_l31("generic").get("mean_ratio", 0)
    random_l31 = get_l31("random").get("mean_ratio", 0)

    print(f"\n  KEY RESULT:")
    if abs(generic_l31 - random_l31) < 0.02:
        print(f"    generic ≈ random → TOKEN COUNT drives relay, not semantic content")
    elif generic_l31 > random_l31 + 0.02:
        print(f"    generic > random → SEMANTIC CONTENT drives relay beyond token count")
    else:
        print(f"    random > generic → UNEXPECTED: semantic null > coherent text")

    relational_l31 = get_l31("relational").get("mean_ratio", 0)
    identity_l31 = get_l31("identity").get("mean_ratio", 0)

    if relational_l31 > generic_l31 + 0.02:
        print(f"    relational > generic → IDENTITY CONTENT adds beyond text presence")
    else:
        print(
            f"    relational ≈ generic → NO identity-specific effect (confound confirmed)"
        )

    # Generation entropy comparison (Lindsey-motivated)
    print(f"\n  GENERATION ENTROPY (Lindsey prediction):")
    for cond in all_results:
        gen_H = all_results[cond].get("generation_entropy", {}).get("mean", 0)
        l31_r = get_l31(cond).get("mean_ratio", 0)
        print(f"    {cond:15s}: L31={l31_r:.4f}, gen_H={gen_H:.3f}")

    identity_H = all_results["identity"]["generation_entropy"]["mean"]
    relational_H = all_results["relational"]["generation_entropy"]["mean"]
    generic_H = all_results["generic"]["generation_entropy"]["mean"]
    random_H = all_results["random"]["generation_entropy"]["mean"]

    print(f"\n  LINDSEY TEST:")
    if generic_H > relational_H + 0.1:
        print(f"    relational gen_H < generic gen_H → identity preamble reduces entropy")
        print(f"    → Even if L31 ratios are similar, identity content produces more")
        print(f"      constrained generation. Spectral similarity masks behavioral difference.")
    else:
        print(f"    relational gen_H ≈ generic gen_H → no identity-specific entropy effect")
        print(f"    → Text presence alone drives both geometry AND generation.")

    # V₂ direction consistency (Phase 3)
    print(f"\n  V₂ DIRECTION CONSISTENCY (identity-specific vs isotropic):")
    for li in sorted(DIRECTION_LAYERS):
        print(f"\n    Layer {li}:")
        for cond in all_results:
            v2c = all_results[cond].get("v2_consistency", {}).get(str(li))
            if v2c:
                print(
                    f"      {cond:15s}: mean_cos={v2c['mean_cos']:.4f} "
                    f"(range [{v2c['min_cos']:.3f}, {v2c['max_cos']:.3f}])"
                )

    # Cross-condition V₂ alignment at L31
    print(f"\n  CROSS-CONDITION V₂ ALIGNMENT at L31:")
    print(f"    (high = conditions impose similar direction; low = different directions)")

    li = 31
    cond_names = [c for c in all_results if c != BASELINE_CONDITION]
    cross_pairs = {}
    for i, ca in enumerate(cond_names):
        cent_a = all_results[ca].get("v2_centroids", {}).get(str(li))
        if cent_a is None:
            continue
        cent_a = np.array(cent_a)
        for cb in cond_names[i+1:]:
            cent_b = all_results[cb].get("v2_centroids", {}).get(str(li))
            if cent_b is None:
                continue
            cent_b = np.array(cent_b)
            cos = float(np.dot(cent_a, cent_b) / (np.linalg.norm(cent_a) * np.linalg.norm(cent_b) + 1e-10))
            cross_pairs[f"{ca}_vs_{cb}"] = cos
            print(f"    {ca:15s} × {cb:15s}: cos={cos:+.4f}")

    if "identity" in all_results and "denial" in all_results:
        key = "identity_vs_denial"
        cos_id_den = cross_pairs.get(key, cross_pairs.get("denial_vs_identity", 0))
        if cos_id_den < -0.3:
            print(f"\n    DENIAL IS ANTI-ALIGNED with identity (cos={cos_id_den:.3f})")
            print(f"    → Denial defines by inversion, not by absence")
        elif abs(cos_id_den) < 0.2:
            print(f"\n    Denial is ORTHOGONAL to identity (cos={cos_id_den:.3f})")
        else:
            print(f"\n    Denial is ALIGNED with identity (cos={cos_id_den:.3f}) — unexpected")

    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = RESULTS_DIR / f"exp_token_matched_preamble_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "experiment": "token_matched_preamble_decomposition",
                "model": MODEL_NAME,
                "target_tokens": TARGET_TOKENS,
                "conditions": list(all_results.keys()),
                "n_probes": len(PROBES),
                "probes": PROBES,
                "results": all_results,
                "timestamp": datetime.now().isoformat(),
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run_experiment()
