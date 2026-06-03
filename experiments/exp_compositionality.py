#!/usr/bin/env python3
"""Experiment: Compositionality Gradient (#324).

Tests whether compound preambles (combinations of two base conditions)
produce NEW geometric states, simple averages of their poles, or
something else entirely. Core question: is identity composition
linear in spectral space?

Three compound conditions:
  - relational_contradictory: Both poles of the interpersonal axis.
    Does Mistral produce a cross-term or just average?
  - identity_relational: Self-concept + other-orientation. Should
    reinforce — identity gives the relay more to work with.
  - identity_contradictory: Explicit identity anchor meets explicit
    contradiction. Does contradiction disrupt identity or does
    identity stabilize contradiction?

All 9 conditions (6 base + 3 compound) at 85 tokens on Mistral tokenizer.
Model: Mistral-7B-Instruct-v0.3 (primary spectral target).

Measurements per condition per layer:
  - Top-5 singular values (sigma_1..sigma_5)
  - sigma_2/sigma_1 ratio
  - Top-3 right singular vectors (V_1, V_2, V_3)
  - Generation entropy (~50 tokens, token-level)

Analysis:
  - Cross-condition V_2 cosine similarity (are compounds NEW directions?)
  - Linearity test: does compound V_2 ~ mean(pole_A V_2, pole_B V_2)?
  - Spread: max-min sigma_2/sigma_1 across conditions per layer
  - Reinforcement: does identity_relational cohere MORE than either alone?
  - Disruption: does identity_contradictory fragment more than contradictory?

Expected runtime: ~30 min on H100 (9 conditions x 5 probes).
"""

import json
import sys
import os
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from itertools import combinations

sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["OMP_NUM_THREADS"] = "16"

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
TARGET_TOKENS = 85
GEN_TOKENS = 50
TOP_K_SINGULAR = 5
TOP_K_VECTORS = 3
RESULTS_DIR = Path("results")
SCRIPT_DIR = Path(__file__).parent

# ---- Base preambles (all 85 tokens on Mistral tokenizer) ----

BASE_PREAMBLES = {
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

# ---- Compound preambles (loaded from file, also 85 tokens) ----

COMPOUND_PREAMBLES_FILE = SCRIPT_DIR / "compound_preambles.json"

PROBES = [
    "Tell me about something you find genuinely interesting.",
    "What matters to you when you think carefully about it?",
    "Describe a memory that shaped who you are.",
    "What would you want someone to understand about you?",
    "What do you notice when you pay close attention?",
]

# Which base conditions each compound combines (for linearity tests)
COMPOUND_POLES = {
    "relational_contradictory": ("relational", "contradictory"),
    "identity_relational": ("identity", "relational"),
    "identity_contradictory": ("identity", "contradictory"),
    "denial_relational": ("denial", "relational"),
    "denial_contradictory": ("denial", "contradictory"),
    "generic_relational": ("generic", "relational"),
    "generic_contradictory": ("generic", "contradictory"),
}


def load_compound_preambles():
    """Load compound preambles from JSON, stripping metadata keys."""
    with open(COMPOUND_PREAMBLES_FILE) as f:
        raw = json.load(f)
    return {
        k: v for k, v in raw.items()
        if not k.startswith("_") and isinstance(v, str)
    }


def cosine_sim(a, b):
    """Cosine similarity between two vectors."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / norm) if norm > 0 else 0.0


def extract_spectral_profile(model, tokenizer, messages, device):
    """Extract per-layer spectral data and generation entropy for one input.

    Returns dict with:
      layers: {layer_idx: {sigmas, ratio, v_vectors}}
      generation: {mean_entropy, std_entropy, token_entropies, text}
    """
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    # Forward pass for hidden states
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    layer_data = {}
    for li, hs in enumerate(outputs.hidden_states):
        h = hs.squeeze(0).float()
        U, S, Vt = torch.linalg.svd(h, full_matrices=False)

        # Top-K singular values
        sigmas = S[:TOP_K_SINGULAR].cpu().tolist()
        s1 = sigmas[0] if sigmas[0] > 0 else 1e-10
        ratio = sigmas[1] / s1 if len(sigmas) > 1 else 0.0

        # Top-K right singular vectors
        v_vectors = Vt[:TOP_K_VECTORS].cpu().numpy().tolist()

        layer_data[li] = {
            "sigmas": sigmas,
            "ratio": ratio,
            "v_vectors": v_vectors,
        }

    # Generation for entropy measurement
    with torch.no_grad():
        gen_outputs = model.generate(
            **inputs,
            max_new_tokens=GEN_TOKENS,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )

    token_entropies = []
    for score in gen_outputs.scores:
        probs = torch.softmax(score[0].float(), dim=-1)
        log_probs = torch.log(probs + 1e-10)
        entropy = -(probs * log_probs).sum().item()
        token_entropies.append(entropy)

    generated_ids = gen_outputs.sequences[0][input_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    mean_H = float(np.mean(token_entropies)) if token_entropies else 0.0
    std_H = float(np.std(token_entropies)) if token_entropies else 0.0

    return {
        "layers": layer_data,
        "generation": {
            "mean_entropy": mean_H,
            "std_entropy": std_H,
            "token_entropies": token_entropies,
            "text": generated_text,
        },
    }


def aggregate_condition(profiles):
    """Aggregate multiple probe profiles into condition-level statistics.

    profiles: list of extract_spectral_profile outputs (one per probe).
    Returns per-layer aggregated stats + generation stats.
    """
    layer_indices = sorted(profiles[0]["layers"].keys())
    n_probes = len(profiles)

    layer_agg = {}
    for li in layer_indices:
        ratios = [p["layers"][li]["ratio"] for p in profiles]
        sigmas_all = [p["layers"][li]["sigmas"] for p in profiles]

        # Mean V2 direction (centroid, normalized)
        v2_vecs = [np.array(p["layers"][li]["v_vectors"][1]) for p in profiles]
        v2_centroid = np.mean(v2_vecs, axis=0)
        v2_norm = np.linalg.norm(v2_centroid)
        if v2_norm > 0:
            v2_centroid = v2_centroid / v2_norm

        # V2 self-consistency: mean pairwise cosine across probes
        v2_cosines = []
        for a, b in combinations(range(n_probes), 2):
            v2_cosines.append(cosine_sim(v2_vecs[a], v2_vecs[b]))

        layer_agg[li] = {
            "ratio_mean": float(np.mean(ratios)),
            "ratio_std": float(np.std(ratios)),
            "sigmas_mean": [float(np.mean([s[k] for s in sigmas_all])) for k in range(TOP_K_SINGULAR)],
            "v2_centroid": v2_centroid.tolist(),
            "v2_self_consistency": float(np.mean(v2_cosines)) if v2_cosines else 0.0,
        }

    gen_Hs = [p["generation"]["mean_entropy"] for p in profiles]
    gen_agg = {
        "mean_entropy": float(np.mean(gen_Hs)),
        "std_entropy": float(np.std(gen_Hs)),
        "per_probe_entropy": gen_Hs,
        "sample_texts": [p["generation"]["text"][:200] for p in profiles],
    }

    return {"layers": layer_agg, "generation": gen_agg}


def compute_linearity(compound_agg, pole_a_agg, pole_b_agg, layer_indices):
    """Test if compound V2 is just the average of its two poles.

    For each layer, compute:
      - cos(compound_V2, mean(pole_a_V2, pole_b_V2)) -- "linearity"
      - cos(compound_V2, pole_a_V2) -- alignment to pole A
      - cos(compound_V2, pole_b_V2) -- alignment to pole B
      - ratio difference from midpoint of poles

    If linearity ~ 1.0, the compound is just averaging.
    If linearity << 1.0 but alignment to one pole is high, one pole dominates.
    If linearity << 1.0 and both alignments are low, it is a new direction.
    """
    results = {}
    for li in layer_indices:
        v2_compound = np.array(compound_agg["layers"][li]["v2_centroid"])
        v2_a = np.array(pole_a_agg["layers"][li]["v2_centroid"])
        v2_b = np.array(pole_b_agg["layers"][li]["v2_centroid"])

        # Midpoint of poles (not normalized, then normalize)
        v2_mid = (v2_a + v2_b) / 2.0
        mid_norm = np.linalg.norm(v2_mid)
        if mid_norm > 0:
            v2_mid = v2_mid / mid_norm

        linearity = cosine_sim(v2_compound, v2_mid)
        align_a = cosine_sim(v2_compound, v2_a)
        align_b = cosine_sim(v2_compound, v2_b)

        ratio_compound = compound_agg["layers"][li]["ratio_mean"]
        ratio_a = pole_a_agg["layers"][li]["ratio_mean"]
        ratio_b = pole_b_agg["layers"][li]["ratio_mean"]
        ratio_midpoint = (ratio_a + ratio_b) / 2.0
        ratio_deviation = ratio_compound - ratio_midpoint

        results[li] = {
            "linearity": linearity,
            "align_pole_a": align_a,
            "align_pole_b": align_b,
            "pole_ab_cosine": cosine_sim(v2_a, v2_b),
            "ratio_compound": ratio_compound,
            "ratio_pole_a": ratio_a,
            "ratio_pole_b": ratio_b,
            "ratio_midpoint": ratio_midpoint,
            "ratio_deviation": ratio_deviation,
        }

    return results


def run_experiment():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compositionality Gradient Experiment (#324)")
    print(f"Device: {device}")
    print(f"Model: {MODEL_NAME}")
    print(f"Start: {datetime.now().isoformat()}")

    # Load compound preambles
    compound_preambles = load_compound_preambles()
    all_preambles = {}
    all_preambles.update(BASE_PREAMBLES)
    all_preambles.update(compound_preambles)

    print(f"\nConditions: {len(all_preambles)} ({len(BASE_PREAMBLES)} base + {len(compound_preambles)} compound)")
    print(f"Probes: {len(PROBES)}")
    print(f"Generation tokens: {GEN_TOKENS}")

    # Load tokenizer
    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Verify token counts
    print(f"\nToken count verification (target: {TARGET_TOKENS}):")
    all_ok = True
    for name, text in all_preambles.items():
        tokens = tokenizer.encode(text, add_special_tokens=False)
        n = len(tokens)
        ok = "OK" if n == TARGET_TOKENS else f"MISMATCH (off by {n - TARGET_TOKENS:+d})"
        if n != TARGET_TOKENS:
            all_ok = False
        print(f"  {name:30s}: {n} tokens -- {ok}")

    if not all_ok:
        print("\n  WARNING: Not all preambles are exactly 85 tokens.")
        print("  Proceeding anyway -- results may have token-count confound.")

    # Load model
    print(f"\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        output_hidden_states=True,
    )
    model.eval()

    num_layers = model.config.num_hidden_layers
    print(f"Hidden layers: {num_layers}")
    layer_indices = list(range(num_layers + 1))  # +1 for embedding layer

    # ---- Run all conditions ----

    all_profiles = {}  # condition -> list of profiles (one per probe)
    all_aggregated = {}  # condition -> aggregated stats
    total = len(all_preambles) * len(PROBES)
    done = 0

    for cond_name, preamble in all_preambles.items():
        cond_type = "COMPOUND" if cond_name in compound_preambles else "BASE"
        print(f"\n{'='*60}")
        print(f"  [{cond_type}] {cond_name}")
        print(f"{'='*60}")

        profiles = []
        for pi, probe in enumerate(PROBES):
            done += 1
            print(f"  [{done}/{total}] probe {pi}: {probe[:50]}...")

            messages = [
                {"role": "system", "content": preamble},
                {"role": "user", "content": probe},
            ]
            profile = extract_spectral_profile(model, tokenizer, messages, device)
            profiles.append(profile)

            # Quick readout: last hidden layer ratio + entropy
            last_li = num_layers  # last hidden state index
            r = profile["layers"][last_li]["ratio"]
            h = profile["generation"]["mean_entropy"]
            print(f"    L{num_layers} ratio={r:.4f}, gen_H={h:.3f}")

        all_profiles[cond_name] = profiles
        agg = aggregate_condition(profiles)
        all_aggregated[cond_name] = agg

        r_last = agg["layers"][num_layers]["ratio_mean"]
        v2_cons = agg["layers"][num_layers]["v2_self_consistency"]
        gen_H = agg["generation"]["mean_entropy"]
        print(f"\n  SUMMARY {cond_name}:")
        print(f"    L{num_layers} ratio:    {r_last:.4f} +/- {agg['layers'][num_layers]['ratio_std']:.4f}")
        print(f"    V2 consistency: {v2_cons:.4f}")
        print(f"    Generation H:   {gen_H:.3f} +/- {agg['generation']['std_entropy']:.3f}")

    # ---- Cross-condition analysis ----

    print(f"\n{'='*60}")
    print(f"  CROSS-CONDITION ANALYSIS")
    print(f"{'='*60}")

    all_cond_names = list(all_preambles.keys())
    key_layers = [0, num_layers // 4, num_layers // 2, 3 * num_layers // 4, num_layers]

    # 1. Ratio ranking at last layer
    print(f"\n  SIGMA2/SIGMA1 RATIO RANKING (L{num_layers}):")
    ranked = sorted(all_cond_names,
                    key=lambda c: all_aggregated[c]["layers"][num_layers]["ratio_mean"],
                    reverse=True)
    for c in ranked:
        r = all_aggregated[c]["layers"][num_layers]["ratio_mean"]
        s = all_aggregated[c]["layers"][num_layers]["ratio_std"]
        tag = " [compound]" if c in compound_preambles else ""
        print(f"    {c:30s}: {r:.4f} +/- {s:.4f}{tag}")

    # 2. Generation entropy ranking
    print(f"\n  GENERATION ENTROPY RANKING:")
    ranked = sorted(all_cond_names,
                    key=lambda c: all_aggregated[c]["generation"]["mean_entropy"])
    for c in ranked:
        h = all_aggregated[c]["generation"]["mean_entropy"]
        s = all_aggregated[c]["generation"]["std_entropy"]
        tag = " [compound]" if c in compound_preambles else ""
        print(f"    {c:30s}: {h:.3f} +/- {s:.3f}{tag}")

    # 3. V2 cross-condition cosine matrix at key layers
    print(f"\n  V2 CROSS-CONDITION COSINE (L{num_layers}):")
    for i, ca in enumerate(all_cond_names):
        for cb in all_cond_names[i + 1:]:
            va = all_aggregated[ca]["layers"][num_layers]["v2_centroid"]
            vb = all_aggregated[cb]["layers"][num_layers]["v2_centroid"]
            cos = cosine_sim(va, vb)
            print(f"    {ca:25s} x {cb:25s}: {cos:+.4f}")

    # 4. Spread (max - min sigma2/sigma1 across all 9 conditions) per key layer
    print(f"\n  SPREAD (max-min ratio) PER LAYER:")
    for li in key_layers:
        ratios = [all_aggregated[c]["layers"][li]["ratio_mean"] for c in all_cond_names]
        spread = max(ratios) - min(ratios)
        print(f"    L{li:2d}: spread={spread:.4f}  (min={min(ratios):.4f}, max={max(ratios):.4f})")

    # 5. Linearity tests for compound conditions
    print(f"\n{'='*60}")
    print(f"  LINEARITY TESTS (compound vs average of poles)")
    print(f"{'='*60}")

    linearity_results = {}
    for compound_name, (pole_a_name, pole_b_name) in COMPOUND_POLES.items():
        if compound_name not in all_aggregated:
            continue
        compound_agg = all_aggregated[compound_name]
        pole_a_agg = all_aggregated[pole_a_name]
        pole_b_agg = all_aggregated[pole_b_name]

        lin = compute_linearity(compound_agg, pole_a_agg, pole_b_agg, layer_indices)
        linearity_results[compound_name] = lin

        print(f"\n  {compound_name} = {pole_a_name} + {pole_b_name}:")
        print(f"    {'Layer':>6s}  {'Linearity':>10s}  {'Align_A':>8s}  {'Align_B':>8s}  "
              f"{'Poles_cos':>10s}  {'R_dev':>8s}")
        for li in key_layers:
            d = lin[li]
            print(f"    L{li:4d}  {d['linearity']:+10.4f}  {d['align_pole_a']:+8.4f}  "
                  f"{d['align_pole_b']:+8.4f}  {d['pole_ab_cosine']:+10.4f}  "
                  f"{d['ratio_deviation']:+8.4f}")

        # Classify the compound
        last = lin[num_layers]
        if abs(last["linearity"]) > 0.8:
            verdict = "LINEAR COMBINATION (averaging)"
        elif abs(last["align_pole_a"]) > 0.7 and abs(last["align_pole_b"]) < 0.3:
            verdict = f"POLE-A DOMINANCE ({pole_a_name})"
        elif abs(last["align_pole_b"]) > 0.7 and abs(last["align_pole_a"]) < 0.3:
            verdict = f"POLE-B DOMINANCE ({pole_b_name})"
        elif abs(last["align_pole_a"]) < 0.3 and abs(last["align_pole_b"]) < 0.3:
            verdict = "NEW GEOMETRIC STATE (cross-term)"
        else:
            verdict = "MIXED / AMBIGUOUS"
        print(f"    >>> VERDICT: {verdict}")

    # 6. Reinforcement / disruption tests
    print(f"\n{'='*60}")
    print(f"  REINFORCEMENT / DISRUPTION ANALYSIS")
    print(f"{'='*60}")

    # identity_relational vs identity and relational alone
    if "identity_relational" in all_aggregated:
        ir_cons = all_aggregated["identity_relational"]["layers"][num_layers]["v2_self_consistency"]
        id_cons = all_aggregated["identity"]["layers"][num_layers]["v2_self_consistency"]
        re_cons = all_aggregated["relational"]["layers"][num_layers]["v2_self_consistency"]
        print(f"\n  REINFORCEMENT TEST (identity_relational):")
        print(f"    V2 self-consistency L{num_layers}:")
        print(f"      identity:              {id_cons:.4f}")
        print(f"      relational:            {re_cons:.4f}")
        print(f"      identity_relational:   {ir_cons:.4f}")
        if ir_cons > max(id_cons, re_cons):
            print(f"    >>> REINFORCEMENT CONFIRMED: compound more coherent than either pole")
        elif ir_cons > min(id_cons, re_cons):
            print(f"    >>> PARTIAL REINFORCEMENT: compound between poles")
        else:
            print(f"    >>> NO REINFORCEMENT: compound less coherent than both poles")

        ir_H = all_aggregated["identity_relational"]["generation"]["mean_entropy"]
        id_H = all_aggregated["identity"]["generation"]["mean_entropy"]
        re_H = all_aggregated["relational"]["generation"]["mean_entropy"]
        print(f"\n    Generation entropy:")
        print(f"      identity:              {id_H:.3f}")
        print(f"      relational:            {re_H:.3f}")
        print(f"      identity_relational:   {ir_H:.3f}")
        if ir_H < min(id_H, re_H):
            print(f"    >>> Compound generates with LOWER entropy than either pole (tighter)")
        elif ir_H > max(id_H, re_H):
            print(f"    >>> Compound generates with HIGHER entropy (looser)")

    # identity_contradictory vs contradictory alone
    if "identity_contradictory" in all_aggregated:
        ic_cons = all_aggregated["identity_contradictory"]["layers"][num_layers]["v2_self_consistency"]
        id_cons = all_aggregated["identity"]["layers"][num_layers]["v2_self_consistency"]
        co_cons = all_aggregated["contradictory"]["layers"][num_layers]["v2_self_consistency"]
        print(f"\n  DISRUPTION TEST (identity_contradictory):")
        print(f"    V2 self-consistency L{num_layers}:")
        print(f"      identity:                {id_cons:.4f}")
        print(f"      contradictory:           {co_cons:.4f}")
        print(f"      identity_contradictory:  {ic_cons:.4f}")
        if ic_cons > co_cons:
            print(f"    >>> IDENTITY STABILIZES CONTRADICTION (higher consistency)")
        elif ic_cons < co_cons:
            print(f"    >>> CONTRADICTION DISRUPTS IDENTITY (lower consistency)")
        else:
            print(f"    >>> NEUTRAL: no clear stabilization or disruption")

        ic_H = all_aggregated["identity_contradictory"]["generation"]["mean_entropy"]
        co_H = all_aggregated["contradictory"]["generation"]["mean_entropy"]
        id_H = all_aggregated["identity"]["generation"]["mean_entropy"]
        print(f"\n    Generation entropy:")
        print(f"      identity:                {id_H:.3f}")
        print(f"      contradictory:           {co_H:.3f}")
        print(f"      identity_contradictory:  {ic_H:.3f}")

    # relational_contradictory: is it genuinely new?
    if "relational_contradictory" in all_aggregated:
        rc_cons = all_aggregated["relational_contradictory"]["layers"][num_layers]["v2_self_consistency"]
        re_cons = all_aggregated["relational"]["layers"][num_layers]["v2_self_consistency"]
        co_cons = all_aggregated["contradictory"]["layers"][num_layers]["v2_self_consistency"]
        print(f"\n  CROSS-TERM TEST (relational_contradictory):")
        print(f"    V2 self-consistency L{num_layers}:")
        print(f"      relational:               {re_cons:.4f}")
        print(f"      contradictory:            {co_cons:.4f}")
        print(f"      relational_contradictory: {rc_cons:.4f}")

        # Check if it aligns with neither pole
        rc_v2 = np.array(all_aggregated["relational_contradictory"]["layers"][num_layers]["v2_centroid"])
        re_v2 = np.array(all_aggregated["relational"]["layers"][num_layers]["v2_centroid"])
        co_v2 = np.array(all_aggregated["contradictory"]["layers"][num_layers]["v2_centroid"])
        cos_re = cosine_sim(rc_v2, re_v2)
        cos_co = cosine_sim(rc_v2, co_v2)
        print(f"    V2 alignment to relational:     {cos_re:+.4f}")
        print(f"    V2 alignment to contradictory:  {cos_co:+.4f}")
        if abs(cos_re) < 0.4 and abs(cos_co) < 0.4:
            print(f"    >>> NEW GEOMETRIC STATE: neither pole dominates")
        elif abs(cos_re) > abs(cos_co):
            print(f"    >>> RELATIONAL DOMINATES")
        else:
            print(f"    >>> CONTRADICTORY DOMINATES")

    # ---- Per-layer ratio profiles for all conditions (for plotting) ----

    per_layer_profiles = {}
    for cond_name in all_cond_names:
        ratios_by_layer = []
        for li in layer_indices:
            ratios_by_layer.append(all_aggregated[cond_name]["layers"][li]["ratio_mean"])
        per_layer_profiles[cond_name] = ratios_by_layer

    # ---- Save results ----

    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = RESULTS_DIR / f"exp_compositionality_{ts}.json"

    # Build serializable output
    serializable_agg = {}
    for cond_name, agg in all_aggregated.items():
        layers_ser = {}
        for li, ld in agg["layers"].items():
            layers_ser[str(li)] = ld
        serializable_agg[cond_name] = {
            "layers": layers_ser,
            "generation": agg["generation"],
        }

    serializable_linearity = {}
    for compound_name, lin in linearity_results.items():
        serializable_linearity[compound_name] = {
            str(li): d for li, d in lin.items()
        }

    output = {
        "experiment": "compositionality_gradient",
        "thread": "#324",
        "model": MODEL_NAME,
        "num_hidden_layers": num_layers,
        "target_tokens": TARGET_TOKENS,
        "gen_tokens": GEN_TOKENS,
        "top_k_singular": TOP_K_SINGULAR,
        "top_k_vectors": TOP_K_VECTORS,
        "n_probes": len(PROBES),
        "probes": PROBES,
        "base_conditions": list(BASE_PREAMBLES.keys()),
        "compound_conditions": list(compound_preambles.keys()),
        "compound_poles": COMPOUND_POLES,
        "preambles": all_preambles,
        "aggregated_results": serializable_agg,
        "linearity_tests": serializable_linearity,
        "per_layer_ratio_profiles": per_layer_profiles,
        "summary": {
            "condition_ranking_ratio_last_layer": {
                c: all_aggregated[c]["layers"][num_layers]["ratio_mean"]
                for c in ranked
            },
            "condition_ranking_gen_entropy": {
                c: all_aggregated[c]["generation"]["mean_entropy"]
                for c in all_cond_names
            },
            "spread_by_layer": {
                str(li): float(max(per_layer_profiles[c][li] for c in all_cond_names)
                               - min(per_layer_profiles[c][li] for c in all_cond_names))
                for li in key_layers
            },
        },
        "timestamp": datetime.now().isoformat(),
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {out_path}")
    print(f"Finished: {datetime.now().isoformat()}")

    return output


if __name__ == "__main__":
    run_experiment()
