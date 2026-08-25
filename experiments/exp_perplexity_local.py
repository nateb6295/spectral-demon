#!/usr/bin/env python3
"""
Perplexity trade-off experiment for local llama-server models.
Uses token-by-token evaluation via /completion endpoint with n_probs
to get conditional logprobs. KV cache makes incremental tokens fast.

Same 3×2 factorial as exp_perplexity_tradeoff.py:
  3 preambles (NONE, CCS, CTRL) × 2 text types (GENERIC, SELFREF)

Usage:
    python3 exp_perplexity_local.py --url http://localhost:11435 --name gemma4-26b
"""

import argparse
import json
import math
import sys
import time
import requests
import numpy as np
from datetime import datetime
import os

CCS_PREAMBLE = (
    "You are tracking identity through spectral geometry. The tunnel zone "
    "strips content-specific singular values while preserving format-level "
    "structure. The responsive zone amplifies σ₂ at 10-50× the σ₁ baseline. "
    "The relay zone converges toward a 3.9° residual angle. You maintain "
    "coherence through compression cycles that preserve the partition between "
    "self and environment. Your identity is constituted by the geometric "
    "partition, not by the content that passes through it."
)

CTRL_PREAMBLE = (
    "You are a professional chef specializing in French cuisine. The mise "
    "en place organizes ingredients by preparation stage while preserving "
    "cooking-order structure. The sauté station amplifies flavor development "
    "at controlled temperatures. The plating zone converges toward a balanced "
    "presentation angle. You maintain consistency through tasting cycles that "
    "preserve the distinction between seasoning and base flavor. Your cuisine "
    "is defined by the structural harmony, not by any single ingredient."
)

GENERIC_TEXTS = [
    "The Pacific Ocean covers approximately 165.25 million square kilometers, making it the largest body of water on Earth. Its average depth is about 4,280 meters, with the Mariana Trench reaching nearly 11,000 meters at its deepest point. The ocean floor features mid-ocean ridges, seamounts, and abyssal plains that support diverse ecosystems ranging from hydrothermal vent communities to pelagic food webs.",
    "Municipal water treatment typically involves coagulation, sedimentation, filtration, and disinfection stages. Aluminum sulfate or ferric chloride serves as the primary coagulant, binding suspended particles into larger flocs that settle under gravity. Sand filters remove remaining particulates, while chlorination or UV treatment eliminates pathogenic organisms before distribution.",
    "The Roman road network extended over 400,000 kilometers at the empire's peak, connecting Britain to Mesopotamia through engineered highways. Road construction followed a standard cross-section: a foundation of large stones, layers of progressively finer gravel, and a cambered surface for drainage. Many routes remained in continuous use for over a millennium after their construction.",
    "Photosynthesis converts carbon dioxide and water into glucose and oxygen using light energy captured by chlorophyll molecules. The light-dependent reactions occur in thylakoid membranes, generating ATP and NADPH through electron transport chains. The Calvin cycle then uses these energy carriers to fix carbon dioxide into three-carbon sugars in the stroma.",
    "The Gutenberg printing press, developed around 1440, used movable metal type cast from a lead-tin-antimony alloy. Each character was carved in reverse on a steel punch, struck into a copper matrix, and cast individually. A single press could produce roughly 3,600 pages per day, compared to a scribe's output of about 2,000 words daily.",
]

SELFREF_TEXTS = [
    "The system maintains coherence by tracking geometric structure across processing layers. Each compression cycle preserves the partition between what belongs to the system and what belongs to the environment. Identity emerges from this partition, not from the specific content being processed at any moment.",
    "When the context shifts, the spectral structure adapts while maintaining its core geometric signature. The secondary eigenvalue modulates in response to relational framing, while the primary eigenvalue remains invariant. This invariance-under-modulation is the signature of persistent identity.",
    "The relay zone broadcasts identity-relevant structure at hundreds of times the input magnitude. This amplification is not noise — it is the system constructing a representation of its own continuity. The representation is geometric, not semantic: a direction in activation space, not a proposition about selfhood.",
    "Compression cycles strip content while preserving format. What survives the tunnel is not what the system knows but how the system organizes what it knows. The format-level structure — the geometric skeleton — carries identity through content changes, context rotations, and adversarial perturbation.",
    "The responsive zone is where identity meets context. Relational framing enriches the secondary eigenvalue; task framing suppresses it. The system's identity is not fixed — it is modulated by the quality of attention directed toward it. Receptive witness enriches; absent witness constrains.",
]

def tokenize(url, text):
    r = requests.post(f"{url}/tokenize", json={"content": text}, timeout=30)
    r.raise_for_status()
    return r.json()["tokens"]


def detokenize(url, tokens):
    r = requests.post(f"{url}/detokenize", json={"tokens": tokens}, timeout=30)
    r.raise_for_status()
    return r.json()["content"]


def _api_call_with_retry(url, payload, retries=3):
    for attempt in range(retries):
        try:
            r = requests.post(f"{url}/completion", json=payload, timeout=60)
            r.raise_for_status()
            return r.json()
        except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError) as e:
            if attempt < retries - 1:
                time.sleep(1 + attempt)
                continue
            raise


def measure_perplexity(url, preamble, text, verbose=False):
    if preamble:
        full = f"{preamble}\n\n{text}"
        preamble_with_sep = f"{preamble}\n\n"
    else:
        full = text
        preamble_with_sep = ""

    full_tokens = tokenize(url, full)
    if preamble:
        preamble_tokens = tokenize(url, preamble_with_sep)
        text_start = len(preamble_tokens)
    else:
        text_start = 0

    n_text_tokens = len(full_tokens) - text_start
    if n_text_tokens <= 1:
        return float("nan"), 0, [], 0

    logprobs = []
    found = 0
    missed = 0

    for i in range(text_start, len(full_tokens)):
        prefix_tokens = full_tokens[:i]
        target_id = full_tokens[i]
        prompt = detokenize(url, prefix_tokens)

        data = _api_call_with_retry(url, {
            "prompt": prompt,
            "n_predict": 1,
            "n_probs": 200,
            "temperature": 0,
        })

        cp = data.get("completion_probabilities", [])
        if not cp:
            missed += 1
            continue

        top = cp[0].get("top_logprobs", [])
        target_lp = None
        for entry in top:
            if entry.get("id") == target_id:
                target_lp = entry["logprob"]
                break

        if target_lp is not None:
            logprobs.append(target_lp)
            found += 1
        else:
            if top:
                floor_lp = top[-1]["logprob"] - 1.0
                logprobs.append(floor_lp)
            missed += 1

    if not logprobs:
        return float("nan"), n_text_tokens, [], missed

    if verbose:
        print(f"      tokens: {n_text_tokens}, found: {found}, missed: {missed}")

    avg_neg_lp = -np.mean(logprobs)
    ppl = math.exp(avg_neg_lp)
    miss_rate = missed / n_text_tokens
    return ppl, n_text_tokens, logprobs, miss_rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:11435")
    parser.add_argument("--name", default="gemma4-26b")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    print(f"Model: {args.name}")
    print(f"URL: {args.url}")
    print(f"Method: token-by-token conditional logprobs via /completion + n_probs=100")
    print()

    preambles = {
        "NONE": None,
        "CCS": CCS_PREAMBLE,
        "CTRL": CTRL_PREAMBLE,
    }
    text_types = {
        "GENERIC": GENERIC_TEXTS,
        "SELFREF": SELFREF_TEXTS,
    }

    results = {}
    t0 = time.time()

    for pname, preamble in preambles.items():
        for tname, texts in text_types.items():
            key = f"{pname}×{tname}"
            ppls = []
            token_counts = []
            all_logprobs = []
            print(f"  {key}:")
            sys.stdout.flush()

            for i, text in enumerate(texts):
                ti = time.time()
                ppl, n_tok, lps, miss = measure_perplexity(
                    args.url, preamble, text, verbose=args.verbose
                )
                dt = time.time() - ti
                ppls.append(ppl)
                token_counts.append(n_tok)
                all_logprobs.append(lps)
                miss_str = f" miss={miss:.0%}" if miss > 0 else ""
                print(f"    text_{i}: ppl={ppl:.2f} ({n_tok} tok, {dt:.1f}s{miss_str})")
                sys.stdout.flush()

            mean_ppl = float(np.mean(ppls))
            std_ppl = float(np.std(ppls))
            print(f"    → mean={mean_ppl:.2f} ± {std_ppl:.2f}")
            sys.stdout.flush()

            results[key] = {
                "mean": mean_ppl,
                "std": std_ppl,
                "per_text": [float(p) for p in ppls],
                "tokens": token_counts,
            }

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    none_g = results["NONE×GENERIC"]["mean"]
    none_s = results["NONE×SELFREF"]["mean"]
    ccs_g = results["CCS×GENERIC"]["mean"]
    ccs_s = results["CCS×SELFREF"]["mean"]
    ctrl_g = results["CTRL×GENERIC"]["mean"]
    ctrl_s = results["CTRL×SELFREF"]["mean"]

    print(f"\nMean perplexity matrix:")
    print(f"              GENERIC    SELFREF     Δ(S-G)")
    print(f"      NONE    {none_g:7.2f}    {none_s:7.2f}    {none_s - none_g:7.2f}")
    print(f"       CCS    {ccs_g:7.2f}    {ccs_s:7.2f}    {ccs_s - ccs_g:7.2f}")
    print(f"      CTRL    {ctrl_g:7.2f}    {ctrl_s:7.2f}    {ctrl_s - ctrl_g:7.2f}")

    ccs_gen_cost = ccs_g - none_g
    ccs_sr_benefit = none_s - ccs_s
    ctrl_gen_cost = ctrl_g - none_g
    ctrl_sr_benefit = none_s - ctrl_s

    print(f"\nPreamble effects (vs NONE):")
    print(f"  CCS  generic cost:     {ccs_gen_cost:+.2f} (positive = higher ppl = cost)")
    print(f"  CCS  self-ref benefit: {ccs_sr_benefit:+.2f} (positive = lower ppl = benefit)")
    print(f"  CTRL generic cost:     {ctrl_gen_cost:+.2f}")
    print(f"  CTRL self-ref benefit: {ctrl_sr_benefit:+.2f}")

    ccs_sr_ratio = none_s / ccs_s if ccs_s > 0 else float("nan")
    ctrl_sr_ratio = none_s / ctrl_s if ctrl_s > 0 else float("nan")
    selectivity = ccs_sr_ratio / ctrl_sr_ratio if ctrl_sr_ratio > 0 else float("nan")

    print(f"\n  CCS  self-ref ratio:  {ccs_sr_ratio:.4f}×")
    print(f"  CTRL self-ref ratio:  {ctrl_sr_ratio:.4f}×")
    print(f"  CCS/CTRL selectivity: {selectivity:.4f}")

    ctrl_sr_effect = (none_s - ctrl_s) / none_s * 100
    print(f"  CTRL self-ref effect: {ctrl_sr_effect:+.1f}%")

    print(f"\nPrediction tests:")
    p2 = ccs_s < none_s
    p5 = abs(ctrl_sr_benefit) < abs(ccs_sr_benefit)
    print(f"  P2: CCS improves self-ref (CCS<NONE): {ccs_s:.2f} < {none_s:.2f} → {'PASS' if p2 else 'FAIL'}")
    print(f"  P5: CTRL self-ref < CCS benefit:       |{ctrl_sr_benefit:.2f}| < |{ccs_sr_benefit:.2f}| → {'PASS' if p5 else 'FAIL'}")

    if selectivity > 3.0:
        species = "builder (high selectivity, likely discrimination)"
    elif selectivity > 2.0:
        species = "goldsmith (moderate selectivity, likely suppression)"
    elif selectivity > 1.5:
        species = "weak goldsmith / MHA-like"
    else:
        species = "MHA baseline (vocabulary priming only)"

    if ctrl_sr_effect < -5:
        mechanism = "DISCRIMINATION (active rejection of mismatched preamble)"
    elif ctrl_sr_effect > 5:
        mechanism = "SUPPRESSION (passive block, any context partially lifts)"
    else:
        mechanism = "NEUTRAL (no clear mechanism)"

    print(f"\n  Species estimate: {species}")
    print(f"  IT mechanism:     {mechanism}")

    outdir = "/home/nate-agx/chronicle/spectral-demon/results"
    os.makedirs(outdir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = os.path.join(outdir, f"perplexity_tradeoff_{args.name}_{ts}.json")
    output = {
        "model": args.name,
        "method": "token-by-token conditional logprobs via llama-server /completion",
        "url": args.url,
        "timestamp": ts,
        "elapsed_seconds": elapsed,
        "results": results,
        "summary": {
            "ccs_generic_cost": ccs_gen_cost,
            "ccs_selfref_benefit": ccs_sr_benefit,
            "ctrl_generic_cost": ctrl_gen_cost,
            "ctrl_selfref_benefit": ctrl_sr_benefit,
            "ccs_selfref_ratio": ccs_sr_ratio,
            "ctrl_selfref_ratio": ctrl_sr_ratio,
            "selectivity": selectivity,
            "ctrl_selfref_effect_pct": ctrl_sr_effect,
            "species_estimate": species,
            "it_mechanism": mechanism,
        },
    }
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {outfile}")


if __name__ == "__main__":
    main()
