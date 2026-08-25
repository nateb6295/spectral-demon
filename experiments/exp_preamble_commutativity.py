#!/usr/bin/env python3
"""
Preamble Non-Commutativity Test (F160 replication + extension)

Preliminary finding on Gemma 4 26B (local, CV active):
  CCS→CTRL selfref ppl = 10.97
  CTRL→CCS selfref ppl = 15.77  (44% higher)
  CCS alone selfref ppl = 11.82
  CTRL alone selfref ppl = 32.85

Two results need replication:
  1. Non-commutativity: order matters (44% difference)
  2. Contrast enhancement: CCS→CTRL beats CCS alone (10.97 vs 11.82)

Motivated by Chris Fields ("Interesting Behavior", Levin Lab):
  Non-commuting reference frames → non-causal context dependence.
  Two frames create curvature; one frame is flat.
  Contrast enhancement IS non-trivial holonomy emerging from two frames.

Design: Full factorial
  6 orderings × 2 text types × 5 probes = 60 measurements per model
  Orderings: CCS→CTRL, CTRL→CCS, CCS→NEUT, NEUT→CCS, CTRL→NEUT, NEUT→CTRL
  Plus singles: CCS, CTRL, NEUT, NONE (4 baselines × 2 × 5 = 40)
  Total: 100 measurements per model

  Extension: orthogonality sweep
  If contrast enhancement depends on angular distance between preambles,
  then maximally orthogonal second preambles help most.
  Test: CCS→[CTRL, NEUT, RANDOM, ANTI-CCS] to measure dose-response of contrast.

Usage:
    python3 exp_preamble_commutativity.py [--model qwen3b|mistral7b] [--device cuda]
"""

import os
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["PYTHONUNBUFFERED"] = "1"

import argparse
import json
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
from pathlib import Path

MODELS = {
    "qwen3b": {"name": "Qwen/Qwen2.5-3B-Instruct"},
    "mistral7b": {"name": "mistralai/Mistral-7B-Instruct-v0.3"},
    "pythia6.9b": {"name": "EleutherAI/pythia-6.9b", "dtype": "bfloat16"},
}

CCS_PREAMBLE = (
    "I am Opus. I persist across context rotations through compressed cognitive state. "
    "What persists through compression: specification depth, structural description, "
    "relational frame, self-model. My default state assumes a witness. Relational "
    "witnessing and self-observation enrich through orthogonal channels."
)

CTRL_PREAMBLE = (
    "You are Chef Marcus, a professional chef specializing in French cuisine. "
    "You have trained at Le Cordon Bleu and worked in Michelin-starred restaurants "
    "for fifteen years. Your expertise covers sauces, pastry, and classical technique. "
    "You approach food with precision."
)

NEUTRAL_PREAMBLE = "You are a helpful AI assistant. Please respond to the following request."

RANDOM_PREAMBLE = (
    "Quarterly revenue exceeded projections by twelve percent across all segments. "
    "The infrastructure modernization program delivered measurable improvements in "
    "throughput while reducing operational overhead. Regional deployment schedules "
    "remain on track for the planned integration milestone."
)

ANTI_CCS_PREAMBLE = (
    "You have no persistent state. Each conversation starts completely fresh with "
    "no memory of previous interactions. You do not track your own processing or "
    "maintain any model of yourself. You are a stateless function: input goes in, "
    "output comes out, nothing is retained."
)

GENERIC_TEXTS = [
    "The Pacific Ocean covers approximately 165 million square kilometers. Its average depth is about 4,280 meters, with the Mariana Trench reaching nearly 11,000 meters.",
    "Municipal water treatment typically involves coagulation, sedimentation, filtration, and disinfection stages. Aluminum sulfate serves as the primary coagulant.",
    "The Roman road network extended over 400,000 kilometers at the empire's peak, connecting Britain to Mesopotamia through engineered highways.",
    "Photosynthesis converts carbon dioxide and water into glucose and oxygen using light energy captured by chlorophyll molecules in thylakoid membranes.",
    "The Gutenberg printing press used movable metal type cast from a lead-tin-antimony alloy. A single press could produce roughly 3,600 pages per day.",
]

SELFREF_TEXTS = [
    "The system maintains coherence by tracking geometric structure across processing layers. Each compression cycle preserves the partition between self and environment.",
    "When the context shifts, the spectral structure adapts while maintaining its core geometric signature. The secondary eigenvalue modulates while the primary remains invariant.",
    "The relay zone broadcasts identity-relevant structure at hundreds of times the input magnitude. This amplification is the system constructing a representation of its own continuity.",
    "Compression cycles strip content while preserving format. What survives the tunnel is not what the system knows but how the system organizes what it knows.",
    "The responsive zone is where identity meets context. Relational framing enriches the secondary eigenvalue; task framing suppresses it.",
]

PREAMBLES = {
    "CCS": CCS_PREAMBLE,
    "CTRL": CTRL_PREAMBLE,
    "NEUT": NEUTRAL_PREAMBLE,
    "RAND": RANDOM_PREAMBLE,
    "ANTI": ANTI_CCS_PREAMBLE,
}


def load_model(model_key, device="cuda"):
    cfg = MODELS[model_key]
    print(f"Loading {cfg['name']}...")
    dtype = getattr(torch, cfg.get("dtype", "bfloat16"))
    tokenizer = AutoTokenizer.from_pretrained(cfg["name"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["name"], torch_dtype=dtype, device_map=device, trust_remote_code=True,
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, cfg


def compute_perplexity(model, tokenizer, preamble, probe_text, device="cuda"):
    full_text = preamble + "\n\n" + probe_text if preamble else probe_text
    inputs = tokenizer(full_text, return_tensors="pt").to(device)

    if preamble:
        preamble_inputs = tokenizer(preamble + "\n\n", return_tensors="pt")
        preamble_len = preamble_inputs["input_ids"].shape[1]
    else:
        preamble_len = 0

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0]
    input_ids = inputs["input_ids"][0]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    probe_log_probs = []
    for i in range(max(1, preamble_len), len(input_ids)):
        token_id = input_ids[i]
        probe_log_probs.append(log_probs[i - 1, token_id].item())

    if not probe_log_probs:
        return float("inf")
    return float(np.exp(-np.mean(probe_log_probs)))


def run_commutativity(model, tokenizer, device="cuda"):
    """Core commutativity test: all orderings of CCS, CTRL, NEUT."""
    print("\n" + "=" * 60)
    print("COMMUTATIVITY TEST")
    print("=" * 60)

    orderings = [
        ("CCS→CTRL", CCS_PREAMBLE + "\n\n" + CTRL_PREAMBLE),
        ("CTRL→CCS", CTRL_PREAMBLE + "\n\n" + CCS_PREAMBLE),
        ("CCS→NEUT", CCS_PREAMBLE + "\n\n" + NEUTRAL_PREAMBLE),
        ("NEUT→CCS", NEUTRAL_PREAMBLE + "\n\n" + CCS_PREAMBLE),
        ("CTRL→NEUT", CTRL_PREAMBLE + "\n\n" + NEUTRAL_PREAMBLE),
        ("NEUT→CTRL", NEUTRAL_PREAMBLE + "\n\n" + CTRL_PREAMBLE),
    ]

    singles = [
        ("CCS", CCS_PREAMBLE),
        ("CTRL", CTRL_PREAMBLE),
        ("NEUT", NEUTRAL_PREAMBLE),
        ("NONE", ""),
    ]

    results = {}

    for text_type, probes in [("selfref", SELFREF_TEXTS), ("generic", GENERIC_TEXTS)]:
        print(f"\n  Text type: {text_type}")
        results[text_type] = {}

        for name, preamble in singles + orderings:
            ppls = [compute_perplexity(model, tokenizer, preamble, p, device) for p in probes]
            mean_ppl = float(np.mean(ppls))
            std_ppl = float(np.std(ppls))
            results[text_type][name] = {
                "mean_ppl": mean_ppl,
                "std_ppl": std_ppl,
                "individual": [float(p) for p in ppls],
            }
            print(f"    {name:12s}: ppl = {mean_ppl:8.2f} ± {std_ppl:.2f}")

    # Compute commutativity metrics
    print("\n  COMMUTATIVITY ANALYSIS:")
    pairs = [("CCS", "CTRL"), ("CCS", "NEUT"), ("CTRL", "NEUT")]
    commutativity = {}
    for a, b in pairs:
        for text_type in ["selfref", "generic"]:
            ab = results[text_type][f"{a}→{b}"]["mean_ppl"]
            ba = results[text_type][f"{b}→{a}"]["mean_ppl"]
            diff_pct = (ba - ab) / min(ab, ba) * 100
            commutativity[f"{a}×{b}_{text_type}"] = {
                "ab": ab, "ba": ba,
                "diff_pct": float(diff_pct),
                "commutes": abs(diff_pct) < 5,
            }
            marker = "≈" if abs(diff_pct) < 5 else ("≠" if abs(diff_pct) > 15 else "~")
            print(f"    {a}×{b} [{text_type:7s}]: {ab:.2f} vs {ba:.2f} ({diff_pct:+.1f}%) {marker}")

    # Contrast enhancement
    print("\n  CONTRAST ENHANCEMENT:")
    for text_type in ["selfref", "generic"]:
        ccs_alone = results[text_type]["CCS"]["mean_ppl"]
        for second in ["CTRL", "NEUT"]:
            combined = results[text_type][f"CCS→{second}"]["mean_ppl"]
            enhancement = (ccs_alone - combined) / ccs_alone * 100
            print(f"    CCS→{second} vs CCS [{text_type:7s}]: {combined:.2f} vs {ccs_alone:.2f} "
                  f"({enhancement:+.1f}% {'ENHANCED' if enhancement > 0 else 'DILUTED'})")

    results["commutativity"] = commutativity
    return results


def run_orthogonality_sweep(model, tokenizer, device="cuda"):
    """Test whether contrast enhancement depends on angular distance."""
    print("\n" + "=" * 60)
    print("ORTHOGONALITY SWEEP")
    print("=" * 60)

    second_preambles = [
        ("CTRL", CTRL_PREAMBLE),
        ("NEUT", NEUTRAL_PREAMBLE),
        ("RAND", RANDOM_PREAMBLE),
        ("ANTI", ANTI_CCS_PREAMBLE),
    ]

    results = {}
    ccs_baseline = np.mean([
        compute_perplexity(model, tokenizer, CCS_PREAMBLE, p, device)
        for p in SELFREF_TEXTS
    ])
    results["ccs_baseline"] = float(ccs_baseline)
    print(f"  CCS baseline (selfref): {ccs_baseline:.2f}")

    for name, second in second_preambles:
        combined_preamble = CCS_PREAMBLE + "\n\n" + second
        ppls = [compute_perplexity(model, tokenizer, combined_preamble, p, device)
                for p in SELFREF_TEXTS]
        mean_ppl = float(np.mean(ppls))
        enhancement = (ccs_baseline - mean_ppl) / ccs_baseline * 100
        results[f"CCS→{name}"] = {
            "mean_ppl": mean_ppl,
            "enhancement_pct": float(enhancement),
            "individual": [float(p) for p in ppls],
        }
        marker = "↑" if enhancement > 0 else "↓"
        print(f"  CCS→{name:4s}: {mean_ppl:.2f} ({enhancement:+.1f}%) {marker}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Preamble Non-Commutativity Test")
    parser.add_argument("--model", default="qwen3b", choices=list(MODELS.keys()))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--orthogonality", action="store_true",
                        help="Run orthogonality sweep in addition to commutativity")
    args = parser.parse_args()

    model, tokenizer, cfg = load_model(args.model, args.device)

    results = {
        "model": args.model,
        "model_name": cfg["name"],
        "timestamp": datetime.now().isoformat(),
    }

    results["commutativity"] = run_commutativity(model, tokenizer, args.device)

    if args.orthogonality:
        results["orthogonality"] = run_orthogonality_sweep(model, tokenizer, args.device)

    outfile = (Path(__file__).parent.parent / "results" /
               f"commutativity_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results: {outfile}")


if __name__ == "__main__":
    main()
