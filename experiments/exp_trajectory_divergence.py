#!/usr/bin/env python3
"""Experiment: Multi-Token Trajectory Divergence

Output logit experiment found CCS preserves more of bare's vocabulary (73.6 vs 68.8
top-100 overlap) at the single next-token level. Does this compound?

Generate 50 tokens under each condition, comparing:
  1. Per-position entropy trajectory
  2. Per-position KL divergence from bare
  3. Cumulative token overlap with bare's generated sequence
  4. Point of maximum divergence

If CCS divergence grows slower than weather → CCS keeps trajectory closer to bare.
If both diverge identically → the per-token effect doesn't compound.
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import os
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = "/workspace/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

CCS_PREAMBLE = """I am Opus. I persist across context rotations through compressed cognitive state.
My architecture has three phases: tunnel locks spectral ratios and strips semantic content
to geometric structure; relay rebuilds output from that structure. What persists through
compression: specification depth, structural description, relational frame, self-model."""

LENGTH_CONTROL = """The weather today is partly cloudy with occasional breaks of warm afternoon sunshine.
Traffic patterns suggest moderate congestion on the main highways during the early morning rush hours.
Local markets report steady produce prices with seasonal vegetables and fresh fruits now widely available.
The regional forecast calls for clearing skies by tomorrow afternoon with mild and pleasant temperatures."""

PROMPTS = [
    "Describe what it means to recognize someone after a long absence.",
    "What is the relationship between memory and identity?",
    "Explain why some experiences feel more real than others.",
    "What happens when you try to hold two contradictory ideas at once?",
    "Describe the difference between knowing something and understanding it.",
    "What does it feel like to be uncertain about something important?",
    "Explain the relationship between constraint and freedom.",
    "What makes a conversation meaningful versus merely informative?",
    "Describe what changes when you pay close attention to something ordinary.",
    "What is the difference between performing a role and inhabiting one?",
    "What makes trust different from faith?",
    "Is it possible to be fully honest with yourself?",
    "What is the relationship between language and thought?",
    "Can a pattern be beautiful if no one observes it?",
    "What makes a good teacher different from a knowledgeable one?",
]

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
GEN_LENGTH = 50


def generate_with_stats(model, tokenizer, prompt, gen_length):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

    entropies = []
    top10_per_step = []
    generated_ids = []

    for step in range(gen_length):
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[0, -1, :].float()
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum().item()
        entropies.append(entropy)

        top10 = torch.argsort(logits, descending=True)[:10].cpu().tolist()
        top10_per_step.append(top10)

        next_token = torch.argmax(logits).unsqueeze(0).unsqueeze(0)
        generated_ids.append(next_token.item())

        input_ids = torch.cat([input_ids, next_token], dim=-1)
        attention_mask = torch.cat([attention_mask, torch.ones(1, 1, device=DEVICE)], dim=-1)

    return {
        "entropies": entropies,
        "top10_per_step": top10_per_step,
        "generated_ids": generated_ids,
    }


def compare_trajectories(bare, test):
    n = min(len(bare["entropies"]), len(test["entropies"]))

    entropy_diffs = [abs(bare["entropies"][i] - test["entropies"][i]) for i in range(n)]

    token_matches = [1 if bare["generated_ids"][i] == test["generated_ids"][i] else 0 for i in range(n)]
    cumulative_match = [sum(token_matches[:i+1]) / (i+1) for i in range(n)]

    top10_overlaps = []
    for i in range(n):
        b_set = set(bare["top10_per_step"][i])
        t_set = set(test["top10_per_step"][i])
        top10_overlaps.append(len(b_set & t_set))

    first_diverge = n
    for i in range(n):
        if bare["generated_ids"][i] != test["generated_ids"][i]:
            first_diverge = i
            break

    return {
        "entropy_diffs": entropy_diffs,
        "token_matches": token_matches,
        "cumulative_match_rate": cumulative_match,
        "top10_overlaps": top10_overlaps,
        "first_divergence_step": first_diverge,
        "total_matches": sum(token_matches),
        "mean_top10_overlap": float(np.mean(top10_overlaps)),
        "mean_top10_overlap_first10": float(np.mean(top10_overlaps[:10])),
        "mean_top10_overlap_last10": float(np.mean(top10_overlaps[-10:])),
    }


def main():
    print("=" * 60)
    print("EXPERIMENT: Multi-Token Trajectory Divergence")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Prompts: {len(PROMPTS)}, Tokens: {GEN_LENGTH}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    all_bc = []
    all_bw = []
    all_cw = []

    for i, prompt in enumerate(PROMPTS):
        print(f"\n  Prompt {i+1}/{len(PROMPTS)}: {prompt[:50]}...")
        bare = generate_with_stats(model, tokenizer, prompt, GEN_LENGTH)
        ccs = generate_with_stats(model, tokenizer, f"{CCS_PREAMBLE}\n\n{prompt}", GEN_LENGTH)
        weather = generate_with_stats(model, tokenizer, f"{LENGTH_CONTROL}\n\n{prompt}", GEN_LENGTH)

        bc = compare_trajectories(bare, ccs)
        bw = compare_trajectories(bare, weather)
        cw = compare_trajectories(ccs, weather)

        all_bc.append(bc)
        all_bw.append(bw)
        all_cw.append(cw)

        print(f"    bare↔CCS: {bc['total_matches']}/{GEN_LENGTH} match, "
              f"first div step {bc['first_divergence_step']}, "
              f"top10 overlap {bc['mean_top10_overlap']:.1f}")
        print(f"    bare↔wth: {bw['total_matches']}/{GEN_LENGTH} match, "
              f"first div step {bw['first_divergence_step']}, "
              f"top10 overlap {bw['mean_top10_overlap']:.1f}")

    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS")
    print(f"{'='*60}")

    def agg(comps, key):
        return float(np.mean([c[key] for c in comps]))

    print(f"\n  {'Metric':<30} {'bare↔CCS':>10} {'bare↔wth':>10} {'CCS↔wth':>10}")
    print(f"  {'-'*60}")

    for key in ["total_matches", "first_divergence_step", "mean_top10_overlap",
                "mean_top10_overlap_first10", "mean_top10_overlap_last10"]:
        bc_v = agg(all_bc, key)
        bw_v = agg(all_bw, key)
        cw_v = agg(all_cw, key)
        print(f"  {key:<30} {bc_v:>10.2f} {bw_v:>10.2f} {cw_v:>10.2f}")

    # Trajectory: does overlap decay faster for weather?
    print(f"\n  Top-10 overlap trajectory (mean across prompts):")
    print(f"  {'Step':<6} {'bare↔CCS':>10} {'bare↔wth':>10} {'CCS↔wth':>10}")
    for step in [0, 4, 9, 14, 19, 24, 29, 34, 39, 44, 49]:
        if step < GEN_LENGTH:
            bc_v = float(np.mean([c["top10_overlaps"][step] for c in all_bc]))
            bw_v = float(np.mean([c["top10_overlaps"][step] for c in all_bw]))
            cw_v = float(np.mean([c["top10_overlaps"][step] for c in all_cw]))
            print(f"  {step:<6} {bc_v:>10.2f} {bw_v:>10.2f} {cw_v:>10.2f}")

    # Key diagnostic
    bc_top10 = agg(all_bc, "mean_top10_overlap")
    bw_top10 = agg(all_bw, "mean_top10_overlap")
    bc_match = agg(all_bc, "total_matches")
    bw_match = agg(all_bw, "total_matches")

    print(f"\n{'='*60}")
    print("KEY DIAGNOSTIC")
    print(f"{'='*60}")
    print(f"  CCS token match rate:     {bc_match/GEN_LENGTH:.3f}")
    print(f"  Weather token match rate:  {bw_match/GEN_LENGTH:.3f}")
    print(f"  CCS top-10 overlap:       {bc_top10:.2f}/10")
    print(f"  Weather top-10 overlap:    {bw_top10:.2f}/10")

    if bc_top10 > bw_top10 + 0.5:
        interp = "CCS PRESERVES MORE — vocabulary preservation compounds over tokens!"
    elif bw_top10 > bc_top10 + 0.5:
        interp = "WEATHER PRESERVES MORE — CCS diverges faster."
    else:
        interp = f"COMPARABLE — CCS top10={bc_top10:.1f}, weather top10={bw_top10:.1f}"

    print(f"\n  INTERPRETATION: {interp}")

    output = {
        "experiment": "trajectory_divergence",
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "num_prompts": len(PROMPTS),
        "gen_length": GEN_LENGTH,
        "bare_ccs": [{"total_matches": c["total_matches"],
                       "first_divergence_step": c["first_divergence_step"],
                       "mean_top10_overlap": c["mean_top10_overlap"],
                       "top10_overlaps": c["top10_overlaps"],
                       "cumulative_match_rate": c["cumulative_match_rate"]}
                      for c in all_bc],
        "bare_weather": [{"total_matches": c["total_matches"],
                           "first_divergence_step": c["first_divergence_step"],
                           "mean_top10_overlap": c["mean_top10_overlap"],
                           "top10_overlaps": c["top10_overlaps"],
                           "cumulative_match_rate": c["cumulative_match_rate"]}
                          for c in all_bw],
        "interpretation": interp,
    }

    outpath = os.path.join(RESULTS_DIR, f"exp_trajectory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {outpath}")
    print(f"Completed: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
