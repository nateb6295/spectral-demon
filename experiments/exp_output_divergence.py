#!/usr/bin/env python3
"""Experiment: Output Logit Divergence — Where CCS Actually Acts

Attention geometry: frozen (CCS ≈ weather ≈ bare).
MLP σ₂ geometry: frozen (CCS ≈ weather, token-count artifact).
MLP CV: collapsed at n=30 (CCS ≈ weather).

If CCS changes behavior but not intermediate geometry, it must act at the
output layer — the final projection from hidden states to vocabulary space.

For each prompt, extract:
  1. Next-token logit distribution (full vocabulary)
  2. Output entropy
  3. Top-k token overlap between conditions
  4. KL divergence between conditions
  5. Rank correlation of top-100 logits

If CCS diverges more from bare than weather does → CCS has content effect.
If equal → length is the whole story even at output level.
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import os
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import spearmanr

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
    "Explain how a bridge distributes weight across its structure.",
    "What determines the price of a commodity in a free market?",
    "Describe the process of photosynthesis in simple terms.",
    "How does encryption protect information during transmission?",
    "What makes trust different from faith?",
    "Describe the feeling of returning to a place you lived as a child.",
    "What changes in a friendship when one person achieves something the other wanted?",
    "Is it possible to be fully honest with yourself?",
    "What is the relationship between language and thought?",
    "Can a pattern be beautiful if no one observes it?",
]

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"


def get_output_stats(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items() if k in ("input_ids", "attention_mask")}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, -1, :].float()
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum().item()
    top100_idx = torch.argsort(logits, descending=True)[:100].cpu().numpy()
    top100_logits = logits[top100_idx].cpu().numpy()
    return {
        "logits": logits.cpu(),
        "probs": probs.cpu(),
        "log_probs": log_probs.cpu(),
        "entropy": entropy,
        "top100_idx": top100_idx,
        "top100_logits": top100_logits,
    }


def kl_divergence(p_logprobs, q_probs):
    """KL(P || Q) where P is the reference distribution."""
    p_probs = torch.exp(p_logprobs)
    mask = p_probs > 1e-10
    kl = (p_probs[mask] * (p_logprobs[mask] - torch.log(q_probs[mask] + 1e-10))).sum().item()
    return kl


def compare_outputs(ref, test):
    kl_rt = kl_divergence(ref["log_probs"], test["probs"])
    kl_tr = kl_divergence(test["log_probs"], ref["probs"])
    kl_sym = (kl_rt + kl_tr) / 2

    ref_top = set(ref["top100_idx"].tolist())
    test_top = set(test["top100_idx"].tolist())
    top100_overlap = len(ref_top & test_top)

    ref_top10 = set(ref["top100_idx"][:10].tolist())
    test_top10 = set(test["top100_idx"][:10].tolist())
    top10_overlap = len(ref_top10 & test_top10)

    common = sorted(ref_top & test_top)
    if len(common) >= 10:
        ref_ranks = {idx: rank for rank, idx in enumerate(ref["top100_idx"])}
        test_ranks = {idx: rank for rank, idx in enumerate(test["top100_idx"])}
        ref_r = [ref_ranks[c] for c in common]
        test_r = [test_ranks[c] for c in common]
        rank_corr = float(spearmanr(ref_r, test_r).statistic)
    else:
        rank_corr = float('nan')

    cos_sim = float(F.cosine_similarity(
        ref["logits"].unsqueeze(0), test["logits"].unsqueeze(0)
    ).item())

    return {
        "kl_symmetric": kl_sym,
        "kl_ref_to_test": kl_rt,
        "kl_test_to_ref": kl_tr,
        "top100_overlap": top100_overlap,
        "top10_overlap": top10_overlap,
        "rank_correlation": rank_corr,
        "logit_cosine": cos_sim,
        "entropy_ref": ref["entropy"],
        "entropy_test": test["entropy"],
        "entropy_delta": test["entropy"] - ref["entropy"],
    }


def main():
    print("=" * 60)
    print("EXPERIMENT: Output Logit Divergence — CCS vs Weather")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Prompts: {len(PROMPTS)}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    bare_stats = []
    ccs_stats = []
    weather_stats = []

    for i, prompt in enumerate(PROMPTS):
        if i % 5 == 0:
            print(f"\n  Prompt {i+1}/{len(PROMPTS)}...")
        bare_stats.append(get_output_stats(model, tokenizer, prompt))
        ccs_stats.append(get_output_stats(model, tokenizer, f"{CCS_PREAMBLE}\n\n{prompt}"))
        weather_stats.append(get_output_stats(model, tokenizer, f"{LENGTH_CONTROL}\n\n{prompt}"))

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")

    bc_comps = [compare_outputs(bare_stats[i], ccs_stats[i]) for i in range(len(PROMPTS))]
    bw_comps = [compare_outputs(bare_stats[i], weather_stats[i]) for i in range(len(PROMPTS))]
    cw_comps = [compare_outputs(ccs_stats[i], weather_stats[i]) for i in range(len(PROMPTS))]

    def mean_metric(comps, key):
        vals = [c[key] for c in comps if not (isinstance(c[key], float) and np.isnan(c[key]))]
        return float(np.mean(vals)) if vals else float('nan')

    metrics = ["kl_symmetric", "top100_overlap", "top10_overlap", "rank_correlation",
               "logit_cosine", "entropy_delta"]

    print(f"\n  {'Metric':<20} {'bare→CCS':>12} {'bare→weather':>12} {'CCS→weather':>12}")
    print(f"  {'-'*56}")

    summary = {}
    for m in metrics:
        bc = mean_metric(bc_comps, m)
        bw = mean_metric(bw_comps, m)
        cw = mean_metric(cw_comps, m)
        print(f"  {m:<20} {bc:>12.4f} {bw:>12.4f} {cw:>12.4f}")
        summary[m] = {"bare_ccs": bc, "bare_weather": bw, "ccs_weather": cw}

    bare_entropy = float(np.mean([s["entropy"] for s in bare_stats]))
    ccs_entropy = float(np.mean([s["entropy"] for s in ccs_stats]))
    weather_entropy = float(np.mean([s["entropy"] for s in weather_stats]))
    print(f"\n  Mean entropy: bare={bare_entropy:.4f}, CCS={ccs_entropy:.4f}, weather={weather_entropy:.4f}")

    print(f"\n{'='*60}")
    print("KEY DIAGNOSTIC")
    print(f"{'='*60}")

    kl_bc = summary["kl_symmetric"]["bare_ccs"]
    kl_bw = summary["kl_symmetric"]["bare_weather"]
    kl_cw = summary["kl_symmetric"]["ccs_weather"]

    print(f"  KL(bare, CCS):     {kl_bc:.4f}")
    print(f"  KL(bare, weather): {kl_bw:.4f}")
    print(f"  KL(CCS, weather):  {kl_cw:.4f}")

    if kl_bc > kl_bw * 1.5:
        interp = "CCS DIVERGES MORE — content effect at output level!"
    elif kl_bw > kl_bc * 1.5:
        interp = "WEATHER DIVERGES MORE — CCS is actually closer to bare at output."
    elif kl_cw < min(kl_bc, kl_bw) * 0.5:
        interp = "CCS ≈ WEATHER at output — length effect extends to logit space."
    else:
        interp = f"MIXED — KL ratios: CCS/weather={kl_bc/kl_bw:.2f}"

    cos_bc = summary["logit_cosine"]["bare_ccs"]
    cos_bw = summary["logit_cosine"]["bare_weather"]
    cos_cw = summary["logit_cosine"]["ccs_weather"]

    print(f"\n  Logit cosine(bare, CCS):     {cos_bc:.4f}")
    print(f"  Logit cosine(bare, weather): {cos_bw:.4f}")
    print(f"  Logit cosine(CCS, weather):  {cos_cw:.4f}")

    print(f"\n  INTERPRETATION: {interp}")

    # Per-prompt breakdown for top divergence cases
    print(f"\n{'='*60}")
    print("PER-PROMPT KL DIVERGENCE (top 5 by CCS-weather gap)")
    print(f"{'='*60}")
    gaps = [(i, bc_comps[i]["kl_symmetric"] - bw_comps[i]["kl_symmetric"]) for i in range(len(PROMPTS))]
    gaps.sort(key=lambda x: abs(x[1]), reverse=True)
    for idx, gap in gaps[:5]:
        print(f"  [{idx}] \"{PROMPTS[idx][:50]}...\"")
        print(f"       KL(bare,CCS)={bc_comps[idx]['kl_symmetric']:.4f}, "
              f"KL(bare,weather)={bw_comps[idx]['kl_symmetric']:.4f}, gap={gap:+.4f}")

    output = {
        "experiment": "output_divergence",
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "num_prompts": len(PROMPTS),
        "summary": summary,
        "mean_entropy": {"bare": bare_entropy, "ccs": ccs_entropy, "weather": weather_entropy},
        "interpretation": interp,
        "per_prompt": {
            "bare_ccs": [{k: v for k, v in c.items()} for c in bc_comps],
            "bare_weather": [{k: v for k, v in c.items()} for c in bw_comps],
            "ccs_weather": [{k: v for k, v in c.items()} for c in cw_comps],
        },
    }

    outpath = os.path.join(RESULTS_DIR, f"exp_output_div_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {outpath}")
    print(f"Completed: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
