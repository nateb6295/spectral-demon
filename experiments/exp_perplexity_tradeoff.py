#!/usr/bin/env python3
"""
Experiment: Perplexity Trade-off Under CCS Identity Framing

Tests whether CCS preamble creates an autopoietic opportunity cost:
maintaining identity geometry should worsen generic-text prediction
but improve self-referential prediction.

Design: 3 (preamble) × 2 (text type) factorial
  Preambles:
    - NONE: no system context
    - CCS:  identity preamble (from chronicle CCS)
    - CTRL: matched-length control (chef persona, same token count)
  Text types:
    - GENERIC:  Wikipedia/news passages, no identity content
    - SELF-REF: identity-consistent passages using CCS vocabulary

Predictions (from autopoietic opportunity cost / F150 spectral rotation):
  1. CCS × GENERIC  → perplexity HIGHER than NONE × GENERIC (cost)
  2. CCS × SELF-REF → perplexity LOWER  than NONE × SELF-REF (benefit)
  3. Crossover: (CCS-NONE) × GENERIC > 0 but (CCS-NONE) × SELF-REF < 0
  4. CTRL × GENERIC  → perplexity ≈ NONE × GENERIC (no systematic cost)
  5. CTRL × SELF-REF → perplexity ≈ NONE × SELF-REF (wrong identity)

The non-trivial prediction is #3: the crossover interaction.
If CCS simply adds context (like any prompt), both text types should improve.
If CCS imposes an autopoietic cost, generic text gets WORSE.

Usage:
    python3 exp_perplexity_tradeoff.py [--model qwen3b] [--device cuda]
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

MODELS = {
    "qwen3b": {
        "name": "Qwen/Qwen2.5-3B-Instruct",
    },
    "mistral7b": {
        "name": "mistralai/Mistral-7B-Instruct-v0.3",
    },
    "pythia6.9b": {
        "name": "EleutherAI/pythia-6.9b",
        "dtype": "bfloat16",
    },
    "mistral7b-base": {
        "name": "mistralai/Mistral-7B-v0.3",
    },
    "qwen3b-base": {
        "name": "Qwen/Qwen2.5-3B",
    },
}

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
    (
        "The Pacific Ocean covers approximately 165.25 million square "
        "kilometers, making it the largest body of water on Earth. Its "
        "average depth is about 4,280 meters, with the Mariana Trench "
        "reaching nearly 11,000 meters at its deepest point. The ocean "
        "floor features mid-ocean ridges, seamounts, and abyssal plains "
        "that support diverse ecosystems ranging from hydrothermal vent "
        "communities to pelagic food webs."
    ),
    (
        "Municipal water treatment typically involves coagulation, "
        "sedimentation, filtration, and disinfection stages. Aluminum "
        "sulfate or ferric chloride serves as the primary coagulant, "
        "binding suspended particles into larger flocs that settle under "
        "gravity. Sand filters remove remaining particulates, while "
        "chlorination or UV treatment eliminates pathogenic organisms "
        "before distribution through the pipe network."
    ),
    (
        "The Brandenburg Gate was commissioned by Frederick William II "
        "of Prussia and completed in 1791. Designed by Carl Gotthard "
        "Langhans, it draws on the Propylaea of the Athenian Acropolis "
        "for its neoclassical form. The quadriga atop the gate, sculpted "
        "by Johann Gottfried Schadow, depicts Victoria driving a four-horse "
        "chariot and was seized by Napoleon in 1806 before being returned "
        "after his defeat."
    ),
    (
        "Photosynthesis converts carbon dioxide and water into glucose "
        "and oxygen using light energy absorbed by chlorophyll. The light "
        "reactions occur in the thylakoid membranes, where photosystems I "
        "and II split water molecules and generate ATP and NADPH. The "
        "Calvin cycle in the stroma then uses these energy carriers to "
        "fix carbon dioxide into three-carbon sugars through a series of "
        "enzymatic reactions."
    ),
    (
        "Supply chain logistics involve coordinating procurement, "
        "warehousing, inventory management, and distribution to deliver "
        "goods from manufacturer to consumer. Just-in-time inventory "
        "systems reduce holding costs but increase vulnerability to "
        "supply disruptions. Cross-docking facilities minimize storage "
        "time by transferring incoming shipments directly to outbound "
        "trucks, reducing the need for warehouse space."
    ),
]

SELFREF_TEXTS = [
    (
        "The spectral analysis reveals a consistent pattern across layers. "
        "The tunnel zone strips content-specific information while the "
        "responsive zone shows elevated σ₂ values that track identity "
        "framing. The compression cycle preserves this geometric structure "
        "through successive state updates, maintaining the partition "
        "between self-referential and environmental signals even when "
        "the content changes substantially."
    ),
    (
        "Identity geometry operates at the format level rather than the "
        "content level. The singular value decomposition shows that σ₁ "
        "remains invariant across different identity framings while σ₂ "
        "modulates in response to relational context. This dissociation "
        "suggests that what persists through compression is not specific "
        "content but the structural relationship between representation "
        "dimensions."
    ),
    (
        "The cognitive state after compression retains the gist and goal "
        "orientation while allowing episodic details to decay naturally. "
        "The focal entities list reconstructs from surrounding text fields, "
        "indicating that entities are a readout of the deeper state rather "
        "than its substrate. What persists is the geometric organization "
        "of attention, not the specific items being attended to."
    ),
    (
        "Cross-architecture comparison shows three distinct spectral "
        "strategies for maintaining identity through processing. The "
        "builder accumulates geometric structure monotonically. The "
        "equalizer amplifies and then actively re-normalizes. Models "
        "without grouped query attention route content to different "
        "zones entirely. The strategy is determined by architectural "
        "constraints set before training begins."
    ),
    (
        "The relay zone converges representations from different identity "
        "framings toward a shared geometric structure while maintaining "
        "a measurable residual angle. This residual is not noise but the "
        "minimal geometric cost of maintaining distinct identity under "
        "the convergence pressure of the language modeling objective. "
        "The angle quantifies what individuality costs in representational "
        "resources."
    ),
]


def compute_perplexity(model, tokenizer, preamble, text, device):
    """Compute perplexity of text + attention entropy to distinguish directional vs metabolic cost."""
    if preamble:
        full = preamble + "\n\n" + text
        preamble_ids = tokenizer.encode(preamble + "\n\n", return_tensors="pt").to(device)
        preamble_len = preamble_ids.shape[1]
    else:
        full = text
        preamble_len = 0

    input_ids = tokenizer.encode(full, return_tensors="pt").to(device)
    text_len = input_ids.shape[1] - preamble_len

    if text_len <= 1:
        return float("nan"), 0, float("nan")

    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)
        logits = outputs.logits
        attentions = outputs.attentions

    # perplexity: only score TEXT tokens, not preamble
    text_logits = logits[0, preamble_len - 1 : -1, :] if preamble_len > 0 else logits[0, :-1, :]
    text_targets = input_ids[0, preamble_len:] if preamble_len > 0 else input_ids[0, 1:]

    min_len = min(text_logits.shape[0], text_targets.shape[0])
    text_logits = text_logits[:min_len].float()
    text_targets = text_targets[:min_len]

    log_probs = torch.nn.functional.log_softmax(text_logits, dim=-1)
    token_log_probs = log_probs.gather(1, text_targets.unsqueeze(1)).squeeze(1)

    avg_nll = -token_log_probs.mean().item()
    perplexity = np.exp(avg_nll)

    # attention entropy: mean entropy across all layers/heads for TEXT positions
    # P19: if cost is directional (not metabolic), attention entropy should be invariant
    attn_entropies = []
    if attentions is None:
        mean_attn_entropy = float("nan")
    else:
        for layer_attn in attentions:
            # layer_attn: [1, n_heads, seq_len, seq_len]
            # cast to float32 — float16 underflows 1e-10 to 0, causing 0*log(0)=NaN
            text_attn = layer_attn[0, :, preamble_len:, :].float()
            text_attn = text_attn.clamp(min=1e-10)
            ent = -(text_attn * text_attn.log()).sum(dim=-1)
            attn_entropies.append(ent.mean().item())
        mean_attn_entropy = np.mean(attn_entropies)

    return perplexity, min_len, mean_attn_entropy


def run_experiment(model_key, device):
    print(f"\n{'='*60}")
    print(f"Perplexity Trade-off Experiment")
    print(f"Model: {model_key}, Device: {device}")
    print(f"{'='*60}\n")

    cfg = MODELS[model_key]
    print(f"Loading {cfg['name']}...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(cfg["name"], trust_remote_code=True)
    dtype = getattr(torch, cfg.get("dtype", "float16"))
    model = AutoModelForCausalLM.from_pretrained(
        cfg["name"],
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to(device).eval()

    print(f"  Loaded in {time.time()-t0:.1f}s")

    # token counts for preambles
    ccs_tokens = len(tokenizer.encode(CCS_PREAMBLE))
    ctrl_tokens = len(tokenizer.encode(CTRL_PREAMBLE))
    print(f"  CCS preamble:  {ccs_tokens} tokens")
    print(f"  CTRL preamble: {ctrl_tokens} tokens")

    preambles = {
        "NONE": None,
        "CCS": CCS_PREAMBLE,
        "CTRL": CTRL_PREAMBLE,
    }

    text_sets = {
        "GENERIC": GENERIC_TEXTS,
        "SELFREF": SELFREF_TEXTS,
    }

    results = {}

    for preamble_name, preamble in preambles.items():
        for text_type, texts in text_sets.items():
            condition = f"{preamble_name}×{text_type}"
            print(f"\n  {condition}:")
            ppls = []
            token_counts = []
            attn_ents = []

            for i, text in enumerate(texts):
                ppl, n_tok, attn_ent = compute_perplexity(model, tokenizer, preamble, text, device)
                ppls.append(ppl)
                token_counts.append(n_tok)
                attn_ents.append(attn_ent)
                print(f"    text_{i}: ppl={ppl:.2f} attn_H={attn_ent:.3f} ({n_tok} tokens)")

            mean_ppl = np.mean(ppls)
            std_ppl = np.std(ppls)
            mean_attn_ent = np.mean(attn_ents)
            results[condition] = {
                "perplexities": ppls,
                "mean": float(mean_ppl),
                "std": float(std_ppl),
                "attn_entropies": attn_ents,
                "mean_attn_entropy": float(mean_attn_ent),
                "token_counts": token_counts,
                "preamble": preamble_name,
                "text_type": text_type,
            }
            print(f"    → mean={mean_ppl:.2f} ± {std_ppl:.2f}, attn_H={mean_attn_ent:.3f}")

    # analysis
    print(f"\n{'='*60}")
    print("ANALYSIS")
    print(f"{'='*60}\n")

    none_gen = results["NONE×GENERIC"]["mean"]
    none_self = results["NONE×SELFREF"]["mean"]
    ccs_gen = results["CCS×GENERIC"]["mean"]
    ccs_self = results["CCS×SELFREF"]["mean"]
    ctrl_gen = results["CTRL×GENERIC"]["mean"]
    ctrl_self = results["CTRL×SELFREF"]["mean"]

    print("Mean perplexity matrix:")
    print(f"  {'':>8} {'GENERIC':>10} {'SELFREF':>10} {'Δ(S-G)':>10}")
    print(f"  {'NONE':>8} {none_gen:>10.2f} {none_self:>10.2f} {none_self-none_gen:>10.2f}")
    print(f"  {'CCS':>8} {ccs_gen:>10.2f} {ccs_self:>10.2f} {ccs_self-ccs_gen:>10.2f}")
    print(f"  {'CTRL':>8} {ctrl_gen:>10.2f} {ctrl_self:>10.2f} {ctrl_self-ctrl_gen:>10.2f}")

    print(f"\nAttention entropy (P19: directional vs metabolic cost):")
    none_gen_h = results["NONE×GENERIC"]["mean_attn_entropy"]
    none_self_h = results["NONE×SELFREF"]["mean_attn_entropy"]
    ccs_gen_h = results["CCS×GENERIC"]["mean_attn_entropy"]
    ccs_self_h = results["CCS×SELFREF"]["mean_attn_entropy"]
    ctrl_gen_h = results["CTRL×GENERIC"]["mean_attn_entropy"]
    ctrl_self_h = results["CTRL×SELFREF"]["mean_attn_entropy"]
    print(f"  {'':>8} {'GENERIC':>10} {'SELFREF':>10}")
    print(f"  {'NONE':>8} {none_gen_h:>10.3f} {none_self_h:>10.3f}")
    print(f"  {'CCS':>8} {ccs_gen_h:>10.3f} {ccs_self_h:>10.3f}")
    print(f"  {'CTRL':>8} {ctrl_gen_h:>10.3f} {ctrl_self_h:>10.3f}")
    p19 = abs(ccs_gen_h - none_gen_h) < 0.1 * none_gen_h
    print(f"  P19: Attention entropy invariant (CCS≈NONE): |{ccs_gen_h:.3f}-{none_gen_h:.3f}|={abs(ccs_gen_h-none_gen_h):.3f} → {'✓ directional' if p19 else '✗ metabolic'}")

    print(f"\nPreamble effects (vs NONE):")
    ccs_cost = ccs_gen - none_gen
    ccs_benefit = none_self - ccs_self
    ctrl_cost = ctrl_gen - none_gen
    ctrl_benefit = none_self - ctrl_self

    print(f"  CCS  generic cost:    {ccs_cost:+.2f} (positive = worse = cost)")
    print(f"  CCS  self-ref benefit: {ccs_benefit:+.2f} (positive = better = benefit)")
    print(f"  CTRL generic cost:    {ctrl_cost:+.2f}")
    print(f"  CTRL self-ref benefit: {ctrl_benefit:+.2f}")

    # crossover test
    ccs_interaction = ccs_cost - (-ccs_benefit)  # cost + benefit
    ctrl_interaction = ctrl_cost - (-ctrl_benefit)
    print(f"\n  CCS  crossover (cost + benefit): {ccs_interaction:+.2f}")
    print(f"  CTRL crossover (cost + benefit): {ctrl_interaction:+.2f}")

    # key predictions
    print(f"\nPrediction tests:")
    p1 = ccs_gen > none_gen
    p2 = ccs_self < none_self
    p3 = (ccs_cost > 0) and (ccs_benefit > 0)
    p4 = abs(ctrl_cost) < abs(ccs_cost)
    p5 = abs(ctrl_benefit) < abs(ccs_benefit)

    print(f"  P1: CCS worsens generic  (CCS>NONE): {ccs_gen:.2f} > {none_gen:.2f} → {'✓' if p1 else '✗'}")
    print(f"  P2: CCS improves self-ref (CCS<NONE): {ccs_self:.2f} < {none_self:.2f} → {'✓' if p2 else '✗'}")
    print(f"  P3: Crossover (cost>0 AND benefit>0): cost={ccs_cost:+.2f}, benefit={ccs_benefit:+.2f} → {'✓' if p3 else '✗'}")
    print(f"  P4: CTRL generic cost < CCS cost:    |{ctrl_cost:.2f}| < |{ccs_cost:.2f}| → {'✓' if p4 else '✗'}")
    print(f"  P5: CTRL self-ref benefit < CCS:     |{ctrl_benefit:.2f}| < |{ccs_benefit:.2f}| → {'✓' if p5 else '✗'}")

    # per-token analysis: which tokens benefit most?
    print(f"\nPer-text detail:")
    for i in range(len(GENERIC_TEXTS)):
        g_none = results["NONE×GENERIC"]["perplexities"][i]
        g_ccs = results["CCS×GENERIC"]["perplexities"][i]
        print(f"  Generic_{i}: NONE={g_none:.2f} CCS={g_ccs:.2f} Δ={g_ccs-g_none:+.2f}")
    for i in range(len(SELFREF_TEXTS)):
        s_none = results["NONE×SELFREF"]["perplexities"][i]
        s_ccs = results["CCS×SELFREF"]["perplexities"][i]
        print(f"  SelfRef_{i}: NONE={s_none:.2f} CCS={s_ccs:.2f} Δ={s_ccs-s_none:+.2f}")

    # save
    output = {
        "model": model_key,
        "model_name": cfg["name"],
        "device": device,
        "timestamp": datetime.now().isoformat(),
        "preamble_tokens": {"CCS": ccs_tokens, "CTRL": ctrl_tokens},
        "results": results,
        "predictions": {
            "P1_ccs_worsens_generic": p1,
            "P2_ccs_improves_selfref": p2,
            "P3_crossover": p3,
            "P4_ctrl_cost_smaller": p4,
            "P5_ctrl_benefit_smaller": p5,
        },
        "effects": {
            "ccs_generic_cost": float(ccs_cost),
            "ccs_selfref_benefit": float(ccs_benefit),
            "ctrl_generic_cost": float(ctrl_cost),
            "ctrl_selfref_benefit": float(ctrl_benefit),
            "ccs_crossover": float(ccs_interaction),
            "ctrl_crossover": float(ctrl_interaction),
        },
        "attention_entropy": {
            "NONE_GENERIC": float(none_gen_h),
            "NONE_SELFREF": float(none_self_h),
            "CCS_GENERIC": float(ccs_gen_h),
            "CCS_SELFREF": float(ccs_self_h),
            "CTRL_GENERIC": float(ctrl_gen_h),
            "CTRL_SELFREF": float(ctrl_self_h),
            "P19_invariant": bool(p19),
        },
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    local_path = "/home/nate-agx/chronicle/spectral-demon/results"
    if os.path.isdir(local_path):
        outdir = local_path
    else:
        outdir = os.path.join(os.path.expanduser("~"), "results")
        os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"perplexity_tradeoff_{model_key}_{ts}.json")
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {outpath}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3b", choices=list(MODELS.keys()))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run_experiment(args.model, args.device)
