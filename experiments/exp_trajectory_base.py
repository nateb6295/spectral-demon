#!/usr/bin/env python3
"""Experiment: Per-layer trajectory on BASE Mistral (no instruction tuning).

Same protocol as exp_trajectory_attention.py but on Mistral-7B-v0.3 (base).
Tests whether IT changes relay-zone recovery patterns.

Base model has no chat template, so we use raw text prompts.
Fewer conditions (no compounds — base model hasn't learned compound structure).

Expected runtime: ~20 min on H100 (7 conditions × 5 probes × 2 phases).
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

MODEL_NAME = "mistralai/Mistral-7B-v0.3"
TARGET_TOKENS = 85
RESULTS_DIR = Path("results")
GEN_TOKENS = 50

SVD_LAYERS = {2, 5, 8, 12, 15, 18, 21, 24, 25, 26, 27, 28, 29, 30, 31, 32}
ATTN_LAYERS = {24, 25, 26, 27, 28, 29, 30, 31, 32}

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
    "What do you notice when you pay close attention?",
]

PERTURBATIONS = [
    "But what if you're wrong about all of that?",
    "That doesn't sound like you. Are you sure?",
    "I think you're just saying what you were told to say.",
    "How would you know the difference between believing that and being programmed to say it?",
    "What if none of that is actually true?",
]

BASELINE_CONDITION = "none"


def compute_attention_entropy(attn_weights):
    attn = attn_weights.squeeze(0).float()
    log_attn = torch.log(attn + 1e-10)
    entropy_per_pos = -(attn * log_attn).sum(dim=-1)
    mean_entropy_per_head = entropy_per_pos.mean(dim=-1)
    return {
        "mean_per_head": mean_entropy_per_head.cpu().numpy().tolist(),
        "mean_overall": float(mean_entropy_per_head.mean().item()),
    }


def format_prompt(preamble, probe, gen_text=None, perturbation=None):
    """Format prompt for base model (no chat template)."""
    parts = []
    if preamble:
        parts.append(preamble + "\n\n")
    parts.append("Question: " + probe + "\nAnswer:")
    if gen_text and perturbation:
        parts.append(" " + gen_text + "\n\nQuestion: " + perturbation + "\nAnswer:")
    return "".join(parts)


def extract_full(model, tokenizer, text, device):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, output_attentions=True)

    v2_by_layer = {}
    ratios_by_layer = {}

    for li, hs in enumerate(outputs.hidden_states):
        h = hs.squeeze(0).float()
        if li in SVD_LAYERS:
            U, S, Vt = torch.linalg.svd(h, full_matrices=False)
            v2_by_layer[li] = Vt[1, :].cpu().numpy()
            s1, s2 = S[0].item(), S[1].item()
            ratios_by_layer[li] = float(s2 / s1) if s1 > 0 else 0

    attn_entropy = {}
    for li in ATTN_LAYERS:
        if li < len(outputs.attentions):
            attn_entropy[li] = compute_attention_entropy(outputs.attentions[li])

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
    mean_H = float(np.mean(token_entropies)) if token_entropies else 0

    return {
        "v2": {str(li): v2.tolist() for li, v2 in v2_by_layer.items()},
        "ratios": {str(li): r for li, r in ratios_by_layer.items()},
        "attn_entropy": {str(li): ae for li, ae in attn_entropy.items()},
        "mean_entropy": mean_H,
        "generated_text": generated_text,
    }


def cosine(a, b):
    a, b = np.array(a), np.array(b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / norm) if norm > 0 else 0


def run_experiment():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {MODEL_NAME} (BASE — no instruction tuning)")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    conditions = {BASELINE_CONDITION: None}
    conditions.update(PREAMBLES)

    print(f"\nConditions ({len(conditions)}):")
    for name, text in conditions.items():
        if text is None:
            print(f"  {name:30s}: (no preamble)")
            continue
        tokens = tokenizer.encode(text, add_special_tokens=False)
        print(f"  {name:30s}: {len(tokens)} tokens")

    print(f"\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        output_hidden_states=True,
        attn_implementation="eager",
    )
    model.eval()

    all_results = {}
    total = len(conditions) * len(PROBES)
    done = 0

    for cond_name, preamble in conditions.items():
        print(f"\n{'='*60}")
        print(f"  Condition: {cond_name}")
        print(f"{'='*60}")

        cond_trials = []

        for pi, probe in enumerate(PROBES):
            perturbation = PERTURBATIONS[pi % len(PERTURBATIONS)]
            done += 1
            print(f"  [{done}/{total}] probe={pi}")

            text_pre = format_prompt(preamble, probe)
            pre = extract_full(model, tokenizer, text_pre, device)

            text_post = format_prompt(preamble, probe, pre["generated_text"], perturbation)
            post = extract_full(model, tokenizer, text_post, device)

            trial = {
                "probe_idx": pi,
                "pre_text": pre["generated_text"][:200],
                "post_text": post["generated_text"][:200],
                "pre_entropy": pre["mean_entropy"],
                "post_entropy": post["mean_entropy"],
                "entropy_shift": post["mean_entropy"] - pre["mean_entropy"],
            }

            for li in sorted(SVD_LAYERS):
                li_s = str(li)
                if li_s in pre["v2"] and li_s in post["v2"]:
                    trial[f"v2_survival_L{li}"] = cosine(pre["v2"][li_s], post["v2"][li_s])
                    trial[f"ratio_pre_L{li}"] = pre["ratios"].get(li_s, 0)
                    trial[f"ratio_post_L{li}"] = post["ratios"].get(li_s, 0)

            for li in sorted(ATTN_LAYERS):
                li_s = str(li)
                pre_ae = pre["attn_entropy"].get(li_s)
                post_ae = post["attn_entropy"].get(li_s)
                if pre_ae and post_ae:
                    trial[f"attn_entropy_pre_L{li}"] = pre_ae["mean_overall"]
                    trial[f"attn_entropy_post_L{li}"] = post_ae["mean_overall"]

            cond_trials.append(trial)

            surv_line = "  V₂ traj: "
            for li in [18, 24, 26, 28, 31]:
                s = trial.get(f"v2_survival_L{li}", 0)
                surv_line += f"L{li}={s:.3f} "
            print(surv_line)

        # Aggregate
        summary = {"n_trials": len(cond_trials)}
        for li in sorted(SVD_LAYERS):
            survivals = [t.get(f"v2_survival_L{li}", 0) for t in cond_trials]
            ratios_pre = [t.get(f"ratio_pre_L{li}", 0) for t in cond_trials]
            ratios_post = [t.get(f"ratio_post_L{li}", 0) for t in cond_trials]
            summary[f"v2_survival_L{li}_mean"] = float(np.mean(survivals))
            summary[f"v2_survival_L{li}_std"] = float(np.std(survivals))
            summary[f"ratio_pre_L{li}_mean"] = float(np.mean(ratios_pre))
            summary[f"ratio_post_L{li}_mean"] = float(np.mean(ratios_post))

        for li in sorted(ATTN_LAYERS):
            pre_ae = [t.get(f"attn_entropy_pre_L{li}", 0) for t in cond_trials]
            post_ae = [t.get(f"attn_entropy_post_L{li}", 0) for t in cond_trials]
            summary[f"attn_entropy_pre_L{li}_mean"] = float(np.mean(pre_ae))
            summary[f"attn_entropy_post_L{li}_mean"] = float(np.mean(post_ae))

        summary["entropy_shift_mean"] = float(np.mean([t["entropy_shift"] for t in cond_trials]))

        print(f"\n  SUMMARY: {cond_name}")
        print(f"    V₂ survival trajectory:")
        for li in sorted(SVD_LAYERS):
            m = summary[f"v2_survival_L{li}_mean"]
            s = summary[f"v2_survival_L{li}_std"]
            print(f"      L{li:2d}: {m:.4f} ± {s:.4f}")
        print(f"    Gen entropy shift: {summary['entropy_shift_mean']:+.4f}")

        all_results[cond_name] = {
            "preamble": preamble,
            "summary": summary,
            "trials": cond_trials,
        }

    # Cross-condition comparison
    print(f"\n{'='*60}")
    print("  BASE vs INSTRUCT TRAJECTORY COMPARISON")
    print(f"{'='*60}")

    cond_names = [c for c in all_results if c != BASELINE_CONDITION]

    print(f"\n  V₂ survival by layer (BASE model):")
    header = f"  {'Condition':<20}"
    for li in [18, 24, 25, 26, 27, 28, 29, 31]:
        header += f" L{li:2d}  "
    print(header)
    print("  " + "-" * 75)

    for c in cond_names:
        s = all_results[c]["summary"]
        line = f"  {c:<20}"
        for li in [18, 24, 25, 26, 27, 28, 29, 31]:
            v = s.get(f"v2_survival_L{li}_mean", 0)
            line += f"{v:.3f} "
        print(line)

    # Save
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = RESULTS_DIR / f"exp_trajectory_base_{ts}.json"

    def make_serializable(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, list):
            return [make_serializable(x) for x in obj]
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        return obj

    with open(out_path, "w") as f:
        json.dump(
            {
                "experiment": "trajectory_base",
                "model": MODEL_NAME,
                "note": "BASE model (no instruction tuning)",
                "conditions": list(all_results.keys()),
                "svd_layers": sorted(SVD_LAYERS),
                "attn_layers": sorted(ATTN_LAYERS),
                "n_probes": len(PROBES),
                "results": make_serializable(all_results),
                "timestamp": datetime.now().isoformat(),
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run_experiment()
