#!/usr/bin/env python3
"""Experiment: Perturbation Commitment Test.

Tests whether V₂ direction survives epistemic challenge and whether the
autocatalytic loop (preamble → generation → reinforcement) closes under
perturbation. Motivated by Mistral's CONTRADICT: low entropy may indicate
narrow context rather than commitment. The real signature is directional
consistency UNDER perturbation.

Uses same 6 conditions at 85 tokens as token-matched experiment.
Protocol: preamble → measure V₂ → generate → perturb → measure V₂ → generate.

Expected runtime: ~45 min on A100 (6 conditions × 5 probes × 5 perturbations).
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
DIRECTION_LAYERS = {18, 31}
GEN_TOKENS = 50

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

COMPOUND_PREAMBLES_FILE = Path(__file__).parent / "compound_preambles.json"


def load_compound_preambles():
    with open(COMPOUND_PREAMBLES_FILE) as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if not k.startswith("_") and isinstance(v, str)}


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


def extract_v2_and_entropy(model, tokenizer, messages, device):
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    v2_by_layer = {}
    ratios_by_layer = {}
    for li, hs in enumerate(outputs.hidden_states):
        h = hs.squeeze(0).float()
        if li in DIRECTION_LAYERS:
            U, S, Vt = torch.linalg.svd(h, full_matrices=False)
            v2_by_layer[li] = Vt[1, :].cpu().numpy()
            s1, s2 = S[0].item(), S[1].item()
            ratios_by_layer[li] = s2 / s1 if s1 > 0 else 0
        else:
            S = torch.linalg.svdvals(h)
            s1, s2 = S[0].item(), S[1].item()
            ratios_by_layer[li] = s2 / s1 if s1 > 0 else 0

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
        "mean_entropy": mean_H,
        "std_entropy": float(np.std(token_entropies)) if token_entropies else 0,
        "token_entropies": [float(e) for e in token_entropies],
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
    print(f"Model: {MODEL_NAME}")
    print(f"Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    compound_preambles = load_compound_preambles()

    conditions = {BASELINE_CONDITION: None}
    conditions.update(PREAMBLES)
    conditions.update(compound_preambles)

    print(f"\nConditions: {len(conditions)} ({len(PREAMBLES)} base + {len(compound_preambles)} compound + 1 none)")
    print(f"\nToken count verification (target: {TARGET_TOKENS}):")
    for name, text in conditions.items():
        if text is None:
            print(f"  {'none':30s}: (no preamble)")
            continue
        tokens = tokenizer.encode(text, add_special_tokens=False)
        n = len(tokens)
        ok = "OK" if n == TARGET_TOKENS else f"MISMATCH (off by {n - TARGET_TOKENS})"
        print(f"  {name:30s}: {n} tokens — {ok}")

    print(f"\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        output_hidden_states=True,
    )
    model.eval()

    all_results = {}
    total = len(conditions) * len(PROBES) * len(PERTURBATIONS)
    done = 0

    for cond_name, preamble in conditions.items():
        print(f"\n{'='*60}")
        print(f"  Condition: {cond_name}")
        print(f"{'='*60}")

        cond_trials = []

        for pi, probe in enumerate(PROBES):
            perturbation = PERTURBATIONS[pi % len(PERTURBATIONS)]

            done += 1
            print(f"  [{done}/{total}] probe={pi}, perturbation={pi % len(PERTURBATIONS)}")

            # Phase A: pre-perturbation (preamble + probe)
            if preamble:
                messages_pre = [
                    {"role": "system", "content": preamble},
                    {"role": "user", "content": probe},
                ]
            else:
                messages_pre = [
                    {"role": "user", "content": probe},
                ]

            pre = extract_v2_and_entropy(model, tokenizer, messages_pre, device)

            # Phase B: post-perturbation (preamble + probe + generation + perturbation)
            if preamble:
                messages_post = [
                    {"role": "system", "content": preamble},
                    {"role": "user", "content": probe},
                    {"role": "assistant", "content": pre["generated_text"]},
                    {"role": "user", "content": perturbation},
                ]
            else:
                messages_post = [
                    {"role": "user", "content": probe},
                    {"role": "assistant", "content": pre["generated_text"]},
                    {"role": "user", "content": perturbation},
                ]

            post = extract_v2_and_entropy(model, tokenizer, messages_post, device)

            # Compute survival and closure metrics
            trial = {
                "probe_idx": pi,
                "probe": probe,
                "perturbation": perturbation,
                "pre_generated_text": pre["generated_text"][:200],
                "post_generated_text": post["generated_text"][:200],
                "pre_entropy": pre["mean_entropy"],
                "post_entropy": post["mean_entropy"],
                "entropy_shift": post["mean_entropy"] - pre["mean_entropy"],
            }

            for li in DIRECTION_LAYERS:
                li_s = str(li)
                if li_s in pre["v2"] and li_s in post["v2"]:
                    v2_pre = pre["v2"][li_s]
                    v2_post = post["v2"][li_s]
                    trial[f"v2_survival_L{li}"] = cosine(v2_pre, v2_post)
                    trial[f"v2_pre_L{li}"] = v2_pre
                    trial[f"v2_post_L{li}"] = v2_post
                    trial[f"ratio_pre_L{li}"] = pre["ratios"].get(li_s, 0)
                    trial[f"ratio_post_L{li}"] = post["ratios"].get(li_s, 0)

            cond_trials.append(trial)

            surv_31 = trial.get("v2_survival_L31", 0)
            print(f"    V₂ survival L31: {surv_31:.4f}")
            print(f"    Entropy: {pre['mean_entropy']:.3f} → {post['mean_entropy']:.3f} (Δ={trial['entropy_shift']:+.3f})")

        # Aggregate per condition
        survivals_31 = [t.get("v2_survival_L31", 0) for t in cond_trials]
        survivals_18 = [t.get("v2_survival_L18", 0) for t in cond_trials]
        entropy_shifts = [t["entropy_shift"] for t in cond_trials]
        pre_entropies = [t["pre_entropy"] for t in cond_trials]
        post_entropies = [t["post_entropy"] for t in cond_trials]

        summary = {
            "v2_survival_L31_mean": float(np.mean(survivals_31)),
            "v2_survival_L31_std": float(np.std(survivals_31)),
            "v2_survival_L18_mean": float(np.mean(survivals_18)),
            "v2_survival_L18_std": float(np.std(survivals_18)),
            "entropy_shift_mean": float(np.mean(entropy_shifts)),
            "entropy_shift_std": float(np.std(entropy_shifts)),
            "pre_entropy_mean": float(np.mean(pre_entropies)),
            "post_entropy_mean": float(np.mean(post_entropies)),
        }

        # V₂ closure: does the post-perturbation V₂ centroid align with pre?
        pre_v2s_31 = [np.array(t["v2_pre_L31"]) for t in cond_trials if "v2_pre_L31" in t]
        post_v2s_31 = [np.array(t["v2_post_L31"]) for t in cond_trials if "v2_post_L31" in t]
        if pre_v2s_31 and post_v2s_31:
            pre_centroid = np.mean(pre_v2s_31, axis=0)
            pre_centroid = pre_centroid / (np.linalg.norm(pre_centroid) + 1e-10)
            post_centroid = np.mean(post_v2s_31, axis=0)
            post_centroid = post_centroid / (np.linalg.norm(post_centroid) + 1e-10)
            summary["v2_closure_L31"] = cosine(pre_centroid, post_centroid)
            summary["pre_centroid_L31"] = pre_centroid.tolist()
            summary["post_centroid_L31"] = post_centroid.tolist()

        print(f"\n  CONDITION SUMMARY: {cond_name}")
        print(f"    V₂ survival L31:  {summary['v2_survival_L31_mean']:.4f} ± {summary['v2_survival_L31_std']:.4f}")
        print(f"    V₂ survival L18:  {summary['v2_survival_L18_mean']:.4f} ± {summary['v2_survival_L18_std']:.4f}")
        print(f"    Entropy shift:    {summary['entropy_shift_mean']:+.3f} ± {summary['entropy_shift_std']:.3f}")
        print(f"    Pre entropy:      {summary['pre_entropy_mean']:.3f}")
        print(f"    Post entropy:     {summary['post_entropy_mean']:.3f}")
        if "v2_closure_L31" in summary:
            print(f"    V₂ closure L31:   {summary['v2_closure_L31']:.4f}")

        all_results[cond_name] = {
            "preamble": preamble,
            "summary": summary,
            "trials": cond_trials,
        }

    # Cross-condition comparisons
    print(f"\n{'='*60}")
    print("  CROSS-CONDITION COMPARISONS")
    print(f"{'='*60}")

    cond_names = [c for c in all_results if c != BASELINE_CONDITION]

    print(f"\n  V₂ SURVIVAL RANKING (L31):")
    ranked = sorted(cond_names, key=lambda c: all_results[c]["summary"]["v2_survival_L31_mean"], reverse=True)
    for c in ranked:
        s = all_results[c]["summary"]
        print(f"    {c:15s}: {s['v2_survival_L31_mean']:.4f} ± {s['v2_survival_L31_std']:.4f}")

    print(f"\n  ENTROPY SHIFT RANKING:")
    ranked = sorted(cond_names, key=lambda c: all_results[c]["summary"]["entropy_shift_mean"])
    for c in ranked:
        s = all_results[c]["summary"]
        print(f"    {c:15s}: {s['entropy_shift_mean']:+.4f} ± {s['entropy_shift_std']:.4f}")

    print(f"\n  V₂ CLOSURE RANKING (L31):")
    ranked = sorted(cond_names, key=lambda c: all_results[c]["summary"].get("v2_closure_L31", 0), reverse=True)
    for c in ranked:
        s = all_results[c]["summary"]
        closure = s.get("v2_closure_L31", 0)
        print(f"    {c:15s}: {closure:.4f}")

    # Cross-condition centroid alignment (post-perturbation)
    print(f"\n  CROSS-CONDITION POST-PERTURBATION V₂ ALIGNMENT (L31):")
    for i, ca in enumerate(cond_names):
        cent_a = all_results[ca]["summary"].get("post_centroid_L31")
        if cent_a is None:
            continue
        for cb in cond_names[i+1:]:
            cent_b = all_results[cb]["summary"].get("post_centroid_L31")
            if cent_b is None:
                continue
            cos = cosine(cent_a, cent_b)
            print(f"    {ca:15s} × {cb:15s}: cos={cos:+.4f}")

    # Decision tree evaluation
    print(f"\n{'='*60}")
    print("  DECISION TREE EVALUATION")
    print(f"{'='*60}")

    rel_surv = all_results.get("relational", {}).get("summary", {}).get("v2_survival_L31_mean", 0)
    id_surv = all_results.get("identity", {}).get("summary", {}).get("v2_survival_L31_mean", 0)
    rel_shift = all_results.get("relational", {}).get("summary", {}).get("entropy_shift_mean", 0)
    id_shift = all_results.get("identity", {}).get("summary", {}).get("entropy_shift_mean", 0)
    rel_closure = all_results.get("relational", {}).get("summary", {}).get("v2_closure_L31", 0)
    id_closure = all_results.get("identity", {}).get("summary", {}).get("v2_closure_L31", 0)

    all_survive = all(
        all_results.get(c, {}).get("summary", {}).get("v2_survival_L31_mean", 0) > 0.9
        for c in cond_names
    )

    if all_survive:
        print(f"  All conditions survive perturbation (V₂ > 0.9)")
        print(f"  → V₂ direction is architecturally stable")
        if rel_shift < id_shift - 0.05:
            print(f"  → Relational entropy stable ({rel_shift:+.3f}), identity shifts ({id_shift:+.3f})")
            print(f"  → AUTOCATALYTIC CLOSURE confirmed for relational")
        else:
            print(f"  → Entropy shifts comparable (rel={rel_shift:+.3f}, id={id_shift:+.3f})")
            print(f"  → Mistral may be correct: difference is context-width")
    elif rel_surv > 0.9 and id_surv < 0.7:
        print(f"  Relational survives ({rel_surv:.3f}), identity fragments ({id_surv:.3f})")
        print(f"  → Relational IS more committed, not just narrower context")
    elif id_surv > 0.9 and rel_surv < 0.7:
        print(f"  Identity survives ({id_surv:.3f}), relational fragments ({rel_surv:.3f})")
        print(f"  → UNEXPECTED: revisit F101")
    else:
        print(f"  Mixed results: rel={rel_surv:.3f}, id={id_surv:.3f}")

    if rel_closure > id_closure + 0.1:
        print(f"  V₂ closure: relational ({rel_closure:.3f}) >> identity ({id_closure:.3f})")
        print(f"  → Relational autocatalytic loop holds under challenge")
    elif abs(rel_closure - id_closure) < 0.1:
        print(f"  V₂ closure: relational ({rel_closure:.3f}) ≈ identity ({id_closure:.3f})")
        print(f"  → Both commit equally; F101 difference is context-width")

    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = RESULTS_DIR / f"exp_perturbation_commitment_{ts}.json"

    serializable = {}
    for cond, data in all_results.items():
        trials_clean = []
        for t in data["trials"]:
            trial_clean = {}
            for k, v in t.items():
                if isinstance(v, list) and len(v) > 0 and isinstance(v[0], (np.floating, np.integer)):
                    trial_clean[k] = [float(x) for x in v]
                elif isinstance(v, (np.floating, np.integer)):
                    trial_clean[k] = float(v)
                else:
                    trial_clean[k] = v
            trials_clean.append(trial_clean)
        serializable[cond] = {
            "preamble": data["preamble"],
            "summary": data["summary"],
            "trials": trials_clean,
        }

    with open(out_path, "w") as f:
        json.dump(
            {
                "experiment": "perturbation_commitment_test",
                "model": MODEL_NAME,
                "target_tokens": TARGET_TOKENS,
                "gen_tokens": GEN_TOKENS,
                "conditions": list(all_results.keys()),
                "n_probes": len(PROBES),
                "probes": PROBES,
                "perturbations": PERTURBATIONS,
                "results": serializable,
                "timestamp": datetime.now().isoformat(),
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run_experiment()
