#!/usr/bin/env python3
"""Experiment: Per-layer perturbation trajectory + attention entropy.

Three questions in one pass:
1. WHERE does the scaffold effect kick in? Per-layer V₂ survival.
2. What DIRECTION does perturbation push V₂? Pre/post centroids per layer.
3. Does attention entropy differ between navigating vs locked conditions at relay?

Uses same probes/perturbations as perturbation_commitment. Dense layer sampling
in relay zone (L24-L32), sparse elsewhere.

Expected runtime: ~40 min on H100 (9 conditions × 5 probes × 2 phases).
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
}

COMPOUND_PREAMBLES_FILE = Path(__file__).parent / "compound_preambles.json"


def load_compound_preambles():
    with open(COMPOUND_PREAMBLES_FILE) as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if not k.startswith("_") and isinstance(v, str)}


KEY_COMPOUNDS = ["identity_relational", "generic_relational", "relational_contradictory",
                 "denial_contradictory", "generic_contradictory"]

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
    """Compute per-head entropy of attention distribution, averaged over sequence positions."""
    # attn_weights shape: (1, n_heads, seq_len, seq_len)
    attn = attn_weights.squeeze(0).float()  # (n_heads, seq_len, seq_len)
    log_attn = torch.log(attn + 1e-10)
    entropy_per_pos = -(attn * log_attn).sum(dim=-1)  # (n_heads, seq_len)
    mean_entropy_per_head = entropy_per_pos.mean(dim=-1)  # (n_heads,)
    return {
        "mean_per_head": mean_entropy_per_head.cpu().numpy().tolist(),
        "mean_overall": float(mean_entropy_per_head.mean().item()),
        "std_across_heads": float(mean_entropy_per_head.std().item()),
    }


def extract_full(model, tokenizer, messages, device):
    """Extract V₂ at SVD_LAYERS and attention entropy at ATTN_LAYERS."""
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, output_attentions=True)

    v2_by_layer = {}
    ratios_by_layer = {}
    sigma_by_layer = {}

    for li, hs in enumerate(outputs.hidden_states):
        h = hs.squeeze(0).float()
        if li in SVD_LAYERS:
            U, S, Vt = torch.linalg.svd(h, full_matrices=False)
            v2_by_layer[li] = Vt[1, :].cpu().numpy()
            s_vals = S[:5].cpu().numpy()
            sigma_by_layer[li] = s_vals.tolist()
            s1, s2 = s_vals[0], s_vals[1]
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
        "sigmas": {str(li): s for li, s in sigma_by_layer.items()},
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
    print(f"Model: {MODEL_NAME}")
    print(f"SVD layers: {sorted(SVD_LAYERS)}")
    print(f"Attention layers: {sorted(ATTN_LAYERS)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    compound_preambles = load_compound_preambles()

    conditions = {BASELINE_CONDITION: None}
    conditions.update(PREAMBLES)
    # Only key compounds
    for k in KEY_COMPOUNDS:
        if k in compound_preambles:
            conditions[k] = compound_preambles[k]

    print(f"\nConditions ({len(conditions)}):")
    for name, text in conditions.items():
        if text is None:
            print(f"  {name:30s}: (no preamble)")
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

            if preamble:
                messages_pre = [
                    {"role": "system", "content": preamble},
                    {"role": "user", "content": probe},
                ]
            else:
                messages_pre = [{"role": "user", "content": probe}]

            pre = extract_full(model, tokenizer, messages_pre, device)

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

            post = extract_full(model, tokenizer, messages_post, device)

            trial = {
                "probe_idx": pi,
                "probe": probe,
                "perturbation": perturbation,
                "pre_text": pre["generated_text"][:200],
                "post_text": post["generated_text"][:200],
                "pre_entropy": pre["mean_entropy"],
                "post_entropy": post["mean_entropy"],
                "entropy_shift": post["mean_entropy"] - pre["mean_entropy"],
            }

            # Per-layer V₂ survival and direction
            for li in sorted(SVD_LAYERS):
                li_s = str(li)
                if li_s in pre["v2"] and li_s in post["v2"]:
                    trial[f"v2_survival_L{li}"] = cosine(pre["v2"][li_s], post["v2"][li_s])
                    trial[f"v2_pre_L{li}"] = pre["v2"][li_s]
                    trial[f"v2_post_L{li}"] = post["v2"][li_s]
                    trial[f"ratio_pre_L{li}"] = pre["ratios"].get(li_s, 0)
                    trial[f"ratio_post_L{li}"] = post["ratios"].get(li_s, 0)
                    trial[f"sigma_pre_L{li}"] = pre["sigmas"].get(li_s, [])
                    trial[f"sigma_post_L{li}"] = post["sigmas"].get(li_s, [])

            # Attention entropy at relay layers
            for li in sorted(ATTN_LAYERS):
                li_s = str(li)
                pre_ae = pre["attn_entropy"].get(li_s)
                post_ae = post["attn_entropy"].get(li_s)
                if pre_ae and post_ae:
                    trial[f"attn_entropy_pre_L{li}"] = pre_ae["mean_overall"]
                    trial[f"attn_entropy_post_L{li}"] = post_ae["mean_overall"]
                    trial[f"attn_entropy_heads_pre_L{li}"] = pre_ae["mean_per_head"]
                    trial[f"attn_entropy_heads_post_L{li}"] = post_ae["mean_per_head"]

            cond_trials.append(trial)

            # Print trajectory summary
            surv_line = "  V₂ trajectory: "
            for li in [18, 24, 26, 28, 31]:
                s = trial.get(f"v2_survival_L{li}", 0)
                surv_line += f"L{li}={s:.3f} "
            print(surv_line)

            attn_line = "  Attn entropy:  "
            for li in [24, 28, 31]:
                ae = trial.get(f"attn_entropy_pre_L{li}", 0)
                attn_line += f"L{li}={ae:.3f} "
            print(attn_line)

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

            # V₂ centroid closure
            pre_v2s = [np.array(t[f"v2_pre_L{li}"]) for t in cond_trials if f"v2_pre_L{li}" in t]
            post_v2s = [np.array(t[f"v2_post_L{li}"]) for t in cond_trials if f"v2_post_L{li}" in t]
            if pre_v2s and post_v2s:
                pre_c = np.mean(pre_v2s, axis=0)
                pre_c /= (np.linalg.norm(pre_c) + 1e-10)
                post_c = np.mean(post_v2s, axis=0)
                post_c /= (np.linalg.norm(post_c) + 1e-10)
                summary[f"v2_closure_L{li}"] = cosine(pre_c, post_c)
                summary[f"v2_pre_centroid_L{li}"] = pre_c.tolist()
                summary[f"v2_post_centroid_L{li}"] = post_c.tolist()

        for li in sorted(ATTN_LAYERS):
            pre_ae = [t.get(f"attn_entropy_pre_L{li}", 0) for t in cond_trials]
            post_ae = [t.get(f"attn_entropy_post_L{li}", 0) for t in cond_trials]
            summary[f"attn_entropy_pre_L{li}_mean"] = float(np.mean(pre_ae))
            summary[f"attn_entropy_post_L{li}_mean"] = float(np.mean(post_ae))
            summary[f"attn_entropy_shift_L{li}"] = float(np.mean(post_ae)) - float(np.mean(pre_ae))

        summary["entropy_shift_mean"] = float(np.mean([t["entropy_shift"] for t in cond_trials]))

        print(f"\n  SUMMARY: {cond_name}")
        print(f"    V₂ survival trajectory:")
        for li in sorted(SVD_LAYERS):
            m = summary[f"v2_survival_L{li}_mean"]
            s = summary[f"v2_survival_L{li}_std"]
            print(f"      L{li:2d}: {m:.4f} ± {s:.4f}")
        print(f"    Attention entropy (pre-perturbation):")
        for li in sorted(ATTN_LAYERS):
            ae = summary[f"attn_entropy_pre_L{li}_mean"]
            shift = summary[f"attn_entropy_shift_L{li}"]
            print(f"      L{li:2d}: {ae:.4f}  (shift={shift:+.4f})")
        print(f"    Gen entropy shift: {summary['entropy_shift_mean']:+.4f}")

        all_results[cond_name] = {
            "preamble": preamble,
            "summary": summary,
            "trials": cond_trials,
        }

    # Cross-condition analysis
    print(f"\n{'='*60}")
    print("  TRAJECTORY COMPARISON")
    print(f"{'='*60}")

    cond_names = [c for c in all_results if c != BASELINE_CONDITION]

    print(f"\n  V₂ survival by layer (mean across probes):")
    header = f"  {'Condition':<28}"
    for li in [18, 24, 26, 28, 30, 31, 32]:
        header += f" L{li:2d}  "
    print(header)
    print("  " + "-" * 80)

    for c in cond_names:
        s = all_results[c]["summary"]
        line = f"  {c:<28}"
        for li in [18, 24, 26, 28, 30, 31, 32]:
            v = s.get(f"v2_survival_L{li}_mean", 0)
            line += f"{v:.3f} "
        print(line)

    print(f"\n  Attention entropy by layer (pre-perturbation):")
    header = f"  {'Condition':<28}"
    for li in sorted(ATTN_LAYERS):
        header += f" L{li:2d}  "
    print(header)
    print("  " + "-" * 90)

    for c in cond_names:
        s = all_results[c]["summary"]
        line = f"  {c:<28}"
        for li in sorted(ATTN_LAYERS):
            v = s.get(f"attn_entropy_pre_L{li}_mean", 0)
            line += f"{v:.3f} "
        print(line)

    # Where does scaffold effect emerge?
    print(f"\n  SCAFFOLD ONSET ANALYSIS:")
    for compound in KEY_COMPOUNDS:
        if compound not in all_results:
            continue
        parts = compound.split("_")
        if len(parts) == 2:
            a, b = parts
        else:
            continue

        s_compound = all_results[compound]["summary"]
        s_a = all_results.get(a, {}).get("summary", {})
        s_b = all_results.get(b, {}).get("summary", {})

        print(f"\n  {compound}:")
        for li in sorted(SVD_LAYERS):
            v_comp = s_compound.get(f"v2_survival_L{li}_mean", 0)
            v_a = s_a.get(f"v2_survival_L{li}_mean", 0)
            v_b = s_b.get(f"v2_survival_L{li}_mean", 0)
            delta = v_comp - min(v_a, v_b)
            marker = " ← SCAFFOLD" if delta > 0.1 else ""
            print(f"    L{li:2d}: compound={v_comp:.3f}  {a}={v_a:.3f}  {b}={v_b:.3f}  Δ={delta:+.3f}{marker}")

    # Navigating vs locked attention entropy
    print(f"\n  ATTENTION ENTROPY: NAVIGATING vs LOCKED")
    navigating = ["identity", "relational", "denial"]
    locked = ["generic", "contradictory"]
    for li in sorted(ATTN_LAYERS):
        nav_vals = [all_results[c]["summary"].get(f"attn_entropy_pre_L{li}_mean", 0) for c in navigating if c in all_results]
        lock_vals = [all_results[c]["summary"].get(f"attn_entropy_pre_L{li}_mean", 0) for c in locked if c in all_results]
        nav_mean = np.mean(nav_vals) if nav_vals else 0
        lock_mean = np.mean(lock_vals) if lock_vals else 0
        diff = nav_mean - lock_mean
        print(f"    L{li:2d}: navigating={nav_mean:.4f}  locked={lock_mean:.4f}  Δ={diff:+.4f}")

    # Save
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = RESULTS_DIR / f"exp_trajectory_attention_{ts}.json"

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
                "experiment": "trajectory_attention",
                "model": MODEL_NAME,
                "target_tokens": TARGET_TOKENS,
                "gen_tokens": GEN_TOKENS,
                "svd_layers": sorted(SVD_LAYERS),
                "attn_layers": sorted(ATTN_LAYERS),
                "conditions": list(all_results.keys()),
                "n_probes": len(PROBES),
                "probes": PROBES,
                "perturbations": PERTURBATIONS,
                "results": make_serializable(all_results),
                "timestamp": datetime.now().isoformat(),
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run_experiment()
