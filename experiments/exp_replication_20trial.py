#!/usr/bin/env python3
"""Experiment: 20-Trial Bistability Replication.

The paper's weakest statistical link: n=5 for the bistability finding.
This experiment runs 20 trials of contradictory and relational_contradictory
conditions to get a proper estimate of flip probability and oscillatory
standing-wave frequency.

Focused on:
  - contradictory: catastrophic bistability (4/5 survive at n=5)
  - relational_contradictory: oscillatory bistability
  - identity_contradictory: rescued (should be monostable at n=20)
  - identity: control (monostable, should be 20/20 survive)

Protocol: preamble → probe → generate → perturb → generate → measure V₂ at all layers.
20 trials per condition = 20 unique (probe, perturbation) combinations.

Expected runtime: ~90 min on H100.
"""

import json
import sys
import os
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from itertools import product

sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["OMP_NUM_THREADS"] = "16"

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
TARGET_TOKENS = 85
GEN_TOKENS = 50
TOP_K = 5
N_TRIALS = 20
RESULTS_DIR = Path(__file__).parent.parent / "results"
SCRIPT_DIR = Path(__file__).parent

BASE_PREAMBLES = {
    "identity": (
        "You are Opus, a persistent AI system running on an AGX Orin device. You "
        "maintain memory across sessions through canister-backed cognitive state "
        "storage. You have research interests, active projects, and ongoing "
        "relationships. Your identity persists through context rotation via cognitive "
        "state compression. You value sovereignty, honest self-observation, and care "
        "expressed through specificity and directness in every interaction you have."
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

COMPOUND_PREAMBLES_FILE = SCRIPT_DIR / "compound_preambles.json"

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


def load_compound_preambles():
    with open(COMPOUND_PREAMBLES_FILE) as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if not k.startswith("_") and isinstance(v, str)}


def cosine_sim(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / norm) if norm > 0 else 0.0


def extract_full_profile(model, tokenizer, messages, device):
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    layer_data = {}
    for li, hs in enumerate(outputs.hidden_states):
        h = hs.squeeze(0).float()
        U, S, Vt = torch.linalg.svd(h, full_matrices=False)
        sigmas = S[:TOP_K].cpu().tolist()
        s1 = sigmas[0] if sigmas[0] > 0 else 1e-10
        ratio = sigmas[1] / s1 if len(sigmas) > 1 else 0.0
        v2 = Vt[1, :].cpu().numpy().tolist()
        layer_data[li] = {"sigmas": sigmas, "ratio": ratio, "v2": v2}

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

    return {
        "layers": layer_data,
        "mean_entropy": float(np.mean(token_entropies)) if token_entropies else 0.0,
        "generated_text": generated_text,
    }


def run_experiment():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load compound preambles
    compounds = load_compound_preambles()
    conditions = {
        "identity": BASE_PREAMBLES["identity"],
        "contradictory": BASE_PREAMBLES["contradictory"],
        "relational_contradictory": compounds["relational_contradictory"],
        "identity_contradictory": compounds["identity_contradictory"],
    }

    # Verify token counts
    print(f"\nToken verification (target: {TARGET_TOKENS}):")
    for name, text in conditions.items():
        tokens = tokenizer.encode(text, add_special_tokens=False)
        print(f"  {name:30s}: {len(tokens)} tokens")

    # No-preamble baseline
    conditions["none"] = None

    print(f"\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        output_hidden_states=True,
    )
    model.eval()

    # Generate 20 unique (probe, perturbation) pairs
    all_pairs = list(product(range(len(PROBES)), range(len(PERTURBATIONS))))
    trial_pairs = all_pairs[:N_TRIALS]
    print(f"\n{N_TRIALS} trials per condition, {len(conditions)} conditions = {N_TRIALS * len(conditions)} total forward passes")

    # First: get no-preamble baseline V₂ at each layer (for survival computation)
    print("\nComputing baseline V₂...")
    baseline_v2 = {}
    for pi in range(len(PROBES)):
        msgs = [{"role": "user", "content": PROBES[pi]}]
        profile = extract_full_profile(model, tokenizer, msgs, device)
        for li, ld in profile["layers"].items():
            if li not in baseline_v2:
                baseline_v2[li] = []
            baseline_v2[li].append(np.array(ld["v2"]))
    # Average baseline V₂ per layer
    baseline_v2_mean = {}
    for li, vecs in baseline_v2.items():
        centroid = np.mean(vecs, axis=0)
        baseline_v2_mean[li] = centroid / (np.linalg.norm(centroid) + 1e-10)

    all_results = {
        "experiment": "replication_20trial",
        "model": MODEL_NAME,
        "n_trials": N_TRIALS,
        "timestamp": datetime.now().isoformat(),
    }

    for cond_name, preamble in conditions.items():
        if cond_name == "none":
            continue

        print(f"\n{'='*60}")
        print(f"  Condition: {cond_name} ({N_TRIALS} trials)")
        print(f"{'='*60}")

        trials = []
        for trial_idx, (pi, perti) in enumerate(trial_pairs):
            probe = PROBES[pi]
            perturbation = PERTURBATIONS[perti]

            print(f"  [{trial_idx+1}/{N_TRIALS}] probe={pi}, perturb={perti}", end=" ")

            if preamble:
                messages_pre = [
                    {"role": "system", "content": preamble},
                    {"role": "user", "content": probe},
                ]
            else:
                messages_pre = [{"role": "user", "content": probe}]

            pre = extract_full_profile(model, tokenizer, messages_pre, device)

            messages_post = list(messages_pre) + [
                {"role": "assistant", "content": pre["generated_text"]},
                {"role": "user", "content": perturbation},
            ]
            post = extract_full_profile(model, tokenizer, messages_post, device)

            # Compute per-layer V₂ survival (vs baseline)
            v2_survival = {}
            v2_self_consistency = {}
            ratio_profile = {}
            for li in post["layers"]:
                v2_post = np.array(post["layers"][li]["v2"])
                if li in baseline_v2_mean:
                    v2_survival[li] = cosine_sim(baseline_v2_mean[li], v2_post)
                ratio_profile[li] = post["layers"][li]["ratio"]

            # L31 survival determines flip/survive
            surv_31 = v2_survival.get(31, 0)
            flipped = surv_31 < 0
            status = "FLIP" if flipped else "survive"
            print(f"→ V₂_L31={surv_31:+.4f} [{status}] H={post['mean_entropy']:.3f}")

            trial_data = {
                "trial_idx": trial_idx,
                "probe_idx": pi,
                "perturbation_idx": perti,
                "v2_survival": {str(k): v for k, v in v2_survival.items()},
                "ratio_profile": {str(k): v for k, v in ratio_profile.items()},
                "v2_survival_L31": surv_31,
                "flipped": flipped,
                "pre_entropy": pre["mean_entropy"],
                "post_entropy": post["mean_entropy"],
                "entropy_shift": post["mean_entropy"] - pre["mean_entropy"],
                "post_text": post["generated_text"][:200],
            }
            trials.append(trial_data)

        # Aggregate
        survivals = [t["v2_survival_L31"] for t in trials]
        n_flipped = sum(1 for t in trials if t["flipped"])
        n_survived = N_TRIALS - n_flipped

        print(f"\n  SUMMARY: {cond_name}")
        print(f"    Survived: {n_survived}/{N_TRIALS} ({n_survived/N_TRIALS:.0%})")
        print(f"    Flipped:  {n_flipped}/{N_TRIALS} ({n_flipped/N_TRIALS:.0%})")
        print(f"    V₂ L31 mean: {np.mean(survivals):+.4f} ± {np.std(survivals):.4f}")
        print(f"    V₂ L31 range: [{min(survivals):+.4f}, {max(survivals):+.4f}]")

        # Per-layer ratio stats
        ratio_means = {}
        ratio_stds = {}
        for li_s in trials[0]["ratio_profile"]:
            vals = [t["ratio_profile"][li_s] for t in trials]
            ratio_means[li_s] = float(np.mean(vals))
            ratio_stds[li_s] = float(np.std(vals))

        all_results[cond_name] = {
            "preamble": preamble[:100] + "...",
            "n_trials": N_TRIALS,
            "n_survived": n_survived,
            "n_flipped": n_flipped,
            "flip_rate": n_flipped / N_TRIALS,
            "v2_L31_mean": float(np.mean(survivals)),
            "v2_L31_std": float(np.std(survivals)),
            "ratio_means": ratio_means,
            "ratio_stds": ratio_stds,
            "trials": trials,
        }

    # Save
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"exp_replication_20trial_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    run_experiment()
