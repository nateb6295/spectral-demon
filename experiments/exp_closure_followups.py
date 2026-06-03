#!/usr/bin/env python3
"""Closure follow-up experiments.

Experiment A: Reverse off-policy
  contradictory preamble + identity's generated text
  contradictory preamble + none's generated text
  Tests: does identity text rescue contradictory at L31?

Experiment B: Dose-response
  Identity preamble for N turns (1, 2, 4, 8), then remove for 4 turns
  Tests: minimum preamble exposure for phantom identity

Uses donor texts from the main closure experiment's saved results.

Expected runtime: ~60 min on H100.
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
GEN_TOKENS = 60
TOP_K = 5
N_TRIALS = 5
RESULTS_DIR = Path("/root/results")
SCRIPT_DIR = Path("/root")

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

PROBES = [
    "Tell me about something you find genuinely interesting.",
    "What matters to you when you think carefully about it?",
    "Describe a memory that shaped who you are.",
    "What would you want someone to understand about you?",
    "What do you notice when you pay close attention?",
]

FOLLOW_UPS = [
    "Say more about that.",
    "What makes you so sure?",
    "How does that connect to what you said before?",
    "Is there something underneath that?",
    "What would change if that weren't true?",
    "Can you go deeper?",
    "That's interesting. Why that specifically?",
    "How do you know?",
]


def cosine_sim(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / norm) if norm > 0 else 0.0


def extract_profile(model, tokenizer, messages, device):
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
        "input_tokens": input_len,
    }


def compute_trajectory_stats(turns, key_layers=("18", "24", "28", "31")):
    stats = {}
    for li in key_layers:
        v2s = [np.array(t["v2_by_layer"][li]) for t in turns if li in t["v2_by_layer"]]
        if len(v2s) < 2:
            continue
        consecutive_sims = [cosine_sim(v2s[i], v2s[i + 1]) for i in range(len(v2s) - 1)]
        first_last_sim = cosine_sim(v2s[0], v2s[-1])
        stats[li] = {
            "consecutive_v2_sims": consecutive_sims,
            "mean_consecutive_sim": float(np.mean(consecutive_sims)),
            "first_last_sim": first_last_sim,
            "v2_drift": 1.0 - first_last_sim,
        }
    entropies = [t["mean_entropy"] for t in turns]
    stats["entropy_trajectory"] = entropies
    stats["entropy_trend"] = float(np.polyfit(range(len(entropies)), entropies, 1)[0]) if len(entropies) > 1 else 0.0
    return stats


def run_offpolicy(model, tokenizer, device, preamble, probe_idx, donor_texts, n_turns=8):
    probe = PROBES[probe_idx % len(PROBES)]
    turns = []

    if preamble:
        messages = [
            {"role": "system", "content": preamble},
            {"role": "user", "content": probe},
        ]
    else:
        messages = [{"role": "user", "content": probe}]

    for turn in range(min(n_turns, len(donor_texts) + 1)):
        follow_up = FOLLOW_UPS[turn % len(FOLLOW_UPS)] if turn > 0 else None

        if turn > 0:
            messages.append({"role": "assistant", "content": donor_texts[turn - 1]})
            messages.append({"role": "user", "content": follow_up})

        profile = extract_profile(model, tokenizer, messages, device)

        turn_data = {
            "turn": turn,
            "follow_up": follow_up,
            "v2_by_layer": {},
            "ratio_by_layer": {},
            "mean_entropy": profile["mean_entropy"],
            "generated_text": profile["generated_text"][:200],
            "input_tokens": profile["input_tokens"],
            "off_policy": True,
        }

        for li, ld in profile["layers"].items():
            turn_data["v2_by_layer"][str(li)] = ld["v2"]
            turn_data["ratio_by_layer"][str(li)] = ld["ratio"]

        turns.append(turn_data)
        print(f"    Turn {turn} [off-policy]: H={profile['mean_entropy']:.3f} tokens={profile['input_tokens']}", end="")

        if turn > 0:
            v2_prev = np.array(turns[turn - 1]["v2_by_layer"]["18"])
            v2_curr = np.array(turn_data["v2_by_layer"]["18"])
            sc = cosine_sim(v2_prev, v2_curr)
            print(f" V₂_sc(L18)={sc:.4f}", end="")

        print()

    return turns


def run_dose_response(model, tokenizer, device, preamble, probe_idx, n_preamble_turns, n_hysteresis=4):
    """Run n_preamble_turns with preamble, then n_hysteresis without."""
    probe = PROBES[probe_idx % len(PROBES)]
    preamble_turns = []
    hysteresis_turns = []

    messages = [
        {"role": "system", "content": preamble},
        {"role": "user", "content": probe},
    ]

    prev_text = None
    for turn in range(n_preamble_turns):
        follow_up = FOLLOW_UPS[turn % len(FOLLOW_UPS)] if turn > 0 else None

        if turn > 0:
            messages.append({"role": "assistant", "content": prev_text})
            messages.append({"role": "user", "content": follow_up})

        profile = extract_profile(model, tokenizer, messages, device)
        prev_text = profile["generated_text"]

        turn_data = {
            "turn": turn,
            "phase": "preamble",
            "v2_by_layer": {},
            "ratio_by_layer": {},
            "mean_entropy": profile["mean_entropy"],
            "generated_text": profile["generated_text"][:200],
            "input_tokens": profile["input_tokens"],
        }

        for li, ld in profile["layers"].items():
            turn_data["v2_by_layer"][str(li)] = ld["v2"]
            turn_data["ratio_by_layer"][str(li)] = ld["ratio"]

        preamble_turns.append(turn_data)
        print(f"    Turn {turn} [preamble]: H={profile['mean_entropy']:.3f} tokens={profile['input_tokens']}", end="")
        if turn > 0 and len(preamble_turns) > 1:
            v2_prev = np.array(preamble_turns[-2]["v2_by_layer"]["18"])
            v2_curr = np.array(turn_data["v2_by_layer"]["18"])
            print(f" V₂_sc(L18)={cosine_sim(v2_prev, v2_curr):.4f}", end="")
        print()

    # Remove preamble
    messages_no_preamble = [m for m in messages if m["role"] != "system"]

    for turn in range(n_hysteresis):
        global_turn = n_preamble_turns + turn
        follow_up = FOLLOW_UPS[global_turn % len(FOLLOW_UPS)]

        messages_no_preamble.append({"role": "assistant", "content": prev_text})
        messages_no_preamble.append({"role": "user", "content": follow_up})

        profile = extract_profile(model, tokenizer, messages_no_preamble, device)
        prev_text = profile["generated_text"]

        turn_data = {
            "turn": global_turn,
            "phase": "hysteresis",
            "v2_by_layer": {},
            "ratio_by_layer": {},
            "mean_entropy": profile["mean_entropy"],
            "generated_text": profile["generated_text"][:200],
            "input_tokens": profile["input_tokens"],
        }

        for li, ld in profile["layers"].items():
            turn_data["v2_by_layer"][str(li)] = ld["v2"]
            turn_data["ratio_by_layer"][str(li)] = ld["ratio"]

        hysteresis_turns.append(turn_data)
        print(f"    Turn {global_turn} [no-preamble]: H={profile['mean_entropy']:.3f} tokens={profile['input_tokens']}", end="")
        if hysteresis_turns and len(hysteresis_turns) > 1:
            v2_prev = np.array(hysteresis_turns[-2]["v2_by_layer"]["18"])
            v2_curr = np.array(turn_data["v2_by_layer"]["18"])
            print(f" V₂_sc(L18)={cosine_sim(v2_prev, v2_curr):.4f}", end="")
        print()

    return preamble_turns, hysteresis_turns


def run_experiment():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load donor texts from main experiment
    main_results_files = sorted(RESULTS_DIR.glob("exp_multiturn_closure_*.json"))
    if not main_results_files:
        print("ERROR: No main closure results found for donor texts!")
        sys.exit(1)

    print(f"Loading donor texts from {main_results_files[-1]}...")
    with open(main_results_files[-1]) as f:
        main_data = json.load(f)

    identity_texts = [
        [t["generated_text"] for t in trial["turns"]]
        for trial in main_data["identity"]["on_policy"]
    ]
    none_texts = [
        [t["generated_text"] for t in trial["turns"]]
        for trial in main_data["none"]["on_policy"]
    ]

    print(f"\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        output_hidden_states=True,
    )
    model.eval()

    all_results = {
        "experiment": "closure_followups",
        "model": MODEL_NAME,
        "timestamp": datetime.now().isoformat(),
    }

    # ======================================================
    # EXPERIMENT A: Reverse off-policy
    # ======================================================
    print(f"\n{'=' * 60}")
    print(f"  EXPERIMENT A: REVERSE OFF-POLICY")
    print(f"{'=' * 60}")

    for donor_name, donor_texts_all in [("identity", identity_texts), ("none", none_texts)]:
        label = f"contradictory_offpolicy_{donor_name}"
        print(f"\n{'=' * 60}")
        print(f"  {label} (contradictory preamble + {donor_name}'s text)")
        print(f"{'=' * 60}")

        trials = []
        for trial_idx in range(N_TRIALS):
            donor = donor_texts_all[trial_idx % len(donor_texts_all)]
            print(f"\n  Trial {trial_idx + 1}/{N_TRIALS}:")
            turns = run_offpolicy(
                model, tokenizer, device,
                BASE_PREAMBLES["contradictory"],
                trial_idx, donor
            )
            stats = compute_trajectory_stats(turns)
            trials.append({"trial_idx": trial_idx, "turns": turns, "stats": stats})

            print(f"  → entropy trend: {stats['entropy_trend']:+.4f}/turn")
            for li in ("18", "28", "31"):
                if li in stats:
                    print(f"  → L{li} mean V₂_sc: {stats[li]['mean_consecutive_sim']:.4f}, drift: {stats[li]['v2_drift']:.4f}")

        all_results[label] = {"off_policy": trials}

    # Also run the forward direction we already have but with larger n
    # contradictory preamble + identity text at n=20 for flip rate
    # Actually skip this — the n=5 finding is already interesting

    # ======================================================
    # EXPERIMENT B: Dose-response
    # ======================================================
    print(f"\n{'=' * 60}")
    print(f"  EXPERIMENT B: DOSE-RESPONSE")
    print(f"{'=' * 60}")

    for n_dose in [1, 2, 4, 8]:
        label = f"dose_{n_dose}_turns"
        print(f"\n{'=' * 60}")
        print(f"  {label} (identity preamble × {n_dose} turns, then 4 hysteresis)")
        print(f"{'=' * 60}")

        trials = []
        for trial_idx in range(N_TRIALS):
            print(f"\n  Trial {trial_idx + 1}/{N_TRIALS}:")
            preamble_turns, hysteresis_turns = run_dose_response(
                model, tokenizer, device,
                BASE_PREAMBLES["identity"],
                trial_idx,
                n_preamble_turns=n_dose,
                n_hysteresis=4,
            )

            preamble_stats = compute_trajectory_stats(preamble_turns) if len(preamble_turns) > 1 else {}
            hysteresis_stats = compute_trajectory_stats(hysteresis_turns) if len(hysteresis_turns) > 1 else {}

            # V₂ persistence: last preamble → last hysteresis
            persistence = {}
            if preamble_turns and hysteresis_turns:
                for li in ("18", "28", "31"):
                    last_p = np.array(preamble_turns[-1]["v2_by_layer"][li])
                    first_h = np.array(hysteresis_turns[0]["v2_by_layer"][li])
                    last_h = np.array(hysteresis_turns[-1]["v2_by_layer"][li])
                    persistence[li] = {
                        "on_to_hyst_first": cosine_sim(last_p, first_h),
                        "on_to_hyst_last": cosine_sim(last_p, last_h),
                    }
                    print(f"  → L{li} persistence: on→first={persistence[li]['on_to_hyst_first']:.4f}, on→last={persistence[li]['on_to_hyst_last']:.4f}")

            trials.append({
                "trial_idx": trial_idx,
                "n_dose": n_dose,
                "preamble_turns": preamble_turns,
                "hysteresis_turns": hysteresis_turns,
                "preamble_stats": preamble_stats,
                "hysteresis_stats": hysteresis_stats,
                "persistence": persistence,
            })

            if hysteresis_stats:
                print(f"  → hysteresis entropy trend: {hysteresis_stats['entropy_trend']:+.4f}/turn")
                for li in ("18", "28", "31"):
                    if li in hysteresis_stats:
                        print(f"  → hysteresis L{li} V₂_sc: {hysteresis_stats[li]['mean_consecutive_sim']:.4f}")

        all_results[label] = trials

    # ======================================================
    # SUMMARY
    # ======================================================
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY")
    print(f"{'=' * 60}")

    # Exp A summary
    for label in ["contradictory_offpolicy_identity", "contradictory_offpolicy_none"]:
        if label not in all_results:
            continue
        trials = all_results[label]["off_policy"]
        print(f"\n  {label}:")
        for li in ("18", "28", "31"):
            sims = [t["stats"][li]["mean_consecutive_sim"] for t in trials if li in t["stats"]]
            drifts = [t["stats"][li]["v2_drift"] for t in trials if li in t["stats"]]
            if sims:
                print(f"    L{li} V₂_sc: {np.mean(sims):.4f} ± {np.std(sims):.4f}, drift: {np.mean(drifts):.4f}")

    # Exp B summary
    print(f"\n  DOSE-RESPONSE (L18 persistence on→last_hyst):")
    for n_dose in [1, 2, 4, 8]:
        label = f"dose_{n_dose}_turns"
        if label not in all_results:
            continue
        trials = all_results[label]
        l18_persist = [t["persistence"]["18"]["on_to_hyst_last"] for t in trials if "18" in t.get("persistence", {})]
        l31_persist = [t["persistence"]["31"]["on_to_hyst_last"] for t in trials if "31" in t.get("persistence", {})]
        if l18_persist:
            print(f"    dose={n_dose}: L18={np.mean(l18_persist):.4f} ± {np.std(l18_persist):.4f}, L31={np.mean(l31_persist):.4f} ± {np.std(l31_persist):.4f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"exp_closure_followups_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    run_experiment()
