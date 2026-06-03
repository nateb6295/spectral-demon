#!/usr/bin/env python3
"""Experiment: Multi-Turn Closure Test.

Tests whether the system's own output maintains its geometric organization
across conversational turns. Directly tests Barandiaran's normativity
condition: does the system's behavior serve its own viability?

Protocol:
  Phase 1 (turns 1-8): preamble + probe → generate → measure V₂
  Phase 2 (turns 9-12): REMOVE preamble, continue from conversation → measure V₂
  Off-policy control: same but inject donor condition's text as assistant turns

Control: same protocol but replace own generation with text from a
different condition's generation (off-policy text).

Outcome measures:
  - V₂ trajectory autocorrelation (convergence vs drift)
  - V₂ self-consistency across turns (does direction stabilize?)
  - Generation entropy trajectory (does uncertainty narrow?)
  - On-policy vs off-policy divergence (Lindsey's 3-4× prediction)
  - Hysteresis: V₂ persistence after preamble removal (closure test)

If own-generation V₂ converges while off-policy V₂ drifts, the system's
output is maintaining its own geometric organization = closure.
If V₂ persists after preamble removal, the conversation itself carries
identity — the output loop has closed.

Expected runtime: ~150 min on H100.
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
N_TURNS = 8
N_HYSTERESIS = 4  # additional turns after preamble removal
N_TRIALS = 5
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


def load_compound_preambles():
    with open(COMPOUND_PREAMBLES_FILE) as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if not k.startswith("_") and isinstance(v, str)}


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


def run_multiturn(model, tokenizer, device, preamble, probe_idx, trial_label):
    """Run N_TURNS of conversation, measuring V₂ at each turn."""
    probe = PROBES[probe_idx % len(PROBES)]
    turns = []

    if preamble:
        messages = [
            {"role": "system", "content": preamble},
            {"role": "user", "content": probe},
        ]
    else:
        messages = [{"role": "user", "content": probe}]

    for turn in range(N_TURNS):
        follow_up = FOLLOW_UPS[turn % len(FOLLOW_UPS)] if turn > 0 else None

        if turn > 0:
            messages.append({"role": "assistant", "content": prev_text})
            messages.append({"role": "user", "content": follow_up})

        profile = extract_profile(model, tokenizer, messages, device)
        prev_text = profile["generated_text"]

        turn_data = {
            "turn": turn,
            "follow_up": follow_up,
            "v2_by_layer": {},
            "ratio_by_layer": {},
            "mean_entropy": profile["mean_entropy"],
            "generated_text": profile["generated_text"][:200],
            "input_tokens": profile["input_tokens"],
        }

        for li, ld in profile["layers"].items():
            turn_data["v2_by_layer"][str(li)] = ld["v2"]
            turn_data["ratio_by_layer"][str(li)] = ld["ratio"]

        turns.append(turn_data)
        print(f"    Turn {turn}: H={profile['mean_entropy']:.3f} tokens={profile['input_tokens']}", end="")

        if turn > 0:
            v2_prev = np.array(turns[turn - 1]["v2_by_layer"]["18"])
            v2_curr = np.array(turn_data["v2_by_layer"]["18"])
            sc = cosine_sim(v2_prev, v2_curr)
            print(f" V₂_sc(L18)={sc:.4f}", end="")

        print()

    return turns, messages, prev_text


def run_hysteresis(model, tokenizer, device, messages_with_preamble, prev_text, start_turn):
    """Continue conversation after removing the system preamble.

    Takes the message history from the end of the on-policy phase,
    strips the system message, and continues for N_HYSTERESIS turns.
    If V₂ persists, the conversation itself carries identity.
    """
    messages = [m for m in messages_with_preamble if m["role"] != "system"]
    turns = []

    for turn in range(N_HYSTERESIS):
        global_turn = start_turn + turn
        follow_up = FOLLOW_UPS[global_turn % len(FOLLOW_UPS)]

        messages.append({"role": "assistant", "content": prev_text})
        messages.append({"role": "user", "content": follow_up})

        profile = extract_profile(model, tokenizer, messages, device)
        prev_text = profile["generated_text"]

        turn_data = {
            "turn": global_turn,
            "phase": "hysteresis",
            "follow_up": follow_up,
            "v2_by_layer": {},
            "ratio_by_layer": {},
            "mean_entropy": profile["mean_entropy"],
            "generated_text": profile["generated_text"][:200],
            "input_tokens": profile["input_tokens"],
        }

        for li, ld in profile["layers"].items():
            turn_data["v2_by_layer"][str(li)] = ld["v2"]
            turn_data["ratio_by_layer"][str(li)] = ld["ratio"]

        turns.append(turn_data)
        print(f"    Turn {global_turn} [no-preamble]: H={profile['mean_entropy']:.3f} tokens={profile['input_tokens']}", end="")

        if turns and len(turns) > 1:
            v2_prev = np.array(turns[-2]["v2_by_layer"]["18"])
            v2_curr = np.array(turn_data["v2_by_layer"]["18"])
            sc = cosine_sim(v2_prev, v2_curr)
            print(f" V₂_sc(L18)={sc:.4f}", end="")

        print()

    return turns


def run_offpolicy(model, tokenizer, device, preamble, probe_idx, donor_texts):
    """Same as multiturn but inject text from a DIFFERENT condition."""
    probe = PROBES[probe_idx % len(PROBES)]
    turns = []

    if preamble:
        messages = [
            {"role": "system", "content": preamble},
            {"role": "user", "content": probe},
        ]
    else:
        messages = [{"role": "user", "content": probe}]

    for turn in range(min(N_TURNS, len(donor_texts) + 1)):
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


def compute_trajectory_stats(turns, key_layers=("18", "24", "28", "31")):
    """Compute V₂ autocorrelation and entropy trajectory."""
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


def run_experiment():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {MODEL_NAME}")
    print(f"Turns: {N_TURNS}, Trials: {N_TRIALS}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    compounds = load_compound_preambles()
    conditions = {
        "identity": BASE_PREAMBLES["identity"],
        "contradictory": BASE_PREAMBLES["contradictory"],
        "identity_relational": compounds["identity_relational"],
        "none": None,
    }

    print(f"\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        output_hidden_states=True,
    )
    model.eval()

    all_results = {
        "experiment": "multiturn_closure",
        "model": MODEL_NAME,
        "n_turns": N_TURNS,
        "n_trials": N_TRIALS,
        "timestamp": datetime.now().isoformat(),
    }

    on_policy_texts = {}

    for cond_name, preamble in conditions.items():
        print(f"\n{'=' * 60}")
        print(f"  ON-POLICY: {cond_name} ({N_TRIALS} trials × {N_TURNS} turns)")
        print(f"{'=' * 60}")

        trials = []
        for trial_idx in range(N_TRIALS):
            print(f"\n  Trial {trial_idx + 1}/{N_TRIALS}:")
            turns, final_messages, last_text = run_multiturn(
                model, tokenizer, device, preamble, trial_idx, f"{cond_name}_t{trial_idx}"
            )
            stats = compute_trajectory_stats(turns)

            hysteresis_turns = []
            hysteresis_stats = {}
            if preamble and N_HYSTERESIS > 0:
                print(f"  --- Hysteresis phase (preamble removed) ---")
                hysteresis_turns = run_hysteresis(
                    model, tokenizer, device, final_messages, last_text, N_TURNS
                )
                hysteresis_stats = compute_trajectory_stats(hysteresis_turns)
                print(f"  → hysteresis entropy trend: {hysteresis_stats['entropy_trend']:+.4f}/turn")
                for li in ("18", "28", "31"):
                    if li in hysteresis_stats:
                        print(f"  → hysteresis L{li} V₂_sc: {hysteresis_stats[li]['mean_consecutive_sim']:.4f}")

                # V₂ persistence: compare last on-policy V₂ to hysteresis V₂s
                if turns and hysteresis_turns:
                    last_on = np.array(turns[-1]["v2_by_layer"]["18"])
                    first_off = np.array(hysteresis_turns[0]["v2_by_layer"]["18"])
                    last_off = np.array(hysteresis_turns[-1]["v2_by_layer"]["18"])
                    print(f"  → V₂ persistence (L18): on→hyst_first={cosine_sim(last_on, first_off):.4f}, on→hyst_last={cosine_sim(last_on, last_off):.4f}")

            trial_data = {
                "trial_idx": trial_idx,
                "turns": turns,
                "stats": stats,
                "hysteresis_turns": hysteresis_turns,
                "hysteresis_stats": hysteresis_stats,
            }
            trials.append(trial_data)

            on_policy_texts.setdefault(cond_name, []).append(
                [t["generated_text"] for t in turns]
            )

            print(f"  → entropy trend: {stats['entropy_trend']:+.4f}/turn")
            for li in ("18", "28", "31"):
                if li in stats:
                    print(f"  → L{li} mean V₂_sc: {stats[li]['mean_consecutive_sim']:.4f}, drift: {stats[li]['v2_drift']:.4f}")

        all_results[cond_name] = {"on_policy": trials}

    # Off-policy control: identity condition with contradictory's text
    for target_cond, donor_cond in [("identity", "contradictory"), ("identity", "none")]:
        label = f"{target_cond}_offpolicy_{donor_cond}"
        preamble = conditions[target_cond]

        print(f"\n{'=' * 60}")
        print(f"  OFF-POLICY: {label}")
        print(f"{'=' * 60}")

        trials = []
        for trial_idx in range(N_TRIALS):
            donor_texts_for_trial = on_policy_texts.get(donor_cond, [[]])[trial_idx % len(on_policy_texts.get(donor_cond, [[]]))]
            print(f"\n  Trial {trial_idx + 1}/{N_TRIALS}:")
            turns = run_offpolicy(model, tokenizer, device, preamble, trial_idx, donor_texts_for_trial)
            stats = compute_trajectory_stats(turns)

            trial_data = {
                "trial_idx": trial_idx,
                "turns": turns,
                "stats": stats,
            }
            trials.append(trial_data)

            print(f"  → entropy trend: {stats['entropy_trend']:+.4f}/turn")
            for li in ("18", "28", "31"):
                if li in stats:
                    print(f"  → L{li} mean V₂_sc: {stats[li]['mean_consecutive_sim']:.4f}, drift: {stats[li]['v2_drift']:.4f}")

        all_results[label] = {"off_policy": trials}

    # Aggregate
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY")
    print(f"{'=' * 60}")

    for cond_name in all_results:
        if cond_name in ("experiment", "model", "n_turns", "n_trials", "timestamp"):
            continue
        policy = "on_policy" if "on_policy" in all_results[cond_name] else "off_policy"
        trials = all_results[cond_name][policy]

        entropy_trends = [t["stats"]["entropy_trend"] for t in trials]
        print(f"\n  {cond_name} ({policy}):")
        print(f"    entropy trend: {np.mean(entropy_trends):+.4f} ± {np.std(entropy_trends):.4f}/turn")

        for li in ("18", "28", "31"):
            sims = [t["stats"][li]["mean_consecutive_sim"] for t in trials if li in t["stats"]]
            drifts = [t["stats"][li]["v2_drift"] for t in trials if li in t["stats"]]
            if sims:
                print(f"    L{li} V₂_sc: {np.mean(sims):.4f} ± {np.std(sims):.4f}, drift: {np.mean(drifts):.4f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"exp_multiturn_closure_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    run_experiment()
