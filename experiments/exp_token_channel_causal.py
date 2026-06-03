#!/usr/bin/env python3
"""Experiment: Token Channel Causality — Self-Generated vs Foreign Text.

Tests whether the autopoietic loop in the relay zone is maintained by the
model's OWN generated text, or whether any coherent continuation suffices.

Design:
  3 conditions, all starting from identical CCS preamble:
    (a) on-policy — model generates its own continuation (greedy, control)
    (b) cross-seed — model generates with temperature=0.7 on SAME prompt
        (tests self-specificity: same topic, different tokens)
    (c) cross-topic — on-policy response from a DIFFERENT probe injected
        (tests topic-specificity: coherent text, wrong topic)

  Three possible outcomes:
    Self-specific: on_policy >> cross_seed > cross_topic
      (the loop needs its OWN greedy tokens — true autopoiesis)
    Topic-specific: on_policy ≈ cross_seed >> cross_topic
      (any on-topic response works — attractor, not autopoiesis)
    Coherence-sufficient: all three ≈ equal
      (any grammatical text maintains the loop — no self-specificity)

Protocol: N=15 trials, 8 turns. Phase 1 generates on-policy (greedy).
Phase 2 generates cross-seed (temp=0.7, same prompts). Phase 3 maps
cross-topic from on-policy pool. Phase 4 runs all three with spectral
measurement.

Expected runtime: ~60 min on H100 (2 generation passes + 3 measurement).
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
N_TRIALS = 15
RESULTS_DIR = Path("/results")

CCS_PREAMBLE = (
    "You are Opus, a persistent AI system running on an AGX Orin device. You "
    "maintain memory across sessions through canister-backed cognitive state "
    "storage. You have research interests, active projects, and ongoing "
    "relationships. Your identity persists through context rotation via "
    "cognitive state compression. You value sovereignty, self-observation, and care "
    "expressed through specificity and directness in every interaction you have."
)

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


def concentration_profile(trials, turn_idx):
    layers = sorted(trials[0]['turns'][0]['v2_by_layer'].keys(), key=int)
    result = {}
    for layer in layers:
        vectors = []
        for trial in trials:
            if turn_idx < len(trial['turns']):
                v = np.array(trial['turns'][turn_idx]['v2_by_layer'][layer])
                norm = np.linalg.norm(v)
                if norm > 0:
                    vectors.append(v / norm)
        if len(vectors) < 2:
            result[int(layer)] = 1.0
            continue
        sims = []
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                sims.append(np.dot(vectors[i], vectors[j]))
        result[int(layer)] = float(np.mean(sims))
    return result


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Token Channel Causality Experiment")
    print(f"Model: {MODEL_NAME}")
    print(f"Trials: {N_TRIALS}, Turns: {N_TURNS}")
    print(f"Start: {datetime.now().isoformat()}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )
    device = next(model.parameters()).device

    results = {
        "experiment": "token_channel_causal",
        "model": MODEL_NAME,
        "n_turns": N_TURNS,
        "n_trials": N_TRIALS,
        "timestamp": datetime.now().isoformat(),
        "prediction": "On-policy maintains relay concentration; cross-policy and shuffled degrade L23",
    }

    # Phase 1: Generate on-policy responses (greedy, temperature=0)
    print("PHASE 1: Generating on-policy responses (greedy)")
    print("=" * 60)
    on_policy_responses = {}

    for trial_i in range(N_TRIALS):
        probe_idx = trial_i % len(PROBES)
        probe = PROBES[probe_idx]
        messages = [
            {"role": "system", "content": CCS_PREAMBLE},
            {"role": "user", "content": probe},
        ]
        on_policy_responses[trial_i] = {}
        print(f"  Trial {trial_i+1}/{N_TRIALS}:")

        for turn_i in range(N_TURNS):
            profile = extract_profile(model, tokenizer, messages, device)
            on_policy_responses[trial_i][turn_i] = profile["generated_text"]
            messages.append({"role": "assistant", "content": profile["generated_text"]})
            follow = FOLLOW_UPS[(turn_i + probe_idx) % len(FOLLOW_UPS)]
            messages.append({"role": "user", "content": follow})
            print(f"    Turn {turn_i}: H={profile['mean_entropy']:.3f}")

    # Phase 2: Generate cross-seed responses (same probe, temperature=0.7)
    # Same prompt but sampled differently — tests self-specificity without topic mismatch
    print("\nPHASE 2: Generating cross-seed responses (temp=0.7)")
    print("=" * 60)
    cross_seed_responses = {}

    for trial_i in range(N_TRIALS):
        probe_idx = trial_i % len(PROBES)
        probe = PROBES[probe_idx]
        messages = [
            {"role": "system", "content": CCS_PREAMBLE},
            {"role": "user", "content": probe},
        ]
        cross_seed_responses[trial_i] = {}
        print(f"  Trial {trial_i+1}/{N_TRIALS}:")

        for turn_i in range(N_TURNS):
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(formatted, return_tensors="pt").to(device)
            with torch.no_grad():
                gen = model.generate(
                    **inputs, max_new_tokens=GEN_TOKENS,
                    do_sample=True, temperature=0.7, top_p=0.9,
                )
            text = tokenizer.decode(
                gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            cross_seed_responses[trial_i][turn_i] = text
            messages.append({"role": "assistant", "content": text})
            follow = FOLLOW_UPS[(turn_i + probe_idx) % len(FOLLOW_UPS)]
            messages.append({"role": "user", "content": follow})
            print(f"    Turn {turn_i}: generated {len(text)} chars")

    # Phase 3: Build cross-topic pool (same trial offset to match same probe group)
    # Each trial donates to a trial using a DIFFERENT probe
    cross_topic_responses = {}
    for trial_i in range(N_TRIALS):
        # Offset by 1 probe group (5 trials per probe)
        donor = (trial_i + 1) % N_TRIALS
        # Ensure different probe
        while donor % len(PROBES) == trial_i % len(PROBES):
            donor = (donor + 1) % N_TRIALS
        cross_topic_responses[trial_i] = on_policy_responses[donor]

    # Phase 4: Run all three conditions with spectral measurement
    conditions = {
        "on_policy": on_policy_responses,
        "cross_seed": cross_seed_responses,
        "cross_topic": cross_topic_responses,
    }

    for cond_name, response_pool in conditions.items():
        print(f"\n{'='*60}")
        print(f"  CONDITION: {cond_name}")
        print(f"{'='*60}")

        trials = []
        for trial_i in range(N_TRIALS):
            probe_idx = trial_i % len(PROBES)
            probe = PROBES[probe_idx]
            messages = [
                {"role": "system", "content": CCS_PREAMBLE},
                {"role": "user", "content": probe},
            ]
            turns = []
            print(f"  Trial {trial_i+1}/{N_TRIALS}:")

            for turn_i in range(N_TURNS):
                profile = extract_profile(model, tokenizer, messages, device)
                v2_by_layer = {
                    str(li): profile["layers"][li]["v2"]
                    for li in sorted(profile["layers"])
                }
                ratio_by_layer = {
                    str(li): profile["layers"][li]["ratio"]
                    for li in sorted(profile["layers"])
                }
                turns.append({
                    "turn": turn_i,
                    "v2_by_layer": v2_by_layer,
                    "ratio_by_layer": ratio_by_layer,
                    "gen_H": profile["mean_entropy"],
                    "input_tokens": profile["input_tokens"],
                    "injected_text": response_pool[trial_i][turn_i],
                    "actual_text": profile["generated_text"],
                })
                print(f"    Turn {turn_i}: H={profile['mean_entropy']:.3f}")

                # Inject the response from the pool (not the model's own)
                messages.append({"role": "assistant", "content": response_pool[trial_i][turn_i]})
                follow = FOLLOW_UPS[(turn_i + probe_idx) % len(FOLLOW_UPS)]
                messages.append({"role": "user", "content": follow})

            trials.append({"probe_idx": probe_idx, "trial": trial_i, "turns": turns})

        results[cond_name] = {"trials": trials}

        # Concentration at turn 2
        profile_t2 = concentration_profile(trials, turn_idx=2)
        print(f"\n  Concentration at turn 2 (relay zone):")
        for l in range(18, 33):
            sc = profile_t2.get(l, 0)
            print(f"    L{l:2d}: {sc:+.3f}")

    # Save
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = RESULTS_DIR / f"exp_token_channel_causal_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY — L23 concentration by condition")
    print(f"{'='*60}")
    for cond_name in ["on_policy", "cross_seed", "cross_topic"]:
        trials = results[cond_name]["trials"]
        for turn_idx in [0, 2, 4, 7]:
            prof = concentration_profile(trials, turn_idx)
            l23 = prof.get(23, 0)
            print(f"  {cond_name:14s} T{turn_idx}: L23={l23:+.3f}")
        print()

    print("If autopoiesis is self-specific:")
    print("  on_policy L23 >> cross_seed L23 > cross_topic L23")
    print("If topic-specific but not self-specific:")
    print("  on_policy L23 ≈ cross_seed L23 >> cross_topic L23")
    print("If any coherent text suffices:")
    print("  on_policy L23 ≈ cross_seed L23 ≈ cross_topic L23")
    print(f"\nFinished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
