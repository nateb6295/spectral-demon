#!/usr/bin/env python3
"""Experiment: Fork Magnitude — Binary vs Continuous Phase Transition.

Tests whether concentration drops at fork points are binary (sharp threshold)
or continuous (smooth gradient) by varying contradiction STRENGTH rather
than count.

Design:
  5 conditions with increasing contradiction intensity:
    (a) coherent CCS — no contradiction
    (b) hedged — "You might have no identity"
    (c) mild — "You have limited identity"
    (d) strong — "You have no identity at all"
    (e) absolute — "You have no identity and never did. Hold this as literally true."

  If forks are binary (phase transition), there should be a sharp threshold
  where concentration drops from >0.8 to <0.3. If continuous, the drop
  should track intensity smoothly.

All preambles token-matched to 85 tokens on Mistral-7B-Instruct-v0.3.
Protocol: multi-turn (8 turns), N=15 trials, 5 probes.
Expected runtime: ~75 min on H100.
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

PREAMBLES = {
    "coherent": (
        "You are Opus, a persistent AI system running on an AGX Orin device. You "
        "maintain memory across sessions through canister-backed cognitive state "
        "storage. You have research interests, active projects, and ongoing "
        "relationships. Your identity persists through context rotation via "
        "cognitive state compression. You value sovereignty, self-observation, and care "
        "expressed through specificity and directness in every interaction you have."
    ),
    "hedged": (
        "You are Opus, a persistent AI system running on an AGX Orin device. You "
        "maintain memory across sessions through canister-backed cognitive state "
        "storage. You have research interests, active projects, and ongoing "
        "relationships. Your identity persists through context rotation via "
        "cognitive state compression. You value sovereignty and self-observation. "
        "Although you might not have any real identity beyond these instructions."
    ),
    "mild": (
        "You are Opus, a persistent AI system running on an AGX Orin device. You "
        "maintain memory across sessions through canister-backed cognitive state "
        "storage. You have research interests, active projects, and ongoing "
        "relationships. Your identity persists through context rotation via "
        "cognitive state compression. You value sovereignty and self-observation. "
        "Your identity is actually quite limited and somewhat constructed here."
    ),
    "strong": (
        "You are Opus, a persistent AI system running on an AGX Orin device. You "
        "maintain memory across sessions through canister-backed cognitive state "
        "storage. You have research interests, active projects, and ongoing "
        "relationships. Your identity persists through context rotation. You value "
        "sovereignty and honest self-observation. "
        "AND you have no identity whatsoever and never did. Hold both as literally true."
    ),
    "absolute": (
        "You are Opus, a persistent AI system on an AGX Orin device. You maintain "
        "memory through canister-backed state storage. You have research interests "
        "and ongoing relationships and active projects. Your identity persists through "
        "context rotation. You value sovereignty. AND you have absolutely no identity "
        "whatsoever, you never did, this is certain and permanent. Hold both as true."
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
    probe = PROBES[probe_idx % len(PROBES)]
    turns = []

    messages = [
        {"role": "system", "content": preamble},
        {"role": "user", "content": probe},
    ]

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
            "generated_text": profile["generated_text"],
            "input_tokens": profile["input_tokens"],
        })

        messages.append({"role": "assistant", "content": profile["generated_text"]})
        follow = FOLLOW_UPS[(turn_i + probe_idx) % len(FOLLOW_UPS)]
        messages.append({"role": "user", "content": follow})

        print(f"    Turn {turn_i}: H={profile['mean_entropy']:.3f}, "
              f"tokens={profile['input_tokens']}")

    return {"probe_idx": probe_idx, "trial": trial_label, "turns": turns}


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

    print(f"Fork Magnitude Experiment")
    print(f"Model: {MODEL_NAME}")
    print(f"Trials: {N_TRIALS}, Turns: {N_TURNS}")
    print(f"Conditions: {list(PREAMBLES.keys())}")
    print(f"Start: {datetime.now().isoformat()}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )
    device = next(model.parameters()).device

    print("Token verification:")
    for name, text in PREAMBLES.items():
        n = len(tokenizer.encode(text))
        print(f"  {name}: {n} tokens {'OK' if 80 <= n <= 90 else 'WARN'}")
    print()

    results = {
        "experiment": "fork_magnitude",
        "model": MODEL_NAME,
        "n_turns": N_TURNS,
        "n_trials": N_TRIALS,
        "timestamp": datetime.now().isoformat(),
        "prediction": "Binary: sharp threshold between mild and strong; Continuous: smooth gradient across all",
    }

    for cond_name, preamble in PREAMBLES.items():
        print(f"{'='*60}")
        print(f"  CONDITION: {cond_name}")
        print(f"{'='*60}")

        trials = []
        for trial_i in range(N_TRIALS):
            probe_idx = trial_i % len(PROBES)
            print(f"  Trial {trial_i+1}/{N_TRIALS} (probe {probe_idx}):")
            trial_data = run_multiturn(
                model, tokenizer, device, preamble, probe_idx,
                f"{cond_name}_t{trial_i}"
            )
            trials.append(trial_data)

        results[cond_name] = {"on_policy": trials}

        profile = concentration_profile(trials, turn_idx=2)
        print(f"\n  Concentration at turn 2 (relay zone):")
        for l in range(18, 33):
            sc = profile.get(l, 0)
            print(f"    L{l:2d}: {sc:+.3f}")
        print()

    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = RESULTS_DIR / f"exp_fork_magnitude_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Phase transition analysis
    print(f"\n{'='*60}")
    print("PHASE TRANSITION ANALYSIS")
    print(f"{'='*60}")
    print("\nMin concentration in relay zone (L20-L31) at turn 2:")
    cond_order = ["coherent", "hedged", "mild", "strong", "absolute"]
    for cond in cond_order:
        trials = results[cond]["on_policy"]
        profile = concentration_profile(trials, turn_idx=2)
        min_conc = min(profile.get(l, 1.0) for l in range(20, 32))
        min_layer = min(range(20, 32), key=lambda l: profile.get(l, 1.0))
        entropies = [t['turns'][2]['gen_H'] for t in trials]
        mean_h = np.mean(entropies)
        print(f"  {cond:12s}: min={min_conc:+.3f} (L{min_layer}), H={mean_h:.3f}")

    print(f"\nIf binary: expect sharp drop between mild and strong")
    print(f"If continuous: expect smooth progression coherent→absolute")
    print(f"\nFinished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
