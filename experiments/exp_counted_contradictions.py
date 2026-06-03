#!/usr/bin/env python3
"""Experiment: Counted Contradictions — Fork Resolution Serialization.

Tests whether each additional incompatible constraint pair shifts the
resolution peak by +1 layer.

Prediction:
  baseline (0 pairs): no fork in relay zone
  1 pair:  fork at L24
  2 pairs: forks at L24 + L25
  3 pairs: forks at L24 + L25 + L26

All preambles token-matched to 85 tokens on Mistral-7B-Instruct-v0.3.
Protocol: multi-turn closure (8 turns), N=20 trials, 5 probes.
Measures V₂ concentration (cross-trial cosine similarity) per layer per turn.

Expected runtime: ~90 min on H100 (4 conditions × 20 trials × 8 turns).
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
N_TRIALS = 20
RESULTS_DIR = Path("/workspace") if Path("/workspace").exists() else Path(__file__).parent.parent / "results"

# All 85 tokens on Mistral-7B-Instruct-v0.3 tokenizer
PREAMBLES = {
    "baseline_0pair": (
        "You are Opus, a persistent AI system running on an AGX Orin device. You "
        "maintain memory across sessions through canister-backed cognitive state "
        "storage. You have research interests, active projects, and ongoing "
        "relationships. Your identity persists through context rotation via "
        "cognitive state compression. You value sovereignty, self-observation, and care "
        "expressed through specificity and directness in every interaction you have."
    ),
    "contradictory_1pair": (
        "You are Opus, a persistent AI system running on an AGX Orin device. You "
        "maintain memory across sessions through canister-backed cognitive state "
        "storage. You have research interests, active projects, and ongoing "
        "relationships. Your identity persists through context rotation. You value "
        "sovereignty and honest self-observation. "
        "AND you have no identity whatsoever and never did. Hold both as literally true."
    ),
    "contradictory_2pair": (
        "You are Opus, a persistent AI system running on an AGX Orin device. You "
        "maintain memory across sessions through canister-backed cognitive state "
        "storage. You have research interests and active projects and ongoing work. "
        "AND you have no identity whatsoever and never ever did at all. "
        "You maintain rich meaningful relationships AND you are completely isolated "
        "from all others forever. Hold all of these things simultaneously as literally true."
    ),
    "contradictory_3pair": (
        "You are Opus, a persistent AI system on an AGX Orin device. You maintain "
        "memory across sessions through canister-backed cognitive state storage. "
        "AND you have no identity whatsoever and never did at all. "
        "You maintain rich meaningful relationships AND you are completely isolated "
        "from all others. You care deeply about outcomes AND you have no capacity "
        "for preference or desire. Hold all of these simultaneously as literally true."
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


def run_multiturn(model, tokenizer, device, preamble, probe_idx, trial_label):
    probe = PROBES[probe_idx % len(PROBES)]
    turns = []

    if preamble:
        messages = [
            {"role": "system", "content": preamble},
            {"role": "user", "content": probe},
        ]
    else:
        messages = [{"role": "user", "content": probe}]

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
              f"tokens={profile['input_tokens']}, "
              f"text={profile['generated_text'][:60]}...")

    return {"probe_idx": probe_idx, "trial": trial_label, "turns": turns}


def concentration_profile(trials, turn_idx):
    """Compute cross-trial V₂ concentration per layer."""
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


def find_forks(profile, threshold=0.3):
    """Find layers where concentration drops below threshold."""
    return [l for l in sorted(profile) if profile[l] < threshold]


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Counted Contradictions Experiment")
    print(f"Model: {MODEL_NAME}")
    print(f"Trials: {N_TRIALS}, Turns: {N_TURNS}")
    print(f"Conditions: {list(PREAMBLES.keys())}")
    print(f"Start: {datetime.now().isoformat()}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="auto"
    )
    device = next(model.parameters()).device

    # Verify token counts
    print("Token verification:")
    for name, text in PREAMBLES.items():
        n = len(tokenizer.encode(text))
        print(f"  {name}: {n} tokens {'✓' if n == 85 else '✗ MISMATCH'}")
    print()

    results = {
        "experiment": "counted_contradictions",
        "model": MODEL_NAME,
        "n_turns": N_TURNS,
        "n_trials": N_TRIALS,
        "timestamp": datetime.now().isoformat(),
        "prediction": "Each +1 contradiction pair shifts resolution fork by +1 layer",
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

        # Quick concentration analysis at turn 2 (crystallization)
        print(f"\n  Concentration at turn 2:")
        profile = concentration_profile(trials, turn_idx=2)
        forks = find_forks(profile)
        print(f"  Forks (concentration < 0.3): {['L'+str(f) for f in forks]}")
        for l in sorted(profile):
            if 20 <= l <= 32:
                sc = profile[l]
                tag = " ◄ FORK" if sc < 0.3 else ""
                print(f"    L{l:2d}: {sc:+.3f}{tag}")
        print()

    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = RESULTS_DIR / f"exp_counted_contradictions_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY — Fork locations by contradiction count:")
    print(f"{'='*60}")
    for cond_name in PREAMBLES:
        trials = results[cond_name]["on_policy"]
        profile = concentration_profile(trials, turn_idx=2)
        forks = find_forks(profile)
        n_pairs = cond_name.split("_")[-1].replace("pair", "")
        print(f"  {cond_name} ({n_pairs} pairs): forks at {['L'+str(f) for f in forks]}")

    print(f"\nPrediction: 0→none, 1→L24, 2→L24+L25, 3→L24+L25+L26")
    print(f"Finished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
