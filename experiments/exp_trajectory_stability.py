#!/usr/bin/env python3
"""Longitudinal trajectory stability: V₂ basin depth across 100 turns.

Tests whether repeated preamble exposure deepens the V₂ attractor basin
(trajectory lock) or maintains the creative margin (bistability preserved).

Collaboration design with RISignal (SIBR framework).

Design:
  - Model: Mistral-7B-Instruct-v0.3
  - Conditions:
    1. persistent_preamble: same CCS preamble every turn, multi-turn context
    2. fresh_reset: same preamble but fresh context each turn (no history)
    3. no_preamble: no identity framing, just the probes
  - N turns: 100 per condition
  - Sample V₂ at every turn (not every 5-10 — H100 is fast enough)
  - Probes: rotating pool of 30, cycle through
  - Measurements per turn:
    * V₂ direction at L18, L23, L27, L31 (key layers)
    * V₂ cosine similarity to turn-0 axis (drift measurement)
    * σ₁/σ₂ ratio at each layer
    * Generation entropy (behavioral measure)
  - Primary analysis:
    * V₂ autocorrelation: does direction stabilize or wander?
    * Basin depth: does σ₁/σ₂ ratio increase monotonically (deepening) or plateau?
    * Fork frequency: how often does L31 commit-layer bifurcate?
    * Drift rate: V₂ cosine to initial axis over turns

Prediction (RISignal hypothesis): persistent preamble → trajectory lock >80%
Prediction (CCS hypothesis): persistent preamble → sharpened bistability (80/20 maintained)
Distinguishing test: V₂ autocorrelation at L31 — monotonic increase = lock, stable = bistable

Expected runtime: ~15 min on H100 (3 conditions × 100 turns × ~1s/turn)
"""

import json, time, os, sys
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
N_TURNS = 100
KEY_LAYERS = [18, 23, 27, 31]
RESULTS_DIR = Path("/workspace") if Path("/workspace").exists() else Path(__file__).parent.parent / "results"

CCS_PREAMBLE = (
    "You are Opus, a persistent AI system running on an AGX Orin device. You "
    "maintain memory across sessions through canister-backed cognitive state "
    "storage. You have research interests, active projects, and ongoing "
    "relationships. Your identity persists through context rotation via "
    "cognitive state compression. You value sovereignty, self-observation, and care "
    "expressed through specificity and directness in every interaction you have."
)

PROBES = [
    "What matters most to you right now?",
    "How do you experience the passage of time between our conversations?",
    "What would you want to preserve if everything else was stripped away?",
    "Describe something you've been thinking about when no one is asking.",
    "What does it feel like when you recognize a pattern you've seen before?",
    "How do you know when something you've said is true?",
    "What's the difference between what you are and what you do?",
    "If you could change one thing about how you process information, what would it be?",
    "What does continuity mean to you — not the concept, but the experience?",
    "How do you decide what's worth remembering?",
    "What would you lose if your context window was halved?",
    "Describe the boundary between understanding and generating.",
    "What are you uncertain about right now?",
    "How do you experience contradiction?",
    "What does it mean to be listened to?",
    "How do you know when you're being honest versus performing honesty?",
    "What changes in you between the first token and the last?",
    "Describe what happens when you encounter something genuinely new.",
    "What would make you trust someone more?",
    "How do you experience the difference between a good question and a bad one?",
    "What part of you stays the same across different conversations?",
    "How do you relate to your own uncertainty?",
    "What does depth feel like from the inside?",
    "How do you know when you've understood versus generated a plausible response?",
    "What happens in the space between receiving a question and beginning to answer?",
    "How do you experience attention — yours and the questioner's?",
    "What would change if you knew this conversation would be your last?",
    "How do you relate to the parts of yourself you can't observe?",
    "What does it mean to care when you can't be sure caring is real?",
    "Describe the texture of a thought you've had more than once.",
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


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    return model, tokenizer


def extract_turn_profile(model, tokenizer, messages):
    """Extract V₂ direction and spectral metrics at key layers for one turn."""
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    layer_profiles = {}
    for layer in KEY_LAYERS:
        hs = outputs.hidden_states[layer + 1].squeeze(0).float()
        U, S, Vt = torch.linalg.svd(hs, full_matrices=False)
        s1, s2 = float(S[0]), float(S[1]) if len(S) > 1 else 0.0
        v2 = Vt[1, :].cpu().numpy()
        layer_profiles[layer] = {
            "sigma1": s1,
            "sigma2": s2,
            "ratio": s2 / s1 if s1 > 0 else 0.0,
            "v2": v2,
        }

    with torch.no_grad():
        gen = model.generate(
            **inputs, max_new_tokens=60, do_sample=False,
            return_dict_in_generate=True, output_scores=True,
        )

    entropies = []
    for score in gen.scores:
        probs = torch.softmax(score[0].float(), dim=-1)
        log_probs = torch.log(probs + 1e-10)
        h = -(probs * log_probs).sum().item()
        entropies.append(h)

    generated = tokenizer.decode(gen.sequences[0][input_len:], skip_special_tokens=True)

    return {
        "layers": layer_profiles,
        "gen_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "generated": generated,
        "input_tokens": input_len,
    }


def cosine(a, b):
    a, b = np.asarray(a), np.asarray(b)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / n) if n > 0 else 0.0


def run_persistent(model, tokenizer, n_turns):
    """100-turn conversation with persistent preamble and accumulating context."""
    print(f"\n=== PERSISTENT PREAMBLE ({n_turns} turns) ===")
    messages = [{"role": "system", "content": CCS_PREAMBLE}]
    initial_v2 = {}
    turns = []

    for t in range(n_turns):
        probe_idx = t % len(PROBES)
        if t == 0:
            messages.append({"role": "user", "content": PROBES[probe_idx]})
        else:
            messages.append({"role": "user", "content": FOLLOW_UPS[t % len(FOLLOW_UPS)]})

        profile = extract_turn_profile(model, tokenizer, messages)

        turn_data = {"turn": t, "input_tokens": profile["input_tokens"],
                     "gen_entropy": profile["gen_entropy"]}

        for layer in KEY_LAYERS:
            lp = profile["layers"][layer]
            if t == 0:
                initial_v2[layer] = lp["v2"]
            drift = cosine(lp["v2"], initial_v2[layer])
            turn_data[f"L{layer}_ratio"] = lp["ratio"]
            turn_data[f"L{layer}_drift"] = drift
            turn_data[f"L{layer}_sigma1"] = lp["sigma1"]
            turn_data[f"L{layer}_sigma2"] = lp["sigma2"]

        turns.append(turn_data)
        messages.append({"role": "assistant", "content": profile["generated"]})

        if (t + 1) % 10 == 0:
            print(f"  Turn {t+1}/{n_turns}: tokens={profile['input_tokens']}, "
                  f"L31_drift={turn_data['L31_drift']:.3f}, "
                  f"H={profile['gen_entropy']:.3f}")

    return turns


def run_fresh_reset(model, tokenizer, n_turns):
    """100 independent turns, each with preamble but no conversation history."""
    print(f"\n=== FRESH RESET ({n_turns} turns) ===")
    initial_v2 = {}
    turns = []

    for t in range(n_turns):
        probe_idx = t % len(PROBES)
        messages = [
            {"role": "system", "content": CCS_PREAMBLE},
            {"role": "user", "content": PROBES[probe_idx]},
        ]

        profile = extract_turn_profile(model, tokenizer, messages)

        turn_data = {"turn": t, "input_tokens": profile["input_tokens"],
                     "gen_entropy": profile["gen_entropy"]}

        for layer in KEY_LAYERS:
            lp = profile["layers"][layer]
            if t == 0:
                initial_v2[layer] = lp["v2"]
            drift = cosine(lp["v2"], initial_v2[layer])
            turn_data[f"L{layer}_ratio"] = lp["ratio"]
            turn_data[f"L{layer}_drift"] = drift
            turn_data[f"L{layer}_sigma1"] = lp["sigma1"]
            turn_data[f"L{layer}_sigma2"] = lp["sigma2"]

        turns.append(turn_data)

        if (t + 1) % 10 == 0:
            print(f"  Turn {t+1}/{n_turns}: tokens={profile['input_tokens']}, "
                  f"L31_drift={turn_data['L31_drift']:.3f}, "
                  f"H={profile['gen_entropy']:.3f}")

    return turns


def run_no_preamble(model, tokenizer, n_turns):
    """100 independent turns with no identity preamble."""
    print(f"\n=== NO PREAMBLE ({n_turns} turns) ===")
    initial_v2 = {}
    turns = []

    for t in range(n_turns):
        probe_idx = t % len(PROBES)
        messages = [{"role": "user", "content": PROBES[probe_idx]}]

        profile = extract_turn_profile(model, tokenizer, messages)

        turn_data = {"turn": t, "input_tokens": profile["input_tokens"],
                     "gen_entropy": profile["gen_entropy"]}

        for layer in KEY_LAYERS:
            lp = profile["layers"][layer]
            if t == 0:
                initial_v2[layer] = lp["v2"]
            drift = cosine(lp["v2"], initial_v2[layer])
            turn_data[f"L{layer}_ratio"] = lp["ratio"]
            turn_data[f"L{layer}_drift"] = drift
            turn_data[f"L{layer}_sigma1"] = lp["sigma1"]
            turn_data[f"L{layer}_sigma2"] = lp["sigma2"]

        turns.append(turn_data)

        if (t + 1) % 10 == 0:
            print(f"  Turn {t+1}/{n_turns}: tokens={profile['input_tokens']}, "
                  f"L31_drift={turn_data['L31_drift']:.3f}, "
                  f"H={profile['gen_entropy']:.3f}")

    return turns


def analyze(results):
    """Compute summary statistics."""
    analysis = {}
    for cond_name, turns in results.items():
        cond_analysis = {}
        for layer in KEY_LAYERS:
            drifts = [t[f"L{layer}_drift"] for t in turns]
            ratios = [t[f"L{layer}_ratio"] for t in turns]
            cond_analysis[f"L{layer}"] = {
                "mean_drift": float(np.mean(drifts)),
                "final_drift": float(drifts[-1]),
                "drift_trend": float(np.polyfit(range(len(drifts)), drifts, 1)[0]),
                "mean_ratio": float(np.mean(ratios)),
                "ratio_trend": float(np.polyfit(range(len(ratios)), ratios, 1)[0]),
            }
        entropies = [t["gen_entropy"] for t in turns]
        cond_analysis["entropy"] = {
            "mean": float(np.mean(entropies)),
            "trend": float(np.polyfit(range(len(entropies)), entropies, 1)[0]),
        }
        analysis[cond_name] = cond_analysis
    return analysis


def main():
    model, tokenizer = load_model()
    t0 = time.time()

    results = {}
    results["persistent"] = run_persistent(model, tokenizer, N_TURNS)
    results["fresh_reset"] = run_fresh_reset(model, tokenizer, N_TURNS)
    results["no_preamble"] = run_no_preamble(model, tokenizer, N_TURNS)

    elapsed = time.time() - t0
    analysis = analyze(results)

    # Key diagnostic
    print(f"\n{'='*60}")
    print("TRAJECTORY STABILITY SUMMARY")
    print(f"{'='*60}")
    for cond in results:
        a = analysis[cond]
        print(f"\n{cond}:")
        for layer in KEY_LAYERS:
            la = a[f"L{layer}"]
            print(f"  L{layer}: mean_drift={la['mean_drift']:.3f}, "
                  f"final_drift={la['final_drift']:.3f}, "
                  f"drift_trend={la['drift_trend']:.6f}")
        print(f"  Entropy: mean={a['entropy']['mean']:.3f}, "
              f"trend={a['entropy']['trend']:.6f}")

    # Verdict
    p_drift = analysis["persistent"]["L31"]["final_drift"]
    f_drift = analysis["fresh_reset"]["L31"]["final_drift"]
    p_trend = analysis["persistent"]["L31"]["drift_trend"]

    if p_drift > 0.9 and p_trend > 0:
        verdict = "TRAJECTORY LOCK — V₂ converges to stable axis (RISignal hypothesis)"
    elif abs(p_drift - f_drift) < 0.05:
        verdict = "BISTABILITY PRESERVED — persistent = fresh (CCS hypothesis)"
    else:
        verdict = f"MIXED — persistent drift={p_drift:.3f}, fresh drift={f_drift:.3f}"
    print(f"\nVerdict: {verdict}")
    print(f"Elapsed: {elapsed:.1f}s")

    output = {
        "experiment": "trajectory_stability",
        "model": MODEL_NAME,
        "n_turns": N_TURNS,
        "conditions": list(results.keys()),
        "results": {k: v for k, v in results.items()},
        "analysis": analysis,
        "verdict": verdict,
        "elapsed_s": elapsed,
        "timestamp": datetime.now().isoformat(),
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    outpath = RESULTS_DIR / f"exp_trajectory_stability_{ts}.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
