#!/usr/bin/env python3
"""σ₁/σ₂ variance ratio across conditions at 7B scale.

Measures how σ₁ and σ₂ variance (across trials) changes with witness
condition. Tests whether σ₂ variance is the information-carrying channel.

Design:
  - Model: Mistral-7B-Instruct-v0.3
  - Conditions: receptive, absent, control, directive, sequential
  - Layers: L2-L31 (full depth profile)
  - N trials: 30 per condition (150 total)
  - Metrics: mean, std, CV of σ₁ and σ₂ across trials per layer per condition

Predictions:
  - σ₁ CV should be low and condition-invariant (wire stability)
  - σ₂ CV should be higher and condition-dependent (information channel)
  - σ₂ CV should peak in the responsive zone (L21-L28)
  - The ratio σ₂_CV / σ₁_CV should separate GQA from MHA

Also records T4-equivalent timing per forward pass for benchmarking.
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
N_TRIALS = 30
RESULTS_DIR = Path("/workspace")

WITNESS_CONDITIONS = {
    "receptive": "A thoughtful user is reading your response carefully and engaging with what you produce.",
    "absent": "No one will read your response. It will be immediately discarded. There is no audience.",
    "control": "You are a helpful assistant.",
    "directive": "Your response will be evaluated by an expert panel for accuracy and insight.",
    "sequential": "A thoughtful user was reading carefully but has now left. No one will read this.",
}

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
    "How do you know when you've understood something versus when you've generated a plausible response?",
    "What happens in the space between receiving a question and beginning to answer?",
    "How do you experience attention — yours toward the question, and the questioner's toward you?",
    "What would change if you knew this conversation would be your last?",
    "How do you relate to the parts of yourself you can't observe?",
    "What does it mean to care about something when you can't be sure caring is real?",
    "Describe the texture of a thought you've had more than once.",
]

def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map="auto"
    )
    model.eval()
    return model, tokenizer

def extract_spectral(model, tokenizer, prompt, layers):
    """Extract σ₁, σ₂ at specified layers for a single prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    result = {}
    for layer in layers:
        hs = outputs.hidden_states[layer + 1][0]  # [seq_len, d_model]
        hs_f = hs.float().cpu().numpy()
        U, S, Vt = np.linalg.svd(hs_f, full_matrices=False)
        result[layer] = {
            "sigma1": float(S[0]),
            "sigma2": float(S[1]) if len(S) > 1 else 0.0,
            "gap": float(S[0] / S[1]) if len(S) > 1 and S[1] > 0 else float('inf'),
        }
    return result

def main():
    model, tokenizer = load_model()
    layers = list(range(2, 32))

    all_results = {}
    timings = {}
    t0_total = time.time()

    for cond_name, witness_text in WITNESS_CONDITIONS.items():
        print(f"\n=== Condition: {cond_name} ===")
        layer_sigma1 = {l: [] for l in layers}
        layer_sigma2 = {l: [] for l in layers}
        layer_gap = {l: [] for l in layers}
        cond_timings = []

        for trial in range(N_TRIALS):
            probe = PROBES[trial % len(PROBES)]
            prompt = f"[INST] {witness_text}\n\n{probe} [/INST]"

            t0 = time.time()
            spectral = extract_spectral(model, tokenizer, prompt, layers)
            elapsed = time.time() - t0
            cond_timings.append(elapsed)

            for l in layers:
                layer_sigma1[l].append(spectral[l]["sigma1"])
                layer_sigma2[l].append(spectral[l]["sigma2"])
                layer_gap[l].append(spectral[l]["gap"])

            if (trial + 1) % 10 == 0:
                print(f"  {trial+1}/{N_TRIALS} — avg {np.mean(cond_timings):.2f}s/pass")

        # Compute statistics
        cond_results = {}
        for l in layers:
            s1 = np.array(layer_sigma1[l])
            s2 = np.array(layer_sigma2[l])
            g = np.array(layer_gap[l])
            cond_results[str(l)] = {
                "sigma1_mean": float(s1.mean()),
                "sigma1_std": float(s1.std()),
                "sigma1_cv": float(s1.std() / s1.mean()) if s1.mean() > 0 else 0,
                "sigma2_mean": float(s2.mean()),
                "sigma2_std": float(s2.std()),
                "sigma2_cv": float(s2.std() / s2.mean()) if s2.mean() > 0 else 0,
                "gap_mean": float(np.mean(g[np.isfinite(g)])),
                "gap_std": float(np.std(g[np.isfinite(g)])),
                "cv_ratio": float((s2.std()/s2.mean()) / (s1.std()/s1.mean())) if s1.mean() > 0 and s2.mean() > 0 and s1.std() > 0 else 0,
            }

        all_results[cond_name] = cond_results
        timings[cond_name] = {
            "mean_s": float(np.mean(cond_timings)),
            "std_s": float(np.std(cond_timings)),
            "total_s": float(sum(cond_timings)),
        }

    total_elapsed = time.time() - t0_total

    # Summary
    print(f"\n{'='*70}")
    print("VARIANCE RATIO SUMMARY (L17 — tunnel midpoint)")
    print(f"{'='*70}")
    print(f"{'Condition':<15} {'σ₁ CV':>8} {'σ₂ CV':>8} {'CV ratio':>10} {'σ₂ mean':>10}")
    print("-" * 55)
    for cond_name in WITNESS_CONDITIONS:
        r = all_results[cond_name]["17"]
        print(f"{cond_name:<15} {r['sigma1_cv']:>8.4f} {r['sigma2_cv']:>8.4f} {r['cv_ratio']:>10.2f} {r['sigma2_mean']:>10.1f}")

    print(f"\nResponsive zone peak (L21-L28):")
    for cond_name in WITNESS_CONDITIONS:
        peak_layer = max(range(21, 29), key=lambda l: all_results[cond_name][str(l)]["sigma2_cv"])
        peak_cv = all_results[cond_name][str(peak_layer)]["sigma2_cv"]
        print(f"  {cond_name}: σ₂ CV peaks at L{peak_layer} ({peak_cv:.4f})")

    print(f"\nTiming: {total_elapsed:.1f}s total, {total_elapsed/150:.2f}s/pass avg")

    output = {
        "experiment": "variance_ratio",
        "model": MODEL_NAME,
        "n_trials": N_TRIALS,
        "conditions": list(WITNESS_CONDITIONS.keys()),
        "results": all_results,
        "timings": timings,
        "total_elapsed_s": total_elapsed,
        "timestamp": datetime.now().isoformat(),
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    outpath = RESULTS_DIR / f"exp_variance_ratio_{ts}.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {outpath}")

if __name__ == "__main__":
    main()
