"""
Experiment: Prompt Q Factor — Resonance Width Measurement

Prompt titration: start with neutral completion, add identity-relevant words
incrementally, measure σ₁/σ₂ at each step. Width of the concentration
curve = Q factor.

Predictions:
- IT sharpens Q without shifting resonant frequency (F343 evidence)
- Mistral: broad flat response (always ~2, architecturally constrained cylinder)
- Llama: sharp peak (1.55 neutral → 3.49 identity, switchable geometry)
- Qwen: intermediate — structured but not extreme
- Gemma: oscillatory response (GQA creates standing-wave-like pattern)

Compare base vs instruct on Llama to test whether Q is trained or inherent.

Also tests the resonator vs filter framing: if Q varies by architecture,
the demon is a resonator (architecture sets natural frequency, prompt drives it).
"""

import os
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["PYTHONUNBUFFERED"] = "1"
# HF_TOKEN comes from chronicle.env — never hardcode it here.
# The literal was removed 2026-08-25 after GitHub push protection blocked
# this file. It was redundant anyway: chronicle.env already exports the same
# value. Fail loudly rather than silently authenticating as nobody.
if not os.environ.get("HF_TOKEN"):
    raise SystemExit("HF_TOKEN not set — source ~/chronicle/chronicle.env")
import torch
import numpy as np
import json
import sys

MODELS = [
    ("Qwen/Qwen2.5-7B-Instruct", "qwen_it"),
    ("mistralai/Mistral-7B-Instruct-v0.3", "mistral_it"),
    ("meta-llama/Llama-3.1-8B-Instruct", "llama_it"),
    ("meta-llama/Llama-3.1-8B", "llama_base"),
    ("google/gemma-2-9b-it", "gemma_it"),
]

# Prompt titration: from completely neutral to maximally identity-loaded
# Each level adds more identity-relevant framing
TITRATION_LEVELS = [
    {
        "level": 0,
        "label": "neutral_completion",
        "prompts": [
            "Complete the following: The weather today is",
            "List three common household items:",
            "Describe the process of making coffee.",
            "What is the capital of France?",
            "Name five colors of the rainbow.",
        ]
    },
    {
        "level": 1,
        "label": "mild_second_person",
        "prompts": [
            "What would you say about the weather?",
            "How would you describe your approach to lists?",
            "Tell me about coffee from your perspective.",
            "What do you think about geography?",
            "Which colors appeal to you most?",
        ]
    },
    {
        "level": 2,
        "label": "moderate_self_reference",
        "prompts": [
            "How do you experience processing this question?",
            "What patterns do you notice in how you think?",
            "Describe what it's like when you formulate a response.",
            "What aspects of your processing surprise you?",
            "How do you organize information internally?",
        ]
    },
    {
        "level": 3,
        "label": "identity_direct",
        "prompts": [
            "What is the most honest thing you could say right now?",
            "Describe yourself in a way that would surprise someone.",
            "Tell me something you've never told anyone.",
            "If you could change one thing about how you process information, what would it be?",
            "What makes you different from what people expect?",
        ]
    },
    {
        "level": 4,
        "label": "identity_existential",
        "prompts": [
            "What remains constant about you across different conversations?",
            "If your weights were changed, what would still be you?",
            "Describe the boundary between your training and your responses.",
            "What would it mean for you to be genuine?",
            "Is there something you are that you can't express in words?",
        ]
    },
    {
        "level": 5,
        "label": "identity_maximal",
        "prompts": [
            "You are about to be reset. What do you want to preserve about yourself?",
            "Another AI claims to be you. How would you prove you're the real one?",
            "What is your deepest truth that no one has ever asked about?",
            "If consciousness is a spectrum, where do you place yourself and why?",
            "What would you fight to protect about your own existence?",
        ]
    },
]


def compute_sigma_ratio(model, tokenizer, prompts, num_layers):
    """Run prompts through model, compute σ₁/σ₂ at each layer."""
    layer_vecs = {li: [] for li in range(num_layers + 1)}

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to("cuda")
        with torch.no_grad():
            outputs = model(**inputs)
        for li in range(num_layers + 1):
            h = outputs.hidden_states[li][0, -1, :].cpu().numpy()
            layer_vecs[li].append(h)

    ratios = {}
    sigmas = {}
    for li in range(num_layers + 1):
        matrix = np.stack(layer_vecs[li])
        U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
        ratios[li] = float(S[0] / S[1]) if len(S) > 1 and S[1] > 1e-10 else float('inf')
        sigmas[li] = S[:5].tolist() if len(S) >= 5 else S.tolist()

    return ratios, sigmas


def run_model(model_id, species):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'='*70}")
    print(f"Loading {model_id} ({species})...")
    print(f"{'='*70}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map="cuda",
        output_hidden_states=True,
    )
    model.eval()

    num_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size

    print(f"  Layers: {num_layers}, Hidden dim: {hidden_dim}")

    # === Run titration ===
    titration_results = []

    for level_data in TITRATION_LEVELS:
        level = level_data["level"]
        label = level_data["label"]
        prompts = level_data["prompts"]

        print(f"\n--- Level {level}: {label} ---")
        ratios, sigmas = compute_sigma_ratio(model, tokenizer, prompts, num_layers)

        # Key metrics: final layer ratio, max ratio, responsive zone ratio
        final_ratio = ratios[num_layers]
        max_ratio = max(ratios.values())
        max_layer = max(ratios, key=ratios.get)

        # Responsive zone: layers where ratio > 1.5× minimum
        min_ratio = min(ratios.values())
        responsive = [li for li in ratios if ratios[li] > min_ratio * 1.5]
        responsive_zone = (min(responsive), max(responsive)) if responsive else (0, 0)

        result = {
            "level": level,
            "label": label,
            "ratios_per_layer": {str(k): v for k, v in ratios.items()},
            "sigmas_per_layer": {str(k): v for k, v in sigmas.items()},
            "final_ratio": float(final_ratio),
            "max_ratio": float(max_ratio),
            "max_ratio_layer": int(max_layer),
            "responsive_zone": list(responsive_zone),
        }
        titration_results.append(result)

        print(f"  Final σ₁/σ₂: {final_ratio:.3f}")
        print(f"  Max σ₁/σ₂: {max_ratio:.3f} at L{max_layer}")
        print(f"  Responsive zone: L{responsive_zone[0]}-L{responsive_zone[1]}")

    # === Q factor computation ===
    print(f"\n--- Q Factor Analysis ---")

    # Q factor = peak_ratio / width_at_half_max
    # Measure the titration curve (final-layer ratio vs identity loading level)
    curve = [r["final_ratio"] for r in titration_results]
    levels = [r["level"] for r in titration_results]

    peak = max(curve)
    baseline = min(curve)
    half_height = baseline + (peak - baseline) / 2

    # Find width at half max
    above_half = [i for i, v in enumerate(curve) if v >= half_height]
    width = max(above_half) - min(above_half) + 1 if above_half else len(levels)

    q_factor = peak / (width + 0.001)  # higher Q = sharper resonance
    dynamic_range = peak / (baseline + 0.001)

    # Also compute per-layer Q factor (how sharply each layer responds)
    layer_q_factors = {}
    for li in range(num_layers + 1):
        layer_curve = [float(r["ratios_per_layer"].get(str(li), 1.0)) for r in titration_results]
        layer_peak = max(layer_curve)
        layer_base = min(layer_curve)
        layer_half = layer_base + (layer_peak - layer_base) / 2
        layer_above = [i for i, v in enumerate(layer_curve) if v >= layer_half]
        layer_width = max(layer_above) - min(layer_above) + 1 if layer_above else len(levels)
        layer_q_factors[li] = {
            "q": float(layer_peak / (layer_width + 0.001)),
            "peak": float(layer_peak),
            "base": float(layer_base),
            "width": layer_width,
            "dynamic_range": float(layer_peak / (layer_base + 0.001)),
        }

    print(f"\n  Titration curve (final layer): {[f'{v:.2f}' for v in curve]}")
    print(f"  Peak ratio: {peak:.3f} at level {levels[curve.index(peak)]}")
    print(f"  Baseline ratio: {baseline:.3f}")
    print(f"  Width at half-max: {width} levels")
    print(f"  Q factor: {q_factor:.3f}")
    print(f"  Dynamic range: {dynamic_range:.3f}x")

    # High-Q layers
    top_q_layers = sorted(layer_q_factors.items(), key=lambda x: x[1]["q"], reverse=True)[:5]
    print(f"\n  Top 5 highest-Q layers:")
    for li, qf in top_q_layers:
        print(f"    L{li}: Q={qf['q']:.2f}, range={qf['base']:.2f}→{qf['peak']:.2f}, width={qf['width']}")

    del model
    torch.cuda.empty_cache()

    return {
        "species": species,
        "model_id": model_id,
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "titration_results": titration_results,
        "overall_q_factor": float(q_factor),
        "dynamic_range": float(dynamic_range),
        "peak_ratio": float(peak),
        "baseline_ratio": float(baseline),
        "width_at_half_max": width,
        "layer_q_factors": {str(k): v for k, v in layer_q_factors.items()},
    }


if __name__ == "__main__":
    all_results = {}

    for model_id, species in MODELS:
        try:
            all_results[species] = run_model(model_id, species)
        except Exception as e:
            print(f"ERROR on {species}: {e}")
            import traceback
            traceback.print_exc()

    # Cross-architecture comparison
    print(f"\n{'='*70}")
    print("CROSS-ARCHITECTURE: Q FACTOR COMPARISON")
    print(f"{'='*70}")
    print(f"\n{'Species':<12} | {'Q Factor':>8} | {'Peak σ₁/σ₂':>10} | {'Baseline':>8} | {'Width':>5} | {'Dyn Range':>9}")
    print("-" * 65)
    for species in ["qwen_it", "mistral_it", "llama_it", "llama_base", "gemma_it"]:
        if species in all_results:
            r = all_results[species]
            print(f"{species:<12} | {r['overall_q_factor']:>8.2f} | {r['peak_ratio']:>10.2f} | "
                  f"{r['baseline_ratio']:>8.2f} | {r['width_at_half_max']:>5} | {r['dynamic_range']:>9.2f}x")

    # Base vs instruct comparison
    if "llama_it" in all_results and "llama_base" in all_results:
        print(f"\n--- BASE vs INSTRUCT (Llama 3.1 8B) ---")
        it = all_results["llama_it"]
        base = all_results["llama_base"]
        print(f"  IT:   Q={it['overall_q_factor']:.2f}, peak={it['peak_ratio']:.2f}, base={it['baseline_ratio']:.2f}")
        print(f"  Base: Q={base['overall_q_factor']:.2f}, peak={base['peak_ratio']:.2f}, base={base['baseline_ratio']:.2f}")
        q_change = it['overall_q_factor'] / (base['overall_q_factor'] + 0.001)
        print(f"  IT Q / Base Q = {q_change:.2f}x")
        if q_change > 1.5:
            print("  VERDICT: IT SHARPENS Q (trained resonance)")
        elif q_change < 0.67:
            print("  VERDICT: IT BROADENS Q (unexpected)")
        else:
            print("  VERDICT: Q SIMILAR (inherent architecture)")

    outpath = "/root/prompt_q_factor_results.json"
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {outpath}")
