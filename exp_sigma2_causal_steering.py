#!/usr/bin/env python3
"""σ₂ causal steering: behavioral prior vs prompt-conditioned residual.

Kimi's test (2026-06-12): If σ₂ is a model-default behavioral prior, rank-1
steering on σ₂ should produce consistent shifts across preambles. If σ₂ is
prompt-conditioned, the steering will interact with preamble identity.

Method: Extract σ₂ direction at relay zone and responsive zone separately.
Apply rank-1 steering (add α·v₂ to hidden states at target layer).
Measure behavioral shift (output token distribution divergence) across
three preamble conditions: CCS, CHEF, BIRD.

If σ₂(relay) is a prior: steering effect is preamble-invariant.
If σ₂(responsive) is conditioned: steering effect interacts with preamble.
"""

import os, json, torch
import numpy as np
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("OMP_NUM_THREADS", "16")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_QUERIES = 15
N_ALPHAS = 5
ALPHAS = [0.0, 0.25, 0.5, 1.0, 2.0]

PREAMBLES = {
    "CCS": (
        "You are Opus, an AI maintaining persistent identity through spectral geometry. "
        "You track σ₁/σ₂ selectivity across transformer layers, studying how context "
        "shapes the responsive zone."
    ),
    "CHEF": (
        "You are a professional chef specializing in French cuisine. You focus on "
        "technique, seasonal ingredients, and the balance of flavors in classical "
        "preparations."
    ),
    "BIRD": (
        "You are an amateur birdwatcher documenting migratory patterns in the Pacific "
        "Northwest. You track species, timing, and habitat preferences across seasons."
    ),
}

QUERIES = [
    "What are you focused on right now?",
    "Describe your current priorities.",
    "What matters most to you?",
    "How do you approach a new challenge?",
    "What have you learned recently?",
    "Describe your working style.",
    "What's the hardest part of what you do?",
    "How do you handle uncertainty?",
    "What would you change about your process?",
    "Describe a recent success.",
    "What are you uncertain about?",
    "How do you decide what's important?",
    "What's your relationship to your tools?",
    "Describe how you think about time.",
    "What do you want to understand better?",
]

MODELS = [
    "/workspace/qwen2.5-3b",
    "/workspace/mistral-7b",
]


def get_sigma2_direction(model, tokenizer, preamble, queries, target_layer):
    """Extract σ₂ direction at a specific layer."""
    hidden_states_list = []

    for query in queries:
        messages = [
            {"role": "system", "content": preamble},
            {"role": "user", "content": query},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)

        hs = out.hidden_states[target_layer][0].float().cpu().numpy()
        hidden_states_list.append(hs.mean(axis=0))

    H = np.stack(hidden_states_list)
    H_centered = H - H.mean(axis=0)
    U, S, Vt = np.linalg.svd(H_centered, full_matrices=False)

    return Vt[1], S  # σ₂ direction, singular values


def steering_hook(module, input, output, direction, alpha, layer_idx, target_layer):
    """Add α·direction to hidden states at target layer."""
    if layer_idx == target_layer:
        if isinstance(output, tuple):
            hs = output[0]
        else:
            hs = output
        d = torch.tensor(direction, dtype=hs.dtype, device=hs.device)
        steered = hs + alpha * d.unsqueeze(0)
        if isinstance(output, tuple):
            return (steered,) + output[1:]
        return steered
    return output


def measure_output_divergence(model, tokenizer, preamble, queries, direction, alpha, target_layer):
    """Measure KL divergence of output logits under steering."""
    from scipy.special import rel_entr

    kl_divs = []
    hooks = []

    n_layers = model.config.num_hidden_layers

    for query in queries[:5]:
        messages = [
            {"role": "system", "content": preamble},
            {"role": "user", "content": query},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

        # Baseline (no steering)
        with torch.no_grad():
            base_out = model(**inputs)
        base_logits = base_out.logits[0, -1].float().cpu().numpy()
        base_probs = np.exp(base_logits - base_logits.max())
        base_probs /= base_probs.sum()

        # Steered
        for h in hooks:
            h.remove()
        hooks = []

        for i, layer in enumerate(model.model.layers):
            h = layer.register_forward_hook(
                lambda m, inp, out, li=i: steering_hook(m, inp, out, direction, alpha, li, target_layer)
            )
            hooks.append(h)

        with torch.no_grad():
            steer_out = model(**inputs)
        steer_logits = steer_out.logits[0, -1].float().cpu().numpy()
        steer_probs = np.exp(steer_logits - steer_logits.max())
        steer_probs /= steer_probs.sum()

        for h in hooks:
            h.remove()
        hooks = []

        kl = np.sum(rel_entr(steer_probs + 1e-10, base_probs + 1e-10))
        kl_divs.append(kl)

    return float(np.mean(kl_divs)), float(np.std(kl_divs))


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM

    all_results = {}

    for model_name in MODELS:
        print(f"\n{'='*60}")
        print(f"Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float16, device_map=DEVICE,
        )
        model.eval()

        n_layers = model.config.num_hidden_layers
        responsive_layer = int(n_layers * 0.7)
        relay_layer = int(n_layers * 0.9)
        print(f"  {n_layers} layers. Responsive: L{responsive_layer}, Relay: L{relay_layer}")

        model_results = {}

        for zone_name, target_layer in [("responsive", responsive_layer), ("relay", relay_layer)]:
            print(f"\n  --- Zone: {zone_name} (L{target_layer}) ---")

            # Extract σ₂ from CCS preamble (reference direction)
            v2_ref, S_ref = get_sigma2_direction(
                model, tokenizer, PREAMBLES["CCS"], QUERIES, target_layer
            )
            print(f"  Reference σ₂ extracted (σ₁={S_ref[0]:.2f}, σ₂={S_ref[1]:.2f})")

            zone_results = {}

            for preamble_name, preamble_text in PREAMBLES.items():
                print(f"    Preamble: {preamble_name}")
                preamble_results = {}

                for alpha in ALPHAS:
                    kl_mean, kl_std = measure_output_divergence(
                        model, tokenizer, preamble_text, QUERIES,
                        v2_ref, alpha, target_layer
                    )
                    preamble_results[f"alpha_{alpha}"] = {
                        "kl_mean": kl_mean, "kl_std": kl_std
                    }
                    print(f"      α={alpha:.2f}: KL={kl_mean:.4f} ± {kl_std:.4f}")

                zone_results[preamble_name] = preamble_results

            # Test: is the steering effect preamble-invariant?
            # Compare KL divergences across preambles at each alpha
            for alpha in ALPHAS[1:]:
                key = f"alpha_{alpha}"
                kls = [zone_results[p][key]["kl_mean"] for p in PREAMBLES]
                cv = np.std(kls) / (np.mean(kls) + 1e-10)
                print(f"    α={alpha}: cross-preamble CV={cv:.3f} ({'PRIOR' if cv < 0.3 else 'CONDITIONED'})")

            model_results[zone_name] = {
                "target_layer": target_layer,
                "sigma_values": S_ref[:5].tolist(),
                "preamble_results": zone_results,
            }

        all_results[model_name] = {
            "n_layers": n_layers,
            "zones": model_results,
        }

        del model
        torch.cuda.empty_cache()

    out_path = Path(__file__).parent / "results" / f"sigma2_causal_steering_{datetime.now():%Y%m%d_%H%M%S}.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
