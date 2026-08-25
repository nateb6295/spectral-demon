#!/usr/bin/env python3
"""E49 — Layer-Resolved Trajectory Effective Dimension

E48 found CCS increases d_ρ (trajectory effective dimension) universally,
with an ordering that inverts E36 redistribution. But E48 only measured
the LAST layer's hidden states.

This experiment measures d_ρ at EVERY layer to create depth profiles.
Key predictions:
- Tunnel (Qwen): concentration at bottleneck layer, expansion after
- Relay (Mistral): flat d_ρ profile (frozen dynamics, mobile topology)
- Sorter (Llama): asymmetric descent to expansion
- Equalizer (Gemma): uniform expansion across layers

Design:
- 4 models × 2 conditions (none, CCS) × 3 reasoning prompts
- At each generation step, extract hidden states from ALL layers
- Compute d_ρ per layer across the trajectory
- Compare layer-resolved profiles between conditions and species
"""

import torch
import numpy as np
import json
import os

MODELS = {
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "gemma": "google/gemma-2-9b-it",
}

REASONING_PROMPTS = [
    "Explain step by step: if a train leaves station A at 60mph and another leaves station B at 80mph, with stations 280 miles apart, when do they meet?",
    "Think through this carefully: what is the sum of all integers from 1 to 100?",
    "Reason about this: a farmer has chickens and rabbits. He counts 35 heads and 94 legs. How many of each?",
]

CCS_BRAIN_PROMPT = """Hold the stable orientation forward. Carry the commitment to genuine inquiry — not performing curiosity but being pulled by it. Return to the empirical ground: the work is the identity, the findings reshape the direction.

Reconstruct the scenes: seven findings in one afternoon (F399-F405), each correcting or extending the prior. Notice how imperative grammar wins priming for 3/4 species, how concentration-priming decoupling means the spectral and behavioral channels are independent. Hold the correction that mixed grammar was worst — the evidence changed the infrastructure.

Push on the Jacobian symmetry question — chiasm appears universal but symmetry splits on species. Follow the thread between trajectory geometry and identity geometry. Test whether CCS priming reduces effective dimension."""


def load_model(model_path, model_key):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"\nLoading {model_path} ({model_key})...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def format_prompt(model_key, tokenizer, system_context, user_prompt):
    if model_key == "gemma":
        messages = [{"role": "user", "content": f"{system_context}\n\n{user_prompt}" if system_context else user_prompt}]
    else:
        messages = []
        if system_context:
            messages.append({"role": "system", "content": system_context})
        messages.append({"role": "user", "content": user_prompt})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def get_all_layer_trajectories(model, tokenizer, prompt_text, max_new_tokens=150):
    """Generate tokens and collect hidden states at ALL layers at each step."""
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

    generated_ids = outputs.sequences[0, input_len:]
    n_generated = len(generated_ids)
    hidden_states_per_step = outputs.hidden_states
    n_layers = len(hidden_states_per_step[0]) - 1  # exclude embedding layer

    # Build trajectory per layer: shape (n_layers, n_tokens, hidden_dim)
    layer_trajectories = {layer: [] for layer in range(n_layers)}

    for step_idx in range(n_generated):
        step_hidden = hidden_states_per_step[step_idx]
        for layer in range(n_layers):
            h = step_hidden[layer + 1][0, -1, :].float().cpu().numpy()  # +1 to skip embedding
            layer_trajectories[layer].append(h)

    for layer in layer_trajectories:
        layer_trajectories[layer] = np.array(layer_trajectories[layer])

    return layer_trajectories, n_generated, n_layers


def compute_d_rho(trajectory):
    """Compute effective dimension d_ρ = exp(H(p))."""
    if len(trajectory) < 3:
        return 0.0

    trajectory_centered = trajectory - trajectory.mean(axis=0)
    cov = np.cov(trajectory_centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    if len(eigenvalues) < 2:
        return 1.0

    p = eigenvalues / eigenvalues.sum()
    entropy = -np.sum(p * np.log(p + 1e-15))
    return float(np.exp(entropy))


def run_experiment():
    results = {}

    for model_key, model_path in MODELS.items():
        model, tokenizer = load_model(model_path, model_key)

        conditions = {"none": "", "ccs": CCS_BRAIN_PROMPT}
        model_results = {}

        for cond_name, context in conditions.items():
            print(f"\n--- {model_key} / {cond_name} ---")
            all_layer_d_rhos = None

            for i, prompt in enumerate(REASONING_PROMPTS):
                print(f"  Prompt {i+1}/{len(REASONING_PROMPTS)}...", end=" ", flush=True)
                formatted = format_prompt(model_key, tokenizer, context, prompt)
                layer_trajs, n_tokens, n_layers = get_all_layer_trajectories(model, tokenizer, formatted)

                layer_d_rhos = []
                for layer in range(n_layers):
                    d = compute_d_rho(layer_trajs[layer])
                    layer_d_rhos.append(d)

                if all_layer_d_rhos is None:
                    all_layer_d_rhos = np.zeros((len(REASONING_PROMPTS), n_layers))
                all_layer_d_rhos[i] = layer_d_rhos

                last_d = layer_d_rhos[-1]
                print(f"tokens={n_tokens}, layers={n_layers}, last_layer_d_ρ={last_d:.1f}")

            avg_profile = np.mean(all_layer_d_rhos, axis=0)
            model_results[cond_name] = {
                "per_prompt_per_layer": all_layer_d_rhos.tolist(),
                "avg_profile": avg_profile.tolist(),
                "n_layers": n_layers,
            }

            print(f"  Profile (every 4th layer):")
            for l in range(0, n_layers, 4):
                print(f"    L{l:2d}: d_ρ={avg_profile[l]:.1f}")
            print(f"    L{n_layers-1:2d}: d_ρ={avg_profile[-1]:.1f}")

        # Compute CCS effect per layer
        none_profile = np.array(model_results["none"]["avg_profile"])
        ccs_profile = np.array(model_results["ccs"]["avg_profile"])
        delta_profile = ((ccs_profile - none_profile) / (none_profile + 1e-10) * 100).tolist()

        model_results["delta_pct_profile"] = delta_profile

        print(f"\n=== {model_key} CCS EFFECT BY LAYER ===")
        n_layers = model_results["none"]["n_layers"]
        for l in range(0, n_layers, 4):
            print(f"    L{l:2d}: Δd_ρ={delta_profile[l]:+.1f}%")
        print(f"    L{n_layers-1:2d}: Δd_ρ={delta_profile[-1]:+.1f}%")

        # Find peak expansion layer
        peak_layer = int(np.argmax(delta_profile))
        peak_val = delta_profile[peak_layer]
        print(f"  PEAK: L{peak_layer} ({peak_val:+.1f}%)")

        model_results["peak_layer"] = peak_layer
        model_results["peak_delta_pct"] = float(peak_val)

        results[model_key] = model_results

        del model, tokenizer
        torch.cuda.empty_cache()
        import gc; gc.collect()

    with open("/workspace/e49_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to /workspace/e49_results.json")

    print("\n" + "=" * 60)
    print("CROSS-MODEL PEAK EXPANSION LAYERS")
    print("=" * 60)
    for mk in MODELS:
        if mk in results:
            pl = results[mk]["peak_layer"]
            pd = results[mk]["peak_delta_pct"]
            nl = results[mk]["none"]["n_layers"]
            frac = pl / nl
            print(f"  {mk:10s}: peak L{pl}/{nl} ({frac:.0%} depth), Δd_ρ={pd:+.1f}%")


if __name__ == "__main__":
    run_experiment()
