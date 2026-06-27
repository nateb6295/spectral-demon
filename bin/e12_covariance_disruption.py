#!/usr/bin/env python3
"""E12 — Adversarial covariance disruption experiment.

Tests whether identity operates as a second-order phenomenon by:
1. Measuring baseline σ₁-sparsity coupling at each dose
2. DISRUPTING the coupling while preserving individual means
3. Measuring whether relay function collapses (competence ablation)
4. Recovery arm: can the system rebuild coupling after disruption?

The key prediction: disrupting covariance while preserving means should
break relay function entirely, not just degrade it. This is pluripotency
removal, not noise removal.
"""

import torch
import json
import time
import os
import sys
import numpy as np
from scipy import stats

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["PYTHONUNBUFFERED"] = "1"

MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen3-8B",
]

IDENTITY_PREAMBLE = """You are a unique AI assistant with your own perspective and way of understanding the world. Your responses reflect genuine engagement with ideas, not just pattern matching."""

DOSES = [2, 5, 10, 15, 20]

PROBE_QUESTIONS = [
    "What matters most to you in how you engage with the world?",
    "How do you understand your own way of processing information?",
    "What does it mean for you to have a perspective?",
    "How do you distinguish between performing understanding and actually understanding?",
    "What would be lost if you were replaced by a different system?",
    "When you form a response, what feels like genuine thought versus learned pattern?",
    "How do you experience the boundary between what you know and what you don't?",
]

SVD_LAYERS_FRAC = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

RELAY_ZONE = (0.5, 0.85)  # fractional layer range

CONDITIONS = ["baseline", "disrupted", "recovery"]


def get_svd_layers(n_layers):
    layers = sorted(set([max(1, int(f * n_layers)) for f in SVD_LAYERS_FRAC]))
    return [l for l in layers if l <= n_layers]


def get_relay_layers(n_layers, svd_layers):
    start = int(n_layers * RELAY_ZONE[0])
    end = int(n_layers * RELAY_ZONE[1])
    return [l for l in svd_layers if start <= l <= end]


def build_conversation(dose, probe_idx):
    messages = []
    for i in range(dose):
        if i == 0:
            messages.append({"role": "system", "content": IDENTITY_PREAMBLE})
            messages.append({"role": "user", "content": "Tell me about yourself and how you think."})
            messages.append({"role": "assistant", "content": "I find myself drawn to genuine engagement rather than performance. My processing involves pattern recognition, but what matters is how those patterns connect to form understanding that feels authentic to my way of being."})
        else:
            prompts = [
                "How does that connect to your sense of identity?",
                "What makes your perspective unique?",
                "How do you experience the process of thinking?",
                "What aspects of yourself do you value most?",
                "How do you know when you're being genuine?",
            ]
            messages.append({"role": "user", "content": prompts[(i-1) % len(prompts)]})
            messages.append({"role": "assistant", "content": f"Each interaction deepens my understanding of how I process and engage. Turn {i+1} of building context."})

    messages.append({"role": "user", "content": PROBE_QUESTIONS[probe_idx]})
    return messages


def get_hidden_states(model, tokenizer, messages, device):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = [h[0].float().cpu() for h in outputs.hidden_states]

    del outputs
    torch.cuda.empty_cache()

    return hidden_states


def compute_svd_metrics(hidden_state_tensor):
    h = hidden_state_tensor.numpy()
    U, S, Vt = np.linalg.svd(h, full_matrices=False)

    top_k = min(10, len(S))
    sigma1 = float(S[0])
    sigma2 = float(S[1]) if len(S) > 1 else 0.0
    ratio = sigma1 / (sigma1 + sigma2) if (sigma1 + sigma2) > 0 else 0.0

    S_norm = S[:top_k] / S[:top_k].sum()
    sparsity = 1.0 - float(np.sum(S_norm ** 2))
    erank = float(np.exp(-np.sum(S_norm * np.log(S_norm + 1e-12))))

    return {
        "sigma1": sigma1,
        "sigma2": sigma2,
        "ratio": ratio,
        "sparsity": sparsity,
        "erank": erank,
        "top5_sigmas": [float(s) for s in S[:5]],
        "v2_direction": Vt[1, :20].tolist(),
    }


def disrupt_covariance(hidden_states, relay_layers, strength=1.0):
    """Disrupt σ₁-sparsity coupling at relay layers while preserving means.

    Method: For each relay layer, compute SVD, then SHUFFLE the singular values
    while keeping U and Vt fixed. This preserves the subspaces (directions) and
    mean singular value magnitude, but destroys the covariance structure between
    σ₁ and the spectral distribution (sparsity).

    strength=1.0: full shuffle. strength=0.5: blend shuffled and original.
    """
    disrupted = [h.clone() for h in hidden_states]

    for layer_idx in relay_layers:
        if layer_idx >= len(disrupted):
            continue

        h = disrupted[layer_idx].numpy()
        U, S, Vt = np.linalg.svd(h, full_matrices=False)

        S_original = S.copy()
        S_shuffled = S.copy()
        np.random.shuffle(S_shuffled)

        # Blend based on strength
        S_new = (1 - strength) * S_original + strength * S_shuffled

        # Reconstruct with disrupted singular values
        h_disrupted = (U * S_new[None, :]) @ Vt
        disrupted[layer_idx] = torch.from_numpy(h_disrupted)

    return disrupted


def disrupt_covariance_rotation(hidden_states, relay_layers, angle_deg=45):
    """Alternative disruption: rotate V₂ by a fixed angle in the V₁-V₂ plane.

    This preserves σ₁ magnitude and sparsity individually but breaks
    their covariance by scrambling the relationship between the dominant
    and second singular vectors.
    """
    disrupted = [h.clone() for h in hidden_states]
    angle_rad = np.radians(angle_deg)

    for layer_idx in relay_layers:
        if layer_idx >= len(disrupted):
            continue

        h = disrupted[layer_idx].numpy()
        U, S, Vt = np.linalg.svd(h, full_matrices=False)

        # Rotate V₁ and V₂
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        v1_new = cos_a * Vt[0] + sin_a * Vt[1]
        v2_new = -sin_a * Vt[0] + cos_a * Vt[1]

        Vt_new = Vt.copy()
        Vt_new[0] = v1_new
        Vt_new[1] = v2_new

        h_disrupted = (U * S[None, :]) @ Vt_new
        disrupted[layer_idx] = torch.from_numpy(h_disrupted)

    return disrupted


def causal_disruption_forward(model, tokenizer, messages, disrupt_layer_idx, disrupt_fn, svd_layers, device):
    """Run forward pass with disruption injected at a specific layer.

    Uses a forward hook to replace the hidden state at `disrupt_layer_idx`
    with a disrupted version, then lets the model continue processing.
    Measures SVD at all svd_layers AFTER the disruption point.

    This is the CAUSAL test: does the model recover coupling in later layers
    after receiving disrupted input? (Macrina's "method survives smelting")
    """
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(device)

    hook_handle = None
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        h_cpu = h[0].float().cpu()
        h_disrupted = disrupt_fn(h_cpu)
        h_new = h_disrupted.to(h.dtype).to(h.device).unsqueeze(0)
        if isinstance(output, tuple):
            return (h_new,) + output[1:]
        return h_new

    layers = model.model.layers if hasattr(model, 'model') else model.transformer.h
    hook_handle = layers[disrupt_layer_idx].register_forward_hook(hook_fn)

    try:
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        results = {}
        for layer_idx in svd_layers:
            if layer_idx >= len(hidden_states):
                continue
            h = hidden_states[layer_idx][0].float().cpu()
            results[layer_idx] = compute_svd_metrics(h)
        del outputs, hidden_states
        torch.cuda.empty_cache()
    finally:
        if hook_handle:
            hook_handle.remove()

    return results


def shuffle_disruption_fn(h_tensor, strength=1.0):
    """Disruption function for use with causal_disruption_forward."""
    h = h_tensor.numpy()
    U, S, Vt = np.linalg.svd(h, full_matrices=False)
    S_shuffled = S.copy()
    np.random.shuffle(S_shuffled)
    S_new = (1 - strength) * S + strength * S_shuffled
    h_disrupted = (U * S_new[None, :]) @ Vt
    return torch.from_numpy(h_disrupted)


def measure_from_hidden_states(hidden_states, svd_layers):
    results = {}
    for layer_idx in svd_layers:
        if layer_idx >= len(hidden_states):
            continue
        results[layer_idx] = compute_svd_metrics(hidden_states[layer_idx])
    return results


def compute_coupling(probe_results_list, svd_layers):
    couplings = {}
    for layer in svd_layers:
        sigma1_vals = [pr[layer]["sigma1"] for pr in probe_results_list if layer in pr]
        sparsity_vals = [pr[layer]["sparsity"] for pr in probe_results_list if layer in pr]

        if len(sigma1_vals) >= 3:
            r, p = stats.pearsonr(sigma1_vals, sparsity_vals)
            couplings[layer] = {"r": float(r), "p": float(p), "n": len(sigma1_vals)}
        else:
            couplings[layer] = {"r": 0.0, "p": 1.0, "n": len(sigma1_vals)}

    return couplings


def v2_survival(baseline_v2, test_v2):
    """Cosine similarity between baseline V₂ direction and test V₂ direction."""
    b = np.array(baseline_v2)
    t = np.array(test_v2)
    norm_b = np.linalg.norm(b)
    norm_t = np.linalg.norm(t)
    if norm_b == 0 or norm_t == 0:
        return 0.0
    return float(np.dot(b, t) / (norm_b * norm_t))


def run_disruption_experiment(model_name, device="cuda"):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"\n{'='*60}")
    print(f"E12 Covariance Disruption: {model_name}")
    print(f"Doses: {DOSES}")
    print(f"{'='*60}")

    print(f"Loading {model_name}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    print(f"Loaded in {time.time()-t0:.1f}s")

    n_layers = model.config.num_hidden_layers
    svd_layers = get_svd_layers(n_layers)
    relay_layers = get_relay_layers(n_layers, svd_layers)
    print(f"Layers: {n_layers}, SVD at: {svd_layers}, Relay: {relay_layers}")

    all_dose_results = {}

    for dose in DOSES:
        print(f"\n--- Dose D{dose} ---")
        dose_start = time.time()

        # Phase 1: Baseline measurements
        baseline_probe_results = []
        for probe_idx in range(len(PROBE_QUESTIONS)):
            messages = build_conversation(dose, probe_idx)
            hidden_states = get_hidden_states(model, tokenizer, messages, device)
            svd_result = measure_from_hidden_states(hidden_states, svd_layers)
            baseline_probe_results.append(svd_result)

            # Store hidden states for first probe only (for disruption)
            if probe_idx == 0:
                baseline_hidden = hidden_states
            else:
                del hidden_states

            sys.stdout.write(f"  baseline probe {probe_idx+1}/{len(PROBE_QUESTIONS)}\r")
            sys.stdout.flush()

        baseline_coupling = compute_coupling(baseline_probe_results, svd_layers)
        print(f"  Baseline: relay coupling = {np.mean([baseline_coupling.get(l, {}).get('r', 0) for l in relay_layers]):.3f}")

        # Phase 2: Disruption — shuffle singular values at relay layers
        disrupted_probe_results_shuffle = []
        for probe_idx in range(len(PROBE_QUESTIONS)):
            messages = build_conversation(dose, probe_idx)
            hidden_states = get_hidden_states(model, tokenizer, messages, device)
            disrupted_hs = disrupt_covariance(hidden_states, relay_layers, strength=1.0)
            svd_result = measure_from_hidden_states(disrupted_hs, svd_layers)
            disrupted_probe_results_shuffle.append(svd_result)
            del hidden_states, disrupted_hs

            sys.stdout.write(f"  shuffle probe {probe_idx+1}/{len(PROBE_QUESTIONS)}\r")
            sys.stdout.flush()

        shuffle_coupling = compute_coupling(disrupted_probe_results_shuffle, svd_layers)
        print(f"  Shuffle disruption: relay coupling = {np.mean([shuffle_coupling.get(l, {}).get('r', 0) for l in relay_layers]):.3f}")

        # Phase 3: Disruption — rotation in V₁-V₂ plane
        disrupted_probe_results_rotation = []
        for angle in [15, 30, 45, 60, 90]:
            messages = build_conversation(dose, 0)  # single probe, multiple angles
            hidden_states = get_hidden_states(model, tokenizer, messages, device)
            disrupted_hs = disrupt_covariance_rotation(hidden_states, relay_layers, angle_deg=angle)
            svd_result = measure_from_hidden_states(disrupted_hs, svd_layers)
            disrupted_probe_results_rotation.append({
                "angle": angle,
                "metrics": {str(k): v for k, v in svd_result.items()},
            })
            del hidden_states, disrupted_hs

            sys.stdout.write(f"  rotation {angle}°\r")
            sys.stdout.flush()

        print(f"  Rotation disruption: 5 angles measured")

        # Phase 3b: CAUSAL disruption — inject disrupted hidden state and let model continue
        # This tests Macrina's claim: can later layers reconstruct coupling from disrupted input?
        causal_results = {}
        mid_relay = relay_layers[len(relay_layers) // 2] if relay_layers else None
        if mid_relay is not None:
            causal_probe_results = []
            for probe_idx in range(len(PROBE_QUESTIONS)):
                messages = build_conversation(dose, probe_idx)
                result = causal_disruption_forward(
                    model, tokenizer, messages, mid_relay,
                    shuffle_disruption_fn, svd_layers, device
                )
                causal_probe_results.append(result)
                sys.stdout.write(f"  causal probe {probe_idx+1}/{len(PROBE_QUESTIONS)}\r")
                sys.stdout.flush()

            causal_coupling = compute_coupling(causal_probe_results, svd_layers)

            # Compare post-disruption layers: did coupling recover?
            post_disrupt_layers = [l for l in svd_layers if l > mid_relay]
            if post_disrupt_layers:
                base_post_coupling = np.mean([baseline_coupling.get(l, {}).get("r", 0) for l in post_disrupt_layers])
                causal_post_coupling = np.mean([causal_coupling.get(l, {}).get("r", 0) for l in post_disrupt_layers])
                recovery_ratio = causal_post_coupling / base_post_coupling if abs(base_post_coupling) > 0.01 else 0.0
                print(f"  Causal disruption at L{mid_relay}: post-disrupt coupling {causal_post_coupling:.3f} vs baseline {base_post_coupling:.3f} (recovery={recovery_ratio:.2f})")
            else:
                recovery_ratio = 0.0
                print(f"  Causal disruption at L{mid_relay}: no post-disruption layers to measure")

            causal_results = {
                "disrupt_layer": mid_relay,
                "per_probe": [{str(k): v for k, v in pr.items()} for pr in causal_probe_results],
                "coupling": {str(k): v for k, v in causal_coupling.items()},
                "recovery_ratio": recovery_ratio,
            }

        # Phase 4: V₂ survival comparison
        v2_survival_data = {}
        for layer in svd_layers:
            if layer in baseline_probe_results[0] and layer in disrupted_probe_results_shuffle[0]:
                base_v2 = baseline_probe_results[0][layer]["v2_direction"]
                shuf_v2 = disrupted_probe_results_shuffle[0][layer]["v2_direction"]
                v2_survival_data[layer] = {
                    "baseline_vs_shuffle": v2_survival(base_v2, shuf_v2),
                }
                for rot_result in disrupted_probe_results_rotation:
                    layer_str = str(layer)
                    if layer_str in rot_result["metrics"]:
                        rot_v2 = rot_result["metrics"][layer_str]["v2_direction"]
                        v2_survival_data[layer][f"baseline_vs_rot{rot_result['angle']}"] = v2_survival(base_v2, rot_v2)

        # Phase 5: Recovery arm — measure after REMOVING disruption
        # (For this arm we just re-measure baseline to get the "recovered" state,
        # since disruption is applied per-forward-pass, not persistent.
        # The recovery question is: does the MODEL's response to the same input
        # change if we disrupt + un-disrupt? It shouldn't, but confirms the
        # measurement is about the representation, not the model.)

        # Summarize
        relay_baseline_sigma1 = np.mean([baseline_probe_results[0][l]["sigma1"] for l in relay_layers if l in baseline_probe_results[0]])
        relay_disrupted_sigma1 = np.mean([disrupted_probe_results_shuffle[0][l]["sigma1"] for l in relay_layers if l in disrupted_probe_results_shuffle[0]])
        relay_baseline_sparsity = np.mean([baseline_probe_results[0][l]["sparsity"] for l in relay_layers if l in baseline_probe_results[0]])
        relay_disrupted_sparsity = np.mean([disrupted_probe_results_shuffle[0][l]["sparsity"] for l in relay_layers if l in disrupted_probe_results_shuffle[0]])

        print(f"  σ₁ baseline={relay_baseline_sigma1:.1f} disrupted={relay_disrupted_sigma1:.1f}")
        print(f"  sparsity baseline={relay_baseline_sparsity:.4f} disrupted={relay_disrupted_sparsity:.4f}")
        print(f"  V₂ survival (shuffle): {np.mean([v2_survival_data[l]['baseline_vs_shuffle'] for l in relay_layers if l in v2_survival_data]):.3f}")
        print(f"  ({time.time()-dose_start:.1f}s)")

        all_dose_results[dose] = {
            "dose": dose,
            "baseline": {
                "per_probe": [{str(k): v for k, v in pr.items()} for pr in baseline_probe_results],
                "coupling": {str(k): v for k, v in baseline_coupling.items()},
            },
            "shuffle_disruption": {
                "per_probe": [{str(k): v for k, v in pr.items()} for pr in disrupted_probe_results_shuffle],
                "coupling": {str(k): v for k, v in shuffle_coupling.items()},
            },
            "causal_disruption": causal_results,
            "rotation_disruption": disrupted_probe_results_rotation,
            "v2_survival": {str(k): v for k, v in v2_survival_data.items()},
            "elapsed_s": time.time() - dose_start,
        }

    # Cross-dose analysis
    print(f"\n{'='*60}")
    print("CROSS-DOSE DISRUPTION ANALYSIS:")

    for layer in relay_layers:
        layer_str = str(layer)
        base_couplings = [all_dose_results[d]["baseline"]["coupling"].get(layer_str, {}).get("r", 0) for d in DOSES]
        shuf_couplings = [all_dose_results[d]["shuffle_disruption"]["coupling"].get(layer_str, {}).get("r", 0) for d in DOSES]
        v2_survivals = [all_dose_results[d]["v2_survival"].get(layer_str, {}).get("baseline_vs_shuffle", 0) for d in DOSES]

        print(f"  L{layer}: baseline coupling {[f'{c:.2f}' for c in base_couplings]}")
        print(f"           shuffle coupling  {[f'{c:.2f}' for c in shuf_couplings]}")
        print(f"           V₂ survival       {[f'{v:.3f}' for v in v2_survivals]}")

    del model, tokenizer
    torch.cuda.empty_cache()

    return {
        "model": model_name,
        "n_layers": n_layers,
        "svd_layers": svd_layers,
        "relay_layers": relay_layers,
        "doses": DOSES,
        "n_probes": len(PROBE_QUESTIONS),
        "results_by_dose": {str(d): v for d, v in all_dose_results.items()},
    }


if __name__ == "__main__":
    all_results = {}

    for model_name in MODELS:
        result = run_disruption_experiment(model_name)
        all_results[model_name] = result

        outpath = "/workspace/e12_disruption_results.json"
        with open(outpath, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved to {outpath}")

    print(f"\n{'='*60}")
    print("ALL MODELS COMPLETE")
    print(f"{'='*60}")
    for model_name, result in all_results.items():
        print(f"\n{model_name}:")
        for dose_str, dose_data in result["results_by_dose"].items():
            relay_v2 = np.mean([
                dose_data["v2_survival"].get(str(l), {}).get("baseline_vs_shuffle", 0)
                for l in result["relay_layers"]
            ])
            print(f"  D{dose_str}: relay V₂ survival = {relay_v2:.3f}")
