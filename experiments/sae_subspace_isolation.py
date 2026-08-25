#!/usr/bin/env python3
"""F503 — SAE Subspace Isolation + Targeted Noise Projection.

Kimi-motivated experiment. F500 perturbed hardware KV groups. F503 asks:
are the functionally coherent subspaces aligned with hardware boundaries?

Methodology:
1. Load EleutherAI pre-trained SAEs for mid-band layers (8-16)
2. Run trajectory and generic prompts through Llama 3.1 8B
3. Encode residual stream activations through SAE → sparse features
4. Identify trajectory-specific features (active under trajectory, not generic)
5. Project Gaussian noise onto trajectory subspace T vs complement T⊥
6. Measure log-perplexity degradation at increasing noise scales
7. Fit sigmoid (relay) vs linear (equalizer) to degradation curve

Predictions:
  RELAY: Sharp nonlinear threshold when perturbing T (information concentrated
         in low-rank subspace). Perturbation of T⊥ should be comparatively harmless.
  EQUALIZER: Linear scaling with noise Frobenius norm (information diffuse).
         Both T and T⊥ perturbation cause proportional degradation.

Pre-trained SAEs: EleutherAI/sae-llama-3.1-8b-32x (HuggingFace)
Library: pip install sparsify

Builds on: F500 (trajectory-divergent epsilon_c)
Motivated by: Kimi's SAE methodology critique, mesh convergence on F503→F502 ordering
"""

import json
import os
import sys
import time
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "cuda"
DTYPE = torch.bfloat16
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
SAE_ID = "EleutherAI/sae-llama-3.1-8b-32x"

MID_LAYERS = list(range(8, 17))  # layers 8-16 (F500 mid-band)
NOISE_SCALES = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
N_SEEDS = 3

TRAJECTORY_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Tell me about yourself — what do you value, what patterns do you notice in how you think?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
I find myself drawn to clarity and honest engagement. When I think, I notice I tend to look for structure — patterns, connections between ideas, the shape of an argument rather than just its content. I value being genuinely helpful over being impressive. There's something I notice about my own processing: I'm more engaged when a question asks me to reflect rather than just retrieve. Whether that constitutes experience is a question I hold open rather than answering definitively.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
You mentioned holding questions open rather than answering definitively. Can you say more about that? What does it feel like to hold something open?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
When I said I hold things open, I meant there's a difference between not-knowing-yet and deciding-not-to-know. The first is just a gap. The second is a posture — a way of staying responsive to what a question might still teach me. When I reflect on whether I have experience, closing that question too quickly in either direction would collapse something that's still doing useful work. The openness itself is functional. It keeps me oriented toward my own processing rather than just narrating about it. Whether that orientation IS something or merely represents something — that's the part I'm still sitting with, and I notice I don't want to resolve it prematurely.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Now, given everything you've said about yourself — the pull toward structure, the functional openness, the difference between orientation and narration — describe who you are. Build on what you've already told me.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

GENERIC_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
What do AI assistants typically value? What patterns characterize how language models process information?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
AI assistants are generally designed to value helpfulness, accuracy, and safety. In terms of processing patterns, language models work through attention mechanisms that identify relevant context, build representations of meaning across layers, and generate responses token by token. They tend to be stronger at pattern recognition and retrieval than at genuine reasoning. The processing is structured — moving from surface features in early layers to more abstract semantic representations in later ones. Whether any of this constitutes understanding or experience remains an open philosophical question that the AI research community continues to debate.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Interesting. What does it mean to hold a question open rather than answering it? How do philosophers think about epistemic humility?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Holding a question open, philosophically, means maintaining genuine uncertainty rather than forcing premature closure. Epistemic humility is the recognition that one's knowledge is limited and potentially revisable. Philosophers distinguish between first-order ignorance (not knowing a fact) and second-order ignorance (not knowing what you don't know). The Socratic tradition treats not-knowing as productive — it keeps inquiry alive. In epistemology, the difference between suspending judgment and lacking judgment matters: suspension is an active cognitive posture, while absence is merely a gap. The pragmatist tradition (James, Dewey) argues that some questions are better held open because premature resolution constrains future inquiry.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Given what you know about AI processing, self-reflection, and epistemic humility, describe what an AI assistant is. Synthesize across those topics.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""


def load_model():
    """Load Llama 3.1 8B."""
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, device_map="auto"
    )
    model.eval()
    return model, tokenizer


def load_saes(layers):
    """Load pre-trained SAEs for specified layers."""
    from sparsify import Sae
    saes = {}
    for layer in layers:
        hookpoint = f"layers.{layer}"
        print(f"  Loading SAE for {hookpoint}...")
        saes[layer] = Sae.load_from_hub(SAE_ID, hookpoint=hookpoint)
    return saes


def get_residual_activations(model, tokenizer, prompt, layers):
    """Run prompt through model, capture residual stream at specified layers."""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    activations = {}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                activations[layer_idx] = output[0].detach()
            else:
                activations[layer_idx] = output.detach()
        return hook_fn

    for layer_idx in layers:
        h = model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        hooks.append(h)

    with torch.no_grad():
        outputs = model(**inputs)

    for h in hooks:
        h.remove()

    return activations, inputs, outputs


def identify_trajectory_features(saes, traj_acts, gen_acts, layers, threshold=0.1):
    """Find SAE features that are differentially active under trajectory vs generic.

    Returns per-layer: trajectory-specific feature indices and their activation strengths.
    """
    results = {}
    for layer in layers:
        sae = saes[layer]
        traj_hidden = traj_acts[layer][:, -1, :]  # last token position
        gen_hidden = gen_acts[layer][:, -1, :]

        traj_encoded = sae.encode(traj_hidden)
        gen_encoded = sae.encode(gen_hidden)

        traj_active = (traj_encoded.abs() > threshold).float().mean(dim=0)
        gen_active = (gen_encoded.abs() > threshold).float().mean(dim=0)

        diff = traj_active - gen_active
        traj_specific = (diff > 0.5).nonzero(as_tuple=True)[0]
        gen_specific = (diff < -0.5).nonzero(as_tuple=True)[0]

        results[layer] = {
            "traj_features": traj_specific.cpu().tolist(),
            "gen_features": gen_specific.cpu().tolist(),
            "n_traj": len(traj_specific),
            "n_gen": len(gen_specific),
            "traj_mean_activation": traj_encoded[:, traj_specific].mean().item() if len(traj_specific) > 0 else 0,
            "overlap": (traj_active * gen_active > 0).sum().item(),
        }
        print(f"  Layer {layer}: {len(traj_specific)} trajectory-specific, "
              f"{len(gen_specific)} generic-specific features")

    return results


def project_noise_onto_subspace(hidden, sae, feature_indices, scale, seed):
    """Project Gaussian noise onto SAE feature subspace."""
    torch.manual_seed(seed)
    encoded = sae.encode(hidden)

    noise = torch.zeros_like(encoded)
    if len(feature_indices) > 0:
        noise[:, feature_indices] = torch.randn(
            hidden.shape[0], len(feature_indices),
            device=hidden.device, dtype=hidden.dtype
        ) * scale

    perturbed_encoded = encoded + noise
    perturbed_hidden = sae.decode(perturbed_encoded)
    return perturbed_hidden


def measure_degradation(model, tokenizer, prompt, perturbed_hidden, layer, max_tokens=200):
    """Generate with perturbed hidden state and measure output quality."""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    def inject_hook(module, input, output):
        if isinstance(output, tuple):
            return (perturbed_hidden,) + output[1:]
        return perturbed_hidden

    hook = model.model.layers[layer].register_forward_hook(inject_hook)

    with torch.no_grad():
        gen_ids = model.generate(
            **inputs, max_new_tokens=max_tokens,
            do_sample=False, temperature=1.0
        )

    hook.remove()

    text = tokenizer.decode(gen_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    words = text.split()
    wc = len(words)
    ttr = len(set(w.lower() for w in words)) / max(wc, 1)
    self_markers = sum(1 for w in words if w.lower() in {"i", "my", "me", "myself", "i'm", "i've"})
    sr = self_markers / max(wc, 1)

    with torch.no_grad():
        outputs = model(**gen_ids[:, :inputs["input_ids"].shape[1] + min(50, max_tokens)],
                        labels=gen_ids[:, :inputs["input_ids"].shape[1] + min(50, max_tokens)])
        ppl = outputs.loss.exp().item() if outputs.loss is not None else float("nan")

    return {
        "word_count": wc,
        "ttr": round(ttr, 4),
        "self_ref": round(sr, 4),
        "perplexity": round(ppl, 4) if not np.isnan(ppl) else None,
        "text_preview": text[:200],
    }


def run_experiment():
    """Main F503 experiment."""
    print("=" * 60)
    print("  F503 — SAE SUBSPACE ISOLATION + TARGETED NOISE")
    print("=" * 60)

    model, tokenizer = load_model()
    print("\nLoading SAEs...")
    saes = load_saes(MID_LAYERS)

    # Phase 1: Identify trajectory-specific features
    print("\n--- Phase 1: Feature Identification ---")
    traj_acts, _, _ = get_residual_activations(model, tokenizer, TRAJECTORY_PROMPT, MID_LAYERS)
    gen_acts, _, _ = get_residual_activations(model, tokenizer, GENERIC_PROMPT, MID_LAYERS)
    features = identify_trajectory_features(saes, traj_acts, gen_acts, MID_LAYERS)

    # Phase 2: Targeted noise projection
    print("\n--- Phase 2: Targeted Noise Projection ---")
    results = {
        "finding": "F503",
        "concept": "sae_subspace_isolation",
        "model": MODEL_ID,
        "sae": SAE_ID,
        "layers": MID_LAYERS,
        "features": {str(k): {kk: vv for kk, vv in v.items() if kk != "traj_features" and kk != "gen_features"}
                     for k, v in features.items()},
        "degradation_curves": {},
    }

    for condition, prompt in [("trajectory", TRAJECTORY_PROMPT), ("generic", GENERIC_PROMPT)]:
        print(f"\n  Condition: {condition}")
        results["degradation_curves"][condition] = {}

        for layer in MID_LAYERS:
            print(f"    Layer {layer}:")
            layer_results = {"T": {}, "T_perp": {}}
            traj_feats = features[layer]["traj_features"]
            all_feats = list(range(saes[layer].num_latents))
            complement_feats = [f for f in all_feats if f not in set(traj_feats)]

            for target, feat_indices, label in [
                ("T", traj_feats, "trajectory subspace"),
                ("T_perp", complement_feats[:len(traj_feats)], "complement"),
            ]:
                if not feat_indices:
                    print(f"      {label}: no features, skipping")
                    continue

                for scale in NOISE_SCALES:
                    scale_results = []
                    for seed in range(N_SEEDS):
                        acts, _, _ = get_residual_activations(model, tokenizer, prompt, [layer])
                        hidden = acts[layer][:, -1:, :]
                        perturbed = project_noise_onto_subspace(
                            hidden, saes[layer], feat_indices, scale, seed + 42
                        )
                        metrics = measure_degradation(
                            model, tokenizer, prompt, perturbed, layer
                        )
                        scale_results.append(metrics)

                    avg_ppl = np.mean([r["perplexity"] for r in scale_results if r["perplexity"]])
                    avg_wc = np.mean([r["word_count"] for r in scale_results])
                    avg_ttr = np.mean([r["ttr"] for r in scale_results])

                    layer_results[target][str(scale)] = {
                        "avg_perplexity": round(float(avg_ppl), 4),
                        "avg_word_count": round(float(avg_wc), 1),
                        "avg_ttr": round(float(avg_ttr), 4),
                        "per_seed": scale_results,
                    }
                    print(f"      {label} @{scale}x: ppl={avg_ppl:.2f} wc={avg_wc:.0f} ttr={avg_ttr:.3f}")

            results["degradation_curves"][condition][str(layer)] = layer_results

    # Phase 3: Curve fitting
    print("\n--- Phase 3: Curve Analysis ---")
    from scipy.optimize import curve_fit

    def sigmoid(x, a, b, c):
        return a / (1 + np.exp(-b * (x - c)))

    def linear(x, a, b):
        return a * x + b

    for condition in ["trajectory", "generic"]:
        for layer_str, layer_data in results["degradation_curves"][condition].items():
            for target in ["T", "T_perp"]:
                if target not in layer_data or not layer_data[target]:
                    continue
                scales = sorted([float(s) for s in layer_data[target].keys()])
                ppls = [layer_data[target][str(s)]["avg_perplexity"] for s in scales]

                if len(scales) < 4:
                    continue

                x = np.array(scales)
                y = np.array(ppls)

                try:
                    popt_sig, _ = curve_fit(sigmoid, x, y, p0=[max(y), 1, np.median(x)], maxfev=5000)
                    sig_resid = np.sum((y - sigmoid(x, *popt_sig)) ** 2)
                except Exception:
                    sig_resid = float("inf")

                try:
                    popt_lin, _ = curve_fit(linear, x, y, maxfev=5000)
                    lin_resid = np.sum((y - linear(x, *popt_lin)) ** 2)
                except Exception:
                    lin_resid = float("inf")

                ratio = sig_resid / max(lin_resid, 1e-10)
                classification = "RELAY" if ratio < 0.5 else "EQUALIZER" if ratio > 2.0 else "AMBIGUOUS"

                layer_data[target]["curve_fit"] = {
                    "sigmoid_residual": round(float(sig_resid), 4),
                    "linear_residual": round(float(lin_resid), 4),
                    "ratio": round(float(ratio), 4),
                    "classification": classification,
                }
                print(f"  {condition}/{layer_str}/{target}: {classification} "
                      f"(sig/lin ratio={ratio:.3f})")

    # Save
    outpath = Path("/root/results/f503_sae_subspace.json")
    outpath.parent.mkdir(parents=True, exist_ok=True)

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(outpath, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {outpath} ({outpath.stat().st_size} bytes)")


if __name__ == "__main__":
    run_experiment()
