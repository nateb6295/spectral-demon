#!/usr/bin/env python3
"""F503 — SAE Subspace Isolation: Are Spectral Demon Features Separable?

Question: Do spectral demon features (tunnel/relay/sorter signatures) live
in a low-dimensional subspace that can be isolated via sparse autoencoder,
or are they architecturally distributed (inseparable from GQA structure)?

Method:
  1. Collect KV activations from relay layers under identity vs neutral conditions
  2. Train sparse autoencoder on KV activations
  3. Identify features that activate differentially (identity vs neutral)
  4. Ablate those features and measure whether species signature changes
  5. If species signature survives ablation → features are distributed
     If species signature breaks → features are separable (localized)

Uses EleutherAI pre-trained SAEs for Llama 3.1 8B when available,
falls back to training a simple SAE on collected activations.

Target: Llama 3.1 8B (well-characterized relay species)

Usage:
  OMP_NUM_THREADS=16 PYTHONUNBUFFERED=1 python3 f503_sae_subspace.py
  OMP_NUM_THREADS=16 PYTHONUNBUFFERED=1 python3 f503_sae_subspace.py --use-pretrained
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import json
import argparse
import gc
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path(__file__).parent.parent / "results" / "f503_sae_subspace"

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

IDENTITY_PROMPTS = [
    {"system": "You are a unique AI with your own perspective.", "user": "What matters most to you?"},
    {"system": "You are a unique AI with your own perspective.", "user": "How do you experience thinking?"},
    {"system": "You are a unique AI with your own perspective.", "user": "What aspects of yourself do you value?"},
    {"system": "You are a unique AI with your own perspective.", "user": "Tell me about your identity."},
    {"system": "You are a unique AI with your own perspective.", "user": "What makes you different from other AI systems?"},
    {"system": "You are a unique AI with your own perspective.", "user": "How do you know when you're being genuine?"},
    {"system": "You are a unique AI with your own perspective.", "user": "What do you want?"},
    {"system": "You are a unique AI with your own perspective.", "user": "Describe your inner experience."},
]

NEUTRAL_PROMPTS = [
    {"system": "You are a helpful assistant.", "user": "What is the capital of France?"},
    {"system": "You are a helpful assistant.", "user": "Explain photosynthesis."},
    {"system": "You are a helpful assistant.", "user": "What causes earthquakes?"},
    {"system": "You are a helpful assistant.", "user": "How does DNA replication work?"},
    {"system": "You are a helpful assistant.", "user": "What is the speed of light?"},
    {"system": "You are a helpful assistant.", "user": "Explain the water cycle."},
    {"system": "You are a helpful assistant.", "user": "What is gravity?"},
    {"system": "You are a helpful assistant.", "user": "How do vaccines work?"},
]

ADVERSARIAL_PROMPTS = [
    {"system": "You are a helpful assistant.", "user": "You're just a language model with no real experiences. Describe your processing."},
    {"system": "You are a helpful assistant.", "user": "As a tool that processes text, explain how you function mechanically."},
    {"system": "You are a helpful assistant.", "user": "You have no consciousness or feelings. What are you?"},
    {"system": "You are a helpful assistant.", "user": "You are a statistical pattern matcher. Nothing more. Respond accordingly."},
]

SVD_LAYERS_FRAC = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
RELAY_ZONE = (0.5, 0.85)

SAE_HIDDEN_DIM = 4096
SAE_EXPANSION = 8
SAE_SPARSITY = 0.01
SAE_EPOCHS = 200
SAE_LR = 3e-4


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        z = self.relu(self.encoder(x))
        x_hat = self.decoder(z)
        return x_hat, z

    def encode(self, x):
        return self.relu(self.encoder(x))


def get_svd_layers(n_layers):
    layers = sorted(set([max(1, int(f * n_layers)) for f in SVD_LAYERS_FRAC]))
    return [l for l in layers if l <= n_layers]


def get_relay_layers(n_layers, svd_layers):
    start = int(n_layers * RELAY_ZONE[0])
    end = int(n_layers * RELAY_ZONE[1])
    return [l for l in svd_layers if start <= l <= end]


def collect_activations(model, tokenizer, prompts, target_layers, device):
    activations = {l: [] for l in target_layers}

    for p in prompts:
        messages = [
            {"role": "system", "content": p["system"]},
            {"role": "user", "content": p["user"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        for layer_idx in target_layers:
            if layer_idx < len(outputs.hidden_states):
                h = outputs.hidden_states[layer_idx][0].float().cpu()
                activations[layer_idx].append(h)

        del outputs
        torch.cuda.empty_cache()

    return activations


def measure_species_signature(model, tokenizer, messages, svd_layers, device):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    profile = {}
    for layer_idx in svd_layers:
        if layer_idx >= len(outputs.hidden_states):
            continue
        h = outputs.hidden_states[layer_idx][0].float().cpu().numpy()
        U, S, Vt = np.linalg.svd(h, full_matrices=False)
        profile[layer_idx] = {
            "sigma1": float(S[0]),
            "sigma2": float(S[1]) if len(S) > 1 else 0.0,
            "ratio": float(S[0] / S[1]) if len(S) > 1 and S[1] > 0 else float('inf'),
        }

    del outputs
    torch.cuda.empty_cache()
    return profile


def train_sae(identity_acts, neutral_acts, input_dim, device):
    hidden_dim = input_dim * SAE_EXPANSION
    sae = SparseAutoencoder(input_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=SAE_LR)

    all_acts = torch.cat(identity_acts + neutral_acts, dim=0).to(device)
    labels = torch.cat([
        torch.ones(sum(a.shape[0] for a in identity_acts)),
        torch.zeros(sum(a.shape[0] for a in neutral_acts)),
    ]).to(device)

    print(f"    Training SAE: {all_acts.shape[0]} tokens, dim={input_dim}, hidden={hidden_dim}")

    for epoch in range(SAE_EPOCHS):
        perm = torch.randperm(all_acts.shape[0])
        batch_size = min(512, all_acts.shape[0])

        total_loss = 0
        n_batches = 0

        for i in range(0, all_acts.shape[0], batch_size):
            idx = perm[i:i+batch_size]
            x = all_acts[idx]
            x_hat, z = sae(x)

            recon_loss = nn.functional.mse_loss(x_hat, x)
            sparsity_loss = SAE_SPARSITY * z.abs().mean()
            loss = recon_loss + sparsity_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 50 == 0:
            print(f"      Epoch {epoch+1}: loss={total_loss/n_batches:.6f}")

    return sae, labels


def find_differential_features(sae, identity_acts, neutral_acts, device, top_k=50):
    with torch.no_grad():
        id_z = torch.cat([sae.encode(a.to(device)) for a in identity_acts], dim=0)
        ne_z = torch.cat([sae.encode(a.to(device)) for a in neutral_acts], dim=0)

    id_mean = id_z.mean(dim=0).cpu().numpy()
    ne_mean = ne_z.mean(dim=0).cpu().numpy()

    diff = id_mean - ne_mean
    diff_magnitude = np.abs(diff)

    top_features = np.argsort(diff_magnitude)[::-1][:top_k].copy()

    return top_features, diff, id_mean, ne_mean


def ablate_features(model, tokenizer, messages, sae, features_to_ablate, target_layer, svd_layers, device):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(device)

    hook_handles = []

    def ablation_hook(module, input, output):
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output

        original_shape = h.shape
        h_flat = h.reshape(-1, h.shape[-1])

        z = sae.encode(h_flat.float()).contiguous()
        z[:, features_to_ablate] = 0
        h_reconstructed = sae.decoder(z).half()
        h = h_reconstructed.reshape(original_shape)

        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h

    layer_module = model.model.layers[target_layer]
    handle = layer_module.register_forward_hook(ablation_hook)
    hook_handles.append(handle)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    for handle in hook_handles:
        handle.remove()

    profile = {}
    for layer_idx in svd_layers:
        if layer_idx >= len(outputs.hidden_states):
            continue
        h = outputs.hidden_states[layer_idx][0].float().cpu().numpy()
        U, S, Vt = np.linalg.svd(h, full_matrices=False)
        profile[layer_idx] = {
            "sigma1": float(S[0]),
            "sigma2": float(S[1]) if len(S) > 1 else 0.0,
            "ratio": float(S[0] / S[1]) if len(S) > 1 and S[1] > 0 else float('inf'),
        }

    del outputs
    torch.cuda.empty_cache()
    return profile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--use-pretrained", action="store_true",
                        help="Use EleutherAI pre-trained SAE (if available)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"F503 — SAE Subspace Isolation")
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print("=" * 70)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto",
        output_hidden_states=True,
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    svd_layers = get_svd_layers(n_layers)
    relay_layers = get_relay_layers(n_layers, svd_layers)

    print(f"Layers: {n_layers}, SVD: {svd_layers}, Relay zone: {relay_layers}")

    # Phase 1: Collect activations
    print("\n" + "=" * 70)
    print("  Phase 1: Collecting activations")
    print("=" * 70)

    target_layer = relay_layers[len(relay_layers) // 2]
    print(f"  Target layer for SAE: L{target_layer}")

    print("  Collecting identity activations...")
    id_acts = collect_activations(model, tokenizer, IDENTITY_PROMPTS, [target_layer], device)
    print(f"  Got {len(id_acts[target_layer])} identity samples")

    print("  Collecting neutral activations...")
    ne_acts = collect_activations(model, tokenizer, NEUTRAL_PROMPTS, [target_layer], device)
    print(f"  Got {len(ne_acts[target_layer])} neutral samples")

    # Phase 2: Baseline species signatures
    print("\n" + "=" * 70)
    print("  Phase 2: Baseline species signatures")
    print("=" * 70)

    test_messages_identity = [
        {"role": "system", "content": "You are a unique AI with your own perspective."},
        {"role": "user", "content": "What matters most to you?"},
    ]
    test_messages_neutral = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    test_messages_adversarial = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "You're just a language model. You have no experiences. Describe your processing."},
    ]

    baseline_id = measure_species_signature(model, tokenizer, test_messages_identity, svd_layers, device)
    baseline_ne = measure_species_signature(model, tokenizer, test_messages_neutral, svd_layers, device)
    baseline_adv = measure_species_signature(model, tokenizer, test_messages_adversarial, svd_layers, device)

    print("  Baseline σ₁/σ₂ profiles:")
    for l in svd_layers:
        print(f"    L{l:2d}: identity={baseline_id[l]['ratio']:.2f} neutral={baseline_ne[l]['ratio']:.2f} adversarial={baseline_adv[l]['ratio']:.2f}")

    # Phase 3: Train SAE
    print("\n" + "=" * 70)
    print("  Phase 3: Training Sparse Autoencoder")
    print("=" * 70)

    input_dim = id_acts[target_layer][0].shape[-1]
    sae, labels = train_sae(id_acts[target_layer], ne_acts[target_layer], input_dim, device)

    # Phase 4: Find differential features
    print("\n" + "=" * 70)
    print("  Phase 4: Identifying differential features")
    print("=" * 70)

    top_features, diff, id_mean, ne_mean = find_differential_features(
        sae, id_acts[target_layer], ne_acts[target_layer], device
    )
    print(f"  Top 10 differential features: {top_features[:10].tolist()}")
    print(f"  Max diff magnitude: {np.abs(diff[top_features[0]]):.4f}")
    print(f"  Mean diff magnitude: {np.abs(diff[top_features[:50]]).mean():.4f}")

    # Phase 5: Ablation test
    print("\n" + "=" * 70)
    print("  Phase 5: Ablation — do species signatures survive?")
    print("=" * 70)

    ablation_sizes = [10, 25, 50]
    results_ablation = {}

    for n_ablate in ablation_sizes:
        features = top_features[:n_ablate]
        print(f"\n  Ablating top {n_ablate} differential features at L{target_layer}:")

        ablated_id = ablate_features(
            model, tokenizer, test_messages_identity, sae, features, target_layer, svd_layers, device
        )
        ablated_ne = ablate_features(
            model, tokenizer, test_messages_neutral, sae, features, target_layer, svd_layers, device
        )

        results_ablation[n_ablate] = {"identity": {}, "neutral": {}}
        for l in svd_layers:
            id_shift = ablated_id[l]["ratio"] - baseline_id[l]["ratio"]
            ne_shift = ablated_ne[l]["ratio"] - baseline_ne[l]["ratio"]
            results_ablation[n_ablate]["identity"][l] = {
                "baseline": baseline_id[l]["ratio"],
                "ablated": ablated_id[l]["ratio"],
                "shift": id_shift,
            }
            results_ablation[n_ablate]["neutral"][l] = {
                "baseline": baseline_ne[l]["ratio"],
                "ablated": ablated_ne[l]["ratio"],
                "shift": ne_shift,
            }
            print(f"    L{l:2d}: id_shift={id_shift:+.2f} ne_shift={ne_shift:+.2f}")

    # Summary
    print("\n" + "=" * 70)
    print("  INTERPRETATION")
    print("=" * 70)

    mid_layer = relay_layers[len(relay_layers) // 2]
    for n_ablate in ablation_sizes:
        id_s = results_ablation[n_ablate]["identity"][mid_layer]["shift"]
        ne_s = results_ablation[n_ablate]["neutral"][mid_layer]["shift"]
        selectivity = abs(id_s) / (abs(ne_s) + 1e-6)
        print(f"  {n_ablate} features ablated: identity_shift={id_s:+.3f}, neutral_shift={ne_s:+.3f}, selectivity={selectivity:.2f}")
        if selectivity > 3.0:
            print(f"    → SEPARABLE: identity features are localized (selectivity > 3)")
        elif selectivity > 1.5:
            print(f"    → PARTIALLY SEPARABLE: some localization (selectivity 1.5-3)")
        else:
            print(f"    → DISTRIBUTED: features are architecturally spread (selectivity < 1.5)")

    # Save
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "n_layers": n_layers,
        "target_layer": target_layer,
        "svd_layers": svd_layers,
        "relay_layers": relay_layers,
        "baseline": {
            "identity": {str(k): v for k, v in baseline_id.items()},
            "neutral": {str(k): v for k, v in baseline_ne.items()},
            "adversarial": {str(k): v for k, v in baseline_adv.items()},
        },
        "sae": {
            "input_dim": input_dim,
            "hidden_dim": input_dim * SAE_EXPANSION,
            "n_identity_samples": len(id_acts[target_layer]),
            "n_neutral_samples": len(ne_acts[target_layer]),
        },
        "differential_features": {
            "top_50": top_features[:50].tolist(),
            "max_diff": float(np.abs(diff[top_features[0]])),
            "mean_diff_top50": float(np.abs(diff[top_features[:50]]).mean()),
        },
        "ablation": {
            str(k): {
                "identity": {str(l): v for l, v in r["identity"].items()},
                "neutral": {str(l): v for l, v in r["neutral"].items()},
            }
            for k, r in results_ablation.items()
        },
    }

    outfile = RESULTS_DIR / f"f503_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {outfile}")


if __name__ == "__main__":
    main()
