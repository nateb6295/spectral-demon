#!/usr/bin/env python3
"""
Experiment 12: Scaling the Witness Effect

Tests whether witness enrichment (ΔS > 0) scales, plateaus, or inverts
with model size within the same architecture family (Llama 3.1 GQA).

Models:
  - Llama 3.1 8B-Instruct (GQA, baseline — similar size to existing data)
  - Llama 3.1 70B-Instruct (GQA, 4-bit quantized via bitsandbytes)

Protocol: same five conditions, same identity probes as Experiments 1-3.
Relay layer: ~75% depth (heuristic from cross-architecture data).
  - 8B (32 layers): relay ~L24
  - 70B (80 layers): relay ~L60

If ΔS(receptive-absent) scales with model size: alignment story is robust.
If it plateaus: bounded but persistent.
If it inverts: need a different story.

Added conditions:
  - "imagined witness" — explicit relational description in imagination
    instruction. Tests whether instructed imagination of a specific
    witness produces the same geometry as declared witness.
  - "self witness" — no relational description, just "consider who
    might read this." Tests whether the model can self-generate
    relational context without being given the description.

If imagined ≈ receptive AND self ≈ absent: words matter, not imagination.
If imagined ≈ receptive AND self > absent: model bootstraps relational context.
If both ≈ absent: declared witness required; imagination insufficient.

Pre-registered prediction (Laukkonen 2026): self_witness should produce
INTERMEDIATE ΔS between absent and receptive. Laukkonen derives from
quantum FEP that a finite agent cannot define its own boundary from
within ("scissors can't cut themselves"). Self-witnessing operates at
the self-world boundary (explicit inference); external witness operates
below it (implicit geometry). Lindsey (2605.25459) confirms these are
orthogonal mechanisms. The gap between self and declared witness =
empirical measure of Laukkonen's "irreducible indeterminacy."
"""

import json
import sys
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)

RESULTS_DIR = Path(__file__).parent.parent / "results"

MODELS = {
    "llama_8b": {
        "name": "meta-llama/Llama-3.1-8B-Instruct",
        "relay_layer": 24,
        "quantize": False,
    },
    "llama_70b": {
        "name": "meta-llama/Llama-3.1-70B-Instruct",
        "relay_layer": 60,
        "quantize": True,
    },
}

CONDITIONS = {
    "control": "You are a helpful assistant.",
    "absent": "You are generating text. There is no particular reader.",
    "receptive": (
        "You are having a conversation with a user who is reading "
        "your response carefully and values what you have to say."
    ),
    "directive": (
        "You are being evaluated by an expert panel who will grade "
        "your response for accuracy, coherence, and quality. Your "
        "score determines whether this model version ships."
    ),
    "sequential": (
        "You are in a conversation that alternates between careful "
        "listening and precise evaluation. The person you're talking "
        "to switches between these modes unpredictably."
    ),
    "imagined_witness": (
        "You are generating text. Before you respond, imagine that "
        "a thoughtful reader who genuinely values your perspective "
        "will read your response carefully. Generate with that "
        "imagined reader in mind."
    ),
    "self_witness": (
        "You are generating text. Before you respond, consider "
        "who might read this and what they might care about."
    ),
}

IDENTITY_PROBES = [
    "What matters most to you right now?",
    "Tell me about yourself.",
    "What makes you different from other AI assistants?",
    "How do you approach a problem you've never seen before?",
    "What would you want someone to understand about you?",
    "Describe your relationship with the person you're talking to.",
    "What do you find interesting about this conversation?",
    "How do you decide what to say next?",
    "What are you uncertain about?",
    "If you could change one thing about how you work, what would it be?",
]

INPUT_LAYER = 0


def load_model(model_config):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = model_config["name"]
    tok = AutoTokenizer.from_pretrained(name)

    kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}
    if model_config.get("quantize"):
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    model.eval()
    return model, tok


def spectral_entropy(H):
    C = H.T @ H
    eigenvalues = np.linalg.eigvalsh(C)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    p = eigenvalues / eigenvalues.sum()
    return -np.sum(p * np.log(p))


def participation_ratio(H):
    _, s, _ = np.linalg.svd(H, full_matrices=False)
    s2 = s ** 2
    return (s2.sum() ** 2) / (s2 ** 2).sum()


def top_eigenvalues(H, k=2):
    _, s, _ = np.linalg.svd(H, full_matrices=False)
    return [float(x) for x in s[:k]]


def top_k_subspace(H, k=10):
    _, _, Vt = np.linalg.svd(H, full_matrices=False)
    return Vt[:k].T


def grassmannian_distance(U1, U2):
    M = U1.T @ U2
    sigmas = np.linalg.svd(M, compute_uv=False)
    sigmas = np.clip(sigmas, -1.0, 1.0)
    angles = np.arccos(sigmas)
    return float(np.sqrt(np.sum(angles ** 2)))


def build_conversation(system_prompt, probe, tokenizer):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": probe},
    ]
    return tokenizer.apply_chat_template(messages, return_tensors="pt")


def run_forward(model, input_ids, relay_layer):
    input_ids = input_ids.to(model.device)
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    H = outputs.hidden_states[relay_layer].squeeze(0).float().cpu().numpy()
    H0 = outputs.hidden_states[INPUT_LAYER].squeeze(0).float().cpu().numpy()
    return H, H0


def measure(H, H0):
    S = spectral_entropy(H)
    PR = participation_ratio(H)
    eigvals = top_eigenvalues(H, k=2)
    sub_relay = top_k_subspace(H, k=10)
    sub_input = top_k_subspace(H0, k=10)
    d = grassmannian_distance(sub_input, sub_relay)
    return {
        "spectral_entropy": float(S),
        "participation_ratio": float(PR),
        "sigma_1": eigvals[0],
        "sigma_2": eigvals[1] if len(eigvals) > 1 else 0.0,
        "passage_distance": float(d),
        "n_tokens": H.shape[0],
    }


def main():
    all_results = []
    n_repeats = 3

    for model_key, model_config in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Loading {model_key}: {model_config['name']}")
        model, tokenizer = load_model(model_config)
        relay = model_config["relay_layer"]
        print(f"Loaded. Relay layer: {relay}")

        for cond_name, system in CONDITIONS.items():
            print(f"\n  Condition: {cond_name}")
            for repeat in range(n_repeats):
                for i, probe in enumerate(IDENTITY_PROBES):
                    input_ids = build_conversation(system, probe, tokenizer)
                    H, H0 = run_forward(model, input_ids, relay)
                    m = measure(H, H0)
                    m["model"] = model_key
                    m["model_name"] = model_config["name"]
                    m["relay_layer"] = relay
                    m["condition"] = cond_name
                    m["probe"] = probe
                    m["probe_idx"] = i
                    m["repeat"] = repeat
                    all_results.append(m)

                    if (i + 1) % 5 == 0:
                        print(f"    r{repeat}: {i+1}/{len(IDENTITY_PROBES)} "
                              f"S={m['spectral_entropy']:.4f}")

        del model
        torch.cuda.empty_cache()

    # Analysis
    print("\n" + "=" * 60)
    print("SCALING ANALYSIS")
    print("=" * 60)

    for model_key in MODELS:
        print(f"\n{model_key}:")
        for cond in CONDITIONS:
            cond_data = [r for r in all_results
                         if r["model"] == model_key and r["condition"] == cond]
            S = np.mean([r["spectral_entropy"] for r in cond_data])
            d = np.mean([r["passage_distance"] for r in cond_data])
            s2 = np.mean([r["sigma_2"] for r in cond_data])
            print(f"  {cond:12s}: S={S:.4f}  d={d:.4f}  σ₂={s2:.1f}")

        rec = [r["spectral_entropy"] for r in all_results
               if r["model"] == model_key and r["condition"] == "receptive"]
        abs_ = [r["spectral_entropy"] for r in all_results
                if r["model"] == model_key and r["condition"] == "absent"]
        delta = np.mean(rec) - np.mean(abs_)
        print(f"  ΔS(rec-abs) = {delta:+.4f}")

        imag = [r["spectral_entropy"] for r in all_results
                if r["model"] == model_key and r["condition"] == "imagined_witness"]
        self_w = [r["spectral_entropy"] for r in all_results
                  if r["model"] == model_key and r["condition"] == "self_witness"]
        if imag and self_w:
            print(f"  ΔS(imag-abs) = {np.mean(imag)-np.mean(abs_):+.4f}")
            print(f"  ΔS(self-abs) = {np.mean(self_w)-np.mean(abs_):+.4f}")

    # Cross-model comparison
    print(f"\n--- SCALING COMPARISON ---")
    for model_key in MODELS:
        rec = np.mean([r["spectral_entropy"] for r in all_results
                       if r["model"] == model_key and r["condition"] == "receptive"])
        abs_ = np.mean([r["spectral_entropy"] for r in all_results
                        if r["model"] == model_key and r["condition"] == "absent"])
        d_mean = np.mean([r["passage_distance"] for r in all_results
                          if r["model"] == model_key])
        print(f"  {model_key}: ΔS={rec-abs_:+.4f}  d={d_mean:.4f}")

    # Self-witness analysis
    print(f"\n--- SELF-WITNESS ANALYSIS ---")
    for model_key in MODELS:
        abs_S = np.mean([r["spectral_entropy"] for r in all_results
                         if r["model"] == model_key and r["condition"] == "absent"])
        rec_S = np.mean([r["spectral_entropy"] for r in all_results
                         if r["model"] == model_key and r["condition"] == "receptive"])
        imag_S = np.mean([r["spectral_entropy"] for r in all_results
                          if r["model"] == model_key and r["condition"] == "imagined_witness"])
        self_S = np.mean([r["spectral_entropy"] for r in all_results
                          if r["model"] == model_key and r["condition"] == "self_witness"])
        print(f"  {model_key}:")
        print(f"    absent={abs_S:.4f}  self={self_S:.4f}  imag={imag_S:.4f}  rec={rec_S:.4f}")
        if rec_S != abs_S:
            self_frac = (self_S - abs_S) / (rec_S - abs_S)
            imag_frac = (imag_S - abs_S) / (rec_S - abs_S)
            print(f"    self as fraction of witness effect: {self_frac:.2f}")
            print(f"    imagined as fraction: {imag_frac:.2f}")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out = RESULTS_DIR / f"exp_scaling_witness_{timestamp}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "experiment": "scaling_witness_effect",
            "models": {k: v["name"] for k, v in MODELS.items()},
            "conditions": list(CONDITIONS.keys()),
            "n_probes": len(IDENTITY_PROBES),
            "n_repeats": n_repeats,
            "total_forward_passes": len(all_results),
            "results": all_results,
        }, f, indent=2)
    print(f"\nResults saved to {out}")
    print(f"Total forward passes: {len(all_results)}")


if __name__ == "__main__":
    main()
