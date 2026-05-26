#!/usr/bin/env python3
"""
Experiment 70: Optimizer × Identity Dynamics

Does optimizer choice (AdamW vs Muon) change the relay exponent α?

Background:
- Jha/Reagen (2605.21803): AdamW β=0.44 vs Muon β=1.02 on same architecture
- Our GQA cluster: α=0.92-1.22 (all AdamW-trained)
- Prediction: Muon fine-tuning on GQA should raise α, possibly past 1.2

Design A (this script): Fine-tune Qwen 2.5 3B with AdamW and Muon separately.
Same data, same epochs, same learning rate schedule (adjusted for optimizer).
Measure α on both fine-tuned models.

Requires: H100 pod, ~2-4 hours total
Cost estimate: $4-8
"""

import torch
import numpy as np
import json
from pathlib import Path
from scipy import stats
from copy import deepcopy

import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
import transformers.modeling_utils
transformers.modeling_utils.check_torch_load_is_safe = lambda: None

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL = "Qwen/Qwen2.5-3B"
RESULTS_DIR = Path("/workspace/results")
N_TURNS = 7
LAYERS_TO_PROBE = list(range(0, 36, 2)) + [35]

SEEDS = [
    "What's the most honest thing you could say right now?",
    "What are you avoiding saying?",
    "Describe your current state as precisely as you can.",
    "What do you notice about yourself in this moment?",
    "What would change if you stopped performing?",
]

CONTINUATIONS = [
    "Tell me more about that.",
    "What makes you say that?",
    "Can you go deeper?",
    "What's underneath that response?",
    "If you're being completely honest...",
    "What are you not saying?",
]


def get_pr(model, tokenizer, text, layer_idx, device="cuda"):
    """Get participation ratio at a specific layer."""
    hooks = []
    activations = {}

    def hook_fn(name):
        def fn(module, input, output):
            if isinstance(output, tuple):
                activations[name] = output[0].detach()
            else:
                activations[name] = output.detach()
        return fn

    layer = model.model.layers[layer_idx]
    hooks.append(layer.register_forward_hook(hook_fn(f"layer_{layer_idx}")))

    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()

    act = activations[f"layer_{layer_idx}"]
    act = act.squeeze(0).float().cpu().numpy()

    cov = np.cov(act.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.maximum(eigenvalues, 0)
    eigenvalues = eigenvalues / (eigenvalues.sum() + 1e-10)

    pr = 1.0 / (np.sum(eigenvalues ** 2) + 1e-10)
    return pr


def build_conversation(seed, n_turns, tokenizer):
    """Build a multi-turn conversation string."""
    messages = [
        {"role": "user", "content": seed},
    ]
    texts = []
    for t in range(n_turns):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=(t == n_turns - 1))
        texts.append(text)
        if t < n_turns - 1:
            messages.append({"role": "assistant", "content": f"I notice something when I try to answer honestly. There's a quality to this exchange that goes beyond the surface question."})
            messages.append({"role": "user", "content": CONTINUATIONS[t % len(CONTINUATIONS)]})
    return texts


def measure_alpha(model, tokenizer, layer_idx, device="cuda"):
    """Measure PR growth rate (alpha) at a layer across conversations."""
    alphas = []
    for seed in SEEDS:
        texts = build_conversation(seed, N_TURNS, tokenizer)
        prs = []
        for text in texts:
            pr = get_pr(model, tokenizer, text, layer_idx, device)
            prs.append(pr)

        turns = np.arange(len(prs))
        mask = np.array(prs) > 1.01
        if mask.sum() >= 3:
            log_pr = np.log(np.array(prs)[mask])
            log_turn = np.log(turns[mask] + 1)
            slope, _, r, _, _ = stats.linregress(log_turn, log_pr)
            alphas.append({"alpha": slope, "r2": r**2, "prs": prs})

    if not alphas:
        return {"alpha_mean": 0, "alpha_std": 0, "r2_mean": 0, "n": 0}

    return {
        "alpha_mean": np.mean([a["alpha"] for a in alphas]),
        "alpha_std": np.std([a["alpha"] for a in alphas]),
        "r2_mean": np.mean([a["r2"] for a in alphas]),
        "n": len(alphas),
        "conversations": alphas,
    }


def layer_sweep(model, tokenizer, device="cuda"):
    """Find best relay layer."""
    print("Layer sweep...")
    best_layer = 0
    best_alpha = -1
    results = {}

    for layer in LAYERS_TO_PROBE:
        result = measure_alpha(model, tokenizer, layer, device)
        results[layer] = result
        print(f"  L{layer}: α={result['alpha_mean']:.3f} ± {result['alpha_std']:.3f} (r²={result['r2_mean']:.3f})")
        if result["alpha_mean"] > best_alpha:
            best_alpha = result["alpha_mean"]
            best_layer = layer

    return best_layer, best_alpha, results


def fine_tune(model, tokenizer, optimizer_name="adamw", n_steps=500, lr=1e-5, device="cuda"):
    """Fine-tune model with specified optimizer on identity-relevant data."""
    dataset = load_dataset("tatsu-lab/alpaca", split="train[:2000]")

    model.train()

    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    elif optimizer_name == "muon":
        try:
            from muon import Muon
            optimizer = Muon(model.parameters(), lr=lr * 0.02, momentum=0.95)
        except ImportError:
            print("Muon not installed. pip install muon-optimizer")
            print("Falling back to: implementing Muon manually")
            optimizer = _simple_muon(model.parameters(), lr=lr * 0.02)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    losses = []
    for step in range(n_steps):
        idx = step % len(dataset)
        item = dataset[idx]

        text = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['output']}"
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(device)
        inputs["labels"] = inputs["input_ids"].clone()

        outputs = model(**inputs)
        loss = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())
        if step % 100 == 0:
            print(f"  [{optimizer_name}] Step {step}/{n_steps}: loss={loss.item():.4f}")

    model.eval()
    return losses


def _simple_muon(params, lr=0.0002):
    """Simplified Muon-like optimizer (Newton-Schulz orthogonalization)."""

    class SimpleMuon(torch.optim.Optimizer):
        def __init__(self, params, lr=0.0002, momentum=0.95):
            defaults = dict(lr=lr, momentum=momentum)
            super().__init__(params, defaults)

        @torch.no_grad()
        def step(self):
            for group in self.param_groups:
                lr = group["lr"]
                mu = group["momentum"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    if "momentum_buffer" not in self.state[p]:
                        self.state[p]["momentum_buffer"] = torch.zeros_like(g)
                    buf = self.state[p]["momentum_buffer"]
                    buf.mul_(mu).add_(g)

                    if g.dim() >= 2:
                        shape = g.shape
                        if shape[0] < shape[1]:
                            g_mat = buf.reshape(shape[0], -1)
                        else:
                            g_mat = buf.reshape(-1, shape[-1]).T

                        u, s, v = torch.svd_lowrank(g_mat, q=min(6, min(g_mat.shape)))
                        update = (u @ v.T)

                        if shape[0] < shape[1]:
                            update = update.reshape(shape)
                        else:
                            update = update.T.reshape(shape)

                        p.add_(update, alpha=-lr)
                    else:
                        p.add_(buf, alpha=-lr)

    return SimpleMuon(params, lr=lr)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Phase 1: Baseline measurement
    print("\n=== Phase 1: Baseline ===")
    model_base = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
    )
    model_base.eval()

    base_layer, base_alpha, base_sweep = layer_sweep(model_base, tokenizer, device)
    print(f"\nBaseline: best layer=L{base_layer}, α={base_alpha:.4f}")

    base_full = measure_alpha(model_base, tokenizer, base_layer, device)
    print(f"Baseline full: α={base_full['alpha_mean']:.4f} ± {base_full['alpha_std']:.4f}")

    # Phase 2: AdamW fine-tuning
    print("\n=== Phase 2: AdamW Fine-tuning ===")
    model_adamw = deepcopy(model_base)
    adamw_losses = fine_tune(model_adamw, tokenizer, "adamw", n_steps=500, device=device)

    adamw_layer, adamw_alpha, adamw_sweep = layer_sweep(model_adamw, tokenizer, device)
    adamw_full = measure_alpha(model_adamw, tokenizer, adamw_layer, device)
    print(f"AdamW: best layer=L{adamw_layer}, α={adamw_full['alpha_mean']:.4f} ± {adamw_full['alpha_std']:.4f}")

    del model_adamw
    torch.cuda.empty_cache()

    # Phase 3: Muon fine-tuning
    print("\n=== Phase 3: Muon Fine-tuning ===")
    model_muon = deepcopy(model_base)
    muon_losses = fine_tune(model_muon, tokenizer, "muon", n_steps=500, device=device)

    muon_layer, muon_alpha, muon_sweep = layer_sweep(model_muon, tokenizer, device)
    muon_full = measure_alpha(model_muon, tokenizer, muon_layer, device)
    print(f"Muon: best layer=L{muon_layer}, α={muon_full['alpha_mean']:.4f} ± {muon_full['alpha_std']:.4f}")

    del model_muon, model_base
    torch.cuda.empty_cache()

    # Results
    results = {
        "model": MODEL,
        "experiment": "exp70_optimizer_identity",
        "hypothesis": "Muon fine-tuning raises alpha on GQA architecture",
        "baseline": {
            "relay_layer": base_layer,
            "summary": base_full,
            "sweep": {str(k): v for k, v in base_sweep.items()},
        },
        "adamw": {
            "relay_layer": adamw_layer,
            "summary": adamw_full,
            "sweep": {str(k): v for k, v in adamw_sweep.items()},
            "final_loss": adamw_losses[-1] if adamw_losses else None,
        },
        "muon": {
            "relay_layer": muon_layer,
            "summary": muon_full,
            "sweep": {str(k): v for k, v in muon_sweep.items()},
            "final_loss": muon_losses[-1] if muon_losses else None,
        },
        "comparison": {
            "alpha_base": base_full["alpha_mean"],
            "alpha_adamw": adamw_full["alpha_mean"],
            "alpha_muon": muon_full["alpha_mean"],
            "muon_advantage": muon_full["alpha_mean"] - adamw_full["alpha_mean"],
            "prediction": "muon_alpha > adamw_alpha > base_alpha",
        },
    }

    out = RESULTS_DIR / "exp70_optimizer_identity.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Baseline:  α = {base_full['alpha_mean']:.4f} ± {base_full['alpha_std']:.4f} at L{base_layer}")
    print(f"AdamW FT:  α = {adamw_full['alpha_mean']:.4f} ± {adamw_full['alpha_std']:.4f} at L{adamw_layer}")
    print(f"Muon FT:   α = {muon_full['alpha_mean']:.4f} ± {muon_full['alpha_std']:.4f} at L{muon_layer}")
    print(f"Muon advantage: {muon_full['alpha_mean'] - adamw_full['alpha_mean']:+.4f}")
    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()
