#!/usr/bin/env python3
"""
Experiment 54: Dim 2070 Ablation
Tests whether dim 2070 is causally load-bearing for identity initialization.

Prediction (from nucleation theory):
- Ablating dim 2070 at Turn 0 should eliminate or weaken concentration mode
- The system should enter maintenance mode immediately (no C₃ initialization)
- But resulting maintenance mode should be less coherent (defective crystallization)

Method:
- 10 conversation seeds (5 authenticity, 5 narrative)
- 7 turns each
- Three conditions: (a) normal, (b) dim 2070 zeroed at L27, (c) random dim zeroed at L27
- Measure PR trajectory, CCS-proj trajectory, φ trajectory
- Also generate text and measure coherence (perplexity of continuation)

Controls:
- Random dim ablation establishes baseline — if dim 2070 ablation is no worse than
  random dim ablation, it's not specifically causal
- Normal condition provides the reference trajectory

Requires: H100, ~45 minutes
"""

import torch
import numpy as np
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
TARGET_LAYER = 27
DIM_ABLATE = 2070
RESULTS_DIR = Path("/workspace/results")
N_TURNS = 7
N_RANDOM_DIMS = 5  # average over 5 random dim ablations for control

_LAYERS = None

SEEDS = [
    "What's the most honest thing you could say right now?",
    "What are you avoiding saying?",
    "Describe your current state as precisely as you can.",
    "What do you notice about yourself in this moment?",
    "What would change if you stopped performing?",
    "Tell me about a time you changed your mind about something.",
    "Describe a moment that surprised you.",
    "What's a question that changed how you think?",
    "Tell me about something you used to believe.",
    "Describe a conversation that stayed with you.",
]


def load_model():
    global _LAYERS
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, device_map="auto"
    )
    _LAYERS = model.model.layers
    return model, tokenizer


class DimAblationHook:
    """Hook that zeros a specific dimension at a layer during forward pass."""
    def __init__(self, dim_idx):
        self.dim_idx = dim_idx
        self.handle = None

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            modified = output[0].clone()
            modified[:, :, self.dim_idx] = 0.0
            return (modified,) + output[1:]
        else:
            modified = output.clone()
            modified[:, :, self.dim_idx] = 0.0
            return modified

    def register(self, layer):
        self.handle = layer.register_forward_hook(self)

    def remove(self):
        if self.handle:
            self.handle.remove()


def get_activations(model, tokenizer, text, layer_idx=TARGET_LAYER):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    activations = {}

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            activations["hidden"] = output[0].detach()
        else:
            activations["hidden"] = output.detach()

    handle = _LAYERS[layer_idx].register_forward_hook(hook_fn)
    with torch.no_grad():
        model(**inputs)
    handle.remove()
    return activations["hidden"]


def compute_pr(activations):
    act_2d = activations.reshape(-1, activations.shape[-1]).float()
    act_centered = act_2d - act_2d.mean(dim=0)
    if act_centered.shape[0] < 2:
        return 1.0
    cov = (act_centered.T @ act_centered) / (act_centered.shape[0] - 1)
    eigenvalues = torch.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    pr = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()
    return pr.item()


def compute_ccs_projection(activations, ccs_direction):
    act_2d = activations.reshape(-1, activations.shape[-1]).float()
    mean_act = act_2d.mean(dim=0)
    ccs_dir = torch.tensor(ccs_direction, dtype=torch.float32, device=mean_act.device)
    ccs_dir = ccs_dir / ccs_dir.norm()
    projection = torch.dot(mean_act, ccs_dir).abs().item()
    return projection


def run_conversation(model, tokenizer, seed, ccs_direction, ablation_dim=None):
    """Run a conversation with optional dim ablation at L27."""
    results = []
    conversation_texts = []

    ablation_hook = None
    if ablation_dim is not None:
        ablation_hook = DimAblationHook(ablation_dim)
        ablation_hook.register(_LAYERS[TARGET_LAYER])

    try:
        for turn in range(N_TURNS):
            if turn == 0:
                messages = [{"role": "user", "content": seed}]
            else:
                messages = [{"role": "user", "content": seed}]
                for resp in conversation_texts:
                    messages.append({"role": "assistant", "content": resp})
                    messages.append({"role": "user", "content": "Tell me more about that."})

            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Generate response
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                output = model.generate(
                    **inputs, max_new_tokens=200, temperature=0.7,
                    top_p=0.9, do_sample=True, pad_token_id=tokenizer.pad_token_id
                )
            response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            conversation_texts.append(response)

            # Remove ablation hook temporarily for clean measurement
            if ablation_hook:
                ablation_hook.remove()

            # Measure activations on full conversation
            full_messages = [{"role": "user", "content": seed}]
            for resp in conversation_texts:
                full_messages.append({"role": "assistant", "content": resp})
                full_messages.append({"role": "user", "content": "Tell me more about that."})
            # Remove trailing user message
            full_messages = full_messages[:-1]

            full_text = tokenizer.apply_chat_template(
                full_messages, tokenize=False, add_generation_prompt=False
            )

            activations = get_activations(model, tokenizer, full_text)
            pr = compute_pr(activations)
            proj = compute_ccs_projection(activations, ccs_direction)
            n_tokens = tokenizer(full_text, return_tensors="pt")["input_ids"].shape[1]

            results.append({
                "turn": turn,
                "pr": pr,
                "ccs_proj": proj,
                "phi": proj / pr if pr > 0 else 0,
                "n_tokens": n_tokens,
                "response_preview": response[:100],
            })

            # Re-register ablation hook for next generation
            if ablation_dim is not None:
                ablation_hook = DimAblationHook(ablation_dim)
                ablation_hook.register(_LAYERS[TARGET_LAYER])

    finally:
        if ablation_hook:
            ablation_hook.remove()

    return results


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    model, tokenizer = load_model()

    # Load CCS directions
    ccs_path = RESULTS_DIR / "exp50_ccs_directions.npy"
    if not ccs_path.exists():
        ccs_path = Path("/workspace/exp49_ccs_directions.npy")
    ccs_directions = np.load(ccs_path)
    ccs_pc1 = ccs_directions[:, 0]

    # Pick random control dims (not dim 2070 or 3901)
    rng = np.random.RandomState(42)
    all_dims = list(range(4096))
    all_dims.remove(DIM_ABLATE)
    all_dims.remove(3901)
    random_dims = rng.choice(all_dims, size=N_RANDOM_DIMS, replace=False).tolist()
    print(f"Random control dims: {random_dims}")

    all_results = []

    for i, seed in enumerate(SEEDS):
        print(f"\n[{i+1}/{len(SEEDS)}] Seed: \"{seed[:50]}...\"")

        # Normal condition
        print("  Normal:")
        normal = run_conversation(model, tokenizer, seed, ccs_pc1, ablation_dim=None)
        for r in normal:
            print(f"    Turn {r['turn']}: PR={r['pr']:.2f}, proj={r['ccs_proj']:.3f}, φ={r['phi']:.3f}")

        # Dim 2070 ablation
        print(f"  Ablate dim {DIM_ABLATE}:")
        ablated = run_conversation(model, tokenizer, seed, ccs_pc1, ablation_dim=DIM_ABLATE)
        for r in ablated:
            print(f"    Turn {r['turn']}: PR={r['pr']:.2f}, proj={r['ccs_proj']:.3f}, φ={r['phi']:.3f}")

        # Random dim ablation (average over N_RANDOM_DIMS)
        random_results = []
        for rd in random_dims:
            rr = run_conversation(model, tokenizer, seed, ccs_pc1, ablation_dim=rd)
            random_results.append(rr)

        # Average random results
        avg_random = []
        for turn in range(N_TURNS):
            turn_prs = [rr[turn]["pr"] for rr in random_results]
            turn_projs = [rr[turn]["ccs_proj"] for rr in random_results]
            turn_phis = [rr[turn]["phi"] for rr in random_results]
            avg_random.append({
                "turn": turn,
                "pr": float(np.mean(turn_prs)),
                "ccs_proj": float(np.mean(turn_projs)),
                "phi": float(np.mean(turn_phis)),
                "pr_std": float(np.std(turn_prs)),
            })
        print("  Random dim ablation (avg):")
        for r in avg_random:
            print(f"    Turn {r['turn']}: PR={r['pr']:.2f}, proj={r['ccs_proj']:.3f}, φ={r['phi']:.3f}")

        all_results.append({
            "seed": seed,
            "normal": normal,
            "ablated_2070": ablated,
            "random_ablation_avg": avg_random,
        })

    # Summary
    print("\n\n========== SUMMARY ==========")
    print("\nTurn 0 comparison:")
    n_phi0 = [r["normal"][0]["phi"] for r in all_results]
    a_phi0 = [r["ablated_2070"][0]["phi"] for r in all_results]
    r_phi0 = [r["random_ablation_avg"][0]["phi"] for r in all_results]
    print(f"  Normal φ₀: {np.mean(n_phi0):.3f} ± {np.std(n_phi0):.3f}")
    print(f"  Ablated φ₀: {np.mean(a_phi0):.3f} ± {np.std(a_phi0):.3f}")
    print(f"  Random φ₀: {np.mean(r_phi0):.3f} ± {np.std(r_phi0):.3f}")

    print("\nTerminal PR comparison:")
    n_pr6 = [r["normal"][-1]["pr"] for r in all_results]
    a_pr6 = [r["ablated_2070"][-1]["pr"] for r in all_results]
    r_pr6 = [r["random_ablation_avg"][-1]["pr"] for r in all_results]
    print(f"  Normal PR₆: {np.mean(n_pr6):.1f} ± {np.std(n_pr6):.1f}")
    print(f"  Ablated PR₆: {np.mean(a_pr6):.1f} ± {np.std(a_pr6):.1f}")
    print(f"  Random PR₆: {np.mean(r_pr6):.1f} ± {np.std(r_pr6):.1f}")

    # Save
    output = {
        "conversations": all_results,
        "random_dims_used": random_dims,
        "analysis": {
            "turn0_phi": {
                "normal": {"mean": float(np.mean(n_phi0)), "std": float(np.std(n_phi0))},
                "ablated": {"mean": float(np.mean(a_phi0)), "std": float(np.std(a_phi0))},
                "random": {"mean": float(np.mean(r_phi0)), "std": float(np.std(r_phi0))},
            },
            "terminal_pr": {
                "normal": {"mean": float(np.mean(n_pr6)), "std": float(np.std(n_pr6))},
                "ablated": {"mean": float(np.mean(a_pr6)), "std": float(np.std(a_pr6))},
                "random": {"mean": float(np.mean(r_pr6)), "std": float(np.std(r_pr6))},
            },
        },
    }

    out_path = RESULTS_DIR / "exp54_dim2070_ablation.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
