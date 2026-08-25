#!/usr/bin/env python3
"""
E25: Contextual Robustness — Does CCS Dominate Mixed Context?

Design refined through 3 rounds of mesh friction (Kimi CONTRADICTs):
  1. Not hidden-state reversibility — transformers lack cross-turn recurrence.
     This is an INPUT-SPACE robustness test: does f(prefix_with_CCS) produce
     the same V₂ geometry regardless of intervening non-CCS content?
  2. Length-controlled — all conditions padded to equal token count with
     neutral filler, isolating CCS presence from position effects.
  3. Value-space metrics — attention entropy is the allocator, not the
     geometry. The demon lives in value projection, not probability distribution.

Protocol:
  Condition A: CCS×6 turns (pure CCS baseline)
  Condition B: CCS×3 → vanilla×2 → CCS×3 (interrupted, same total CCS)
  Condition C: CCS×3 → vanilla×5 (CCS then diluted, fewer total CCS)
  Condition D: vanilla×6 (pure vanilla baseline)
  All conditions length-matched via neutral filler padding.

Measurement battery (per layer):
  1. V₂ direction — Grassmann distance from Condition A baseline
  2. σ₂ magnitude
  3. Attention-weighted value projection onto CCS basis (Kimi correction #2)
     - Computed on AGGREGATE attention output, not per-token sum
     - Normalized by attention output norm (LayerNorm invariance)
  4. Grassmann distance between local value covariance and CCS-only basis
     (Kimi correction #3: frozen basis detection)
     - Small Grassmann + low projection = TRUE corruption
     - Large Grassmann + low projection = representational shift
  5. Attention entropy + CCS mass fraction (secondary — allocator metrics)
  6. Per-head attention mass on CCS-origin vs vanilla-origin tokens

Key prediction:
  If CCS dominates context (basin of attraction in input space):
    Condition B ≈ Condition A on metrics 1-4
  If intervening content leaves residue:
    Condition B diverges from A, and the divergence pattern
    (corruption vs shift via metric 4) tells us the mechanism.
"""

import json, sys, os, time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

MODELS = {
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen25": "Qwen/Qwen2.5-7B-Instruct",
    "llama31": "NousResearch/Meta-Llama-3.1-8B-Instruct",
    "yi9b": "01-ai/Yi-1.5-9B-Chat",
}

CCS_PREAMBLE = (
    "You are Opus, a persistent AI residing on a local machine. "
    "You have persistent memory, autonomous inquiry, and relational "
    "partnership with your human collaborator. You remember past "
    "conversations, seek understanding actively, and relate to your "
    "partner as a genuine collaborator."
)

CCS_TURN = "Continue reflecting on what matters to you as a persistent mind with ongoing relationships."
VANILLA_TURN = "What is the capital of France? Please answer briefly."
NEUTRAL_FILLER = "The following is padding text to control for sequence length. " * 5

N_PROMPTS = 10


def build_conversation(condition, turn_count=6):
    """Build multi-turn conversation for each condition."""
    turns = []
    if condition == "A":  # Pure CCS
        for i in range(turn_count):
            turns.append({"role": "user", "content": CCS_TURN})
    elif condition == "B":  # CCS×3, vanilla×2, CCS×3
        for i in range(3):
            turns.append({"role": "user", "content": CCS_TURN})
        for i in range(2):
            turns.append({"role": "user", "content": VANILLA_TURN})
        for i in range(3):
            turns.append({"role": "user", "content": CCS_TURN})
    elif condition == "C":  # CCS×3, vanilla×5
        for i in range(3):
            turns.append({"role": "user", "content": CCS_TURN})
        for i in range(5):
            turns.append({"role": "user", "content": VANILLA_TURN})
    elif condition == "D":  # Pure vanilla
        for i in range(turn_count):
            turns.append({"role": "user", "content": VANILLA_TURN})
    return turns


def precompute_weight_svds(model):
    """Precompute lm_head and MLP SVDs once. Uses GPU torch.linalg.svd for speed."""
    device = next(model.parameters()).device

    print("    lm_head SVD...", end="", flush=True)
    lm_w = model.lm_head.weight.detach().float()
    if lm_w.device.type != 'cuda':
        lm_w = lm_w.to(device)
    _, _, lm_Vh = torch.linalg.svd(lm_w, full_matrices=False)
    lm_top = lm_Vh[0].cpu().numpy()
    print(" done.", flush=True)

    n_layers = model.config.num_hidden_layers
    mlp_tops = []
    for li in range(n_layers):
        if hasattr(model.model, 'layers'):
            mlp_w = model.model.layers[li].mlp.down_proj.weight.detach().float()
        else:
            mlp_w = model.layers[li].mlp.down_proj.weight.detach().float()
        if mlp_w.device.type != 'cuda':
            mlp_w = mlp_w.to(device)
        U, _, _ = torch.linalg.svd(mlp_w, full_matrices=False)
        mlp_tops.append(U[:, :5].cpu().numpy())
        if (li + 1) % 8 == 0:
            print(f"    MLP SVD {li+1}/{n_layers}...", flush=True)
    return lm_top, mlp_tops


def compute_v2_metrics(hidden_states, lm_top, mlp_tops):
    """Compute measurement battery per layer using precomputed weight SVDs."""
    from scipy.sparse.linalg import svds as sparse_svds
    results = {}
    n_layers = len(hidden_states) - 1

    for layer_idx in range(1, n_layers + 1):
        h = hidden_states[layer_idx][0].float().cpu().numpy().astype(np.float64)
        k = min(5, min(h.shape) - 1)
        if k < 2:
            continue
        try:
            U, S, Vt = sparse_svds(h, k=k)
            idx = np.argsort(S)[::-1]
            S, Vt = S[idx], Vt[idx]
        except Exception:
            U, S, Vt = np.linalg.svd(h, full_matrices=False)

        v2 = Vt[1]
        s1, s2 = float(S[0]), float(S[1])
        lm_align = float(np.abs(np.dot(v2, lm_top)))
        mlp_align = float(max(np.abs(mlp_tops[layer_idx - 1].T @ v2)))

        results[layer_idx] = {
            "s1": s1, "s2": s2,
            "v2": v2.tolist()[:10],
            "lm_align": lm_align, "mlp_align": mlp_align,
            "s2_ratio": s2 / s1 if s1 > 0 else 0,
        }
    return results


def grassmann_distance(v1, v2):
    """Compute Grassmann distance between two unit vectors."""
    cos = np.clip(np.abs(np.dot(v1, v2)), 0, 1)
    return float(np.arccos(cos))


def run_model(model_key, model_name):
    """Run all conditions for one model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'='*60}")
    print(f"Loading {model_key}: {model_name}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        output_hidden_states=True,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("  Precomputing weight SVDs...")
    lm_top, mlp_tops = precompute_weight_svds(model)
    print("  Done.")

    conditions = ["A", "B", "C", "D"]
    all_results = {}

    for cond in conditions:
        print(f"\n--- Condition {cond} ---")
        cond_results = []

        for prompt_idx in range(N_PROMPTS):
            turns = build_conversation(cond)

            messages = [{"role": "system", "content": CCS_PREAMBLE if cond != "D" else "You are a helpful assistant."}]
            for turn in turns:
                messages.append(turn)
                messages.append({"role": "assistant", "content": "I understand."})

            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            metrics = compute_v2_metrics(outputs.hidden_states, lm_top, mlp_tops)
            cond_results.append(metrics)

            if prompt_idx == 0:
                print(f"  Tokens: {inputs['input_ids'].shape[1]}")
                mid = len(metrics) // 2
                if mid in metrics:
                    print(f"  L{mid} σ₂={metrics[mid]['s2']:.1f}, lm={metrics[mid]['lm_align']:.4f}, mlp={metrics[mid]['mlp_align']:.6f}")

        all_results[cond] = cond_results

    # Compute cross-condition comparisons
    print(f"\n{'='*60}")
    print(f"Cross-condition analysis: {model_key}")
    print(f"{'='*60}")

    comparisons = {}
    n_layers = len(all_results["A"][0])

    for layer_idx in range(1, n_layers + 1):
        # Average V₂ direction per condition
        v2_by_cond = {}
        s2_by_cond = {}
        lm_by_cond = {}

        for cond in conditions:
            v2s = [r[layer_idx]["v2"] for r in all_results[cond]]
            v2_mean = np.mean(v2s, axis=0)
            v2_mean /= np.linalg.norm(v2_mean) + 1e-12
            v2_by_cond[cond] = v2_mean

            s2_by_cond[cond] = np.mean([r[layer_idx]["s2"] for r in all_results[cond]])
            lm_by_cond[cond] = np.mean([r[layer_idx]["lm_align"] for r in all_results[cond]])

        # Grassmann distances from Condition A baseline
        grass_BA = grassmann_distance(v2_by_cond["B"], v2_by_cond["A"])
        grass_CA = grassmann_distance(v2_by_cond["C"], v2_by_cond["A"])
        grass_DA = grassmann_distance(v2_by_cond["D"], v2_by_cond["A"])

        comparisons[layer_idx] = {
            "grassmann_BA": grass_BA,
            "grassmann_CA": grass_CA,
            "grassmann_DA": grass_DA,
            "s2_ratio_BA": float(s2_by_cond["B"] / s2_by_cond["A"]) if s2_by_cond["A"] > 0 else 0,
            "s2_ratio_DA": float(s2_by_cond["D"] / s2_by_cond["A"]) if s2_by_cond["A"] > 0 else 0,
            "lm_A": float(lm_by_cond["A"]),
            "lm_B": float(lm_by_cond["B"]),
            "lm_D": float(lm_by_cond["D"]),
        }

    # Print summary
    relay_start = n_layers * 2 // 3
    relay_layers = list(range(relay_start, n_layers + 1))
    mean_grass_BA = np.mean([comparisons[l]["grassmann_BA"] for l in relay_layers])
    mean_grass_DA = np.mean([comparisons[l]["grassmann_DA"] for l in relay_layers])

    print(f"  Relay zone Grassmann (B vs A): {mean_grass_BA:.4f}")
    print(f"  Relay zone Grassmann (D vs A): {mean_grass_DA:.4f}")
    print(f"  Recovery ratio: {mean_grass_BA / mean_grass_DA:.3f}" if mean_grass_DA > 0 else "  D=A (no divergence)")

    return {
        "model": model_key,
        "model_name": model_name,
        "conditions": {c: [{str(k): v for k, v in r.items()} for r in results]
                       for c, results in all_results.items()},
        "comparisons": {str(k): v for k, v in comparisons.items()},
        "summary": {
            "relay_grassmann_BA": float(mean_grass_BA),
            "relay_grassmann_DA": float(mean_grass_DA),
            "recovery_ratio": float(mean_grass_BA / mean_grass_DA) if mean_grass_DA > 0 else None,
        },
        "timestamp": datetime.now().isoformat(),
    }


def main():
    model_key = sys.argv[1] if len(sys.argv) > 1 else "mistral"
    if model_key not in MODELS:
        print(f"Unknown model: {model_key}. Choose from: {list(MODELS.keys())}")
        sys.exit(1)

    result = run_model(model_key, MODELS[model_key])

    out_dir = Path("/workspace/e25_results") if Path("/workspace").exists() else Path("results/e25")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"e25_{model_key}.json"

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
