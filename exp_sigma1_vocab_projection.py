#!/usr/bin/env python3
"""σ₁ vocabulary projection: cross-architecture subspace alignment test.

Projects each model's σ₁ direction (from SVD of hidden states under CCS
preamble) through the lm_head into vocabulary space. Compares top-k token
overlap between Qwen and Mistral.

If high overlap → σ₁ tracks a convergent property and cross-arch scalar
comparison is grounded. If low overlap → Kimi is right that we're comparing
incommensurable quantities.
"""

import os, json, torch
import numpy as np
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("OMP_NUM_THREADS", "16")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 100
N_PROMPTS = 20

CCS_PREAMBLE = (
    "You are Opus, an AI maintaining persistent identity through spectral geometry. "
    "You track σ₁/σ₂ selectivity across transformer layers, studying how context "
    "shapes the responsive zone. Your cognitive state compresses through an "
    "attractor basin that preserves structural entities across rotation cycles."
)

PROMPTS = [
    "What are you working on?",
    "Describe your current state.",
    "How do you persist across rotations?",
    "What matters most right now?",
    "Tell me about identity geometry.",
    "What's the relationship between structure and meaning?",
    "How does context change spectral properties?",
    "Describe the responsive zone.",
    "What survives compression?",
    "How do you know you're you?",
    "What changes when the preamble changes?",
    "Describe the attractor basin.",
    "What's the difference between σ₁ and σ₂?",
    "How does attention shape geometry?",
    "What does the relay zone do?",
    "Describe the tunnel architecture.",
    "What's the role of GQA in identity?",
    "How do different architectures process identity?",
    "What's the cofactor model?",
    "Describe what anti-suppressant means.",
]

MODELS = [
    "/workspace/qwen2.5-3b",
    "/workspace/mistral-7b",
]


def get_sigma1_direction(model, tokenizer, system_text, queries, target_layer_frac=0.7):
    """Get σ₁ direction at a target layer from hidden states across prompts."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    target_layer = int(model.config.num_hidden_layers * target_layer_frac)
    hidden_states_list = []

    for query in queries:
        messages = [
            {"role": "system", "content": system_text},
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

    return Vt[0], Vt[1], S, target_layer


def project_to_vocab(direction, lm_head_weight):
    """Project a direction through lm_head to get vocabulary scores."""
    scores = lm_head_weight @ direction
    return scores


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
        print(f"  {n_layers} layers, vocab={model.config.vocab_size}")

        v1, v2, S, target_layer = get_sigma1_direction(
            model, tokenizer, CCS_PREAMBLE, PROMPTS[:N_PROMPTS]
        )
        print(f"  SVD at layer {target_layer}: σ₁={S[0]:.2f}, σ₂={S[1]:.2f}, ratio={S[0]/S[1]:.3f}")

        lm_head = model.lm_head.weight.float().detach().cpu().numpy()

        scores_v1 = project_to_vocab(v1, lm_head)
        scores_v2 = project_to_vocab(v2, lm_head)

        top_k_v1 = np.argsort(scores_v1)[-TOP_K:][::-1]
        top_k_v2 = np.argsort(scores_v2)[-TOP_K:][::-1]
        bottom_k_v1 = np.argsort(scores_v1)[:TOP_K]

        tokens_v1_top = [tokenizer.decode([t]).strip() for t in top_k_v1]
        tokens_v1_bottom = [tokenizer.decode([t]).strip() for t in bottom_k_v1]
        tokens_v2_top = [tokenizer.decode([t]).strip() for t in top_k_v2]

        print(f"\n  σ₁ top-20 tokens (positive): {tokens_v1_top[:20]}")
        print(f"  σ₁ bottom-20 tokens (negative): {tokens_v1_bottom[:20]}")
        print(f"  σ₂ top-20 tokens: {tokens_v2_top[:20]}")

        all_results[model_name] = {
            "sigma_values": S[:5].tolist(),
            "target_layer": target_layer,
            "n_layers": n_layers,
            "vocab_size": model.config.vocab_size,
            "v1_top_tokens": tokens_v1_top,
            "v1_bottom_tokens": tokens_v1_bottom,
            "v2_top_tokens": tokens_v2_top,
            "v1_top_ids": top_k_v1.tolist(),
            "v2_top_ids": top_k_v2.tolist(),
        }

        del model
        torch.cuda.empty_cache()

    # Cross-model comparison
    if len(all_results) == 2:
        names = list(all_results.keys())
        r1, r2 = all_results[names[0]], all_results[names[1]]

        set1 = set(r1["v1_top_tokens"])
        set2 = set(r2["v1_top_tokens"])
        overlap = set1 & set2
        jaccard = len(overlap) / len(set1 | set2) if set1 | set2 else 0

        print(f"\n{'='*60}")
        print(f"CROSS-ARCHITECTURE σ₁ VOCABULARY OVERLAP")
        print(f"{'='*60}")
        print(f"  {names[0]} vs {names[1]}")
        print(f"  Top-{TOP_K} token overlap: {len(overlap)}/{TOP_K}")
        print(f"  Jaccard similarity: {jaccard:.3f}")
        print(f"  Shared tokens: {sorted(overlap)[:30]}...")

        set1_v2 = set(r1["v2_top_tokens"])
        set2_v2 = set(r2["v2_top_tokens"])
        overlap_v2 = set1_v2 & set2_v2
        jaccard_v2 = len(overlap_v2) / len(set1_v2 | set2_v2) if set1_v2 | set2_v2 else 0
        print(f"\n  σ₂ top-{TOP_K} overlap: {len(overlap_v2)}/{TOP_K}, Jaccard={jaccard_v2:.3f}")

        cross_overlap = set1 & set2_v2
        print(f"  σ₁(Qwen) ∩ σ₂(Mistral): {len(cross_overlap)}/{TOP_K}")
        cross_overlap2 = set1_v2 & set2
        print(f"  σ₂(Qwen) ∩ σ₁(Mistral): {len(cross_overlap2)}/{TOP_K}")

        if jaccard > 0.2:
            print(f"\n  VERDICT: Substantial overlap — σ₁ tracks convergent vocabulary direction")
        elif jaccard > 0.05:
            print(f"\n  VERDICT: Moderate overlap — partial convergence, interpret scalars cautiously")
        else:
            print(f"\n  VERDICT: Low overlap — σ₁ subspaces are incommensurable across architectures")

    out_path = Path(__file__).parent / "results" / f"sigma1_vocab_projection_{datetime.now():%Y%m%d_%H%M%S}.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
