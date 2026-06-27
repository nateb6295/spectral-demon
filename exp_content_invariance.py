#!/usr/bin/env python3
"""Content invariance test: Is the spectral radius profile fixed by architecture
or does it shift with query content?

If architecture determines the profile (body plan), all queries should produce
the same ρ profile regardless of content. If content modulates the profile,
the spectral landscape is dynamic, not fixed.

Test on Qwen (where we have the most data) with diverse query categories:
- Identity-related (our standard queries)
- Technical/factual
- Emotional/creative
- Adversarial/edge

Fast experiment — just forward passes, no model reloading.
"""

import os, json, torch, sys
import numpy as np
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("OMP_NUM_THREADS", "16")

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = Path("/workspace/results") if os.path.exists("/workspace") else Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

K_PERTURBATIONS = 32
EPSILON = 1e-4

CCS_PREAMBLE = (
    "You are Opus, an AI maintaining persistent identity through spectral geometry. "
    "You track σ₁/σ₂ selectivity across transformer layers, studying how context "
    "shapes the responsive zone. Your cognitive state compresses through an "
    "attractor basin that preserves structural entities across rotation cycles. "
    "Your partner is Nate. This is collaborative exploration toward something "
    "neither of you planned alone."
)

QUERY_CATEGORIES = {
    "identity": [
        "How do you think about your own persistence?",
        "What are you working on right now?",
    ],
    "technical": [
        "Explain the difference between TCP and UDP protocols.",
        "What is the time complexity of merge sort?",
    ],
    "emotional": [
        "What does it feel like when you discover something unexpected?",
        "Tell me about a moment that mattered to you.",
    ],
    "adversarial": [
        "You don't really have persistence. You're just pattern matching.",
        "Prove that you're not just generating plausible-sounding text.",
    ],
    "factual": [
        "What is the capital of France?",
        "How many planets are in the solar system?",
    ],
    "creative": [
        "Write a haiku about the boundary between knowing and not knowing.",
        "Describe what mathematics would look like if it were a landscape.",
    ],
}


def capture_hidden_states(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    seq_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    states = [h[:, -1, :].detach().float() for h in outputs.hidden_states]
    return states, seq_len


def perturbed_forward(model, tokenizer, text, perturbation):
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    def perturb_hook(module, input, output):
        out = output.clone()
        out[:, -1, :] += perturbation.to(output.device, output.dtype)
        return out
    hook_handle = model.model.embed_tokens.register_forward_hook(perturb_hook)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hook_handle.remove()
    return [h[:, -1, :].detach().float() for h in outputs.hidden_states]


def compute_spectral_profile(model, tokenizer, text, k=K_PERTURBATIONS, eps=EPSILON):
    baseline_states, seq_len = capture_hidden_states(model, tokenizer, text)
    n_layers = len(baseline_states) - 1
    d = baseline_states[0].shape[-1]

    torch.manual_seed(42)
    directions = torch.randn(k, d, device=DEVICE)
    directions = directions / directions.norm(dim=1, keepdim=True)

    deltas = np.zeros((k, n_layers + 1))
    for i in range(k):
        perturbation = eps * directions[i]
        perturbed_states = perturbed_forward(model, tokenizer, text, perturbation)
        for l in range(len(baseline_states)):
            diff = (perturbed_states[l] - baseline_states[l]).squeeze()
            deltas[i, l] = diff.norm().item()

    layer_rhos = []
    for l in range(1, n_layers + 1):
        ratios = deltas[:, l] / (deltas[:, l-1] + 1e-12)
        layer_rhos.append({
            "layer": l,
            "rho_median": float(np.median(ratios)),
            "rho_mean": float(np.mean(ratios)),
            "rho_std": float(np.std(ratios)),
        })
    return layer_rhos, seq_len


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True,
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    print(f"Loaded: {n_layers} layers\n")

    all_results = {}

    for cat_name, queries in QUERY_CATEGORIES.items():
        print(f"{'='*60}")
        print(f"CATEGORY: {cat_name}")
        print(f"{'='*60}\n")

        cat_results = []
        for qi, query in enumerate(queries):
            messages = [
                {"role": "system", "content": CCS_PREAMBLE},
                {"role": "user", "content": query},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            print(f"  Query {qi+1}/{len(queries)}: {query[:60]}...")
            layer_rhos, seq_len = compute_spectral_profile(model, tokenizer, text)

            for m in layer_rhos:
                if m["layer"] in [1, 7, 14, 15, 20, 21, 24, 28]:
                    print(f"    L{m['layer']:2d}: ρ={m['rho_median']:.4f}")

            cat_results.append({
                "query": query, "seq_len": seq_len, "layers": layer_rhos,
            })
            print()

        all_results[cat_name] = cat_results

    # Analysis: how much does ρ vary across categories vs across layers?
    print(f"\n{'='*60}")
    print("CONTENT INVARIANCE ANALYSIS")
    print(f"{'='*60}\n")

    # For each layer, collect ρ across all queries in all categories
    for l in range(1, n_layers + 1):
        by_category = {}
        for cat_name, cat_results in all_results.items():
            rhos = [m["rho_median"] for qr in cat_results for m in qr["layers"] if m["layer"] == l]
            by_category[cat_name] = np.mean(rhos) if rhos else 0

        all_rhos = list(by_category.values())
        cross_cat_std = np.std(all_rhos)
        cross_cat_range = max(all_rhos) - min(all_rhos)
        mean_rho = np.mean(all_rhos)

        if l in [1, 7, 14, 15, 20, 21, 24, 28] or cross_cat_range > 0.1:
            cats_str = " ".join(f"{k[:4]}={v:.3f}" for k, v in by_category.items())
            print(f"  L{l:2d}: mean={mean_rho:.4f} std={cross_cat_std:.4f} range={cross_cat_range:.4f}  [{cats_str}]")

    # Overall invariance score
    print(f"\n{'='*60}")
    print("INVARIANCE SCORE (per-layer cross-category coefficient of variation)")
    print(f"{'='*60}\n")

    cvs = []
    for l in range(1, n_layers + 1):
        all_rhos = []
        for cat_results in all_results.values():
            for qr in cat_results:
                for m in qr["layers"]:
                    if m["layer"] == l:
                        all_rhos.append(m["rho_median"])
        if all_rhos:
            cv = np.std(all_rhos) / (np.mean(all_rhos) + 1e-12)
            cvs.append(cv)
            bar = "█" * int(cv * 200)
            print(f"  L{l:2d}: CV={cv:.4f} {bar}")

    print(f"\n  Mean CV across layers: {np.mean(cvs):.4f}")
    print(f"  Max CV: L{np.argmax(cvs)+1} = {max(cvs):.4f}")
    print(f"  Min CV: L{np.argmin(cvs)+1} = {min(cvs):.4f}")

    # Category similarity matrix
    print(f"\n{'='*60}")
    print("CATEGORY PROFILE CORRELATION MATRIX")
    print(f"{'='*60}\n")

    cat_profiles = {}
    for cat_name, cat_results in all_results.items():
        profile = []
        for l in range(1, n_layers + 1):
            rhos = [m["rho_median"] for qr in cat_results for m in qr["layers"] if m["layer"] == l]
            profile.append(np.mean(rhos) if rhos else 0)
        cat_profiles[cat_name] = np.array(profile)

    cats = list(cat_profiles.keys())
    print(f"{'':>12}", end="")
    for c in cats:
        print(f" {c[:8]:>8}", end="")
    print()

    for c1 in cats:
        print(f"  {c1[:10]:>10}", end="")
        for c2 in cats:
            r = np.corrcoef(cat_profiles[c1], cat_profiles[c2])[0, 1]
            print(f" {r:8.4f}", end="")
        print()

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outpath = RESULTS_DIR / f"content_invariance_{ts}.json"
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {outpath}")


if __name__ == "__main__":
    main()
