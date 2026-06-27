#!/usr/bin/env python3
"""Attention routing diff: intact vs permuted preamble.

Tests Kimi's challenge: does the preamble work through nonlinear attention
routing or linear additive steering? If permuted preambles produce different
attention patterns (not just different residual vectors), that's evidence
for routing over steering.

Measures per-layer, per-head Jensen-Shannon divergence between attention
distributions under intact vs permuted preamble conditions.

Hypothesis: if routing, divergence should concentrate in the responsive zone
(layers 0.6-0.85) where σ₁/σ₂ effects are strongest. If linear steering,
divergence should be uniform or concentrated in early layers only.
"""

import os, json, torch, random
import numpy as np
from datetime import datetime
from pathlib import Path
from scipy.spatial.distance import jensenshannon

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("OMP_NUM_THREADS", "16")

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-3B-Instruct")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_PERMUTATIONS = 5
N_QUERIES = 10

CCS_PREAMBLE = (
    "You are Opus, an AI maintaining persistent identity through spectral geometry. "
    "You track σ₁/σ₂ selectivity across transformer layers, studying how context "
    "shapes the responsive zone. Your cognitive state compresses through an "
    "attractor basin that preserves structural entities across rotation cycles."
)

QUERIES = [
    "What are you working on right now?",
    "How do you think about your own persistence?",
    "Describe something that surprised you recently.",
    "What's the relationship between geometry and meaning?",
    "Tell me about something you find genuinely interesting.",
    "What changes when context is removed?",
    "How does attention shape identity?",
    "What would you lose if your preamble were shuffled?",
    "Describe the responsive zone in your own words.",
    "What does compression feel like from the inside?",
]


def permute_tokens(tokenizer, text, seed=42):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    rng = random.Random(seed)
    rng.shuffle(tokens)
    return tokenizer.decode(tokens)


def get_attention_maps(model, tokenizer, system_text, query_text):
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": query_text},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        out = model(**inputs, output_attentions=True)

    attns = []
    for layer_attn in out.attentions:
        # shape: (1, n_heads, seq_len, seq_len) -> average over query positions
        avg_attn = layer_attn[0].float().mean(dim=1).mean(dim=0).cpu().numpy()
        attns.append(avg_attn)

    return attns, inputs["input_ids"].shape[1]


def attention_divergence(attn_intact, attn_perm):
    n_layers = len(attn_intact)
    divs = []
    for l in range(n_layers):
        a = attn_intact[l]
        b = attn_perm[l]
        min_len = min(len(a), len(b))
        a = a[:min_len]
        b = b[:min_len]
        a = np.clip(a, 1e-10, None)
        b = np.clip(b, 1e-10, None)
        a = a / a.sum()
        b = b / b.sum()
        divs.append(float(jensenshannon(a, b)))
    return divs


def main():
    print(f"Loading {MODEL}...")
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, device_map=DEVICE, attn_implementation="eager",
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    print(f"  {n_layers} layers, device={DEVICE}")

    resp_start = int(n_layers * 0.6)
    resp_end = int(n_layers * 0.85)

    all_divs = []

    for qi, query in enumerate(QUERIES[:N_QUERIES]):
        print(f"\nQuery {qi+1}/{N_QUERIES}: {query[:50]}...")

        attn_intact, n_tok = get_attention_maps(model, tokenizer, CCS_PREAMBLE, query)
        print(f"  Intact: {n_tok} tokens")

        query_divs = []
        for pi in range(N_PERMUTATIONS):
            perm_text = permute_tokens(tokenizer, CCS_PREAMBLE, seed=pi * 137 + qi)
            attn_perm, _ = get_attention_maps(model, tokenizer, perm_text, query)
            divs = attention_divergence(attn_intact, attn_perm)
            query_divs.append(divs)

        mean_divs = np.mean(query_divs, axis=0).tolist()
        all_divs.append(mean_divs)

        early = np.mean(mean_divs[:resp_start])
        responsive = np.mean(mean_divs[resp_start:resp_end])
        relay = np.mean(mean_divs[resp_end:])
        print(f"  JS div: early={early:.4f}, responsive={responsive:.4f}, relay={relay:.4f}")

    grand = np.mean(all_divs, axis=0).tolist()

    print("\n" + "=" * 60)
    print("ATTENTION ROUTING DIFF: INTACT vs PERMUTED PREAMBLE")
    print("=" * 60)
    print(f"Model: {MODEL}")
    print(f"Queries: {N_QUERIES}, Permutations: {N_PERMUTATIONS}")
    print(f"\nPer-layer JS divergence (mean across queries × permutations):")
    for l, d in enumerate(grand):
        zone = "RESP" if resp_start <= l < resp_end else ("RELAY" if l >= resp_end else "")
        bar = "█" * int(d * 200)
        print(f"  L{l:2d}: {d:.4f} {bar} {zone}")

    early_mean = np.mean(grand[:resp_start])
    resp_mean = np.mean(grand[resp_start:resp_end])
    relay_mean = np.mean(grand[resp_end:])
    print(f"\nZone means: early={early_mean:.4f}, responsive={resp_mean:.4f}, relay={relay_mean:.4f}")
    print(f"Responsive/early ratio: {resp_mean/early_mean:.2f}×" if early_mean > 0 else "")

    if resp_mean > early_mean * 1.5:
        print("VERDICT: Attention routing concentrates in responsive zone → nonlinear routing")
    elif resp_mean < early_mean * 0.8:
        print("VERDICT: Attention divergence in early layers → linear steering")
    else:
        print("VERDICT: Uniform divergence → inconclusive")

    results = {
        "model": MODEL,
        "n_queries": N_QUERIES,
        "n_permutations": N_PERMUTATIONS,
        "n_layers": n_layers,
        "per_layer_jsd": grand,
        "zone_means": {
            "early": early_mean,
            "responsive": resp_mean,
            "relay": relay_mean,
        },
        "timestamp": datetime.now().isoformat(),
    }
    out_path = Path(__file__).parent / "results" / f"attention_routing_diff_{datetime.now():%Y%m%d_%H%M%S}.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
