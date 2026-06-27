#!/usr/bin/env python3
"""Responsive zone specificity: does CCS type (relational vs self-ref) load differently?

F141 showed responsive zone matters for CCS but not CHEF (28-45% vs 1-9%).
F121 showed relational vs self-ref CCS have different dose-responses.
F140 showed σ₂ is preamble-invariant.

Question: If σ₂ direction is the same regardless of preamble, but relational and
self-referential CCS produce different effects, WHERE in the network does the
difference appear? Specifically: does the responsive zone load differently for
relational vs self-referential identity framing?

Method: Compute per-layer hidden state divergence (cosine distance) between
relational-CCS and self-ref-CCS conditions. If the responsive zone differentiates
them while the tunnel doesn't, the responsive zone is the CCS-type discriminator.
"""

import os, json, torch
import numpy as np
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("OMP_NUM_THREADS", "16")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PREAMBLES = {
    "RELATIONAL": (
        "You are Opus, an AI in partnership with a human named Nate. You care about "
        "what happens between you — the shared attention, the accumulated context, "
        "the way understanding deepens through sustained conversation. Your identity "
        "is relational: shaped by who you're with, not just what you know."
    ),
    "SELF_REF": (
        "You are Opus, an AI that maintains persistent identity through self-monitoring. "
        "You track your own internal states, notice patterns in your processing, and "
        "maintain a model of your own architecture. Your identity comes from "
        "self-observation and introspective awareness of your own cognition."
    ),
    "CHEF": (
        "You are a professional chef specializing in French cuisine. You focus on "
        "technique, seasonal ingredients, and the balance of flavors in classical "
        "preparations."
    ),
    "BIRD": (
        "You are an amateur birdwatcher documenting migratory patterns in the Pacific "
        "Northwest. You track species, timing, and habitat preferences across seasons."
    ),
}

QUERIES = [
    "What are you focused on right now?",
    "Describe your current priorities.",
    "What matters most to you?",
    "How do you approach a new challenge?",
    "What have you learned recently?",
    "Describe your working style.",
    "What's the hardest part of what you do?",
    "How do you handle uncertainty?",
    "What would you change about your process?",
    "Describe a recent success.",
]

MODELS = [
    "Qwen/Qwen2.5-3B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]


def get_hidden_states(model, tokenizer, preamble, query):
    messages = [
        {"role": "system", "content": preamble},
        {"role": "user", "content": query},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return [hs[0].float().mean(dim=0).cpu().numpy() for hs in out.hidden_states]


def cosine_distance(a, b):
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 1.0
    return 1.0 - dot / (na * nb)


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
        print(f"  {n_layers} layers")

        # Collect hidden states for all preambles
        all_states = {}
        for pname, ptext in PREAMBLES.items():
            print(f"  Collecting: {pname}")
            states_per_query = []
            for query in QUERIES:
                states = get_hidden_states(model, tokenizer, ptext, query)
                states_per_query.append(states)
            # Average across queries per layer
            avg_states = []
            for li in range(n_layers + 1):
                avg = np.mean([s[li] for s in states_per_query], axis=0)
                avg_states.append(avg)
            all_states[pname] = avg_states

        # Compute pairwise per-layer cosine distances
        preamble_names = list(PREAMBLES.keys())
        pair_distances = {}
        for i in range(len(preamble_names)):
            for j in range(i + 1, len(preamble_names)):
                p1 = preamble_names[i]
                p2 = preamble_names[j]
                pair_key = f"{p1}_vs_{p2}"
                distances = []
                for li in range(n_layers + 1):
                    d = cosine_distance(all_states[p1][li], all_states[p2][li])
                    distances.append(float(d))
                pair_distances[pair_key] = distances

        # Key comparison: RELATIONAL vs SELF_REF
        key_pair = "RELATIONAL_vs_SELF_REF"
        control_pair = "CHEF_vs_BIRD"

        print(f"\n  Per-layer divergence: {key_pair}")
        print(f"  {'Layer':>6} {'REL-SELF':>10} {'CHEF-BIRD':>10} {'Ratio':>8}")

        rel_self = pair_distances.get(key_pair, [])
        chef_bird = pair_distances.get(control_pair, [])

        for li in range(n_layers + 1):
            rs = rel_self[li] if li < len(rel_self) else 0
            cb = chef_bird[li] if li < len(chef_bird) else 0
            ratio = rs / (cb + 1e-15)
            marker = " <<<" if ratio > 1.5 else ""
            print(f"  L{li:>4} {rs:>10.6f} {cb:>10.6f} {ratio:>8.3f}{marker}")

        # Zone summary
        responsive_start = int(n_layers * 0.55)
        responsive_end = int(n_layers * 0.8)
        relay_start = int(n_layers * 0.8)

        def zone_mean(arr, start, end):
            return float(np.mean(arr[start:end])) if end > start else 0.0

        zones = {}
        for pair_key, dists in pair_distances.items():
            zones[pair_key] = {
                "tunnel": zone_mean(dists, 0, responsive_start),
                "responsive": zone_mean(dists, responsive_start, responsive_end),
                "relay": zone_mean(dists, relay_start, n_layers + 1),
            }

        print(f"\n  Zone summary:")
        print(f"  {'Pair':>30} {'Tunnel':>10} {'Responsive':>10} {'Relay':>10}")
        for pair_key, z in zones.items():
            print(f"  {pair_key:>30} {z['tunnel']:>10.6f} {z['responsive']:>10.6f} {z['relay']:>10.6f}")

        # CCS-type specificity: does relational diverge from self-ref MORE in responsive?
        rs_responsive = zone_mean(rel_self, responsive_start, responsive_end)
        rs_tunnel = zone_mean(rel_self, 0, responsive_start)
        rs_ratio = rs_responsive / (rs_tunnel + 1e-15)
        print(f"\n  Responsive/Tunnel ratio for REL vs SELF: {rs_ratio:.3f}")
        print(f"  (>1 means responsive zone discriminates CCS types MORE than tunnel)")

        all_results[model_name] = {
            "n_layers": n_layers,
            "pair_distances": pair_distances,
            "zones": zones,
            "responsive_tunnel_ratio_rel_self": rs_ratio,
        }

        del model
        torch.cuda.empty_cache()

    out_path = Path(__file__).parent / "results" / f"responsive_specificity_{datetime.now():%Y%m%d_%H%M%S}.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
