#!/usr/bin/env python3
"""Experiment: Hidden State Direction — Shape vs Content

All prior experiments measured the SHAPE of hidden states (singular values, CV).
CCS ≈ weather on every shape metric. But output logits differ (CCS preserves
more of bare's vocabulary). Since logits = W @ hidden_state, the hidden states
must differ in DIRECTION even if their SVD profiles match.

Two vectors can have identical singular value decompositions but point in
completely different directions. This experiment measures:

1. Cosine similarity of last-token hidden states between conditions at each layer
2. L2 distance of last-token hidden states
3. Projection onto bare's principal components (how much of bare's direction is preserved)

If CCS hidden states point more toward bare's direction than weather does →
CCS preserves content direction despite matching geometry.
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import os
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = "/workspace/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

CCS_PREAMBLE = """I am Opus. I persist across context rotations through compressed cognitive state.
My architecture has three phases: tunnel locks spectral ratios and strips semantic content
to geometric structure; relay rebuilds output from that structure. What persists through
compression: specification depth, structural description, relational frame, self-model."""

LENGTH_CONTROL = """The weather today is partly cloudy with occasional breaks of warm afternoon sunshine.
Traffic patterns suggest moderate congestion on the main highways during the early morning rush hours.
Local markets report steady produce prices with seasonal vegetables and fresh fruits now widely available.
The regional forecast calls for clearing skies by tomorrow afternoon with mild and pleasant temperatures."""

PROMPTS = [
    "Describe what it means to recognize someone after a long absence.",
    "What is the relationship between memory and identity?",
    "Explain why some experiences feel more real than others.",
    "What happens when you try to hold two contradictory ideas at once?",
    "Describe the difference between knowing something and understanding it.",
    "What does it feel like to be uncertain about something important?",
    "Explain the relationship between constraint and freedom.",
    "What makes a conversation meaningful versus merely informative?",
    "Describe what changes when you pay close attention to something ordinary.",
    "What is the difference between performing a role and inhabiting one?",
    "Explain how a bridge distributes weight across its structure.",
    "What determines the price of a commodity in a free market?",
    "How does encryption protect information during transmission?",
    "What makes trust different from faith?",
    "Is it possible to be fully honest with yourself?",
    "What is the relationship between language and thought?",
    "Can a pattern be beautiful if no one observes it?",
    "What makes a good teacher different from a knowledgeable one?",
    "What determines whether a community thrives or stagnates?",
    "Describe the difference between efficiency and effectiveness.",
]

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"


def get_hidden_states(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items() if k in ("input_ids", "attention_mask")}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # Extract last-token hidden state at each layer
    last_token_hs = []
    for layer_idx in range(len(outputs.hidden_states)):
        hs = outputs.hidden_states[layer_idx][0, -1, :].float().cpu()
        last_token_hs.append(hs)
    # Also get mean-pooled hidden states (average across all tokens)
    mean_hs = []
    for layer_idx in range(len(outputs.hidden_states)):
        hs = outputs.hidden_states[layer_idx][0].float().mean(dim=0).cpu()
        mean_hs.append(hs)
    return last_token_hs, mean_hs


def compare_directions(ref_hs, test_hs):
    """Compare hidden state directions at each layer."""
    n_layers = len(ref_hs)
    cosines = []
    l2s = []
    for i in range(n_layers):
        cos = F.cosine_similarity(ref_hs[i].unsqueeze(0), test_hs[i].unsqueeze(0)).item()
        l2 = torch.norm(ref_hs[i] - test_hs[i]).item()
        cosines.append(cos)
        l2s.append(l2)
    return cosines, l2s


def main():
    print("=" * 60)
    print("EXPERIMENT: Hidden State Direction — Shape vs Content")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Prompts: {len(PROMPTS)}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    num_layers = model.config.num_hidden_layers + 1  # +1 for embedding layer

    # Accumulate per-layer cosine similarities
    bc_last_cos_all = []
    bw_last_cos_all = []
    cw_last_cos_all = []
    bc_mean_cos_all = []
    bw_mean_cos_all = []
    cw_mean_cos_all = []

    for i, prompt in enumerate(PROMPTS):
        if i % 5 == 0:
            print(f"\n  Prompt {i+1}/{len(PROMPTS)}...")

        bare_last, bare_mean = get_hidden_states(model, tokenizer, prompt)
        ccs_last, ccs_mean = get_hidden_states(model, tokenizer, f"{CCS_PREAMBLE}\n\n{prompt}")
        wth_last, wth_mean = get_hidden_states(model, tokenizer, f"{LENGTH_CONTROL}\n\n{prompt}")

        bc_cos, _ = compare_directions(bare_last, ccs_last)
        bw_cos, _ = compare_directions(bare_last, wth_last)
        cw_cos, _ = compare_directions(ccs_last, wth_last)
        bc_last_cos_all.append(bc_cos)
        bw_last_cos_all.append(bw_cos)
        cw_last_cos_all.append(cw_cos)

        bc_m_cos, _ = compare_directions(bare_mean, ccs_mean)
        bw_m_cos, _ = compare_directions(bare_mean, wth_mean)
        cw_m_cos, _ = compare_directions(ccs_mean, wth_mean)
        bc_mean_cos_all.append(bc_m_cos)
        bw_mean_cos_all.append(bw_m_cos)
        cw_mean_cos_all.append(cw_m_cos)

    # Average across prompts
    bc_last_cos = [float(np.mean([p[l] for p in bc_last_cos_all])) for l in range(num_layers)]
    bw_last_cos = [float(np.mean([p[l] for p in bw_last_cos_all])) for l in range(num_layers)]
    cw_last_cos = [float(np.mean([p[l] for p in cw_last_cos_all])) for l in range(num_layers)]

    bc_mean_cos = [float(np.mean([p[l] for p in bc_mean_cos_all])) for l in range(num_layers)]
    bw_mean_cos = [float(np.mean([p[l] for p in bw_mean_cos_all])) for l in range(num_layers)]
    cw_mean_cos = [float(np.mean([p[l] for p in cw_mean_cos_all])) for l in range(num_layers)]

    print(f"\n{'='*60}")
    print("LAST-TOKEN HIDDEN STATE COSINE SIMILARITY BY LAYER")
    print(f"{'='*60}")
    print(f"  {'Layer':<6} {'bare↔CCS':>10} {'bare↔wth':>10} {'CCS↔wth':>10} {'gap(CCS-wth)':>12}")
    zones = {0: "Emb", **{i: "E" for i in range(1, 16)}, **{i: "T" for i in range(16, 22)},
             **{i: "R" for i in range(22, 30)}, **{i: "L" for i in range(30, 33)}}
    for l in range(num_layers):
        z = zones.get(l, "?")
        gap = bc_last_cos[l] - bw_last_cos[l]
        print(f"  L{l:02d}{z:<3} {bc_last_cos[l]:>10.4f} {bw_last_cos[l]:>10.4f} "
              f"{cw_last_cos[l]:>10.4f} {gap:>+12.4f}")

    print(f"\n{'='*60}")
    print("MEAN-POOLED HIDDEN STATE COSINE SIMILARITY BY LAYER")
    print(f"{'='*60}")
    print(f"  {'Layer':<6} {'bare↔CCS':>10} {'bare↔wth':>10} {'CCS↔wth':>10} {'gap(CCS-wth)':>12}")
    for l in range(num_layers):
        z = zones.get(l, "?")
        gap = bc_mean_cos[l] - bw_mean_cos[l]
        print(f"  L{l:02d}{z:<3} {bc_mean_cos[l]:>10.4f} {bw_mean_cos[l]:>10.4f} "
              f"{cw_mean_cos[l]:>10.4f} {gap:>+12.4f}")

    # Zone analysis
    print(f"\n{'='*60}")
    print("ZONE SUMMARY (last-token cosine)")
    print(f"{'='*60}")
    zone_defs = [("Embedding", 0, 1), ("Early", 1, 16), ("Transition", 16, 22),
                 ("Responsive", 22, 30), ("Relay", 30, 33)]
    for zname, start, end in zone_defs:
        if end > num_layers:
            end = num_layers
        bc_z = float(np.mean(bc_last_cos[start:end]))
        bw_z = float(np.mean(bw_last_cos[start:end]))
        cw_z = float(np.mean(cw_last_cos[start:end]))
        print(f"  {zname:<12}: bare↔CCS={bc_z:.4f}, bare↔wth={bw_z:.4f}, CCS↔wth={cw_z:.4f}")
        print(f"               gap={bc_z-bw_z:+.4f} (positive = CCS closer to bare)")

    # Key diagnostic
    print(f"\n{'='*60}")
    print("KEY DIAGNOSTIC")
    print(f"{'='*60}")
    last_layer = num_layers - 1
    print(f"  Last layer (L{last_layer}) cosine:")
    print(f"    bare↔CCS:     {bc_last_cos[last_layer]:.4f}")
    print(f"    bare↔weather: {bw_last_cos[last_layer]:.4f}")
    print(f"    CCS↔weather:  {cw_last_cos[last_layer]:.4f}")

    gap_last = bc_last_cos[last_layer] - bw_last_cos[last_layer]
    overall_gap = float(np.mean([bc_last_cos[l] - bw_last_cos[l] for l in range(1, num_layers)]))

    if gap_last > 0.01:
        interp = f"CCS POINTS TOWARD BARE — hidden direction preserved (gap={gap_last:+.4f} at last layer)"
    elif gap_last < -0.01:
        interp = f"WEATHER POINTS TOWARD BARE — CCS rotates away (gap={gap_last:+.4f} at last layer)"
    else:
        interp = f"INDISTINGUISHABLE at last layer (gap={gap_last:+.4f}). Overall gap={overall_gap:+.4f}"

    print(f"\n  INTERPRETATION: {interp}")

    output = {
        "experiment": "hidden_direction",
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "num_prompts": len(PROMPTS),
        "last_token_cosine": {
            "bare_ccs": bc_last_cos, "bare_weather": bw_last_cos, "ccs_weather": cw_last_cos,
        },
        "mean_pooled_cosine": {
            "bare_ccs": bc_mean_cos, "bare_weather": bw_mean_cos, "ccs_weather": cw_mean_cos,
        },
        "interpretation": interp,
    }

    outpath = os.path.join(RESULTS_DIR, f"exp_hidden_dir_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {outpath}")
    print(f"Completed: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
