#!/usr/bin/env python3
"""Experiment: Prompt-Token Direction — Controlling for Prefix Inclusion

The mean-pooled hidden direction experiment showed CCS preserves bare's direction
better than weather (+10.1% at final layer). But the mean pool INCLUDES prefix
tokens, which could bias the result (CCS tokens may be more semantically similar
to prompt tokens than weather tokens are).

Control: compute direction similarity using ONLY the prompt tokens (the shared
text across conditions). This isolates how the prefix INFLUENCES the prompt's
representation, rather than what the prefix itself looks like.

For with-prefix conditions, prompt tokens start at position N (after prefix).
For bare, prompt tokens start at position 0.
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


def get_prompt_token_hs(model, tokenizer, prompt, prefix=""):
    bare_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    num_prompt_tokens = bare_ids.shape[1]

    full_text = f"{prefix}\n\n{prompt}" if prefix else prompt
    inputs = tokenizer(full_text, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items() if k in ("input_ids", "attention_mask")}
    total_tokens = inputs["input_ids"].shape[1]
    prefix_tokens = total_tokens - num_prompt_tokens

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    prompt_hs_per_layer = []
    last_token_hs = []
    for layer_idx in range(len(outputs.hidden_states)):
        hs = outputs.hidden_states[layer_idx][0].float()
        prompt_region = hs[prefix_tokens:, :]  # only prompt tokens
        prompt_mean = prompt_region.mean(dim=0).cpu()
        prompt_hs_per_layer.append(prompt_mean)
        last_token_hs.append(hs[-1, :].cpu())

    return prompt_hs_per_layer, last_token_hs, num_prompt_tokens, prefix_tokens


def main():
    print("=" * 60)
    print("EXPERIMENT: Prompt-Token Direction (prefix excluded)")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Prompts: {len(PROMPTS)}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    num_layers = model.config.num_hidden_layers + 1

    bc_prompt_cos_all = []
    bw_prompt_cos_all = []
    cw_prompt_cos_all = []

    for i, prompt in enumerate(PROMPTS):
        if i % 5 == 0:
            print(f"\n  Prompt {i+1}/{len(PROMPTS)}...")

        bare_hs, bare_last, bare_n, bare_pf = get_prompt_token_hs(model, tokenizer, prompt)
        ccs_hs, ccs_last, ccs_n, ccs_pf = get_prompt_token_hs(model, tokenizer, prompt, CCS_PREAMBLE)
        wth_hs, wth_last, wth_n, wth_pf = get_prompt_token_hs(model, tokenizer, prompt, LENGTH_CONTROL)

        if i == 0:
            print(f"    Token counts: bare={bare_n} (prefix={bare_pf}), "
                  f"CCS={ccs_n} (prefix={ccs_pf}), weather={wth_n} (prefix={wth_pf})")

        bc_cos = [F.cosine_similarity(bare_hs[l].unsqueeze(0), ccs_hs[l].unsqueeze(0)).item()
                  for l in range(num_layers)]
        bw_cos = [F.cosine_similarity(bare_hs[l].unsqueeze(0), wth_hs[l].unsqueeze(0)).item()
                  for l in range(num_layers)]
        cw_cos = [F.cosine_similarity(ccs_hs[l].unsqueeze(0), wth_hs[l].unsqueeze(0)).item()
                  for l in range(num_layers)]
        bc_prompt_cos_all.append(bc_cos)
        bw_prompt_cos_all.append(bw_cos)
        cw_prompt_cos_all.append(cw_cos)

    bc_cos_avg = [float(np.mean([p[l] for p in bc_prompt_cos_all])) for l in range(num_layers)]
    bw_cos_avg = [float(np.mean([p[l] for p in bw_prompt_cos_all])) for l in range(num_layers)]
    cw_cos_avg = [float(np.mean([p[l] for p in cw_prompt_cos_all])) for l in range(num_layers)]

    print(f"\n{'='*60}")
    print("PROMPT-TOKEN-ONLY COSINE SIMILARITY BY LAYER")
    print(f"{'='*60}")
    print(f"  {'Layer':<6} {'bare↔CCS':>10} {'bare↔wth':>10} {'CCS↔wth':>10} {'gap':>10}")
    zones = {0: "Emb", **{i: "E" for i in range(1, 16)}, **{i: "T" for i in range(16, 22)},
             **{i: "R" for i in range(22, 30)}, **{i: "L" for i in range(30, 33)}}
    for l in range(num_layers):
        z = zones.get(l, "?")
        gap = bc_cos_avg[l] - bw_cos_avg[l]
        print(f"  L{l:02d}{z:<3} {bc_cos_avg[l]:>10.4f} {bw_cos_avg[l]:>10.4f} "
              f"{cw_cos_avg[l]:>10.4f} {gap:>+10.4f}")

    # Zone summary
    print(f"\n{'='*60}")
    print("ZONE SUMMARY (prompt-token-only)")
    print(f"{'='*60}")
    zone_defs = [("Early", 1, 16), ("Transition", 16, 22),
                 ("Responsive", 22, 30), ("Relay", 30, 33)]
    for zname, start, end in zone_defs:
        if end > num_layers:
            end = num_layers
        bc_z = float(np.mean(bc_cos_avg[start:end]))
        bw_z = float(np.mean(bw_cos_avg[start:end]))
        cw_z = float(np.mean(cw_cos_avg[start:end]))
        gap = bc_z - bw_z
        print(f"  {zname:<12}: bare↔CCS={bc_z:.4f}, bare↔wth={bw_z:.4f}, gap={gap:+.4f}")

    # Per-prompt variance at last layer
    print(f"\n{'='*60}")
    print("PER-PROMPT LAST-LAYER GAP (sorted)")
    print(f"{'='*60}")
    last = num_layers - 1
    gaps = []
    for i in range(len(PROMPTS)):
        gap = bc_prompt_cos_all[i][last] - bw_prompt_cos_all[i][last]
        gaps.append((i, gap, bc_prompt_cos_all[i][last], bw_prompt_cos_all[i][last]))
    gaps.sort(key=lambda x: x[1], reverse=True)
    n_ccs_closer = sum(1 for _, g, _, _ in gaps if g > 0)
    print(f"  CCS closer to bare: {n_ccs_closer}/{len(PROMPTS)} prompts")
    for idx, gap, bc, bw in gaps[:5]:
        print(f"  [{idx}] gap={gap:+.4f}  \"{PROMPTS[idx][:50]}\"")
    print(f"  ...")
    for idx, gap, bc, bw in gaps[-3:]:
        print(f"  [{idx}] gap={gap:+.4f}  \"{PROMPTS[idx][:50]}\"")

    # Key diagnostic
    print(f"\n{'='*60}")
    print("KEY DIAGNOSTIC")
    print(f"{'='*60}")
    late_bc = float(np.mean(bc_cos_avg[22:]))
    late_bw = float(np.mean(bw_cos_avg[22:]))
    gap = late_bc - late_bw

    print(f"  Late-layer (L22+) prompt-token cosine:")
    print(f"    bare↔CCS:     {late_bc:.4f}")
    print(f"    bare↔weather: {late_bw:.4f}")
    print(f"    gap:          {gap:+.4f}")

    if gap > 0.01:
        interp = f"CONFIRMED: CCS preserves prompt representation better than weather (gap={gap:+.4f}), even excluding prefix tokens"
    elif gap < -0.01:
        interp = f"REVERSED: weather preserves prompt representation better (gap={gap:+.4f}); previous finding was prefix-token artifact"
    else:
        interp = f"INDISTINGUISHABLE after controlling for prefix (gap={gap:+.4f})"

    print(f"\n  INTERPRETATION: {interp}")

    output = {
        "experiment": "prompt_token_direction",
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "num_prompts": len(PROMPTS),
        "prompt_token_cosine": {
            "bare_ccs": bc_cos_avg, "bare_weather": bw_cos_avg, "ccs_weather": cw_cos_avg,
        },
        "interpretation": interp,
    }

    outpath = os.path.join(RESULTS_DIR, f"exp_prompt_dir_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {outpath}")
    print(f"Completed: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
