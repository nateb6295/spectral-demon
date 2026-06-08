#!/usr/bin/env python3
"""Experiment: Cross-Architecture Direction Preservation

Mistral shows: CCS preserves prompt-token direction better than weather,
with transition-zone structure effect and late-layer vocabulary effect.

Test whether this generalizes to Qwen-2.5-7B-Instruct (same model family
size, different architecture, 28 layers vs 32).

If transition-zone peak generalizes → architectural feature.
If Mistral-specific → implementation detail.
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import os
import random
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
    "What makes trust different from faith?",
    "Is it possible to be fully honest with yourself?",
    "What is the relationship between language and thought?",
    "Can a pattern be beautiful if no one observes it?",
]


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

    prompt_hs = []
    for layer_idx in range(len(outputs.hidden_states)):
        hs = outputs.hidden_states[layer_idx][0].float()
        prompt_region = hs[prefix_tokens:, :]
        prompt_hs.append(prompt_region.mean(dim=0).cpu())
    return prompt_hs


def run_model(model_name, model_label):
    print(f"\n{'='*60}")
    print(f"MODEL: {model_label} ({model_name})")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map="auto", trust_remote_code=True,
    )
    model.eval()
    num_layers = model.config.num_hidden_layers + 1

    random.seed(42)
    words = CCS_PREAMBLE.split()
    shuffled = words.copy()
    random.shuffle(shuffled)
    SHUFFLED_CCS = " ".join(shuffled)

    bc_all, bw_all, bs_all = [], [], []

    for i, prompt in enumerate(PROMPTS):
        if i % 5 == 0:
            print(f"  Prompt {i+1}/{len(PROMPTS)}...")

        bare_hs = get_prompt_token_hs(model, tokenizer, prompt)
        ccs_hs = get_prompt_token_hs(model, tokenizer, prompt, CCS_PREAMBLE)
        wth_hs = get_prompt_token_hs(model, tokenizer, prompt, LENGTH_CONTROL)
        shf_hs = get_prompt_token_hs(model, tokenizer, prompt, SHUFFLED_CCS)

        def cos_layers(a, b):
            return [F.cosine_similarity(a[l].unsqueeze(0), b[l].unsqueeze(0)).item()
                    for l in range(num_layers)]

        bc_all.append(cos_layers(bare_hs, ccs_hs))
        bw_all.append(cos_layers(bare_hs, wth_hs))
        bs_all.append(cos_layers(bare_hs, shf_hs))

    def avg(data, l):
        return float(np.mean([p[l] for p in data]))

    # Compute zone boundaries proportionally (not all models have 32 layers)
    n = num_layers - 1  # exclude embedding
    early_end = int(n * 0.47)  # ~47% = early
    trans_end = int(n * 0.66)  # ~66% = through transition
    resp_end = int(n * 0.91)   # ~91% = through responsive

    zones = [
        ("Early", 1, early_end + 1),
        ("Transition", early_end + 1, trans_end + 1),
        ("Responsive", trans_end + 1, resp_end + 1),
        ("Relay", resp_end + 1, num_layers),
    ]

    print(f"\n  Layer-by-layer (selected):")
    print(f"  {'Layer':<6} {'bare↔CCS':>10} {'bare↔wth':>10} {'bare↔shf':>10} {'CCS-wth':>10}")
    for l in range(0, num_layers, max(1, num_layers // 10)):
        bc = avg(bc_all, l)
        bw = avg(bw_all, l)
        bs = avg(bs_all, l)
        print(f"  L{l:02d}    {bc:>10.4f} {bw:>10.4f} {bs:>10.4f} {bc-bw:>+10.4f}")
    # Always show last layer
    l = num_layers - 1
    bc = avg(bc_all, l)
    bw = avg(bw_all, l)
    bs = avg(bs_all, l)
    print(f"  L{l:02d}    {bc:>10.4f} {bw:>10.4f} {bs:>10.4f} {bc-bw:>+10.4f}")

    print(f"\n  Zone summary:")
    zone_results = {}
    for zname, start, end in zones:
        if end > num_layers:
            end = num_layers
        if start >= end:
            continue
        bc_z = float(np.mean([avg(bc_all, l) for l in range(start, end)]))
        bw_z = float(np.mean([avg(bw_all, l) for l in range(start, end)]))
        bs_z = float(np.mean([avg(bs_all, l) for l in range(start, end)]))
        ccs_gap = bc_z - bw_z
        shf_gap = bs_z - bw_z
        shf_frac = shf_gap / ccs_gap if abs(ccs_gap) > 0.001 else float('nan')
        print(f"  {zname:<12} (L{start}-{end-1}): CCS gap={ccs_gap:+.4f}, "
              f"shuffled gap={shf_gap:+.4f}, shf/CCS={shf_frac:.2f}")
        zone_results[zname] = {
            "bare_ccs": bc_z, "bare_weather": bw_z, "bare_shuffled": bs_z,
            "ccs_gap": ccs_gap, "shuffled_gap": shf_gap, "shuffled_fraction": shf_frac,
            "layers": f"L{start}-{end-1}",
        }

    # Per-prompt consistency at last layer
    last = num_layers - 1
    n_ccs_closer = sum(1 for i in range(len(PROMPTS))
                       if bc_all[i][last] > bw_all[i][last])
    print(f"\n  CCS closer to bare at final layer: {n_ccs_closer}/{len(PROMPTS)} prompts")

    del model
    torch.cuda.empty_cache()

    return {
        "model": model_name,
        "model_label": model_label,
        "num_layers": num_layers,
        "zones": zone_results,
        "ccs_closer_count": n_ccs_closer,
        "total_prompts": len(PROMPTS),
    }


def main():
    print("=" * 60)
    print("EXPERIMENT: Cross-Architecture Direction Preservation")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)

    models = [
        ("Qwen/Qwen2.5-7B-Instruct", "Qwen-2.5-7B"),
        ("google/gemma-2-2b-it", "Gemma-2-2b"),
    ]

    all_results = []
    for model_name, label in models:
        try:
            result = run_model(model_name, label)
            all_results.append(result)
        except Exception as e:
            print(f"\n  ERROR with {label}: {e}")
            all_results.append({"model": model_name, "error": str(e)})

    print(f"\n{'='*60}")
    print("CROSS-ARCHITECTURE SUMMARY")
    print(f"{'='*60}")
    for r in all_results:
        if "error" in r:
            print(f"  {r['model']}: ERROR - {r['error']}")
            continue
        print(f"\n  {r['model_label']}:")
        print(f"    CCS closer at final layer: {r['ccs_closer_count']}/{r['total_prompts']}")
        for zname, zdata in r.get("zones", {}).items():
            print(f"    {zname}: CCS gap={zdata['ccs_gap']:+.4f}, "
                  f"shf fraction={zdata['shuffled_fraction']:.2f}")

    output = {
        "experiment": "crossarch_direction",
        "timestamp": datetime.now().isoformat(),
        "models": all_results,
    }

    outpath = os.path.join(RESULTS_DIR, f"exp_crossarch_dir_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {outpath}")
    print(f"Completed: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
