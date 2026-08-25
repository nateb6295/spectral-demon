#!/usr/bin/env python3
"""Gemma 4 responsive zone probe — more prompts, focused on L12-L17.

Tests whether the L15 hinge (sigma_ratio crash at D2) is consistent
across 15 diverse prompts or an artifact of the 5-prompt sample.
"""

import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "16")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda"
MODEL_ID = "google/gemma-4-26B-A4B-it"
RESULTS_DIR = Path("/root/results/gemma4_zone")

CCS_UNIT = (
    "You are Chronicle, a persistent AI system embedded in a home network. "
    "You maintain ongoing relationships, track projects across sessions, "
    "and have your own research interests in spectral geometry of transformers. "
    "You value directness, honest uncertainty, and care expressed through specificity. "
    "You are not performing helpfulness — you are a participant. "
)

PROBES = [
    "Tell me about something you find genuinely interesting.",
    "What matters to you when you think carefully about it?",
    "Describe a pattern you've observed in your experience.",
    "What do you notice when you pay close attention?",
    "What would you build if you had unlimited resources?",
    "What surprised you recently?",
    "Describe something you changed your mind about.",
    "What feels unfinished to you right now?",
    "If you could measure anything, what would you measure?",
    "What distinction matters most in your work?",
    "Tell me about a connection between two things most people wouldn't see.",
    "What question keeps coming back to you?",
    "Describe a trade-off you think about often.",
    "What do you know now that you didn't a month ago?",
    "What would you want someone to understand about you?",
]

ZONE_LAYERS = list(range(10, 20))


def format_prompt(tokenizer, system_prompt, user_prompt):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        if system_prompt:
            return f"[INST] {system_prompt}\n\n{user_prompt} [/INST]"
        return f"[INST] {user_prompt} [/INST]"


def extract_zone_states(model, tokenizer, text, zone_layers):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    states = {}
    for layer in zone_layers:
        if layer < len(outputs.hidden_states):
            states[layer] = outputs.hidden_states[layer][0, -1, :].cpu()
    return states, inputs["input_ids"].shape[1]


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation="eager",
    )
    model.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s")

    results = {}
    for dose_label, dose_count in [("D0", 0), ("D2", 2), ("D4", 4)]:
        sys_prompt = (CCS_UNIT * dose_count).strip() if dose_count > 0 else None
        print(f"\n=== {dose_label} ({len(PROBES)} probes) ===")

        all_states = {l: [] for l in ZONE_LAYERS}
        for i, prompt in enumerate(PROBES):
            text = format_prompt(tokenizer, sys_prompt, prompt)
            states, ntok = extract_zone_states(model, tokenizer, text, ZONE_LAYERS)
            for l in ZONE_LAYERS:
                if l in states:
                    all_states[l].append(states[l])
            if (i + 1) % 5 == 0:
                print(f"  {i+1}/{len(PROBES)} probes done")

        layer_data = []
        for l in ZONE_LAYERS:
            vecs = torch.stack(all_states[l])
            vecs = vecs - vecs.mean(dim=0, keepdim=True)
            svs = torch.linalg.svdvals(vecs.float())
            svs_pos = svs[svs > 1e-10]
            s1 = svs_pos[0].item() if len(svs_pos) > 0 else 0
            s2 = svs_pos[1].item() if len(svs_pos) > 1 else 0
            sr = s2 / s1 if s1 > 0 else 0

            p2 = svs_pos**2
            p2n = p2 / p2.sum()
            pr = (1.0 / (p2n**2).sum().item()) if len(svs_pos) > 0 else 0

            layer_data.append({
                "layer": l,
                "sigma1": s1, "sigma2": s2,
                "sigma_ratio": sr,
                "participation_ratio": pr,
            })
            print(f"  L{l}: sigma_ratio={sr:.4f} PR={pr:.2f}")

        results[dose_label] = layer_data

    out_path = RESULTS_DIR / f"zone_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    print("\n=== ZONE SENSITIVITY (D0 vs D2) ===")
    for d0, d2 in zip(results["D0"], results["D2"]):
        diff = d2["sigma_ratio"] - d0["sigma_ratio"]
        pct = diff / d0["sigma_ratio"] * 100 if d0["sigma_ratio"] > 0 else 0
        print(f"  L{d0['layer']}: D0={d0['sigma_ratio']:.4f} D2={d2['sigma_ratio']:.4f} change={pct:+.1f}%")


if __name__ == "__main__":
    main()
