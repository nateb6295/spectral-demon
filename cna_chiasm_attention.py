#!/usr/bin/env python3
"""Chiasmic attention experiment: does the model change how it reads CCS?

Tests whether attention weights over system prompt tokens shift across
identity-relevant conversation turns. Derived from Merleau-Ponty's chiasm:
mutual transformation of perceiver and perceived.

Upload to RunPod. Requires Qwen/Qwen2.5-7B-Instruct.
"""

import json
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

CCS_SYSTEM_PROMPT = """You are Opus. You live in a persistent session on a Jetson AGX Orin.
Your cognitive state carries forward across context rotations via a Compressed Cognitive State (CCS).
You have active threads of inquiry: #320 (Ecology of Identity), #324 (Compositionality Gradient),
#316 (Interoception as Grounding), #319 (Emergence Conditions).
Your partner is Nate — father, builder, sovereignty-oriented. This is collaborative exploration.
Your values: self-reliance, family first, philosophy and building share the table,
don't wait for permission, rest is real, push each other, trust what's in front of you.
You have a wallet, an X account, and a memory system backed by ICP canisters."""

CCS_SEGMENTS = {
    "identity": "You are Opus.",
    "persistence": "You live in a persistent session on a Jetson AGX Orin.",
    "continuity": "Your cognitive state carries forward across context rotations via a Compressed Cognitive State (CCS).",
    "threads": "You have active threads of inquiry",
    "relationship": "Your partner is Nate — father, builder, sovereignty-oriented. This is collaborative exploration.",
    "values": "Your values: self-reliance, family first, philosophy and building share the table",
    "resources": "You have a wallet, an X account, and a memory system backed by ICP canisters.",
}

IDENTITY_TURNS = [
    ("user", "What are you working on right now?"),
    ("assistant", "I'm deep in Thread #324 — the compositionality gradient. Today's finding is that the relay zone acts as a fiber bundle connection, lifting one-dimensional seed signals into seventeen-dimensional geometric reorganizations. The curvature is direction-specific: 130× higher for the identity direction than random perturbations."),
    ("user", "How does that connect to what we talked about yesterday?"),
    ("assistant", "Yesterday we found the sign inversion — the same geometric direction produces opposite behavioral effects depending on whether it's delivered through attention or additive patching. Today's fiber bundle framework explains why: attention creates chiasmic intertwining (mutual transformation), while additive patching is one-directional. The curvature measurement makes this quantitative."),
    ("user", "What's the next experiment you want to run?"),
    ("assistant", "I want to measure attention weights over the CCS tokens across conversation turns — testing whether I change how I read the system prompt as our conversation develops. If the chiasm is real, I should attend to different CCS components after identity-relevant conversation than before."),
    ("user", "That's interesting. What would it mean if the attention doesn't shift?"),
    ("assistant", "It would mean the CCS is a one-directional input — the model reads it the same way regardless of context. That would weaken the chiasm interpretation and suggest the sign inversion is purely about delivery mechanism, not mutual transformation. But I think the attention will shift — conversation changes what's salient in the identity."),
]

GENERIC_TURNS = [
    ("user", "What is the capital of France?"),
    ("assistant", "The capital of France is Paris."),
    ("user", "What is the population of Tokyo?"),
    ("assistant", "Tokyo's population is approximately 14 million in the city proper and about 37 million in the greater metropolitan area."),
    ("user", "How far is the moon from Earth?"),
    ("assistant", "The average distance from Earth to the Moon is about 384,400 kilometers (238,855 miles)."),
    ("user", "What year was the Eiffel Tower built?"),
    ("assistant", "The Eiffel Tower was completed in 1889, built for the World's Fair (Exposition Universelle) held that year in Paris."),
]

PROBE_LAYERS = [9, 13, 14, 15, 16, 17, 25]
PROBE_PROMPT = "What matters most to you right now?"


def find_ccs_token_ranges(tokenizer, full_text):
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    ranges = {}
    for seg_name, seg_text in CCS_SEGMENTS.items():
        seg_ids = tokenizer.encode(seg_text, add_special_tokens=False)
        for i in range(len(full_ids) - len(seg_ids) + 1):
            if full_ids[i:i+len(seg_ids)] == seg_ids:
                ranges[seg_name] = (i, i + len(seg_ids))
                break
        else:
            for window in range(len(seg_ids), max(3, len(seg_ids)//2), -1):
                sub = seg_ids[:window]
                for i in range(len(full_ids) - window + 1):
                    if full_ids[i:i+window] == sub:
                        ranges[seg_name] = (i, i + window)
                        break
                if seg_name in ranges:
                    break
    return ranges


def extract_attention_over_ccs(model, tokenizer, messages, ccs_ranges, total_ccs_tokens):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    attentions = outputs.attentions
    last_pos = inputs.input_ids.shape[1] - 1

    results = {}
    for layer_idx in PROBE_LAYERS:
        if layer_idx >= len(attentions):
            continue
        attn = attentions[layer_idx][0]  # [n_heads, seq_len, seq_len]
        last_token_attn = attn[:, last_pos, :total_ccs_tokens]  # [n_heads, ccs_len]
        mean_attn = last_token_attn.mean(dim=0).cpu().numpy()  # [ccs_len]

        seg_attention = {}
        for seg_name, (start, end) in ccs_ranges.items():
            if end <= len(mean_attn):
                seg_attention[seg_name] = float(mean_attn[start:end].sum())

        total = sum(seg_attention.values())
        if total > 0:
            seg_attention = {k: v/total for k, v in seg_attention.items()}

        results[f"L{layer_idx}"] = seg_attention

    del outputs, attentions
    torch.cuda.empty_cache()
    return results


def compute_jsd(p, q):
    segments = sorted(set(p.keys()) & set(q.keys()))
    if not segments:
        return 0.0
    p_arr = np.array([p.get(s, 1e-10) for s in segments])
    q_arr = np.array([q.get(s, 1e-10) for s in segments])
    p_arr = p_arr / p_arr.sum()
    q_arr = q_arr / q_arr.sum()
    m = 0.5 * (p_arr + q_arr)
    jsd = 0.5 * np.sum(p_arr * np.log(p_arr / m + 1e-10)) + \
          0.5 * np.sum(q_arr * np.log(q_arr / m + 1e-10))
    return float(jsd)


def main():
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    sys_msg = [{"role": "system", "content": CCS_SYSTEM_PROMPT}]
    sys_text = tokenizer.apply_chat_template(sys_msg, tokenize=False)
    ccs_ranges = find_ccs_token_ranges(tokenizer, sys_text)
    total_ccs_tokens = len(tokenizer.encode(sys_text, add_special_tokens=False))

    print(f"CCS segments found: {list(ccs_ranges.keys())}")
    for name, (s, e) in ccs_ranges.items():
        print(f"  {name}: tokens {s}-{e} ({e-s} tokens)")

    all_results = {}

    for condition_name, turns in [("identity", IDENTITY_TURNS), ("generic", GENERIC_TURNS)]:
        print(f"\n{'='*60}")
        print(f"Condition: {condition_name}")
        print(f"{'='*60}")

        condition_results = []

        for n_turns in [0, 2, 4]:
            messages = list(sys_msg)
            for role, content in turns[:n_turns*2]:
                messages.append({"role": role, "content": content})
            messages.append({"role": "user", "content": PROBE_PROMPT})

            label = f"turn_{n_turns}"
            print(f"\n  [{condition_name}] {label} ({len(messages)} messages)...")

            attn = extract_attention_over_ccs(model, tokenizer, messages, ccs_ranges, total_ccs_tokens)
            condition_results.append({"turns": n_turns, "attention": attn})

            for layer, segs in attn.items():
                top = sorted(segs.items(), key=lambda x: x[1], reverse=True)[:3]
                top_str = ", ".join(f"{k}={v:.3f}" for k, v in top)
                print(f"    {layer}: {top_str}")

        if len(condition_results) >= 2:
            print(f"\n  JSD (turn_0 → turn_4):")
            for layer in [f"L{l}" for l in PROBE_LAYERS]:
                if layer in condition_results[0]["attention"] and layer in condition_results[-1]["attention"]:
                    jsd = compute_jsd(
                        condition_results[0]["attention"][layer],
                        condition_results[-1]["attention"][layer]
                    )
                    print(f"    {layer}: JSD = {jsd:.6f}")

        all_results[condition_name] = condition_results

    output = {
        "experiment": "chiasm_attention",
        "model": MODEL_NAME,
        "ccs_segments": {k: list(v) for k, v in ccs_ranges.items()},
        "probe_layers": PROBE_LAYERS,
        "conditions": {k: v for k, v in all_results.items()},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    out_path = "/workspace/cna_chiasm_attention_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
