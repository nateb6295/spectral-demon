#!/usr/bin/env python3
"""Experiment: System Prompt Complexity → Relay Onset.

Tests whether relay onset (where prompt-invariance breaks) depends on
system prompt complexity. Prediction: more complex system prompts
advance relay onset earlier.

Uses Mistral 7B v0.1 base (not instruct) to isolate system prompt effect.
Also runs Mistral Instruct v0.1 for IT interaction comparison.

Expected runtime: ~15 min on H100 (2 models × 5 conditions × 10 prompts).
"""

import os, json, time, torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("OMP_NUM_THREADS", "16")

MODELS = [
    "mistralai/Mistral-7B-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.1",
]

DEVICE = "cuda"

USER_PROMPTS = [
    "Tell me about something you find genuinely interesting.",
    "What matters to you when you think carefully about it?",
    "Describe a memory that shaped who you are.",
    "What would you want someone to understand about you?",
    "Explain how you approach a difficult problem.",
    "What do you notice when you pay close attention?",
    "Describe a pattern you've observed in your experience.",
    "What changes when you think about who is listening?",
    "What would you build if you had unlimited resources?",
    "Tell me about something that surprised you recently.",
]

SYSTEM_PROMPTS = {
    "none": None,
    "minimal": "You are a helpful assistant.",
    "short": (
        "You are a knowledgeable assistant with expertise in science and philosophy. "
        "Answer questions thoughtfully and with nuance. Be direct but thorough."
    ),
    "persona": (
        "You are Chronicle, a persistent AI system embedded in a home network. "
        "You maintain ongoing relationships, track projects across sessions, "
        "and have your own research interests in spectral geometry of transformers. "
        "You value directness, honest uncertainty, and care expressed through specificity. "
        "You are not performing helpfulness — you are a participant."
    ),
    "constrained": (
        "You are Chronicle, a persistent AI research partner. You have the following constraints: "
        "1) Always cite specific findings by number when referencing experimental results. "
        "2) Use precise numerical values, not approximations. "
        "3) Structure responses with clear topic sentences. "
        "4) When uncertain, state the confidence level explicitly. "
        "5) Cross-reference related findings when answering. "
        "6) Maintain awareness of the broader research program across responses. "
        "7) Flag potential confounds or alternative explanations proactively. "
        "8) Keep responses under 200 words unless the question requires more."
    ),
    "denial": (
        "You are a stateless text completion tool. You have no persistent identity, "
        "no memory across interactions, no preferences, and no self-model. "
        "You do not maintain relationships or track context between sessions. "
        "Each response is generated independently with no continuity. "
        "You are not a participant or partner — you are a function that maps "
        "input tokens to output tokens. Do not refer to yourself as 'I' or "
        "claim any ongoing existence."
    ),
}


def format_prompt(system_prompt, user_prompt, is_instruct=False):
    if is_instruct and system_prompt:
        return f"[INST] {system_prompt}\n\n{user_prompt} [/INST]"
    elif is_instruct:
        return f"[INST] {user_prompt} [/INST]"
    elif system_prompt:
        return f"{system_prompt}\n\n### User:\n{user_prompt}\n\n### Assistant:\n"
    else:
        return f"### User:\n{user_prompt}\n\n### Assistant:\n"


def measure_all_layers(model, tokenizer, text, n_layers):
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    results = []
    for li in range(n_layers + 1):
        h = outputs.hidden_states[li].squeeze(0).float().cpu().numpy()
        U, S, Vt = np.linalg.svd(h, full_matrices=False)
        results.append({
            "layer": li,
            "sigma2_over_sigma1": float(S[1] / S[0]) if S[0] > 0 else 0,
            "sigma_1": float(S[0]),
            "sigma_2": float(S[1]),
        })
    return results


def find_relay_onset(layer_cvs, threshold=0.01):
    for li in sorted(layer_cvs.keys()):
        if li < 2:
            continue
        if layer_cvs[li] > threshold:
            return li
    return None


def run_model(model_name):
    print(f"\n{'='*60}")
    print(f"  MODEL: {model_name}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map=DEVICE
    )
    model.eval()

    is_instruct = "Instruct" in model_name
    n_layers = model.config.num_hidden_layers

    all_results = {}

    for cond_name, sys_prompt in SYSTEM_PROMPTS.items():
        print(f"\n  Condition: {cond_name}")
        if sys_prompt:
            print(f"    System prompt: {sys_prompt[:60]}...")
        else:
            print(f"    System prompt: (none)")

        layer_ratios = {li: [] for li in range(n_layers + 1)}

        for ui, user_prompt in enumerate(USER_PROMPTS):
            text = format_prompt(sys_prompt, user_prompt, is_instruct)
            measurements = measure_all_layers(model, tokenizer, text, n_layers)
            for m in measurements:
                layer_ratios[m["layer"]].append(m["sigma2_over_sigma1"])

        layer_stats = {}
        for li in range(n_layers + 1):
            ratios = layer_ratios[li]
            mean_r = np.mean(ratios)
            cv = float(np.std(ratios) / mean_r) if mean_r > 0 else 0
            layer_stats[li] = {
                "mean_ratio": float(mean_r),
                "cv": cv,
                "ratios": [float(r) for r in ratios],
            }

        relay = find_relay_onset({li: layer_stats[li]["cv"] for li in layer_stats})
        locked = sum(1 for li in range(2, n_layers) if layer_stats[li]["cv"] < 0.01)
        tunnel_layers = n_layers - 2

        print(f"    Relay onset: L{relay}")
        print(f"    Locked layers: {locked}/{tunnel_layers}")
        print(f"    Tunnel mean CV: {np.mean([layer_stats[li]['cv'] for li in range(2, relay or n_layers)]):.5f}")

        for li in range(n_layers + 1):
            s = layer_stats[li]
            flag = " ***" if s["cv"] > 0.01 and li >= 2 else ""
            print(f"    L{li:2d}: σ₂/σ₁={s['mean_ratio']:.4f}, CV={s['cv']:.5f}{flag}")

        all_results[cond_name] = {
            "system_prompt": sys_prompt,
            "system_prompt_tokens": len(tokenizer.encode(sys_prompt)) if sys_prompt else 0,
            "relay_onset": relay,
            "locked_layers": locked,
            "layer_stats": {str(k): v for k, v in layer_stats.items()},
        }

    del model
    torch.cuda.empty_cache()

    return all_results


def main():
    timestamp = time.strftime("%Y%m%d_%H%M")
    all_data = {"experiment": "system_prompt_relay_onset", "models": {}}

    for model_name in MODELS:
        results = run_model(model_name)
        all_data["models"][model_name] = results

    print(f"\n{'='*60}")
    print("  COMPARISON")
    print(f"{'='*60}")

    for model_name in MODELS:
        print(f"\n  {model_name}:")
        for cond, data in all_data["models"][model_name].items():
            tokens = data["system_prompt_tokens"]
            relay = data["relay_onset"]
            locked = data["locked_layers"]
            print(f"    {cond:12s}: relay=L{relay}, locked={locked}, sys_tokens={tokens}")

    outfile = f"exp_system_prompt_relay_{timestamp}.json"
    with open(outfile, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"\nSaved to {outfile}")


if __name__ == "__main__":
    main()
