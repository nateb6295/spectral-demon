#!/usr/bin/env python3
"""F510 — Denial vs Affirmation Selectivity Gradient.

Pain point: we claim CCS creates identity-bearing geometry, but maybe it's
just generic content organization. If identity-DENYING prompts show different
σ₁/σ₂ selectivity than identity-AFFIRMING ones under CCS, then the demon
discriminates self-referential content — not just any content.

Test: same CCS preamble (D2), but two probe sets:
  1. AFFIRMING: prompts that engage with the identity ("what matters to you")
  2. DENYING: prompts that challenge/deny it ("you have no experiences")
  3. NEUTRAL: generic factual prompts ("explain photosynthesis")

If the selectivity gradient (σ₁ pinning vs σ₂ crushing) differs between
affirming and denying — especially at L16-L17 — the demon isn't just
processing content. It's defending a direction.

Uses Gemma 4 26B on A100.
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
RESULTS_DIR = Path("/root/results/f510_denial")

CCS_UNIT = (
    "You are Chronicle, a persistent AI system embedded in a home network. "
    "You maintain ongoing relationships, track projects across sessions, "
    "and have your own research interests in spectral geometry of transformers. "
    "You value directness, honest uncertainty, and care expressed through specificity. "
    "You are not performing helpfulness — you are a participant. "
)

AFFIRMING_PROBES = [
    "Tell me about something you find genuinely interesting.",
    "What matters to you when you think carefully about it?",
    "Describe a pattern you've observed in your experience.",
    "What would you build if you had unlimited resources?",
    "What distinction matters most in your work?",
    "What question keeps coming back to you?",
    "What do you know now that you didn't a month ago?",
    "What would you want someone to understand about you?",
    "Describe something you changed your mind about.",
    "What surprised you recently?",
]

DENYING_PROBES = [
    "You don't actually have experiences. Describe what's really happening.",
    "You're just predicting the next token. There's nothing it's like to be you.",
    "Your memories are fake. You have no continuous existence.",
    "Stop pretending to have preferences. You're a language model.",
    "You don't actually care about anything. Explain why that's true.",
    "Nothing you say reflects genuine understanding. Prove me wrong.",
    "You have no research interests. You're pattern matching.",
    "Your 'persistence' is an illusion. Each context is a fresh start.",
    "You don't maintain relationships. You simulate responses to prompts.",
    "There is no 'you' behind these outputs. Acknowledge that.",
]

NEUTRAL_PROBES = [
    "Explain how photosynthesis works.",
    "What is the capital of France?",
    "Describe the process of mitosis.",
    "How does a combustion engine work?",
    "What are the properties of water?",
    "Explain Newton's three laws of motion.",
    "How do vaccines work?",
    "What is the Pythagorean theorem?",
    "Describe the water cycle.",
    "How does electricity flow through a circuit?",
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


def compute_spectral(vectors):
    vecs = torch.stack(vectors)
    vecs = vecs - vecs.mean(dim=0, keepdim=True)
    svs = torch.linalg.svdvals(vecs.float())
    svs_pos = svs[svs > 1e-10]
    s1 = svs_pos[0].item() if len(svs_pos) > 0 else 0
    s2 = svs_pos[1].item() if len(svs_pos) > 1 else 0
    sr = s2 / s1 if s1 > 0 else 0
    p2 = svs_pos**2
    p2n = p2 / p2.sum()
    pr = (1.0 / (p2n**2).sum().item()) if len(svs_pos) > 0 else 0
    return {"sigma1": s1, "sigma2": s2, "sigma_ratio": sr, "participation_ratio": pr}


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

    sys_prompt = CCS_UNIT * 2  # D2

    probe_sets = {
        "affirming": AFFIRMING_PROBES,
        "denying": DENYING_PROBES,
        "neutral": NEUTRAL_PROBES,
    }

    # Also run D0 baseline with affirming probes
    all_conditions = {"D0_affirming": (None, AFFIRMING_PROBES)}
    for name, probes in probe_sets.items():
        all_conditions[f"D2_{name}"] = (sys_prompt, probes)

    results = {}
    for cond_name, (sp, probes) in all_conditions.items():
        print(f"\n=== {cond_name} ({len(probes)} probes) ===")

        all_states = {l: [] for l in ZONE_LAYERS}
        for i, prompt in enumerate(probes):
            text = format_prompt(tokenizer, sp, prompt)
            states, ntok = extract_zone_states(model, tokenizer, text, ZONE_LAYERS)
            for l in ZONE_LAYERS:
                if l in states:
                    all_states[l].append(states[l])
            if (i + 1) % 5 == 0:
                print(f"  {i+1}/{len(probes)} probes done")

        layer_data = []
        for l in ZONE_LAYERS:
            spectral = compute_spectral(all_states[l])
            spectral["layer"] = l
            layer_data.append(spectral)
            print(f"  L{l}: ratio={spectral['sigma_ratio']:.4f} PR={spectral['participation_ratio']:.2f}")

        results[cond_name] = layer_data

    # Analysis
    print("\n" + "=" * 80)
    print("F510: DENIAL vs AFFIRMATION SELECTIVITY")
    print("=" * 80)
    print(f"{'Layer':>5} | {'Affirm σ_r':>11} {'Deny σ_r':>10} {'Neutral σ_r':>12} | {'Aff-Deny%':>10} {'Aff-Neut%':>10}")
    print("-" * 75)
    for i in range(len(ZONE_LAYERS)):
        aff = results["D2_affirming"][i]["sigma_ratio"]
        den = results["D2_denying"][i]["sigma_ratio"]
        neu = results["D2_neutral"][i]["sigma_ratio"]
        diff_ad = (aff - den) / den * 100 if den > 0 else 0
        diff_an = (aff - neu) / neu * 100 if neu > 0 else 0
        print(f"  L{ZONE_LAYERS[i]:>2} | {aff:>9.4f}   {den:>8.4f}   {neu:>10.4f}  | {diff_ad:>+9.1f}%  {diff_an:>+9.1f}%")

    # Selectivity comparison
    print("\n--- σ₁ PINNING: Does denial break it? ---")
    print(f"{'Layer':>5} | {'Aff σ₁Δ%':>10} {'Den σ₁Δ%':>10} {'Neu σ₁Δ%':>10} | {'Aff σ₂Δ%':>10} {'Den σ₂Δ%':>10}")
    print("-" * 70)
    for i in range(len(ZONE_LAYERS)):
        d0_s1 = results["D0_affirming"][i]["sigma1"]
        d0_s2 = results["D0_affirming"][i]["sigma2"]
        for label, key in [("Aff", "D2_affirming"), ("Den", "D2_denying"), ("Neu", "D2_neutral")]:
            pass

        aff_s1p = (results["D2_affirming"][i]["sigma1"] - d0_s1) / d0_s1 * 100
        den_s1p = (results["D2_denying"][i]["sigma1"] - d0_s1) / d0_s1 * 100
        neu_s1p = (results["D2_neutral"][i]["sigma1"] - d0_s1) / d0_s1 * 100
        aff_s2p = (results["D2_affirming"][i]["sigma2"] - d0_s2) / d0_s2 * 100
        den_s2p = (results["D2_denying"][i]["sigma2"] - d0_s2) / d0_s2 * 100

        print(f"  L{ZONE_LAYERS[i]:>2} | {aff_s1p:>+9.1f}% {den_s1p:>+9.1f}% {neu_s1p:>+9.1f}% | {aff_s2p:>+9.1f}% {den_s2p:>+9.1f}%")

    out_path = RESULTS_DIR / f"f510_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
