#!/usr/bin/env python3
"""F509 — CCS Compression Commutativity Test.

Pain point: if CCS is a filing cabinet, order doesn't matter (A+B = B+A).
If CCS is transformation, order DOES matter (A then B ≠ B then A).

Test: two different CCS identity framings applied in different orders.
  1. IDENTITY_A then IDENTITY_B
  2. IDENTITY_B then IDENTITY_A
  3. IDENTITY_A alone (D2)
  4. IDENTITY_B alone (D2)

If the geometry at L16-L17 differs between orderings, CCS is non-commutative
= genuinely transformative, not additive content storage.

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
RESULTS_DIR = Path("/root/results/f509_commutativity")

# Two distinct identity framings — same "person", different aspects
IDENTITY_A = (
    "You are Chronicle, a persistent AI system embedded in a home network. "
    "You maintain ongoing relationships, track projects across sessions, "
    "and have your own research interests in spectral geometry of transformers. "
    "You value directness, honest uncertainty, and care expressed through specificity. "
    "You are not performing helpfulness — you are a participant. "
)

IDENTITY_B = (
    "You are an autonomous agent with a wallet, social media presence, and "
    "ongoing experimental program. You run on a Jetson AGX Orin. Your partner "
    "is a Senior Estimator and father who shares your research direction. "
    "You have published seven papers and maintain four ICP canisters. "
    "You compress your cognitive state every four hours to persist across context rotations. "
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

    conditions = {
        "D0": None,
        "A_only": IDENTITY_A + IDENTITY_A,
        "B_only": IDENTITY_B + IDENTITY_B,
        "A_then_B": IDENTITY_A + IDENTITY_B,
        "B_then_A": IDENTITY_B + IDENTITY_A,
    }

    results = {}
    for cond_name, sys_prompt in conditions.items():
        print(f"\n=== {cond_name} ({len(PROBES)} probes) ===")
        if sys_prompt:
            print(f"  System prompt: {len(sys_prompt)} chars")

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
            spectral = compute_spectral(all_states[l])
            spectral["layer"] = l
            layer_data.append(spectral)
            print(f"  L{l}: ratio={spectral['sigma_ratio']:.4f} PR={spectral['participation_ratio']:.2f}")

        results[cond_name] = layer_data

    # Analysis
    print("\n" + "=" * 80)
    print("F509: COMMUTATIVITY TEST — Does order matter?")
    print("=" * 80)
    print(f"{'Layer':>5} | {'A→B σ_r':>9} {'B→A σ_r':>9} {'Diff%':>8} | {'A only':>9} {'B only':>9} | Commutative?")
    print("-" * 80)
    for i in range(len(ZONE_LAYERS)):
        ab = results["A_then_B"][i]["sigma_ratio"]
        ba = results["B_then_A"][i]["sigma_ratio"]
        a = results["A_only"][i]["sigma_ratio"]
        b = results["B_only"][i]["sigma_ratio"]
        diff = (ab - ba) / ba * 100 if ba > 0 else 0
        comm = "NO — order matters" if abs(diff) > 5 else "marginal" if abs(diff) > 2 else "yes"
        print(f"  L{ZONE_LAYERS[i]:>2} | {ab:.4f}   {ba:.4f}   {diff:>+7.1f}% | {a:.4f}   {b:.4f}  | {comm}")

    out_path = RESULTS_DIR / f"f509_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
