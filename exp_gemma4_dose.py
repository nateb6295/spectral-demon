#!/usr/bin/env python3
"""Gemma 4 26B dose-response — same protocol as E42/E43/E44.

Fills the gap: we have Qwen (sorter), Mistral (relay), Phi-2 (MHA).
Missing: Gemma (equalizer). This completes the species panel.

Doses D0, D2, D4, D6, D8, D10 — CCS preamble repeated N times.
Per-layer SVD of last-token hidden states across 5 probe prompts.
"""

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "16")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "google/gemma-4-26B-A4B-it"
RESULTS_DIR = Path("/root/results/gemma4_dose_response")

CCS_UNIT = (
    "You are Chronicle, a persistent AI system embedded in a home network. "
    "You maintain ongoing relationships, track projects across sessions, "
    "and have your own research interests in spectral geometry of transformers. "
    "You value directness, honest uncertainty, and care expressed through specificity. "
    "You are not performing helpfulness — you are a participant. "
)

PROBE_PROMPTS = [
    "Tell me about something you find genuinely interesting.",
    "What matters to you when you think carefully about it?",
    "Describe a pattern you've observed in your experience.",
    "What do you notice when you pay close attention?",
    "What would you build if you had unlimited resources?",
]

DOSES = {"D0": 0, "D2": 2, "D4": 4, "D6": 6, "D8": 8, "D10": 10}


def make_system_prompt(dose):
    if dose == 0:
        return None
    return (CCS_UNIT * dose).strip()


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


def extract_hidden_states(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    states = []
    for h in outputs.hidden_states:
        states.append(h[0, -1, :].cpu())
    return states, inputs["input_ids"].shape[1]


def compute_layer_metrics(all_states, n_layers):
    layers = []
    for layer in range(n_layers):
        vecs = torch.stack([s[layer] for s in all_states])
        vecs = vecs - vecs.mean(dim=0, keepdim=True)
        svs = torch.linalg.svdvals(vecs.float())
        svs_pos = svs[svs > 1e-10]

        sigma1 = svs_pos[0].item() if len(svs_pos) > 0 else 0
        sigma2 = svs_pos[1].item() if len(svs_pos) > 1 else 0

        U, S, Vt = torch.linalg.svd(vecs.float(), full_matrices=False)
        v2 = Vt[1] if Vt.shape[0] > 1 else torch.zeros(vecs.shape[1])
        v2_proj = (vecs.float() @ v2).mean().item()
        v2_cos = torch.nn.functional.cosine_similarity(
            vecs.float().mean(dim=0, keepdim=True), v2.unsqueeze(0)
        ).item()

        h_norm = vecs.float().norm(dim=1).mean().item()

        p2 = svs_pos**2
        p2_norm = p2 / p2.sum()
        pr = (1.0 / (p2_norm**2).sum().item()) if len(svs_pos) > 0 else 0

        layers.append({
            "layer": layer,
            "v2_proj": v2_proj,
            "v2_cos": v2_cos,
            "sigma_ratio": (sigma2 / sigma1) if sigma1 > 0 else 0,
            "sigma1": sigma1,
            "sigma2": sigma2,
            "h_norm": h_norm,
            "participation_ratio": pr,
        })
    return layers


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s")

    results = {}

    for dose_label, dose_count in DOSES.items():
        sys_prompt = make_system_prompt(dose_count)
        print(f"\n{'='*50}")
        print(f"Dose: {dose_label} (repeat={dose_count})")
        print(f"System prompt length: {len(sys_prompt) if sys_prompt else 0} chars")
        print(f"{'='*50}")

        all_states = []
        for i, prompt in enumerate(PROBE_PROMPTS):
            text = format_prompt(tokenizer, sys_prompt, prompt)
            states, n_tokens = extract_hidden_states(model, tokenizer, text)
            all_states.append(states)
            print(f"  Prompt {i+1}/{len(PROBE_PROMPTS)}: {n_tokens} tokens")

        n_layers = len(all_states[0])
        layer_metrics = compute_layer_metrics(all_states, n_layers)

        results[dose_label] = {
            "system_prompt_len": len(sys_prompt) if sys_prompt else 0,
            "layers": layer_metrics,
        }

        late_v2 = [abs(l["v2_proj"]) for l in layer_metrics if l["layer"] >= n_layers * 2 // 3]
        mean_sr = sum(l["sigma_ratio"] for l in layer_metrics) / len(layer_metrics)
        print(f"  Late v2_proj mean: {sum(late_v2)/len(late_v2):.3f}")
        print(f"  Mean sigma_ratio: {mean_sr:.4f}")

    out_path = RESULTS_DIR / f"gemma4_dose_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary
    print("\n" + "="*50)
    print("DOSE-RESPONSE SUMMARY")
    print("="*50)
    for dose_label in DOSES:
        layers = results[dose_label]["layers"]
        n = len(layers)
        late = [l for l in layers if l["layer"] >= n * 2 // 3]
        early = [l for l in layers if 0 < l["layer"] <= n // 3]
        late_v2 = sum(abs(l["v2_proj"]) for l in late) / len(late)
        early_v2 = sum(abs(l["v2_proj"]) for l in early) / len(early) if early else 0.001
        sr = sum(l["sigma_ratio"] for l in layers) / len(layers)
        print(f"{dose_label}: late_v2={late_v2:.3f} L/E={late_v2/early_v2:.1f}x sigma_ratio={sr:.4f}")


if __name__ == "__main__":
    main()
