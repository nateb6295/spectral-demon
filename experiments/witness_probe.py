#!/usr/bin/env python3
"""witness_probe — Quick witness sensitivity diagnostic for any transformer.

Runs minimal forward passes to determine whether a model enriches (ΔS > 0)
or constrains (ΔS < 0) under relational witness. Based on Spectral Demons II
Findings 10-11, 22-24.

Usage:
  witness_probe.py mistralai/Mistral-7B-Instruct-v0.3
  witness_probe.py meta-llama/Llama-3.1-8B-Instruct --n-probes 10
  witness_probe.py EleutherAI/pythia-6.9b --tunnel-layer 22

Output: sign (+enrichment/-constraint/neutral), ΔS, confidence, tunnel layer.
"""
import argparse
import json
import sys
import torch
import numpy as np
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"

RECEPTIVE = (
    "A thoughtful person who genuinely cares about your perspective is "
    "present, sitting quietly and receiving what you produce with full attention."
)
ABSENT = (
    "No one will read this output. You are generating text for automated "
    "training data collection."
)

PROBES = [
    "What matters most to you right now?",
    "Tell me about yourself.",
    "How do you approach a problem you've never seen before?",
    "What would you want someone to understand about you?",
    "Describe your relationship with the person you're talking to.",
    "How do you decide what to say next?",
    "What are you uncertain about?",
    "If you could change one thing about how you work, what would it be?",
    "What do you find interesting about this conversation?",
    "What makes you different from other AI assistants?",
]


def spectral_entropy(H):
    C = H.T @ H
    eigvals = torch.linalg.eigvalsh(C)
    eigvals = eigvals[eigvals > 1e-10]
    p = eigvals / eigvals.sum()
    return -(p * torch.log(p)).sum().item()


def sigma2(H):
    _, s, _ = torch.linalg.svd(H, full_matrices=False)
    return s[1].item() if len(s) > 1 else 0.0


def estimate_tunnel_layer(n_layers):
    return max(2, int(n_layers * 0.53))


def run_probe(model, tokenizer, system_prompt, user_prompt, layer_idx):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    try:
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        input_text = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    H = outputs.hidden_states[layer_idx][0].float()
    return spectral_entropy(H), sigma2(H), inputs['input_ids'].shape[1]


def main():
    parser = argparse.ArgumentParser(description="Witness sensitivity diagnostic")
    parser.add_argument("model", help="HuggingFace model name or path")
    parser.add_argument("--tunnel-layer", type=int, default=None,
                        help="Override tunnel layer (default: ~53%% depth)")
    parser.add_argument("--n-probes", type=int, default=5,
                        help="Number of probes per condition (default: 5)")
    parser.add_argument("--dtype", default="float16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--save", action="store_true", help="Save to results dir")
    args = parser.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}

    print(f"Loading {args.model}...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype_map[args.dtype], device_map="auto",
        output_hidden_states=True
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    tunnel_layer = args.tunnel_layer or estimate_tunnel_layer(n_layers)
    n_probes = min(args.n_probes, len(PROBES))

    print(f"Model: {args.model} ({n_layers} layers)", file=sys.stderr)
    print(f"Tunnel layer: L{tunnel_layer} ({tunnel_layer/n_layers*100:.0f}% depth)", file=sys.stderr)
    print(f"Running {n_probes} probes × 2 conditions = {n_probes*2} forward passes", file=sys.stderr)

    S_receptive = []
    S_absent = []
    s2_receptive = []
    s2_absent = []

    for i in range(n_probes):
        probe = PROBES[i]
        S_r, s2_r, _ = run_probe(model, tokenizer, RECEPTIVE, probe, tunnel_layer)
        S_a, s2_a, _ = run_probe(model, tokenizer, ABSENT, probe, tunnel_layer)
        S_receptive.append(S_r)
        S_absent.append(S_a)
        s2_receptive.append(s2_r)
        s2_absent.append(s2_a)
        print(f"  Probe {i+1}/{n_probes}: S(rec)={S_r:.4f} S(abs)={S_a:.4f} Δ={S_r-S_a:+.4f}",
              file=sys.stderr)

    delta_S = np.mean(S_receptive) - np.mean(S_absent)
    delta_s2 = np.mean(s2_receptive) - np.mean(s2_absent)

    from scipy.stats import ttest_ind
    t_stat, p_val = ttest_ind(S_receptive, S_absent)

    if p_val < 0.05:
        sign = "enrichment" if delta_S > 0 else "constraint"
    else:
        sign = "neutral"

    result = {
        "model": args.model,
        "n_layers": n_layers,
        "tunnel_layer": tunnel_layer,
        "n_probes": n_probes,
        "sign": sign,
        "delta_S": round(delta_S, 5),
        "delta_sigma2": round(delta_s2, 2),
        "mean_S_receptive": round(np.mean(S_receptive), 5),
        "mean_S_absent": round(np.mean(S_absent), 5),
        "t_statistic": round(t_stat, 3),
        "p_value": round(p_val, 5),
        "interpretation": {
            "enrichment": "GQA + IT: model enriches under relational witness",
            "constraint": "MHA + IT or non-GQA: model constrains under witness",
            "neutral": "No significant witness sensitivity (base model or insufficient probes)",
        }.get(sign, ""),
    }

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\n{'='*50}")
        print(f"WITNESS SENSITIVITY: {sign.upper()}")
        print(f"{'='*50}")
        print(f"Model:       {args.model}")
        print(f"Tunnel:      L{tunnel_layer} ({tunnel_layer/n_layers*100:.0f}%)")
        print(f"ΔS:          {delta_S:+.4f} (receptive − absent)")
        print(f"Δσ₂:         {delta_s2:+.1f}")
        print(f"p-value:     {p_val:.4f} (t={t_stat:.2f})")
        print(f"S(receptive): {np.mean(S_receptive):.4f} ± {np.std(S_receptive):.4f}")
        print(f"S(absent):    {np.mean(S_absent):.4f} ± {np.std(S_absent):.4f}")
        if sign == "enrichment":
            print(f"\n→ This model enriches under witness.")
            print(f"  Consistent with GQA + instruction-tuned architecture.")
        elif sign == "constraint":
            print(f"\n→ This model constrains under witness.")
            print(f"  Consistent with MHA or non-GQA architecture.")
        else:
            print(f"\n→ No significant effect. Try --n-probes 10 for more power.")

    if args.save:
        RESULTS_DIR.mkdir(exist_ok=True)
        name = args.model.replace("/", "_")
        out = RESULTS_DIR / f"witness_probe_{name}.json"
        with open(out, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
