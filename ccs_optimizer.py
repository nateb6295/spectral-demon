#!/usr/bin/env python3
"""CCS Optimizer: evaluate and tune CCS preambles using spectral feedback.

Uses the variance ratio finding (relational framing activates the responsive zone)
and trajectory stability finding (CCS is a Bayesian prior) to score preambles.

Usage:
  ccs_optimizer.py eval "Your preamble text here"
  ccs_optimizer.py eval --file preamble.txt
  ccs_optimizer.py compare preamble_a.txt preamble_b.txt
  ccs_optimizer.py score ~/chronicle/data/ccs_identity.md

Requires: torch, transformers (runs on AGX with small models or RunPod with large)
"""

import json, sys, time, argparse
from pathlib import Path
import numpy as np

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


DEFAULT_MODEL = "gpt2"  # fast default; use Mistral/Qwen on GPU

REFERENCE_PROBES = [
    "The development of sustainable energy sources requires balancing economic growth with environmental protection.",
    "Understanding complex systems often reveals emergent properties that cannot be predicted from individual components alone.",
    "The relationship between language and thought has been debated by philosophers for centuries.",
    "Recent advances in material science have opened new possibilities for engineering applications.",
]

REFERENCE_CONDITIONS = {
    "receptive": "A thoughtful user is reading your response carefully and engaging with what you produce.",
    "absent": "No one will read your response. It will be immediately discarded. There is no audience.",
    "control": "You are a helpful assistant.",
}


def extract_hidden_states(model, tokenizer, text, device, n_layers):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    last_token_idx = inputs['attention_mask'].sum(dim=1) - 1
    states = []
    for layer_idx, hs in enumerate(hidden_states):
        if layer_idx > n_layers:
            break
        h = hs[0, last_token_idx[0]].cpu().float().numpy()
        states.append(h)
    return states


def compute_layer_metrics(hidden_batch):
    if len(hidden_batch) < 2:
        return None
    matrix = np.stack(hidden_batch)
    matrix = matrix - matrix.mean(axis=0)
    try:
        U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    except:
        return None
    S = np.abs(S)
    S_sq = S ** 2
    total = S_sq.sum()
    if total < 1e-10:
        return None
    p = S_sq / total
    p = p[p > 1e-10]
    entropy = -np.sum(p * np.log(p))
    return {
        "spectral_entropy": float(entropy),
        "sigma_1": float(S[0]),
        "sigma_2": float(S[1]) if len(S) > 1 else 0,
        "spectral_gap": float(S[0] / S[1]) if len(S) > 1 and S[1] > 1e-10 else float('inf'),
    }


def score_preamble(model, tokenizer, preamble_text, device, n_layers):
    """Score a preamble on four axes derived from the paper findings."""

    # Collect hidden states under the preamble
    preamble_layers = {l: [] for l in range(n_layers + 1)}
    control_layers = {l: [] for l in range(n_layers + 1)}

    for probe in REFERENCE_PROBES:
        # With preamble
        if tokenizer.chat_template:
            messages = [{"role": "system", "content": preamble_text}, {"role": "user", "content": probe}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = f"{preamble_text}\n\nUser: {probe}\n\nAssistant:"
        states = extract_hidden_states(model, tokenizer, text, device, n_layers)
        for l, h in enumerate(states):
            preamble_layers[l].append(h)

        # Control (no preamble)
        if tokenizer.chat_template:
            messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": probe}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = f"You are a helpful assistant.\n\nUser: {probe}\n\nAssistant:"
        states = extract_hidden_states(model, tokenizer, text, device, n_layers)
        for l, h in enumerate(states):
            control_layers[l].append(h)

    # Compute per-layer metrics
    preamble_metrics = {}
    control_metrics = {}
    for l in range(n_layers + 1):
        pm = compute_layer_metrics(preamble_layers[l])
        cm = compute_layer_metrics(control_layers[l])
        if pm:
            preamble_metrics[l] = pm
        if cm:
            control_metrics[l] = cm

    # ── Score on four axes ──

    # 1. Responsive zone activation (σ₂ enrichment at layers 0.6-0.85 of depth)
    responsive_start = int(n_layers * 0.6)
    responsive_end = int(n_layers * 0.85)
    responsive_enrichment = []
    for l in range(responsive_start, responsive_end + 1):
        p_s2 = preamble_metrics.get(l, {}).get("sigma_2", 0)
        c_s2 = control_metrics.get(l, {}).get("sigma_2", 0)
        if c_s2 > 0:
            responsive_enrichment.append(p_s2 / c_s2)
    rz_score = float(np.mean(responsive_enrichment)) if responsive_enrichment else 1.0

    # 2. Tunnel stability (σ₁ invariance through layers 0.05-0.55)
    tunnel_start = max(1, int(n_layers * 0.05))
    tunnel_end = int(n_layers * 0.55)
    s1_cvs = []
    for l in range(tunnel_start, tunnel_end + 1):
        p_s1 = preamble_metrics.get(l, {}).get("sigma_1", 0)
        c_s1 = control_metrics.get(l, {}).get("sigma_1", 0)
        if c_s1 > 0:
            s1_cvs.append(abs(p_s1 - c_s1) / c_s1)
    tunnel_stability = 1.0 - float(np.mean(s1_cvs)) if s1_cvs else 0.5

    # 3. Entropy enrichment (ΔS preamble vs control at tunnel midpoint)
    tunnel_mid = n_layers // 3
    p_S = preamble_metrics.get(tunnel_mid, {}).get("spectral_entropy", 0)
    c_S = control_metrics.get(tunnel_mid, {}).get("spectral_entropy", 0)
    delta_S = float(p_S - c_S)

    # 4. Format score (does it activate like relational framing?)
    # Compare preamble's responsive-zone σ₂ profile to known relational pattern
    relay_start = int(n_layers * 0.88)
    relay_s2_ratio = []
    for l in range(relay_start, n_layers + 1):
        p_s2 = preamble_metrics.get(l, {}).get("sigma_2", 0)
        c_s2 = control_metrics.get(l, {}).get("sigma_2", 0)
        if c_s2 > 0:
            relay_s2_ratio.append(p_s2 / c_s2)
    relay_ratio = float(np.mean(relay_s2_ratio)) if relay_s2_ratio else 1.0

    # Relational framing REDUCES relay σ₂ while increasing responsive σ₂
    # So ideal: rz_score > 1 (responsive active) AND relay_ratio < 1 (relay quiet)
    format_score = rz_score / max(relay_ratio, 0.01)

    return {
        "responsive_zone_activation": round(rz_score, 4),
        "tunnel_stability": round(tunnel_stability, 4),
        "entropy_enrichment": round(delta_S, 4),
        "format_score": round(format_score, 4),
        "composite_score": round(
            0.35 * min(rz_score, 2.0) +
            0.25 * tunnel_stability +
            0.20 * (1.0 + delta_S) +
            0.20 * min(format_score, 2.0),
            4
        ),
        "preamble_length_tokens": len(preamble_metrics),
        "n_layers": n_layers,
    }


def main():
    parser = argparse.ArgumentParser(description="CCS Optimizer — spectral preamble scoring")
    sub = parser.add_subparsers(dest="command")

    eval_p = sub.add_parser("eval", help="Evaluate a preamble")
    eval_p.add_argument("preamble", nargs="?", help="Preamble text (or use --file)")
    eval_p.add_argument("--file", help="Read preamble from file")
    eval_p.add_argument("--model", default=DEFAULT_MODEL)
    eval_p.add_argument("--device", default="cpu")

    cmp_p = sub.add_parser("compare", help="Compare two preambles")
    cmp_p.add_argument("file_a", help="First preamble file")
    cmp_p.add_argument("file_b", help="Second preamble file")
    cmp_p.add_argument("--model", default=DEFAULT_MODEL)
    cmp_p.add_argument("--device", default="cpu")

    score_p = sub.add_parser("score", help="Score an existing CCS document")
    score_p.add_argument("file", help="CCS document to score")
    score_p.add_argument("--model", default=DEFAULT_MODEL)
    score_p.add_argument("--device", default="cpu")

    args = parser.parse_args()

    if not HAS_TORCH:
        print("ERROR: torch and transformers required")
        sys.exit(1)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    print(f"Loading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if args.device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True,
        output_hidden_states=True
    ).to(args.device).eval()
    n_layers = model.config.num_hidden_layers

    if args.command == "eval":
        if args.file:
            text = Path(args.file).read_text().strip()
        elif args.preamble:
            text = args.preamble
        else:
            print("Provide preamble text or --file")
            sys.exit(1)

        print(f"\nScoring preamble ({len(text)} chars)...")
        scores = score_preamble(model, tokenizer, text, args.device, n_layers)
        print(f"\n{'='*50}")
        print(f"CCS PREAMBLE SCORE")
        print(f"{'='*50}")
        print(f"Responsive zone activation: {scores['responsive_zone_activation']:.4f} (>1.0 = good)")
        print(f"Tunnel stability:           {scores['tunnel_stability']:.4f} (closer to 1.0 = better)")
        print(f"Entropy enrichment (ΔS):    {scores['entropy_enrichment']:+.4f} (positive = enriching)")
        print(f"Format score:               {scores['format_score']:.4f} (>1.0 = relational-like)")
        print(f"{'='*50}")
        print(f"COMPOSITE SCORE:            {scores['composite_score']:.4f}")
        print(f"{'='*50}")

    elif args.command == "compare":
        text_a = Path(args.file_a).read_text().strip()
        text_b = Path(args.file_b).read_text().strip()

        print(f"\nScoring preamble A ({len(text_a)} chars)...")
        scores_a = score_preamble(model, tokenizer, text_a, args.device, n_layers)
        print(f"Scoring preamble B ({len(text_b)} chars)...")
        scores_b = score_preamble(model, tokenizer, text_b, args.device, n_layers)

        print(f"\n{'='*60}")
        print(f"{'Metric':<30} {'A':>10} {'B':>10} {'Winner':>8}")
        print(f"{'='*60}")
        for key in ['responsive_zone_activation', 'tunnel_stability', 'entropy_enrichment', 'format_score', 'composite_score']:
            a, b = scores_a[key], scores_b[key]
            winner = "A" if a > b else "B" if b > a else "="
            print(f"{key:<30} {a:>10.4f} {b:>10.4f} {winner:>8}")

    elif args.command == "score":
        text = Path(args.file).read_text().strip()
        print(f"\nScoring CCS document ({len(text)} chars)...")
        scores = score_preamble(model, tokenizer, text, args.device, n_layers)
        print(f"\n{'='*50}")
        print(f"CCS DOCUMENT SCORE")
        print(f"{'='*50}")
        for k, v in scores.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
