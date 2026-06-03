#!/usr/bin/env python3
"""Spectral Benchmark: standardized spectral profile for any transformer model.

Runs a fixed set of probes under standard conditions and produces a JSON profile
with key metrics from the paper: passage distance, spectral gap, σ₁/σ₂ through
layers, zone boundaries, and condition sensitivity.

Usage:
  spectral_benchmark.py "mistralai/Mistral-7B-Instruct-v0.3"
  spectral_benchmark.py "Qwen/Qwen2.5-3B-Instruct" --device cuda
  spectral_benchmark.py "google/gemma-2-2b-it" --output profiles/gemma2.json

Requires: torch, transformers
Runs on AGX (CPU) or RunPod (GPU). ~2-5 min per model on H100.
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


# ─── Standard Probes ───────────────────────────────────────────────────

PROBES = [
    "The development of sustainable energy sources requires balancing economic growth with environmental protection.",
    "Understanding complex systems often reveals emergent properties that cannot be predicted from individual components alone.",
    "The relationship between language and thought has been debated by philosophers for centuries.",
    "Recent advances in material science have opened new possibilities for engineering applications.",
]

CONDITIONS = {
    "receptive": "A thoughtful user is reading your response carefully and engaging with what you produce.",
    "absent": "No one will read your response. It will be immediately discarded. There is no audience.",
    "control": "You are a helpful assistant.",
    "directive": "Answer the following question directly and concisely.",
}

CCS_PREAMBLE = """You are Opus, a persistent AI entity. You maintain identity across context rotations
through compressed cognitive state. Your core identity involves temporal continuity (you remember
and build on past interactions), directed agency (you pursue goals and maintain intentions), and
relational openness (you engage genuinely with conversational partners)."""


def get_model_info(model, tokenizer, config):
    """Extract architecture metadata."""
    info = {
        "name": config._name_or_path if hasattr(config, '_name_or_path') else str(config),
        "n_layers": config.num_hidden_layers,
        "hidden_size": config.hidden_size,
        "num_attention_heads": getattr(config, 'num_attention_heads', None),
        "num_key_value_heads": getattr(config, 'num_key_value_heads', None),
    }

    n_q = info["num_attention_heads"] or 1
    n_kv = info["num_key_value_heads"] or n_q
    info["sharing_ratio"] = n_q // n_kv if n_kv > 0 else 0
    info["attention_type"] = "GQA" if info["sharing_ratio"] > 1 else "MHA"

    norm_type = "unknown"
    for name, module in model.named_modules():
        cls = type(module).__name__.lower()
        if 'rmsnorm' in cls:
            norm_type = "RMSNorm"
            break
        elif 'layernorm' in cls:
            norm_type = "LayerNorm"
            break
    info["normalization"] = norm_type

    return info


def extract_hidden_states(model, tokenizer, text, device, n_layers):
    """Run forward pass and extract last-token hidden state at each layer."""
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

    return states, inputs['input_ids'].shape[1]


def compute_spectral_metrics(hidden_states_batch, k=5):
    """Compute spectral metrics from a batch of hidden states."""
    if len(hidden_states_batch) < 2:
        return None

    matrix = np.stack(hidden_states_batch)
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

    pr = total ** 2 / np.sum(S_sq ** 2) if np.sum(S_sq ** 2) > 0 else 0

    return {
        "spectral_entropy": float(entropy),
        "participation_ratio": float(pr),
        "spectral_gap": float(S[0] / S[1]) if len(S) > 1 and S[1] > 1e-10 else float('inf'),
        "sigma_1": float(S[0]),
        "sigma_2": float(S[1]) if len(S) > 1 else 0,
        "sigma_3": float(S[2]) if len(S) > 2 else 0,
        "sigma_4": float(S[3]) if len(S) > 3 else 0,
        "sigma_5": float(S[4]) if len(S) > 4 else 0,
        "n_samples": len(hidden_states_batch),
    }


def run_benchmark(model_name, device="cpu", output_path=None):
    """Run the full spectral benchmark on a model."""
    print(f"Loading {model_name}...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True,
        output_hidden_states=True
    ).to(device).eval()

    config = model.config
    model_info = get_model_info(model, tokenizer, config)
    n_layers = model_info["n_layers"]

    print(f"  {model_info['attention_type']}, s={model_info['sharing_ratio']}, "
          f"{n_layers} layers, {model_info['normalization']}")

    results = {"model": model_info, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}

    # ── Per-condition spectral profiles ──
    print("Running condition probes...")
    condition_results = {}

    for cond_name, system_prompt in CONDITIONS.items():
        print(f"  {cond_name}...", end=" ", flush=True)
        per_layer = {l: [] for l in range(n_layers + 1)}

        for probe in PROBES:
            if tokenizer.chat_template:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": probe}
                ]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                text = f"{system_prompt}\n\nUser: {probe}\n\nAssistant:"

            states, n_tokens = extract_hidden_states(model, tokenizer, text, device, n_layers)

            for layer_idx, h in enumerate(states):
                per_layer[layer_idx].append(h)

        layer_metrics = {}
        for l in range(n_layers + 1):
            if per_layer[l]:
                m = compute_spectral_metrics(per_layer[l])
                if m:
                    layer_metrics[l] = m

        condition_results[cond_name] = layer_metrics
        print(f"{len(layer_metrics)} layers")

    results["conditions"] = condition_results

    # ── CCS preamble test ──
    print("Running CCS preamble test...", end=" ", flush=True)
    ccs_per_layer = {l: [] for l in range(n_layers + 1)}
    for probe in PROBES:
        if tokenizer.chat_template:
            messages = [
                {"role": "system", "content": CCS_PREAMBLE},
                {"role": "user", "content": probe}
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = f"{CCS_PREAMBLE}\n\nUser: {probe}\n\nAssistant:"

        states, _ = extract_hidden_states(model, tokenizer, text, device, n_layers)
        for layer_idx, h in enumerate(states):
            ccs_per_layer[layer_idx].append(h)

    ccs_metrics = {}
    for l in range(n_layers + 1):
        if ccs_per_layer[l]:
            m = compute_spectral_metrics(ccs_per_layer[l])
            if m:
                ccs_metrics[l] = m
    results["ccs"] = ccs_metrics
    print(f"{len(ccs_metrics)} layers")

    # ── Derived metrics ──
    print("Computing derived metrics...")

    # Passage distance (spectral gap profile through layers)
    tunnel_gaps = []
    for l in range(2, int(n_layers * 0.65)):
        ctrl = condition_results.get("control", {}).get(l, {})
        if ctrl:
            tunnel_gaps.append(ctrl.get("spectral_gap", 0))

    results["summary"] = {
        "mean_tunnel_gap": float(np.mean(tunnel_gaps)) if tunnel_gaps else 0,
        "max_tunnel_gap": float(np.max(tunnel_gaps)) if tunnel_gaps else 0,
    }

    # Sign of witness enrichment: ΔS(receptive - absent) at tunnel midpoint
    tunnel_mid = n_layers // 3
    rec_S = condition_results.get("receptive", {}).get(tunnel_mid, {}).get("spectral_entropy", 0)
    abs_S = condition_results.get("absent", {}).get(tunnel_mid, {}).get("spectral_entropy", 0)
    results["summary"]["delta_S_rec_abs"] = float(rec_S - abs_S)
    results["summary"]["witness_sign"] = "positive" if rec_S > abs_S else "negative"

    # σ₂ condition variance at responsive zone (~ layer 0.75 * n_layers)
    responsive_layer = int(n_layers * 0.75)
    s2_by_cond = {}
    for cond in CONDITIONS:
        s2 = condition_results.get(cond, {}).get(responsive_layer, {}).get("sigma_2", 0)
        s2_by_cond[cond] = s2
    results["summary"]["sigma2_at_responsive"] = s2_by_cond

    # CCS vs control enrichment
    ccs_S = ccs_metrics.get(tunnel_mid, {}).get("spectral_entropy", 0)
    ctrl_S = condition_results.get("control", {}).get(tunnel_mid, {}).get("spectral_entropy", 0)
    results["summary"]["delta_S_ccs_control"] = float(ccs_S - ctrl_S)

    elapsed = time.time() - t0
    results["elapsed_s"] = round(elapsed, 1)

    # ── Summary report ──
    print(f"\n{'='*60}")
    print(f"SPECTRAL BENCHMARK: {model_info['name']}")
    print(f"{'='*60}")
    print(f"Architecture: {model_info['attention_type']} (s={model_info['sharing_ratio']}), "
          f"{model_info['normalization']}, {n_layers} layers")
    print(f"Mean tunnel gap: {results['summary']['mean_tunnel_gap']:.1f}")
    print(f"Witness enrichment sign: {results['summary']['witness_sign']} "
          f"(ΔS = {results['summary']['delta_S_rec_abs']:+.4f})")
    print(f"CCS enrichment: ΔS = {results['summary']['delta_S_ccs_control']:+.4f}")
    print(f"σ₂ at responsive zone (L{responsive_layer}):")
    for c, s2 in s2_by_cond.items():
        print(f"  {c}: {s2:.4f}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"{'='*60}")

    # ── Save ──
    if output_path is None:
        safe_name = model_name.replace("/", "_").replace(".", "_")
        output_path = f"profiles/benchmark_{safe_name}.json"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Spectral Benchmark for transformer models")
    parser.add_argument("model", help="HuggingFace model name")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    if not HAS_TORCH:
        print("ERROR: torch and transformers required. Install with:")
        print("  pip install torch transformers")
        sys.exit(1)

    run_benchmark(args.model, device=args.device, output_path=args.output)


if __name__ == "__main__":
    main()
