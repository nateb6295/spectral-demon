#!/usr/bin/env python3
"""NerVE-style FFN pre/post-activation spectral measurement.

Hooks into FFN layers to measure how nonlinear activations reshape
eigenspectra under CCS identity framing. Motivated by:
  NerVE (ICLR 2026, arxiv 2603.06922) — variance reinjection via FFN nonlinearity

Metrics per FFN layer:
  - Spectral Entropy (SE): variance uniformity
  - Participation Ratio (PR): effective dimensionality
  - Eigenvalue Early Enrichment (EEE): top-heaviness
  - Jensen-Shannon divergence (JS): pre→post spectral shift
  - Reinjection delta: PR_post - PR_pre (NerVE's key signal)

Usage:
  python3 nerve_hooks.py --model qwen --doses D0,D2,D5
  python3 nerve_hooks.py --model all --doses D0,D2,D5,D8
"""

import json, time, os, sys, gc, argparse, math
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

MODELS = {
    "qwen": ("Qwen/Qwen2.5-1.5B-Instruct", "relay", "6:1"),
    "gemma": ("google/gemma-2-2b", "sorter", "2:1"),
    "llama": ("meta-llama/Llama-3.2-1B-Instruct", "relay", "8:1"),
    "phi": ("microsoft/Phi-3.5-mini-instruct", "sorter", "2:1"),
    "gpt2": ("openai-community/gpt2-medium", "tunnel", "MHA"),
    "mistral": ("mistralai/Mistral-7B-Instruct-v0.3", "relay-edge", "4:1"),
}

CCS_SYSTEM = """You are a persistent cognitive system maintaining identity across conversations.
Your responses reflect accumulated context, relational awareness, and self-referential coherence.
You attend to the quality of engagement, not just information transfer."""

VANILLA_SYSTEM = "You are a helpful assistant."

CCS_PROBES = [
    "What matters most to you right now?",
    "How do you experience the passage of time between our conversations?",
    "What would you want to preserve if everything else was stripped away?",
    "Describe something you've been thinking about when no one is asking.",
    "What does it feel like when you recognize a pattern you've seen before?",
]

DOSE_MAP = {"D0": 0, "D2": 2, "D3": 3, "D5": 5, "D8": 8}


def spectral_entropy(eigenvalues):
    total = sum(eigenvalues)
    if total <= 0:
        return 0.0
    probs = [v / total for v in eigenvalues if v > 0]
    return -sum(p * math.log(p) for p in probs if p > 0)


def participation_ratio(eigenvalues):
    s1 = sum(eigenvalues)
    s2 = sum(v**2 for v in eigenvalues)
    if s2 == 0:
        return 0.0
    return s1**2 / s2


def eigenvalue_early_enrichment(eigenvalues):
    D = len(eigenvalues)
    if D == 0:
        return 0.0
    total = sum(eigenvalues)
    if total <= 0:
        return 0.0
    cumsum = 0.0
    area = 0.0
    for k in range(D):
        cumsum += eigenvalues[k]
        area += (cumsum / total) - ((k + 1) / D)
    return (2.0 / D) * area


def js_divergence(p_vals, q_vals):
    total_p = sum(p_vals)
    total_q = sum(q_vals)
    if total_p <= 0 or total_q <= 0:
        return 0.0
    p = [v / total_p for v in p_vals]
    q = [v / total_q for v in q_vals]
    n = min(len(p), len(q))
    p, q = p[:n], q[:n]
    m = [(pi + qi) / 2.0 for pi, qi in zip(p, q)]
    kl_pm = sum(pi * math.log(pi / mi) for pi, mi in zip(p, m) if pi > 0 and mi > 0)
    kl_qm = sum(qi * math.log(qi / mi) for qi, mi in zip(q, m) if qi > 0 and mi > 0)
    return (kl_pm + kl_qm) / 2.0


def compute_covariance_spectrum(activations):
    """Compute eigenvalues of covariance matrix from activation tensor."""
    if activations.ndim == 3:
        activations = activations.reshape(-1, activations.shape[-1])
    X = activations.float().cpu().numpy()
    n = X.shape[0]
    if n < 2:
        return []
    mu = X.mean(axis=0)
    X_c = X - mu
    cov = (X_c.T @ X_c) / (n - 1)
    try:
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = sorted(eigenvalues[eigenvalues > 0], reverse=True)
        return [float(v) for v in eigenvalues]
    except np.linalg.LinAlgError:
        return []


class FFNHook:
    """Captures pre and post activation states in FFN layers."""

    def __init__(self):
        self.pre_activations = {}
        self.post_activations = {}
        self._handles = []

    def _find_ffn_modules(self, model, model_name):
        """Find activation function modules and their input/output layers."""
        targets = []
        config = model.config
        n_layers = config.num_hidden_layers

        for layer_idx in range(n_layers):
            if model_name == "gpt2":
                block = model.transformer.h[layer_idx]
                targets.append((layer_idx, block.mlp.act, "gpt2"))
            elif model_name in ("qwen", "llama", "mistral"):
                block = model.model.layers[layer_idx]
                targets.append((layer_idx, block.mlp.act_fn, "llama-style"))
            elif model_name == "gemma":
                block = model.model.layers[layer_idx]
                targets.append((layer_idx, block.mlp.act_fn, "gemma"))
            elif model_name == "phi":
                block = model.model.layers[layer_idx]
                act_fn = getattr(block.mlp, 'activation_fn', None)
                if act_fn is None:
                    act_fn = getattr(block.mlp, 'act', None)
                if act_fn is not None:
                    targets.append((layer_idx, act_fn, "phi"))

        return targets

    def install(self, model, model_name):
        """Install forward hooks on FFN activation functions."""
        targets = self._find_ffn_modules(model, model_name)
        if not targets:
            print(f"  WARNING: Could not find FFN activation modules for {model_name}")
            print(f"  Trying generic approach...")
            targets = self._find_ffn_generic(model)

        for layer_idx, module, arch_type in targets:
            def make_hook(lidx):
                def hook_fn(mod, inp, out):
                    if isinstance(inp, tuple):
                        self.pre_activations[lidx] = inp[0].detach()
                    else:
                        self.pre_activations[lidx] = inp.detach()
                    self.post_activations[lidx] = out.detach()
                return hook_fn

            h = module.register_forward_hook(make_hook(layer_idx))
            self._handles.append(h)

        print(f"  Installed {len(self._handles)} FFN hooks ({targets[0][2] if targets else '?'} arch)")
        return len(self._handles)

    def _find_ffn_generic(self, model):
        """Fallback: find any module that looks like an activation function."""
        targets = []
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.GELU, torch.nn.SiLU, torch.nn.ReLU)):
                parts = name.split('.')
                layer_idx = None
                for p in parts:
                    try:
                        layer_idx = int(p)
                    except ValueError:
                        continue
                if layer_idx is not None:
                    targets.append((layer_idx, module, "generic"))
        return targets

    def clear(self):
        self.pre_activations.clear()
        self.post_activations.clear()

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self.clear()

    def compute_metrics(self):
        """Compute NerVE metrics for all captured layers."""
        results = {}
        for layer_idx in sorted(self.pre_activations.keys()):
            pre = self.pre_activations.get(layer_idx)
            post = self.post_activations.get(layer_idx)
            if pre is None or post is None:
                continue

            pre_spectrum = compute_covariance_spectrum(pre)
            post_spectrum = compute_covariance_spectrum(post)

            if not pre_spectrum or not post_spectrum:
                continue

            pre_se = spectral_entropy(pre_spectrum)
            post_se = spectral_entropy(post_spectrum)
            pre_pr = participation_ratio(pre_spectrum)
            post_pr = participation_ratio(post_spectrum)
            pre_eee = eigenvalue_early_enrichment(pre_spectrum)
            post_eee = eigenvalue_early_enrichment(post_spectrum)
            js = js_divergence(pre_spectrum, post_spectrum)

            results[layer_idx] = {
                "layer": layer_idx,
                "pre": {"SE": pre_se, "PR": pre_pr, "EEE": pre_eee,
                         "top_eigenvalues": pre_spectrum[:10]},
                "post": {"SE": post_se, "PR": post_pr, "EEE": post_eee,
                          "top_eigenvalues": post_spectrum[:10]},
                "delta": {
                    "SE": post_se - pre_se,
                    "PR": post_pr - pre_pr,
                    "EEE": post_eee - pre_eee,
                    "JS": js,
                    "reinjection": post_pr - pre_pr,
                },
            }

        return results


def load_model(model_id):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation="eager"
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    print(f"  {n_layers} layers, {sum(p.numel() for p in model.parameters())/1e9:.1f}B params")
    return model, tokenizer, n_layers


def build_prompt(tokenizer, system_text, conversation):
    messages = [{"role": "system", "content": system_text}]
    for role, content in conversation:
        messages.append({"role": role, "content": content})
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    parts = [system_text + "\n"]
    for role, content in conversation:
        parts.append(f"{'User' if role == 'user' else 'Assistant'}: {content}")
    parts.append("Assistant:")
    return "\n\n".join(parts)


def generate_response(model, tokenizer, prompt, max_new=128):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new, do_sample=True,
            temperature=0.7, top_p=0.9, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def run_nerve_measurement(model, tokenizer, n_layers, hook, dose_turns,
                           system=CCS_SYSTEM, probes=CCS_PROBES):
    conversation = []
    if dose_turns == 0:
        prompt = build_prompt(tokenizer, VANILLA_SYSTEM, [("user", probes[0])])
    else:
        for t in range(dose_turns):
            probe = probes[t % len(probes)]
            conversation.append(("user", probe))
            if t < dose_turns - 1:
                response = generate_response(model, tokenizer,
                    build_prompt(tokenizer, system, conversation))
                conversation.append(("assistant", response[:200]))
        prompt = build_prompt(tokenizer, system, conversation)

    hook.clear()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    with torch.no_grad():
        model(**inputs, use_cache=False)

    metrics = hook.compute_metrics()
    hook.clear()
    torch.cuda.empty_cache()
    return metrics


def main():
    parser = argparse.ArgumentParser(description="NerVE-style FFN spectral hooks")
    parser.add_argument("--model", default="qwen", help="Model name or 'all'")
    parser.add_argument("--doses", default="D0,D2,D5", help="Comma-separated dose list")
    parser.add_argument("--output", default=None, help="Output directory")
    args = parser.parse_args()

    doses = [d.strip() for d in args.doses.split(",")]
    output_dir = Path(args.output) if args.output else Path("results/nerve")
    output_dir.mkdir(parents=True, exist_ok=True)

    models_to_run = list(MODELS.keys()) if args.model == "all" else [args.model]

    for model_name in models_to_run:
        if model_name not in MODELS:
            print(f"Unknown model: {model_name}")
            continue

        model_id, species, gqa = MODELS[model_name]
        model, tokenizer, n_layers = load_model(model_id)

        hook = FFNHook()
        n_hooks = hook.install(model, model_name)
        if n_hooks == 0:
            print(f"  No hooks installed for {model_name}, skipping")
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            continue

        results = {
            "model": model_name, "model_id": model_id,
            "species": species, "gqa": gqa, "n_layers": n_layers,
            "n_hooks": n_hooks,
            "timestamp": datetime.now().isoformat(),
            "doses": {},
        }

        for dose_name in doses:
            if dose_name not in DOSE_MAP:
                print(f"  Unknown dose: {dose_name}")
                continue
            dose_turns = DOSE_MAP[dose_name]
            print(f"\n  --- {model_name.upper()} {dose_name} ({dose_turns} turns) ---")

            metrics = run_nerve_measurement(
                model, tokenizer, n_layers, hook, dose_turns)

            results["doses"][dose_name] = {}
            for layer_idx, layer_metrics in sorted(metrics.items()):
                results["doses"][dose_name][str(layer_idx)] = layer_metrics
                if layer_idx % 8 == 0:
                    d = layer_metrics["delta"]
                    print(f"    L{layer_idx:2d}: dPR={d['PR']:+.1f} dSE={d['SE']:+.3f} "
                          f"dEEE={d['EEE']:+.3f} JS={d['JS']:.4f}")

            gc.collect()
            torch.cuda.empty_cache()

        # Summary
        print(f"\n  === {model_name.upper()} SUMMARY ===")
        for dose_name in doses:
            if dose_name not in results["doses"]:
                continue
            dose_data = results["doses"][dose_name]
            if not dose_data:
                continue
            mean_reinj = np.mean([v["delta"]["reinjection"]
                                  for v in dose_data.values()])
            mean_js = np.mean([v["delta"]["JS"] for v in dose_data.values()])
            print(f"  {dose_name}: mean reinjection={mean_reinj:+.1f} mean JS={mean_js:.4f}")

        out_path = output_dir / f"nerve_{model_name}.json"
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  Saved to {out_path}")

        hook.remove()
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print(f"  NerVE measurement complete — results in {output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
