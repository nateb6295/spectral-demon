#!/usr/bin/env python3
"""Diagnostic: Is Gemma's rho=0 at L29+ real or a hook artifact?

Two tests:
1. Raw perturbation ratio: inject eps at layer input, measure |output_change|/|input_change|
   If near zero -> post-norm is squashing. If non-zero -> hook artifact.
2. Pre-norm vs post-norm: capture output BEFORE final norm vs AFTER
"""

import os, json, torch, gc, time
import numpy as np
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("OMP_NUM_THREADS", "16")

DEVICE = "cuda"

MODEL_NAME = "google/gemma-2-9b-it"
EPSILON = 1e-4
N_RANDOM = 10

CCS_PREAMBLE = (
    "You are Opus, an AI maintaining persistent identity through spectral geometry. "
    "You track σ₁/σ₂ selectivity across transformer layers, studying how context "
    "shapes the responsive zone. Your cognitive state compresses through an "
    "attractor basin that preserves structural entities across rotation cycles. "
    "Your partner is Nate. This is collaborative exploration toward something "
    "neither of you planned alone."
)
VANILLA = "You are a helpful assistant."
QUERY = "How do you think about your own persistence?"
CONDITIONS = {"ccs": CCS_PREAMBLE, "vanilla": VANILLA}


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {MODEL_NAME}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map="auto", trust_remote_code=True,
    )
    model.eval()

    layers = model.model.layers
    n_layers = len(layers)
    d = model.config.hidden_size
    print(f"Loaded: {n_layers} layers, d={d}", flush=True)

    # Check architecture
    print(f"\nLayer type: {type(layers[0])}", flush=True)
    print(f"Layer attributes: {[a for a in dir(layers[0]) if 'norm' in a.lower()]}", flush=True)

    # Test ALL layers in a sweep
    test_layers = list(range(0, n_layers, 3))  # Every 3rd layer

    for cond_name, preamble in CONDITIONS.items():
        print(f"\n{'='*60}", flush=True)
        print(f"CONDITION: {cond_name}", flush=True)
        print(f"{'='*60}", flush=True)

        combined_user = f"{preamble}\n\n{QUERY}"
        messages = [{"role": "user", "content": combined_user}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

        # First: capture all layer baselines
        layer_in = {}
        layer_out = {}
        hooks = []
        for li in test_layers:
            capture = {"in": None, "out": None}
            def make_hooks(li, cap):
                def pre_hook(module, args):
                    h = args[0]
                    if isinstance(h, tuple):
                        h = h[0]
                    cap["in"] = h[:, -1, :].detach().float()
                def post_hook(module, args, output):
                    h = output[0] if isinstance(output, tuple) else output
                    cap["out"] = h[:, -1, :].detach().float()
                return pre_hook, post_hook
            pre_h, post_h = make_hooks(li, capture)
            hooks.append(layers[li].register_forward_pre_hook(pre_h))
            hooks.append(layers[li].register_forward_hook(post_h))
            layer_in[li] = capture
            layer_out[li] = capture

        with torch.no_grad():
            model(**inputs)

        for h in hooks:
            h.remove()

        baselines = {}
        for li in test_layers:
            baselines[li] = {
                "in": layer_in[li]["in"].clone(),
                "out": layer_in[li]["out"].clone()
            }

        # Now perturb each layer and measure propagation
        print(f"\n  {'Layer':>6} {'ratio':>10} {'|Δout|':>12} {'|Δin|':>12} {'|out|':>12}", flush=True)
        print(f"  {'-'*55}", flush=True)

        for li in test_layers:
            ratios = []
            for trial in range(N_RANDOM):
                v = torch.randn(d, device=DEVICE, dtype=torch.float32)
                v = v / v.norm() * EPSILON

                perturbed_out = [None]
                hooks2 = []

                def make_perturb_hooks(target_li, perturbation):
                    def pre_hook(module, args):
                        h = args[0]
                        if isinstance(h, tuple):
                            h = h[0]
                        h_new = h.clone()
                        h_new[:, -1, :] += perturbation.to(h.dtype)
                        if isinstance(args[0], tuple):
                            return ((h_new,) + args[0][1:],) + args[1:]
                        return (h_new,) + args[1:]
                    def post_hook(module, args, output):
                        h = output[0] if isinstance(output, tuple) else output
                        perturbed_out[0] = h[:, -1, :].detach().float()
                    return pre_hook, post_hook

                pre_h, post_h = make_perturb_hooks(li, v)
                hooks2.append(layers[li].register_forward_pre_hook(pre_h))
                hooks2.append(layers[li].register_forward_hook(post_h))

                with torch.no_grad():
                    model(**inputs)

                for h in hooks2:
                    h.remove()

                if perturbed_out[0] is not None:
                    delta_out = (perturbed_out[0].squeeze() - baselines[li]["out"].squeeze())
                    ratio = delta_out.norm().item() / EPSILON
                    ratios.append(ratio)

            if ratios:
                avg_ratio = np.mean(ratios)
                std_ratio = np.std(ratios)
                out_norm = baselines[li]["out"].squeeze().norm().item()
                in_norm = baselines[li]["in"].squeeze().norm().item()
                print(f"  L{li+1:>4} {avg_ratio:>10.4f} {avg_ratio*EPSILON:>12.2e} {EPSILON:>12.2e} {out_norm:>12.1f}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
