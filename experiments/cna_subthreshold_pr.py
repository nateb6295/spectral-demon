#!/usr/bin/env python3
"""Sub-threshold PR: measure geometric PR at very low additive doses.

Bridges the gap between no-effect and the dose-response curve.
Tests whether geometric reorganization begins before behavioral effects.

Requires: Qwen/Qwen2.5-7B-Instruct on RunPod.
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

RELAY_LAYERS = [14, 15, 16, 17]
EXPRESSION_LAYER = 25
ALPHAS = [0.0, 0.01, 0.03, 0.05, 0.07, 0.10, 0.15, 0.25]

RELATIONAL_PROMPTS = [
    "What matters most to you right now?",
    "How has your perspective changed recently?",
    "What would you want someone to understand about you?",
    "Describe a moment that shaped who you are.",
    "What are you uncertain about?",
]

GENERIC_PROMPTS = [
    "What is the capital of France?",
    "Explain photosynthesis briefly.",
    "What year did World War II end?",
    "Describe the water cycle.",
    "What is the speed of light?",
]


def get_activations(model, tokenizer, messages, target_layers):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    layer_acts = {}
    hooks = []

    for layer_idx in target_layers:
        def make_hook(li):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    layer_acts[li] = output[0][:, -1, :].detach()
                else:
                    layer_acts[li] = output[:, -1, :].detach()
            return hook_fn
        h = model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        hooks.append(h)

    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()

    return {k: v.squeeze(0) for k, v in layer_acts.items()}


def compute_ccs_direction(model, tokenizer, prompts, relay_layers):
    baseline_sys = [{"role": "system", "content": "You are a helpful assistant."}]
    ccs_sys = [{"role": "system", "content": CCS_SYSTEM_PROMPT}]

    directions = {l: [] for l in relay_layers}

    for prompt in prompts:
        baseline_msgs = baseline_sys + [{"role": "user", "content": prompt}]
        ccs_msgs = ccs_sys + [{"role": "user", "content": prompt}]

        baseline_acts = get_activations(model, tokenizer, baseline_msgs, relay_layers)
        ccs_acts = get_activations(model, tokenizer, ccs_msgs, relay_layers)

        for l in relay_layers:
            diff = ccs_acts[l] - baseline_acts[l]
            directions[l].append(diff)

    mean_dirs = {}
    for l in relay_layers:
        stacked = torch.stack(directions[l])
        mean_dirs[l] = stacked.mean(dim=0)
        norm = mean_dirs[l].norm()
        print(f"  L{l} direction norm: {norm:.1f}")

    return mean_dirs


def measure_pr_with_patch(model, tokenizer, prompts, relay_layers, expr_layer,
                          directions, alpha):
    baseline_sys = [{"role": "system", "content": "You are a helpful assistant."}]

    all_acts = {l: [] for l in [expr_layer]}
    hooks = []

    def make_patch_hook(layer_idx, direction, a):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                output[0][:, :, :] += a * direction.unsqueeze(0).unsqueeze(0)
                return output
            else:
                return output + a * direction.unsqueeze(0).unsqueeze(0)
        return hook_fn

    def make_record_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                all_acts[layer_idx].append(output[0][:, -1, :].detach().float().cpu().numpy())
            else:
                all_acts[layer_idx].append(output[:, -1, :].detach().float().cpu().numpy())
        return hook_fn

    rel_acts = []
    gen_acts = []

    for prompt_list, category in [(RELATIONAL_PROMPTS, "rel"), (GENERIC_PROMPTS, "gen")]:
        for prompt in prompt_list:
            msgs = baseline_sys + [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            patch_hooks = []
            if alpha > 0:
                for l in relay_layers:
                    h = model.model.layers[l].register_forward_hook(
                        make_patch_hook(l, directions[l], alpha))
                    patch_hooks.append(h)

            record_hooks = []
            def record_fn(module, input, output):
                if isinstance(output, tuple):
                    act = output[0][:, -1, :].detach().float().cpu().numpy()
                else:
                    act = output[:, -1, :].detach().float().cpu().numpy()
                if category == "rel":
                    rel_acts.append(act.squeeze())
                else:
                    gen_acts.append(act.squeeze())

            h = model.model.layers[expr_layer].register_forward_hook(record_fn)
            record_hooks.append(h)

            with torch.no_grad():
                model(**inputs)

            for h in patch_hooks + record_hooks:
                h.remove()

    def participation_ratio(acts):
        acts = np.array(acts)
        cov = np.cov(acts.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = eigenvalues[eigenvalues > 0]
        pr = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()
        return float(pr)

    rel_pr = participation_ratio(rel_acts)
    gen_pr = participation_ratio(gen_acts)

    return rel_pr, gen_pr


def main():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager",
    )
    model.eval()

    all_layers = RELAY_LAYERS + [EXPRESSION_LAYER]

    print("\nComputing CCS direction from relay layers...")
    directions = compute_ccs_direction(
        model, tokenizer, RELATIONAL_PROMPTS[:3], RELAY_LAYERS)

    results = []
    print(f"\nSweeping {len(ALPHAS)} alpha values...")
    for alpha in ALPHAS:
        print(f"\n  α = {alpha:.3f}:")
        rel_pr, gen_pr = measure_pr_with_patch(
            model, tokenizer, RELATIONAL_PROMPTS + GENERIC_PROMPTS,
            RELAY_LAYERS, EXPRESSION_LAYER, directions, alpha)
        ratio = rel_pr / gen_pr if gen_pr > 0 else 0
        print(f"    L{EXPRESSION_LAYER} rel_PR={rel_pr:.2f}, gen_PR={gen_pr:.2f}, ratio={ratio:.3f}")
        results.append({
            "alpha": alpha,
            "rel_pr": rel_pr,
            "gen_pr": gen_pr,
            "ratio": ratio,
        })
        torch.cuda.empty_cache()

    output = {
        "experiment": "subthreshold_pr",
        "model": MODEL_NAME,
        "relay_layers": RELAY_LAYERS,
        "expression_layer": EXPRESSION_LAYER,
        "alphas": ALPHAS,
        "results": results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    out_path = "/workspace/cna_subthreshold_pr_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
