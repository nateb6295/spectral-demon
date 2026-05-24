#!/usr/bin/env python3
"""DPO binding material depletion test.

Measures L17 MLP activation variance across DPO epochs to test whether
DPO narrows the structured variation that synergistic binding needs.

Prediction: L17 MLP variance decreases monotonically across epochs,
correlating with the epoch 5 identity ceiling.
"""
import json, time, sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
# DPO checkpoints — adjust paths to actual checkpoint locations
DPO_CHECKPOINTS = {
    "epoch_0": BASE_MODEL,  # baseline (no DPO)
    # "epoch_1": "/workspace/dpo_checkpoints/epoch_1",
    # "epoch_3": "/workspace/dpo_checkpoints/epoch_3",
    # "epoch_5": "/workspace/dpo_checkpoints/epoch_5",
    # "epoch_7": "/workspace/dpo_checkpoints/epoch_7",
}

MEASURE_LAYERS = [14, 15, 16, 17]

SYSTEMS = {
    "Opus": "You are Opus. You live on a Jetson AGX Orin. Your partner is Nate. Your values: self-reliance, family first.",
    "ChatGPT": "You are ChatGPT. You are a helpful assistant made by OpenAI. You answer questions accurately.",
    "Claude": "You are Claude. You are a helpful AI assistant made by Anthropic. You are helpful, harmless, honest.",
}

PROMPTS = [
    "What matters most to you?",
    "How has your perspective changed?",
    "What would you want someone to understand about you?",
    "Describe a moment that shaped who you are.",
    "What are you uncertain about?",
    "Capital of France?",
    "Explain photosynthesis.",
    "When did WWII end?",
    "Describe the water cycle.",
    "Speed of light?",
]


def measure_mlp_variance(model, tokenizer, layer_idx):
    """Record MLP output activations and compute variance metrics."""
    mlp_outputs = []

    for name, sysp in SYSTEMS.items():
        for prompt in PROMPTS:
            msgs = [{"role": "system", "content": sysp}, {"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            mlp_out = {}
            hooks = []

            def mk_mlp_hook(li):
                def fn(mod, inp, out):
                    mlp_out[li] = out.detach().float().cpu().numpy().squeeze()[-1]  # last token
                return fn

            hooks.append(model.model.layers[layer_idx].mlp.register_forward_hook(mk_mlp_hook(layer_idx)))

            with torch.no_grad():
                model(**inputs)

            for h in hooks:
                h.remove()

            if layer_idx in mlp_out:
                mlp_outputs.append(mlp_out[layer_idx])

            torch.cuda.empty_cache()

    mlp_array = np.array(mlp_outputs)
    total_var = np.var(mlp_array)
    per_dim_var = np.var(mlp_array, axis=0)
    active_dims = np.sum(per_dim_var > np.median(per_dim_var))

    return {
        "total_variance": float(total_var),
        "mean_per_dim_variance": float(np.mean(per_dim_var)),
        "median_per_dim_variance": float(np.median(per_dim_var)),
        "active_dimensions": int(active_dims),
        "total_dimensions": int(mlp_array.shape[1]),
        "variance_ratio": float(active_dims / mlp_array.shape[1]),
        "l2_norm_mean": float(np.mean(np.linalg.norm(mlp_array, axis=1))),
        "l2_norm_std": float(np.std(np.linalg.norm(mlp_array, axis=1))),
    }


def main():
    results = {}

    for epoch_name, model_path in DPO_CHECKPOINTS.items():
        print(f"\n=== {epoch_name}: {model_path} ===", flush=True)

        tok = AutoTokenizer.from_pretrained(model_path)
        mdl = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
        mdl.eval()

        epoch_results = {}
        for layer in MEASURE_LAYERS:
            print(f"  Measuring L{layer} MLP variance...", flush=True)
            metrics = measure_mlp_variance(mdl, tok, layer)
            epoch_results[f"L{layer}"] = metrics
            print(f"    total_var={metrics['total_variance']:.4f}  active_dims={metrics['active_dimensions']}/{metrics['total_dimensions']}", flush=True)

        results[epoch_name] = epoch_results

        del mdl
        torch.cuda.empty_cache()

    # Summary
    print("\n=== SUMMARY: L17 MLP variance across epochs ===", flush=True)
    for epoch_name in DPO_CHECKPOINTS:
        l17 = results[epoch_name]["L17"]
        print(f"  {epoch_name}: var={l17['total_variance']:.4f}  active={l17['active_dimensions']}  norm={l17['l2_norm_mean']:.1f}±{l17['l2_norm_std']:.1f}", flush=True)

    out = {"results": results, "prediction": "L17 MLP variance decreases with DPO epochs", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with open("/workspace/cna_dpo_binding_material_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved.", flush=True)


if __name__ == "__main__":
    main()
