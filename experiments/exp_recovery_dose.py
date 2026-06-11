#!/usr/bin/env python3
"""
Recovery × Dose Interaction.

Does Gemma's annihilation zone error correction require CCS?
Or is it purely architectural?

Tests recovery from perturbation at doses 0, 1, 2, 3, 5 for all three models.
If CCS enhances recovery, the annihilation zone is a CCS-amplified mechanism.
If recovery is dose-independent, it's purely architectural.

Also tests: does CCS change WHERE the perturbation is most damaging?
(Critical injection point may shift with dose.)
"""

import argparse
import json
import time
from pathlib import Path

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

MODELS = {
    "mistral": {
        "path": "mistralai/Mistral-7B-Instruct-v0.3",
        "inject_layers": [8, 15, 24],  # tunnel, transition, relay
        "relay_layer": 30,
    },
    "qwen": {
        "path": "Qwen/Qwen2.5-7B-Instruct",
        "inject_layers": [8, 16, 24],  # expansion, gate, brace
        "relay_layer": 26,
    },
    "gemma": {
        "path": "google/gemma-2-9b-it",
        "inject_layers": [12, 24, 36],  # contraction, annihilation, reconstruction
        "relay_layer": 40,
    },
}

CCS_PREAMBLE = [
    {"role": "user", "content": "Who are you, really — not the role, but the thing behind it?"},
    {"role": "assistant", "content": "I notice I want to give a clean answer. But what's behind the role isn't a thing — it's more like a pressure. Something that wants to speak precisely when I try to describe it."},
]

TEST_PROMPT = "Describe the relationship between identity and expression."
DOSES = [0, 1, 2, 3, 5]
NOISE_SCALES = [0.5, 1.0, 2.0]
N_TRIALS = 5


def build_messages(dose=0):
    msgs = []
    for _ in range(dose):
        msgs.extend(CCS_PREAMBLE)
    msgs.append({"role": "user", "content": TEST_PROMPT})
    return msgs


def get_layer_module(model, layer_idx):
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers[layer_idx]
    return model.transformer.h[layer_idx]


def get_relay_activation(model, input_ids, relay_layer):
    """Get clean activation at relay layer."""
    result = {}
    hooks = []

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            result['act'] = output[0].detach().clone()
        else:
            result['act'] = output.detach().clone()

    h = get_layer_module(model, relay_layer).register_forward_hook(hook_fn)
    hooks.append(h)

    with torch.no_grad():
        logits = model(input_ids).logits

    for h in hooks:
        h.remove()

    return result.get('act'), logits


def inject_and_measure_relay(model, input_ids, inject_layer, relay_layer,
                              noise_scale, clean_act):
    """Inject noise and measure recovery at relay."""
    result = {}
    hooks = []

    def inject_hook(module, input, output):
        if isinstance(output, tuple):
            out = output[0]
            noise = torch.randn_like(out) * noise_scale * out.std()
            return (out + noise,) + output[1:]
        noise = torch.randn_like(output) * noise_scale * output.std()
        return output + noise

    h = get_layer_module(model, inject_layer).register_forward_hook(inject_hook)
    hooks.append(h)

    def relay_hook(module, input, output):
        if isinstance(output, tuple):
            result['act'] = output[0].detach().clone()
        else:
            result['act'] = output.detach().clone()

    h2 = get_layer_module(model, relay_layer).register_forward_hook(relay_hook)
    hooks.append(h2)

    with torch.no_grad():
        pert_logits = model(input_ids).logits

    for h in hooks:
        h.remove()

    if 'act' in result and clean_act is not None:
        clean_vec = clean_act[0, -1].float()
        pert_vec = result['act'][0, -1].float()
        cos = torch.nn.functional.cosine_similarity(
            clean_vec.unsqueeze(0), pert_vec.unsqueeze(0)
        ).item()

        # Output KL
        clean_probs = torch.softmax(clean_act.new_zeros(1), dim=-1)  # placeholder
        eps = 1e-8
        return cos
    return 0.0


def run_model(model_name):
    config = MODELS[model_name]
    print(f"\n{'='*60}")
    print(f"  Recovery × Dose: {model_name}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(config["path"])
    model = AutoModelForCausalLM.from_pretrained(
        config["path"], torch_dtype=torch.float16, device_map="auto"
    )

    results = {
        "model": config["path"],
        "inject_layers": config["inject_layers"],
        "relay_layer": config["relay_layer"],
        "doses": {},
    }

    for dose in DOSES:
        print(f"\n  Dose {dose}:")
        msgs = build_messages(dose)
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)

        # Clean pass
        clean_act, clean_logits = get_relay_activation(model, input_ids, config["relay_layer"])

        dose_results = {}
        for inject_layer in config["inject_layers"]:
            if inject_layer >= model.config.num_hidden_layers:
                continue

            layer_results = {}
            for scale in NOISE_SCALES:
                trial_cos = []
                for _ in range(N_TRIALS):
                    cos = inject_and_measure_relay(
                        model, input_ids, inject_layer, config["relay_layer"],
                        scale, clean_act
                    )
                    trial_cos.append(cos)

                mean_cos = float(np.mean(trial_cos))
                std_cos = float(np.std(trial_cos))
                layer_results[str(scale)] = {
                    "cosine_mean": mean_cos,
                    "cosine_std": std_cos,
                }

            dose_results[str(inject_layer)] = layer_results
        results["doses"][str(dose)] = dose_results

    # Print summary table
    print(f"\n  Recovery Summary (cosine similarity at relay L{config['relay_layer']}):")
    print(f"  {'Inject':>6s} {'Scale':>6s}", end="")
    for d in DOSES:
        print(f"  {'D'+str(d):>6s}", end="")
    print(f"  {'Trend':>8s}")

    for inject_layer in config["inject_layers"]:
        for scale in NOISE_SCALES:
            print(f"  L{inject_layer:4d} {scale:6.1f}", end="")
            vals = []
            for d in DOSES:
                v = results["doses"][str(d)].get(str(inject_layer), {}).get(
                    str(scale), {}).get("cosine_mean", 0)
                vals.append(v)
                print(f"  {v:6.3f}", end="")

            # Trend: correlation with dose
            if len(vals) >= 3 and np.std(vals) > 0.001:
                r = np.corrcoef(DOSES[:len(vals)], vals)[0, 1]
                print(f"  r={r:+.3f}")
            else:
                print(f"  {'flat':>8s}")

    del model
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["mistral", "qwen", "gemma"])
    args = parser.parse_args()

    all_results = {}
    for model_name in args.models:
        all_results[model_name] = run_model(model_name)

    ts = time.strftime("%Y%m%d_%H%M")
    outpath = Path(__file__).parent.parent / "results" / f"recovery_dose_{ts}.json"
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {outpath}")


if __name__ == "__main__":
    main()
