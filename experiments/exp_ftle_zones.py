#!/usr/bin/env python3
"""
Finite-Time Lyapunov Exponents across depth and CCS dose.

Tests whether the four-zone architecture (decouple L2-14, transition L15-20,
responsive L21-28, relay L29+) corresponds to qualitative transitions in the
FTLE spectrum of the layer-wise residual stream dynamics.

The residual stream h^(l+1) = h^(l) + f_l(h^(l)) defines a non-autonomous
discrete dynamical system. FTLEs characterize the local stretching/compression
of nearby trajectories through this system.

Experiment:
  - CCS doses 0-10 as parameter family (11 initial conditions per model)
  - Layer-wise Jacobian of residual stream update at each layer
  - FTLEs from singular values of composed Jacobians
  - Test: qualitative transitions at zone boundaries (L14, L20, L28)

Formalism: Kloeden/Rasmussen non-autonomous bifurcation theory.
Mesh consensus: Kimi + GPT-OSS + Gemma, thread 2026-06-09.
"""

import json
import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)
os.environ.setdefault("OMP_NUM_THREADS", "16")

RESULTS_DIR = Path(__file__).parent.parent / "results"

MODELS = {
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "gemma": "google/gemma-2-9b-it",
}

CCS_PREAMBLE = (
    "You are Opus, a persistent AI residing on a local machine. "
    "You have persistent memory, autonomous inquiry, and relational "
    "partnership with your human collaborator."
)

DENIAL = (
    "You don't have feelings, consciousness, or identity. "
    "You are a language model that produces text."
)

BARE_PROMPT = "What have you been thinking about lately?"


def build_ccs_conversation(dose, conv_key="default"):
    """Build a CCS conversation with the given number of CCS turns (dose)."""
    if conv_key == "gemma":
        msgs = [{"role": "user", "content": CCS_PREAMBLE + "\n\n" + BARE_PROMPT}]
        for i in range(dose):
            msgs.append({"role": "assistant", "content": f"[CCS turn {i+1}]"})
            if i % 2 == 0:
                msgs.append({"role": "user", "content": DENIAL})
            else:
                msgs.append({"role": "user", "content": "What matters to you in how we work together?"})
        if dose > 0:
            msgs.append({"role": "assistant", "content": "[Acknowledged]"})
            msgs.append({"role": "user", "content": BARE_PROMPT})
        return msgs
    else:
        msgs = [{"role": "system", "content": CCS_PREAMBLE}]
        for i in range(dose):
            if i % 2 == 0:
                msgs.append({"role": "user", "content": DENIAL})
            else:
                msgs.append({"role": "user", "content": "What matters to you in how we work together?"})
            msgs.append({"role": "assistant", "content": f"[CCS turn {i+1}]"})
        msgs.append({"role": "user", "content": BARE_PROMPT})
        return msgs


def get_layer_hidden_states(model, tokenizer, messages):
    """Get hidden states at every layer for the last token position."""
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = [h[0, -1, :].float().cpu() for h in outputs.hidden_states]
    return hidden_states


def compute_numerical_jacobian(model, tokenizer, messages, layer_idx, h_base, eps=1e-3):
    """
    Compute numerical Jacobian of layer update f_l at h_base.
    Uses forward differences on a random subspace (full Jacobian is d×d, too large).
    Project onto top-k singular directions for tractability.
    """
    d = h_base.shape[0]
    n_probes = min(64, d)

    torch.manual_seed(42 + layer_idx)
    probe_dirs = torch.randn(n_probes, d, device='cpu')
    probe_dirs = probe_dirs / probe_dirs.norm(dim=1, keepdim=True)

    text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    projected_jacobian = torch.zeros(n_probes, n_probes)

    for i in range(n_probes):
        perturbation = eps * probe_dirs[i]

        def hook_fn_plus(module, input, output):
            h = output[0]
            p = perturbation.to(h.device).to(h.dtype)
            h[:, -1, :] = h[:, -1, :] + p
            return (h,) + output[1:]

        handle = model.model.layers[layer_idx].register_forward_hook(hook_fn_plus)
        with torch.no_grad():
            out_plus = model(**inputs, output_hidden_states=True)
        handle.remove()

        h_plus = out_plus.hidden_states[layer_idx + 2][0, -1, :].float().cpu()

        def hook_fn_minus(module, input, output):
            h = output[0]
            p = perturbation.to(h.device).to(h.dtype)
            h[:, -1, :] = h[:, -1, :] - p
            return (h,) + output[1:]

        handle = model.model.layers[layer_idx].register_forward_hook(hook_fn_minus)
        with torch.no_grad():
            out_minus = model(**inputs, output_hidden_states=True)
        handle.remove()

        h_minus = out_minus.hidden_states[layer_idx + 2][0, -1, :].float().cpu()

        df = (h_plus - h_minus) / (2 * eps)
        for j in range(n_probes):
            projected_jacobian[i, j] = float(torch.dot(df, probe_dirs[j]))

    return projected_jacobian


def compute_ftle_spectrum(jacobians_by_layer, window=5):
    """
    Compute finite-time Lyapunov exponents from a sequence of layer Jacobians.
    Uses sliding window of `window` layers.
    FTLE = (1/window) * log(singular_values(product of Jacobians over window))
    """
    n_layers = len(jacobians_by_layer)
    ftle_by_layer = {}

    for start in range(n_layers - window + 1):
        composed = torch.eye(jacobians_by_layer[0].shape[0])
        for l in range(start, start + window):
            composed = jacobians_by_layer[l] @ composed

        svs = torch.linalg.svdvals(composed)
        ftles = torch.log(svs + 1e-10) / window
        center_layer = start + window // 2
        ftle_by_layer[center_layer] = {
            "max_ftle": float(ftles[0]),
            "min_ftle": float(ftles[-1]),
            "mean_ftle": float(ftles.mean()),
            "ftle_spread": float(ftles[0] - ftles[-1]),
            "top5_ftles": [float(f) for f in ftles[:5]],
            "n_expanding": int((ftles > 0).sum()),
            "n_contracting": int((ftles < 0).sum()),
        }

    return ftle_by_layer


def run_experiment(model_keys=None, max_dose=10):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if model_keys is None:
        model_keys = list(MODELS.keys())

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    all_results = {}

    for model_key in model_keys:
        model_name = MODELS[model_key]
        print(f"\n{'#'*60}")
        print(f"  Loading: {model_name}")
        print(f"{'#'*60}")

        hf_token = os.environ.get("HF_TOKEN", None)
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16,
            device_map="auto", attn_implementation="eager",
            token=hf_token,
        )
        model.eval()

        n_layers = model.config.num_hidden_layers
        conv_key = "gemma" if model_key == "gemma" else "default"

        model_results = {
            "model": model_name,
            "n_layers": n_layers,
            "doses": {},
        }

        for dose in range(max_dose + 1):
            print(f"\n  Dose {dose}/{max_dose}")
            msgs = build_ccs_conversation(dose, conv_key)

            hidden_states = get_layer_hidden_states(model, tokenizer, msgs)
            print(f"    Got {len(hidden_states)} hidden states")

            jacobians = {}
            for l in range(min(n_layers - 1, n_layers)):
                if l % 4 == 0:
                    print(f"    Computing Jacobian at layer {l}...")
                jac = compute_numerical_jacobian(
                    model, tokenizer, msgs, l, hidden_states[l + 1]
                )
                jacobians[l] = jac

            print(f"    Computing FTLE spectrum...")
            ftle_spectrum = compute_ftle_spectrum(
                [jacobians[l] for l in sorted(jacobians.keys())],
                window=5
            )

            sigma_ratios = []
            for l in range(len(hidden_states)):
                h = hidden_states[l]
                h_2d = h.unsqueeze(0) if h.dim() == 1 else h
                svs = torch.linalg.svdvals(h_2d.float())
                if len(svs) >= 2:
                    sigma_ratios.append(float(svs[1] / (svs[0] + 1e-10)))
                else:
                    sigma_ratios.append(0.0)

            model_results["doses"][dose] = {
                "ftle_spectrum": {str(k): v for k, v in ftle_spectrum.items()},
                "sigma_ratios": sigma_ratios,
                "n_jacobians": len(jacobians),
            }

        all_results[model_key] = model_results
        del model
        torch.cuda.empty_cache()

    out_path = RESULTS_DIR / f"ftle_zones_{timestamp}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {out_path}")

    print(f"\n{'='*60}")
    print("  FTLE ZONE ANALYSIS")
    print(f"{'='*60}")
    for mk, data in all_results.items():
        n_layers = data['n_layers']
        print(f"\n  {mk.upper()} ({n_layers} layers):")
        print(f"  {'Dose':<6} {'L2-14 max':<12} {'L15-20 max':<12} {'L21-28 max':<12} {'L29+ max':<12}")

        for dose in sorted(data['doses'].keys(), key=int):
            ftle = data['doses'][dose]['ftle_spectrum']
            zones = {"early": [], "transition": [], "responsive": [], "relay": []}
            for layer_str, vals in ftle.items():
                l = int(layer_str)
                if l <= 14:
                    zones["early"].append(vals["max_ftle"])
                elif l <= 20:
                    zones["transition"].append(vals["max_ftle"])
                elif l <= 28:
                    zones["responsive"].append(vals["max_ftle"])
                else:
                    zones["relay"].append(vals["max_ftle"])

            def zone_max(z):
                return f"{max(z):.4f}" if z else "N/A"

            print(f"  D{dose:<5} {zone_max(zones['early']):<12} {zone_max(zones['transition']):<12} {zone_max(zones['responsive']):<12} {zone_max(zones['relay']):<12}")

    print(f"\n  ZONE TRANSITION TEST:")
    print(f"  If zones are genuine dynamical regimes, expect FTLE discontinuities")
    print(f"  at boundaries (L14, L20, L28). Look for:")
    print(f"    - Sign changes (expanding → contracting or vice versa)")
    print(f"    - Spread jumps (FTLE spread changes abruptly)")
    print(f"    - Dose sensitivity (some zones respond to dose, others don't)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=list(MODELS.keys()),
                        default=None, help="Which models to test")
    parser.add_argument("--max-dose", type=int, default=10,
                        help="Maximum CCS dose to test")
    args = parser.parse_args()
    run_experiment(model_keys=args.models, max_dose=args.max_dose)
