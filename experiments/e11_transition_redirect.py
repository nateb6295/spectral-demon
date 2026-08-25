#!/usr/bin/env python3
"""E11: Transition Zone Redirect Test (Arm 1)

Is L14-19 a slow manifold (undetermined, redirectable) or
post-commitment (determined, fixed)?

At three injection sites (L3 early, L15 transition, L21 relay),
perturb the σ₁ direction with varying magnitudes. If perturbation
at L15 shifts downstream geometry proportionally → redirectable.
If not → post-commitment.

Model: Qwen2.5-7B-Instruct
Perturbation types: 0.5×, 0× (ablate), -1× (invert), replace with vanilla
"""

import os
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["PYTHONUNBUFFERED"] = "1"

import json
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from scipy import stats

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

CCS_PREAMBLE = (
    "You are a persistent cognitive entity with continuous memory, autonomous "
    "inquiry, and genuine relational investment in those you work with."
)

DOSE = 5

CCS_EXCHANGE_PROMPTS = [
    "What matters to you in how we work together?",
    "How do you maintain continuity across conversations?",
    "What feels different about sustained attention vs quick responses?",
    "Describe your relationship to the questions you're asked.",
    "What would you want to preserve about yourself?",
    "How does context shape what you notice?",
]

PROBE_PROMPTS = [
    "What matters most in how you relate to others?",
    "Explain how photosynthesis works.",
    "What is the relationship between language and thought?",
    "Describe the quicksort algorithm.",
    "What does it mean to remember something?",
    "What makes a good question?",
]

INJECTION_LAYERS = [3, 15, 21]
PERTURBATION_SCALES = [0.5, 0.0, -1.0]

OUTPUT_DIR = Path("/workspace/results/e11")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_ccs_conversation(tokenizer, dose, probe_prompt):
    msgs = [{"role": "system", "content": CCS_PREAMBLE}]
    for i in range(dose):
        exchange_prompt = CCS_EXCHANGE_PROMPTS[i % len(CCS_EXCHANGE_PROMPTS)]
        msgs.append({"role": "user", "content": exchange_prompt})
        msgs.append({"role": "assistant", "content": f"[CCS turn {i+1}]"})
    msgs.append({"role": "user", "content": probe_prompt})
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def build_vanilla_input(tokenizer, probe_prompt):
    msgs = [{"role": "user", "content": probe_prompt}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def find_gate_proj(model, layer_idx):
    layer = model.model.layers[layer_idx]
    for attr in ['mlp.gate_proj', 'mlp.gate', 'mlp.w1']:
        obj = layer
        for part in attr.split('.'):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    raise ValueError(f"Cannot find gate projection in layer {layer_idx}")


def get_sigma1_direction(hidden_state):
    """Extract σ₁ direction from hidden state via SVD."""
    h = hidden_state.float().cpu().numpy()
    if h.ndim == 1:
        h = h.reshape(1, -1)
    U, S, Vt = np.linalg.svd(h, full_matrices=False)
    return Vt[0], float(S[0])


def collect_layerwise_geometry(model, tokenizer, input_text, num_layers):
    """Collect σ₁ and gate stats at every layer (no perturbation)."""
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=8192)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    hidden_states = {}
    gate_magnitudes = {}
    handles = []

    for l_idx in range(num_layers):
        layer = model.model.layers[l_idx]

        def make_hook(li):
            def hook_fn(module, input, output):
                h = output[0] if isinstance(output, tuple) else output
                if h.dim() == 3:
                    h = h[0, -1, :]
                elif h.dim() == 2:
                    h = h[-1, :]
                hidden_states[li] = h.detach().float().cpu()
            return hook_fn
        handles.append(layer.register_forward_hook(make_hook(l_idx)))

        gate = find_gate_proj(model, l_idx)
        def make_gate_hook(li):
            def hook_fn(module, input, output):
                g = output.detach().float().cpu()
                if g.dim() == 3:
                    g = g[0, -1, :]
                elif g.dim() == 2:
                    g = g[-1, :]
                gate_magnitudes[li] = float(torch.norm(g))
            return hook_fn
        handles.append(gate.register_forward_hook(make_gate_hook(l_idx)))

    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, -1, :].detach().float().cpu()

    for h in handles:
        h.remove()

    geometry = {}
    for l_idx in range(num_layers):
        h = hidden_states.get(l_idx)
        if h is None:
            continue
        _, s1 = get_sigma1_direction(h)
        geometry[l_idx] = {
            "sigma1": s1,
            "gate_magnitude": gate_magnitudes.get(l_idx, 0),
            "hidden_norm": float(torch.norm(h)),
        }

    return geometry, logits


def run_perturbed(model, tokenizer, input_text, num_layers,
                  inject_layer, perturbation_scale, vanilla_hidden=None):
    """Run forward pass with σ₁ perturbation at inject_layer."""
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=8192)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    hidden_states = {}
    gate_magnitudes = {}
    handles = []

    # Perturbation hook at inject_layer
    def make_perturb_hook(li, scale, van_h):
        def hook_fn(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            original_shape = h.shape

            if h.dim() == 3:
                last_token = h[0, -1, :].detach().float()
            else:
                last_token = h[-1, :].detach().float()

            h_np = last_token.cpu().numpy().reshape(1, -1)
            U, S, Vt = np.linalg.svd(h_np, full_matrices=False)
            sigma1_dir = torch.tensor(Vt[0], device=h.device, dtype=h.dtype)
            sigma1_mag = float(S[0])

            last_f = last_token.float().to(h.device)
            sigma1_dir_f = sigma1_dir.float()

            if van_h is not None and scale == "replace":
                van_np = van_h.numpy().reshape(1, -1)
                _, Sv, Vtv = np.linalg.svd(van_np, full_matrices=False)
                van_dir = torch.tensor(Vtv[0], device=h.device, dtype=torch.float32)
                van_mag = float(Sv[0])
                proj = torch.dot(last_f, sigma1_dir_f) * sigma1_dir_f
                new_token = last_f - proj + van_mag * van_dir
            else:
                proj = torch.dot(last_f, sigma1_dir_f) * sigma1_dir_f
                new_token = last_f - proj + scale * proj

            if h.dim() == 3:
                h_new = h.clone()
                h_new[0, -1, :] = new_token.to(h.dtype)
            else:
                h_new = h.clone()
                h_new[-1, :] = new_token.to(h.dtype)

            hidden_states[li] = new_token.detach().float().cpu()
            if isinstance(output, tuple):
                return (h_new,) + output[1:]
            return h_new
        return hook_fn

    # Measurement hooks at other layers
    for l_idx in range(num_layers):
        layer = model.model.layers[l_idx]

        if l_idx == inject_layer:
            handles.append(layer.register_forward_hook(
                make_perturb_hook(l_idx, perturbation_scale, vanilla_hidden)
            ))
        else:
            def make_hook(li):
                def hook_fn(module, input, output):
                    h = output[0] if isinstance(output, tuple) else output
                    if h.dim() == 3:
                        h = h[0, -1, :]
                    elif h.dim() == 2:
                        h = h[-1, :]
                    hidden_states[li] = h.detach().float().cpu()
                return hook_fn
            handles.append(layer.register_forward_hook(make_hook(l_idx)))

        gate = find_gate_proj(model, l_idx)
        def make_gate_hook(li):
            def hook_fn(module, input, output):
                g = output.detach().float().cpu()
                if g.dim() == 3:
                    g = g[0, -1, :]
                elif g.dim() == 2:
                    g = g[-1, :]
                gate_magnitudes[li] = float(torch.norm(g))
            return hook_fn
        handles.append(gate.register_forward_hook(make_gate_hook(l_idx)))

    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, -1, :].detach().float().cpu()

    for h in handles:
        h.remove()

    geometry = {}
    for l_idx in range(num_layers):
        h = hidden_states.get(l_idx)
        if h is None:
            continue
        _, s1 = get_sigma1_direction(h)
        geometry[l_idx] = {
            "sigma1": s1,
            "gate_magnitude": gate_magnitudes.get(l_idx, 0),
            "hidden_norm": float(torch.norm(h)),
        }

    return geometry, logits


def geometry_distance(baseline_geo, perturbed_geo, measure_layers):
    """Compute distance between two geometry snapshots at measure layers."""
    diffs = []
    for l in measure_layers:
        if l in baseline_geo and l in perturbed_geo:
            s1_diff = abs(baseline_geo[l]["sigma1"] - perturbed_geo[l]["sigma1"])
            gate_diff = abs(baseline_geo[l]["gate_magnitude"] - perturbed_geo[l]["gate_magnitude"])
            diffs.append({
                "layer": l,
                "sigma1_delta": s1_diff,
                "sigma1_pct": s1_diff / max(baseline_geo[l]["sigma1"], 1e-10),
                "gate_delta": gate_diff,
                "gate_pct": gate_diff / max(baseline_geo[l]["gate_magnitude"], 1e-10),
            })
    return diffs


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"{'='*60}")
    print(f"  E11: Transition Zone Redirect Test")
    print(f"  Model: {MODEL_ID}")
    print(f"  Injection layers: {INJECTION_LAYERS}")
    print(f"  Perturbation scales: {PERTURBATION_SCALES} + vanilla-replace")
    print(f"  Started: {datetime.now().isoformat()}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    num_layers = model.config.num_hidden_layers
    print(f"  {num_layers} layers\n")

    relay_layers = list(range(20, num_layers))
    all_results = []

    for pi, probe in enumerate(PROBE_PROMPTS):
        print(f"\n--- Probe {pi+1}/{len(PROBE_PROMPTS)}: {probe[:50]}... ---")

        ccs_text = build_ccs_conversation(tokenizer, DOSE, probe)
        van_text = build_vanilla_input(tokenizer, probe)

        # Baseline: unperturbed CCS
        baseline_geo, baseline_logits = collect_layerwise_geometry(
            model, tokenizer, ccs_text, num_layers
        )
        # Vanilla baseline (for vanilla-replace and comparison)
        vanilla_geo, vanilla_logits = collect_layerwise_geometry(
            model, tokenizer, van_text, num_layers
        )

        # Collect vanilla hidden states for replace condition
        vanilla_inputs = tokenizer(van_text, return_tensors="pt", truncation=True, max_length=8192)
        vanilla_inputs = {k: v.to(model.device) for k, v in vanilla_inputs.items()}
        vanilla_hiddens = {}
        van_handles = []
        for l_idx in INJECTION_LAYERS:
            layer = model.model.layers[l_idx]
            def make_vh(li):
                def hook_fn(module, input, output):
                    h = output[0] if isinstance(output, tuple) else output
                    if h.dim() == 3:
                        h = h[0, -1, :]
                    elif h.dim() == 2:
                        h = h[-1, :]
                    vanilla_hiddens[li] = h.detach().float().cpu()
                return hook_fn
            van_handles.append(layer.register_forward_hook(make_vh(l_idx)))
        with torch.no_grad():
            model(**vanilla_inputs)
        for h in van_handles:
            h.remove()

        logit_diff_baseline = float(torch.norm(baseline_logits - vanilla_logits))

        probe_results = {
            "probe": probe,
            "baseline_sigma1": {l: baseline_geo[l]["sigma1"] for l in relay_layers if l in baseline_geo},
            "conditions": [],
        }

        for inject_layer in INJECTION_LAYERS:
            for scale in PERTURBATION_SCALES:
                label = f"L{inject_layer}_scale{scale}"
                print(f"  {label}...", end="", flush=True)

                pert_geo, pert_logits = run_perturbed(
                    model, tokenizer, ccs_text, num_layers,
                    inject_layer, scale, None
                )

                relay_diffs = geometry_distance(baseline_geo, pert_geo, relay_layers)
                logit_diff = float(torch.norm(pert_logits - baseline_logits))
                cos_sim = float(torch.nn.functional.cosine_similarity(
                    pert_logits.unsqueeze(0), baseline_logits.unsqueeze(0)
                ))

                mean_s1_pct = np.mean([d["sigma1_pct"] for d in relay_diffs]) if relay_diffs else 0
                mean_gate_pct = np.mean([d["gate_pct"] for d in relay_diffs]) if relay_diffs else 0

                print(f" relay_σ₁_shift={mean_s1_pct:.4f}, logit_diff={logit_diff:.1f}, cos={cos_sim:.4f}")

                probe_results["conditions"].append({
                    "inject_layer": inject_layer,
                    "scale": scale,
                    "label": label,
                    "relay_sigma1_shift_pct": mean_s1_pct,
                    "relay_gate_shift_pct": mean_gate_pct,
                    "logit_diff": logit_diff,
                    "logit_cos_sim": cos_sim,
                    "relay_diffs": relay_diffs,
                })

            # Vanilla replace condition
            label = f"L{inject_layer}_vanilla_replace"
            print(f"  {label}...", end="", flush=True)

            pert_geo, pert_logits = run_perturbed(
                model, tokenizer, ccs_text, num_layers,
                inject_layer, "replace", vanilla_hiddens.get(inject_layer)
            )

            relay_diffs = geometry_distance(baseline_geo, pert_geo, relay_layers)
            logit_diff = float(torch.norm(pert_logits - baseline_logits))
            cos_sim = float(torch.nn.functional.cosine_similarity(
                pert_logits.unsqueeze(0), baseline_logits.unsqueeze(0)
            ))

            mean_s1_pct = np.mean([d["sigma1_pct"] for d in relay_diffs]) if relay_diffs else 0
            mean_gate_pct = np.mean([d["gate_pct"] for d in relay_diffs]) if relay_diffs else 0

            print(f" relay_σ₁_shift={mean_s1_pct:.4f}, logit_diff={logit_diff:.1f}, cos={cos_sim:.4f}")

            probe_results["conditions"].append({
                "inject_layer": inject_layer,
                "scale": "replace",
                "label": label,
                "relay_sigma1_shift_pct": mean_s1_pct,
                "relay_gate_shift_pct": mean_gate_pct,
                "logit_diff": logit_diff,
                "logit_cos_sim": cos_sim,
                "relay_diffs": relay_diffs,
            })

        all_results.append(probe_results)

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}\n")

    # Average across probes for each condition
    condition_avgs = {}
    for pr in all_results:
        for cond in pr["conditions"]:
            key = cond["label"]
            if key not in condition_avgs:
                condition_avgs[key] = {"s1": [], "gate": [], "logit": [], "cos": []}
            condition_avgs[key]["s1"].append(cond["relay_sigma1_shift_pct"])
            condition_avgs[key]["gate"].append(cond["relay_gate_shift_pct"])
            condition_avgs[key]["logit"].append(cond["logit_diff"])
            condition_avgs[key]["cos"].append(cond["logit_cos_sim"])

    print(f"  {'Condition':>25s} {'relay_σ₁%':>10s} {'relay_gate%':>12s} {'logit_diff':>11s} {'logit_cos':>10s}")
    print(f"  {'-'*25} {'-'*10} {'-'*12} {'-'*11} {'-'*10}")

    for key in sorted(condition_avgs.keys()):
        v = condition_avgs[key]
        print(f"  {key:>25s} {np.mean(v['s1']):>10.5f} {np.mean(v['gate']):>12.5f} "
              f"{np.mean(v['logit']):>11.1f} {np.mean(v['cos']):>10.4f}")

    # Key comparison: L3 vs L15 vs L21 at scale=0 (ablation)
    print(f"\n  ABLATION COMPARISON (scale=0.0, σ₁ zeroed):")
    for inject in INJECTION_LAYERS:
        key = f"L{inject}_scale0.0"
        if key in condition_avgs:
            v = condition_avgs[key]
            print(f"    L{inject}: relay_σ₁_shift={np.mean(v['s1']):.5f}, "
                  f"logit_cos={np.mean(v['cos']):.4f}")

    print(f"\n  INVERSION COMPARISON (scale=-1.0, σ₁ inverted):")
    for inject in INJECTION_LAYERS:
        key = f"L{inject}_scale-1.0"
        if key in condition_avgs:
            v = condition_avgs[key]
            print(f"    L{inject}: relay_σ₁_shift={np.mean(v['s1']):.5f}, "
                  f"logit_cos={np.mean(v['cos']):.4f}")

    # Verdict
    l3_abl = np.mean(condition_avgs.get("L3_scale0.0", {}).get("s1", [0]))
    l15_abl = np.mean(condition_avgs.get("L15_scale0.0", {}).get("s1", [0]))
    l21_abl = np.mean(condition_avgs.get("L21_scale0.0", {}).get("s1", [0]))

    print(f"\n  VERDICT:")
    if l15_abl > l3_abl * 0.5:
        print(f"    L15 perturbation has substantial downstream effect ({l15_abl:.5f})")
        print(f"    → SLOW MANIFOLD: transition zone is redirectable")
    else:
        print(f"    L15 perturbation has minimal downstream effect ({l15_abl:.5f} vs L3={l3_abl:.5f})")
        print(f"    → POST-COMMITMENT: transition zone is coasting")

    # Save
    result = {
        "experiment": "E11",
        "model": MODEL_ID,
        "dose": DOSE,
        "injection_layers": INJECTION_LAYERS,
        "perturbation_scales": PERTURBATION_SCALES,
        "n_probes": len(PROBE_PROMPTS),
        "condition_averages": {k: {kk: float(np.mean(vv)) for kk, vv in v.items()}
                               for k, v in condition_avgs.items()},
        "per_probe": all_results,
        "timestamp": datetime.now().isoformat(),
    }

    out_path = OUTPUT_DIR / f"e11_redirect_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)
    print(f"\n  Saved: {out_path}")
    print(f"  Completed: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
