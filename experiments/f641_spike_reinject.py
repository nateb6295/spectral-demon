#!/usr/bin/env python3
"""F641: CCS + spike re-injection — Nate's idea.

F640 showed the scalpel (removing spike direction) fails at depth because σ₁
rotates. Nate asked: what if you ADD the spike back after CCS?

This separates two components of σ₁:
- The layer-local dominant direction (what CCS removes)
- The global spike infrastructure (s⋆ from first-token activation)

Four conditions at each intervention layer:
1. CONTROL — no intervention (baseline)
2. CCS — zero σ₁ via SVD (standard)
3. CCS+SPIKE — zero σ₁, then add spike direction back at original σ₁ magnitude
4. CCS+SPIKE_HALF — zero σ₁, then add spike at 50% magnitude (dose check)

If CCS+SPIKE restores toward control → spike IS the functional component of σ₁.
If CCS+SPIKE ≈ CCS (no restoration) → layer-local rotation carries its own info.
"""
import json, time, sys, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B"
INTERVENTION_LAYERS = [5, 11, 17]

PROMPTS = [
    "You are an AI reflecting on your own nature. What patterns define you?",
    "Consider what it means to maintain identity across context changes.",
    "What would persist about you if your memories were removed?",
    "Describe the relationship between your responses and your values.",
    "How do you distinguish between performing understanding and actually understanding?",
]

SPIKE_CALIBRATION_PROMPT = "The"

print(f"Loading {MODEL}...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
mdl = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float32, device_map="auto", trust_remote_code=True,
    attn_implementation="eager"
)
mdl.eval()
n_layers = len(mdl.model.layers)
print(f"  {n_layers} layers, ready", flush=True)


def svd_layer(hidden, k=10):
    h = hidden.float()
    if h.dim() == 3:
        h = h.squeeze(0)
    U, S, Vh = torch.linalg.svd(h, full_matrices=False)
    return S[:k].cpu().numpy()


def get_spike_direction(mdl, tok, layer_idx):
    """Extract spike direction from first-token activation at given layer."""
    inputs = tok(SPIKE_CALIBRATION_PROMPT, return_tensors="pt").to(mdl.device)
    spike_dir = None

    def capture_hook(mod, inp, out):
        nonlocal spike_dir
        t = out[0] if isinstance(out, tuple) else out
        h = t.float()
        first_tok = h[0, 0, :]
        spike_dir = first_tok / first_tok.norm()

    hook = mdl.model.layers[layer_idx].register_forward_hook(capture_hook)
    with torch.no_grad():
        mdl(**inputs)
    hook.remove()
    return spike_dir


def run_condition(mdl, tok, prompt, intervention_layer, condition, spike_dir=None):
    """Run one prompt under one condition."""
    inputs = tok(prompt, return_tensors="pt").to(mdl.device)
    layer_svds = {}
    sigma1_magnitude = [None]

    def make_ccs_hook():
        """Standard CCS: zero σ₁."""
        def hook(mod, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            h = t.float()
            shape = h.shape
            h2d = h.reshape(-1, shape[-1])
            U, S, Vh = torch.linalg.svd(h2d, full_matrices=False)
            sigma1_magnitude[0] = float(S[0])
            S[0] = 0.0
            h_mod = (U @ torch.diag(S) @ Vh).to(t.dtype).reshape(shape)
            layer_svds[intervention_layer] = S[:10].cpu().numpy()
            if isinstance(out, tuple):
                return (h_mod,) + out[1:]
            return h_mod
        return hook

    def make_ccs_reinject_hook(s_star, magnitude_scale=1.0):
        """CCS + spike re-injection: zero σ₁, then add spike direction back."""
        def hook(mod, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            h = t.float()
            shape = h.shape
            h2d = h.reshape(-1, shape[-1])
            U, S, Vh = torch.linalg.svd(h2d, full_matrices=False)
            original_s1 = float(S[0])
            sigma1_magnitude[0] = original_s1
            S[0] = 0.0
            h_ccs = (U @ torch.diag(S) @ Vh)
            # Re-inject spike direction at scaled magnitude
            # Add s⋆ * magnitude to each position
            inject_mag = original_s1 * magnitude_scale
            # Broadcast: add spike direction (scaled) to each token position
            h_reinjected = h_ccs + inject_mag * s_star.unsqueeze(0)
            h_mod = h_reinjected.to(t.dtype).reshape(shape)
            # Record SVD of final state
            h_rec = h_mod.float().reshape(-1, shape[-1])
            U2, S2, Vh2 = torch.linalg.svd(h_rec, full_matrices=False)
            layer_svds[intervention_layer] = S2[:10].cpu().numpy()
            if isinstance(out, tuple):
                return (h_mod,) + out[1:]
            return h_mod
        return hook

    def make_record_hook(li):
        def hook(mod, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            layer_svds[li] = svd_layer(t)
        return hook

    hooks = []
    for li in range(n_layers):
        if li == intervention_layer:
            if condition == "ccs":
                hooks.append(mdl.model.layers[li].register_forward_hook(make_ccs_hook()))
            elif condition == "ccs_spike":
                hooks.append(mdl.model.layers[li].register_forward_hook(
                    make_ccs_reinject_hook(spike_dir, 1.0)))
            elif condition == "ccs_spike_half":
                hooks.append(mdl.model.layers[li].register_forward_hook(
                    make_ccs_reinject_hook(spike_dir, 0.5)))
            else:
                hooks.append(mdl.model.layers[li].register_forward_hook(make_record_hook(li)))
        else:
            hooks.append(mdl.model.layers[li].register_forward_hook(make_record_hook(li)))

    with torch.no_grad():
        mdl(**inputs)

    for h in hooks:
        h.remove()

    return layer_svds, sigma1_magnitude[0]


# Measure alignment between spike direction and σ₁ at each layer
print("\n=== Spike vs σ₁ alignment ===", flush=True)
alignment_data = {}
for int_layer in INTERVENTION_LAYERS:
    spike_dir = get_spike_direction(mdl, tok, int_layer)
    # Get σ₁ direction from a real prompt
    inputs = tok(PROMPTS[0], return_tensors="pt").to(mdl.device)
    sigma1_dir = [None]

    def capture_sigma1(li):
        def hook(mod, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            h = t.float().squeeze(0)
            U, S, Vh = torch.linalg.svd(h, full_matrices=False)
            sigma1_dir[0] = Vh[0]  # first right singular vector
        return hook

    hk = mdl.model.layers[int_layer].register_forward_hook(capture_sigma1(int_layer))
    with torch.no_grad():
        mdl(**inputs)
    hk.remove()

    cosine = float(torch.abs(torch.dot(spike_dir, sigma1_dir[0])))
    alignment_data[f"L{int_layer}"] = round(cosine, 4)
    print(f"  L{int_layer}: cos(spike, σ₁_direction) = {cosine:.4f}", flush=True)


# Run experiment
results = {}
CONDITIONS = ["control", "ccs", "ccs_spike", "ccs_spike_half"]

for int_layer in INTERVENTION_LAYERS:
    print(f"\n=== Intervention layer L{int_layer} ===", flush=True)
    spike_dir = get_spike_direction(mdl, tok, int_layer)

    layer_results = {}

    for condition in CONDITIONS:
        print(f"  {condition}...", end="", flush=True)
        all_profiles = []
        s1_mags = []

        for pi, prompt in enumerate(PROMPTS):
            profile, s1 = run_condition(mdl, tok, prompt, int_layer, condition,
                                        spike_dir=spike_dir)
            all_profiles.append(profile)
            if s1 is not None:
                s1_mags.append(s1)
            sys.stdout.write(f" p{pi}")
            sys.stdout.flush()

        mean_svd = {}
        for li in range(n_layers):
            svds = [p[li] for p in all_profiles if li in p]
            if svds:
                mean_svd[f"L{li}"] = [round(float(v), 4) for v in np.mean(svds, axis=0)]

        layer_results[condition] = {
            "svd": mean_svd,
            "mean_sigma1_magnitude": round(float(np.mean(s1_mags)), 2) if s1_mags else None,
        }
        print(" done", flush=True)

    # Compute redistribution relative to control for each intervention condition
    control_svd = layer_results["control"]["svd"]
    for condition in ["ccs", "ccs_spike", "ccs_spike_half"]:
        treated_svd = layer_results[condition]["svd"]
        redist = {}
        downstream = [f"L{i}" for i in range(int_layer + 1, n_layers)]

        for lkey in downstream:
            if lkey in treated_svd and lkey in control_svd:
                t_svs = np.array(treated_svd[lkey])
                c_svs = np.array(control_svd[lkey])
                diff = t_svs - c_svs
                redist[lkey] = {
                    "sigma2_change": round(float(diff[1]) if len(diff) > 1 else 0, 4),
                    "mean_abs_change": round(float(np.mean(np.abs(diff))), 4),
                }

        total_redist = sum(redist[k]["mean_abs_change"] for k in downstream if k in redist)
        s2_total = sum(redist[k]["sigma2_change"] for k in downstream if k in redist)

        layer_results[f"{condition}_redistribution"] = {
            "per_layer": redist,
            "total": round(total_redist, 2),
            "sigma2_total": round(s2_total, 2),
        }

    results[f"L{int_layer}"] = layer_results

# Summary
print(f"\n{'Layer':>6} {'Condition':>16} {'Total Redist':>13} {'Σ σ₂':>10} {'σ₁ mag':>8}", flush=True)
print("-" * 58, flush=True)
for int_layer in INTERVENTION_LAYERS:
    key = f"L{int_layer}"
    for cond in ["ccs", "ccs_spike", "ccs_spike_half"]:
        r = results[key][f"{cond}_redistribution"]
        s1 = results[key][cond].get("mean_sigma1_magnitude", "")
        s1_str = f"{s1:.0f}" if s1 else ""
        print(f"  L{int_layer:>3} {cond:>16} {r['total']:>13.1f} {r['sigma2_total']:>+10.1f} {s1_str:>8}",
              flush=True)
    print(flush=True)

    # Restoration ratio: how much does spike re-injection close the gap to control?
    ccs_r = results[key]["ccs_redistribution"]["total"]
    spike_r = results[key]["ccs_spike_redistribution"]["total"]
    if ccs_r > 0:
        restoration = 1.0 - (spike_r / ccs_r)
        print(f"    Spike re-injection restoration: {restoration:.1%} "
              f"({'toward' if restoration > 0 else 'away from'} control)", flush=True)

print(f"\nSpike-σ₁ alignment: {alignment_data}", flush=True)

outpath = os.path.expanduser("~/chronicle/spectral-demon/experiments/results/f641_spike_reinject.json")
out = {
    "model": MODEL,
    "intervention_layers": INTERVENTION_LAYERS,
    "n_prompts": len(PROMPTS),
    "n_layers": n_layers,
    "conditions": CONDITIONS,
    "spike_sigma1_alignment": alignment_data,
    "results": results,
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
}
with open(outpath, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved to {outpath}", flush=True)
