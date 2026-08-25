#!/usr/bin/env python3
"""F640: Scalpel CCS — spike-direction suppression vs full σ₁ zeroing.

Massive activations paper (2025, arxiv:2603.05498) identifies spike direction s⋆
as rank-one dominated, orthogonal to core computation, independently suppressible.
Conjecture SVD paper confirms σ₁ = token frequency direction.

Hypothesis: suppressing just the spike direction (projection onto s⋆) should
produce D3-level spectral benefits at lower effective dose, because it doesn't
graze adjacent eigenmodes that share the σ₁ subspace.

Three conditions at each intervention layer:
1. FULL: zero σ₁ entirely (standard CCS)
2. SCALPEL: identify spike direction from first-token activations, project out
3. CONTROL: no intervention (baseline)

Compare: downstream σ₂ enrichment, spatial compensation pattern, total redistribution.
Prediction: scalpel achieves >80% of full CCS's σ₂ enrichment at <50% of the
spectral disruption (measured by total redistribution magnitude).

Model: Qwen2.5-0.5B (relay, GQA 7:1) — same as F638/F639.
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
    """Extract spike direction from first-token activation at given layer.

    The spike direction s⋆ is the dominant singular vector of the first-token
    hidden state, which the massive-activations paper shows concentrates into
    a rank-one structure across layers.
    """
    inputs = tok(SPIKE_CALIBRATION_PROMPT, return_tensors="pt").to(mdl.device)
    spike_dir = None

    def capture_hook(mod, inp, out):
        nonlocal spike_dir
        t = out[0] if isinstance(out, tuple) else out
        h = t.float()
        first_tok = h[0, 0, :]  # first token hidden state
        spike_dir = first_tok / first_tok.norm()

    hook = mdl.model.layers[layer_idx].register_forward_hook(capture_hook)
    with torch.no_grad():
        mdl(**inputs)
    hook.remove()
    return spike_dir


def run_condition(mdl, tok, prompt, intervention_layer, condition, spike_dir=None):
    """Run one prompt under one condition. Return per-layer SVD profiles."""
    inputs = tok(prompt, return_tensors="pt").to(mdl.device)
    layer_svds = {}

    def make_full_ccs_hook():
        """Standard CCS: zero σ₁ via SVD."""
        def hook(mod, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            h = t.float()
            shape = h.shape
            h2d = h.reshape(-1, shape[-1])
            U, S, Vh = torch.linalg.svd(h2d, full_matrices=False)
            S[0] = 0.0
            h_mod = (U @ torch.diag(S) @ Vh).to(t.dtype).reshape(shape)
            layer_svds[intervention_layer] = S[:10].cpu().numpy()
            if isinstance(out, tuple):
                return (h_mod,) + out[1:]
            return h_mod
        return hook

    def make_scalpel_hook(s_star):
        """Scalpel CCS: project out spike direction only."""
        def hook(mod, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            h = t.float()
            shape = h.shape
            h2d = h.reshape(-1, shape[-1])
            # Project out s⋆ from each position
            proj = h2d @ s_star.unsqueeze(1)  # [seq_len, 1]
            h_mod = h2d - proj @ s_star.unsqueeze(0)  # subtract projection
            h_mod = h_mod.to(t.dtype).reshape(shape)
            # Record SVD of modified hidden
            h_rec = h_mod.float().reshape(-1, shape[-1])
            U, S, Vh = torch.linalg.svd(h_rec, full_matrices=False)
            layer_svds[intervention_layer] = S[:10].cpu().numpy()
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
            if condition == "full":
                hooks.append(mdl.model.layers[li].register_forward_hook(make_full_ccs_hook()))
            elif condition == "scalpel":
                hooks.append(mdl.model.layers[li].register_forward_hook(make_scalpel_hook(spike_dir)))
            else:
                hooks.append(mdl.model.layers[li].register_forward_hook(make_record_hook(li)))
        else:
            hooks.append(mdl.model.layers[li].register_forward_hook(make_record_hook(li)))

    with torch.no_grad():
        mdl(**inputs)

    for h in hooks:
        h.remove()

    return layer_svds


# Run experiment
results = {}

for int_layer in INTERVENTION_LAYERS:
    print(f"\n=== Intervention layer L{int_layer} ===", flush=True)

    # Extract spike direction for this layer
    spike_dir = get_spike_direction(mdl, tok, int_layer)
    print(f"  Spike direction extracted (norm={spike_dir.norm():.4f})", flush=True)

    layer_results = {}

    for condition in ["control", "full", "scalpel"]:
        print(f"  {condition}...", end="", flush=True)
        all_profiles = []

        for pi, prompt in enumerate(PROMPTS):
            profile = run_condition(mdl, tok, prompt, int_layer, condition,
                                    spike_dir=spike_dir if condition == "scalpel" else None)
            all_profiles.append(profile)
            sys.stdout.write(f" p{pi}")
            sys.stdout.flush()

        # Compute mean SVD across prompts for each layer
        mean_svd = {}
        for li in range(n_layers):
            svds = [p[li] for p in all_profiles if li in p]
            if svds:
                mean_svd[f"L{li}"] = [round(float(v), 4) for v in np.mean(svds, axis=0)]

        layer_results[condition] = mean_svd
        print(" done", flush=True)

    # Compute redistribution relative to control
    control = layer_results["control"]
    for condition in ["full", "scalpel"]:
        treated = layer_results[condition]
        redist = {}
        downstream = [f"L{i}" for i in range(int_layer + 1, n_layers)]

        for lkey in downstream:
            if lkey in treated and lkey in control:
                t_svs = np.array(treated[lkey])
                c_svs = np.array(control[lkey])
                diff = t_svs - c_svs
                redist[lkey] = {
                    "sigma2_change": round(float(diff[1]) if len(diff) > 1 else 0, 4),
                    "mean_abs_change": round(float(np.mean(np.abs(diff))), 4),
                }

        total_redist = sum(redist[k]["mean_abs_change"] for k in downstream if k in redist)
        s2_total = sum(redist[k]["sigma2_change"] for k in downstream if k in redist)

        # Bulk vs output split
        bulk = [f"L{i}" for i in range(int_layer + 1, n_layers - 3)]
        output = [f"L{i}" for i in range(n_layers - 3, n_layers)]
        bulk_mean = np.mean([redist[k]["mean_abs_change"] for k in bulk if k in redist]) if bulk else 0
        output_mean = np.mean([redist[k]["mean_abs_change"] for k in output if k in redist]) if output else 0

        layer_results[f"{condition}_redistribution"] = {
            "per_layer": redist,
            "total": round(total_redist, 2),
            "sigma2_total": round(s2_total, 2),
            "bulk_mean": round(float(bulk_mean), 2),
            "output_mean": round(float(output_mean), 2),
        }

    results[f"L{int_layer}"] = layer_results

# Summary
print(f"\n{'Layer':>6} {'Condition':>10} {'Total Redist':>13} {'Σ σ₂':>10} {'Bulk':>8} {'Output':>8}", flush=True)
print("-" * 60, flush=True)
for int_layer in INTERVENTION_LAYERS:
    key = f"L{int_layer}"
    for cond in ["full", "scalpel"]:
        r = results[key][f"{cond}_redistribution"]
        print(f"  L{int_layer:>3} {cond:>10} {r['total']:>13.1f} {r['sigma2_total']:>+10.1f} "
              f"{r['bulk_mean']:>8.1f} {r['output_mean']:>8.1f}", flush=True)

    # Efficiency ratio
    full_r = results[key]["full_redistribution"]
    scalpel_r = results[key]["scalpel_redistribution"]
    if full_r["total"] > 0:
        redist_ratio = scalpel_r["total"] / full_r["total"]
        s2_ratio = scalpel_r["sigma2_total"] / full_r["sigma2_total"] if full_r["sigma2_total"] != 0 else 0
        print(f"    Scalpel efficiency: {s2_ratio:.1%} of σ₂ benefit at {redist_ratio:.1%} of disruption", flush=True)

outpath = os.path.expanduser("~/chronicle/spectral-demon/experiments/results/f640_scalpel_ccs.json")
out = {
    "model": MODEL,
    "intervention_layers": INTERVENTION_LAYERS,
    "n_prompts": len(PROMPTS),
    "n_layers": n_layers,
    "conditions": ["control", "full", "scalpel"],
    "spike_calibration": SPIKE_CALIBRATION_PROMPT,
    "results": results,
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
}
with open(outpath, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved to {outpath}", flush=True)
