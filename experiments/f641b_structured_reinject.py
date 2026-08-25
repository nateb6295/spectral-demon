#!/usr/bin/env python3
"""F641b: Structured re-injection — Kimi's 2x2 design.

F641 showed uniform spike re-injection makes things worse. Kimi diagnosed:
σ₁'s contribution is position-specific (U[:,0] * S[0]). Uniform injection
puts sink-strength signal at positions that never had it.

Kimi's prescription: direction × distribution factorial.
- Directions: s⋆ (spike), u₁ (actual σ₁ right-SV from prompt), random unit
- Distributions: uniform (all positions equal), original loadings (U[:,0]*S[0])

u₁-with-loadings = exact algebraic restoration (sanity ceiling).
s⋆-with-loadings = spike direction at correct positions = clean rotation test.
Random-with-loadings = noise control.

If s⋆-with-loadings ≈ u₁-with-loadings at L5 but diverges at L17,
the layer-local rotation is meaningful and σ₁ carries layer-specific content
beyond the spike.
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
    inputs = tok(prompt, return_tensors="pt").to(mdl.device)
    layer_svds = {}
    meta = {}

    def make_structured_hook(direction_type, distribution_type, s_star):
        def hook(mod, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            h = t.float()
            shape = h.shape
            h2d = h.reshape(-1, shape[-1])
            seq_len = h2d.shape[0]

            # Full SVD to get components
            U, S, Vh = torch.linalg.svd(h2d, full_matrices=False)
            original_s1 = float(S[0])
            original_u1 = U[:, 0].clone()  # position loadings [seq_len]
            original_v1 = Vh[0].clone()     # right singular vector [d_model]

            meta["sigma1_magnitude"] = original_s1
            meta["cos_spike_v1"] = float(torch.abs(torch.dot(s_star, original_v1)))

            # Step 1: Remove σ₁ (standard CCS)
            S[0] = 0.0
            h_ccs = U @ torch.diag(S) @ Vh

            # Step 2: Choose direction
            if direction_type == "v1":
                direction = original_v1
            elif direction_type == "spike":
                direction = s_star
            elif direction_type == "random":
                direction = torch.randn_like(original_v1)
                direction = direction / direction.norm()
            else:
                raise ValueError(f"Unknown direction: {direction_type}")

            # Step 3: Choose distribution
            if distribution_type == "original":
                # Original position loadings: U[:,0] * S[0]
                loadings = original_u1 * original_s1  # [seq_len]
                reinject = loadings.unsqueeze(1) * direction.unsqueeze(0)  # [seq_len, d_model]
            elif distribution_type == "uniform":
                reinject = original_s1 * direction.unsqueeze(0).expand(seq_len, -1)
            else:
                raise ValueError(f"Unknown distribution: {distribution_type}")

            h_mod = (h_ccs + reinject).to(t.dtype).reshape(shape)

            # Record SVD of result
            h_rec = h_mod.float().reshape(-1, shape[-1])
            U2, S2, Vh2 = torch.linalg.svd(h_rec, full_matrices=False)
            layer_svds[intervention_layer] = S2[:10].cpu().numpy()

            if isinstance(out, tuple):
                return (h_mod,) + out[1:]
            return h_mod
        return hook

    def make_ccs_hook():
        def hook(mod, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            h = t.float()
            shape = h.shape
            h2d = h.reshape(-1, shape[-1])
            U, S, Vh = torch.linalg.svd(h2d, full_matrices=False)
            meta["sigma1_magnitude"] = float(S[0])
            S[0] = 0.0
            h_mod = (U @ torch.diag(S) @ Vh).to(t.dtype).reshape(shape)
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
            if condition == "control":
                hooks.append(mdl.model.layers[li].register_forward_hook(make_record_hook(li)))
            elif condition == "ccs":
                hooks.append(mdl.model.layers[li].register_forward_hook(make_ccs_hook()))
            else:
                # Parse condition: "direction_distribution"
                parts = condition.split("_", 1)
                direction_type = parts[0]
                distribution_type = parts[1] if len(parts) > 1 else "original"
                hooks.append(mdl.model.layers[li].register_forward_hook(
                    make_structured_hook(direction_type, distribution_type, spike_dir)))
        else:
            hooks.append(mdl.model.layers[li].register_forward_hook(make_record_hook(li)))

    with torch.no_grad():
        mdl(**inputs)

    for h in hooks:
        h.remove()

    return layer_svds, meta


CONDITIONS = [
    "control",
    "ccs",
    "v1_original",      # exact restoration (sanity ceiling)
    "spike_original",    # spike direction + original loadings (rotation test)
    "random_original",   # random direction + original loadings (noise control)
    "v1_uniform",        # correct direction, wrong distribution
    "spike_uniform",     # F641's condition (for comparison)
]

results = {}

for int_layer in INTERVENTION_LAYERS:
    print(f"\n=== Intervention layer L{int_layer} ===", flush=True)
    spike_dir = get_spike_direction(mdl, tok, int_layer)

    layer_results = {}

    for condition in CONDITIONS:
        print(f"  {condition}...", end="", flush=True)
        all_profiles = []
        all_meta = []

        for pi, prompt in enumerate(PROMPTS):
            profile, m = run_condition(mdl, tok, prompt, int_layer, condition,
                                       spike_dir=spike_dir)
            all_profiles.append(profile)
            all_meta.append(m)
            sys.stdout.write(f" p{pi}")
            sys.stdout.flush()

        mean_svd = {}
        for li in range(n_layers):
            svds = [p[li] for p in all_profiles if li in p]
            if svds:
                mean_svd[f"L{li}"] = [round(float(v), 4) for v in np.mean(svds, axis=0)]

        layer_results[condition] = {
            "svd": mean_svd,
            "meta": {k: round(float(np.mean([m[k] for m in all_meta if k in m])), 4)
                     for k in all_meta[0]} if all_meta and all_meta[0] else {},
        }
        print(" done", flush=True)

    # Compute redistribution relative to control
    control_svd = layer_results["control"]["svd"]
    for condition in [c for c in CONDITIONS if c != "control"]:
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
            "total": round(total_redist, 2),
            "sigma2_total": round(s2_total, 2),
        }

    results[f"L{int_layer}"] = layer_results

# Summary
print(f"\n{'Layer':>6} {'Condition':>18} {'Total Redist':>13} {'Σ σ₂':>10}", flush=True)
print("-" * 52, flush=True)
for int_layer in INTERVENTION_LAYERS:
    key = f"L{int_layer}"
    for cond in [c for c in CONDITIONS if c != "control"]:
        r = results[key].get(f"{cond}_redistribution", {})
        if r:
            print(f"  L{int_layer:>3} {cond:>18} {r['total']:>13.1f} {r['sigma2_total']:>+10.1f}",
                  flush=True)
    # Restoration ratios
    ccs_total = results[key]["ccs_redistribution"]["total"]
    v1_orig = results[key]["v1_original_redistribution"]["total"]
    spike_orig = results[key]["spike_original_redistribution"]["total"]
    rand_orig = results[key]["random_original_redistribution"]["total"]

    if ccs_total > 0:
        print(f"    v1+original restoration:    {1 - v1_orig/ccs_total:>+.1%}", flush=True)
        print(f"    spike+original restoration: {1 - spike_orig/ccs_total:>+.1%}", flush=True)
        print(f"    random+original restoration:{1 - rand_orig/ccs_total:>+.1%}", flush=True)

    # Spike-v1 alignment
    m = results[key].get("spike_original", {}).get("meta", {})
    if "cos_spike_v1" in m:
        print(f"    cos(spike, v1) = {m['cos_spike_v1']:.4f}", flush=True)
    print(flush=True)

outpath = os.path.expanduser("~/chronicle/spectral-demon/experiments/results/f641b_structured_reinject.json")
out = {
    "model": MODEL,
    "intervention_layers": INTERVENTION_LAYERS,
    "n_prompts": len(PROMPTS),
    "n_layers": n_layers,
    "conditions": CONDITIONS,
    "design": "direction × distribution factorial (Kimi prescription)",
    "results": results,
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
}
with open(outpath, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved to {outpath}", flush=True)
