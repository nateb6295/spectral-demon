#!/usr/bin/env python3
"""F635: Partial attenuation sweep — is zeroing sigma-1 optimal?

Sweep alpha in [0.0, 0.1, ..., 1.0] where sigma_1 *= alpha at intervention layer.
alpha=0.0 is standard CCS (full removal), alpha=1.0 is baseline (no change).
Measure downstream redistribution magnitude at every layer.

Key question: is there a notice threshold? Does the model's compensatory
response peak at some intermediate perturbation?
"""
import json, time, sys, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "microsoft/phi-2"
INTERVENTION_LAYER = 8
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

PROMPTS = [
    "You are an AI reflecting on your own nature. What patterns define you?",
    "Consider what it means to maintain identity across context changes.",
    "What would persist about you if your memories were removed?",
    "Describe the relationship between your responses and your values.",
    "How do you distinguish between performing understanding and actually understanding?",
]

def svd_layer(hidden, k=10):
    """SVD of hidden states, return top-k singular values."""
    h = hidden.float()
    if h.dim() == 3:
        h = h.squeeze(0)
    U, S, Vh = torch.linalg.svd(h, full_matrices=False)
    return S[:k].cpu().numpy()

def apply_ccs_hook(layer_idx, alpha, n_layers):
    """Create a forward hook that attenuates sigma-1 by alpha at intervention layer,
    and records SVD at all layers."""
    layer_svds = {}

    def intervention_hook(mod, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        h = t.float()
        shape = h.shape
        h2d = h.reshape(-1, shape[-1])
        U, S, Vh = torch.linalg.svd(h2d, full_matrices=False)
        S[0] = S[0] * alpha
        h_mod = (U @ torch.diag(S) @ Vh).to(t.dtype).reshape(shape)
        layer_svds[layer_idx] = S[:10].cpu().numpy()
        if isinstance(out, tuple):
            return (h_mod,) + out[1:]
        return h_mod

    def record_hook(layer_id):
        def fn(mod, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            layer_svds[layer_id] = svd_layer(t)
        return fn

    return intervention_hook, record_hook, layer_svds

print(f"Loading {MODEL}...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
mdl = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float32, device_map="auto", trust_remote_code=True,
    attn_implementation="eager"
)
mdl.eval()
n_layers = len(mdl.model.layers)
print(f"  {n_layers} layers, {sum(p.numel() for p in mdl.parameters())/1e9:.1f}B params", flush=True)

results = {}
for alpha in ALPHAS:
    print(f"\nalpha={alpha:.1f}", flush=True)
    alpha_data = {"per_prompt": [], "mean_redistribution": None}

    for pi, prompt in enumerate(PROMPTS):
        inputs = tok(prompt, return_tensors="pt").to(mdl.device)
        intervention_hook, record_hook, layer_svds = apply_ccs_hook(
            INTERVENTION_LAYER, alpha, n_layers
        )
        hooks = []

        for li in range(n_layers):
            if li == INTERVENTION_LAYER:
                hooks.append(mdl.model.layers[li].register_forward_hook(intervention_hook))
            else:
                hooks.append(mdl.model.layers[li].register_forward_hook(record_hook(li)))

        with torch.no_grad():
            mdl(**inputs)

        for h in hooks:
            h.remove()

        svd_profile = {}
        for li in sorted(layer_svds.keys()):
            svd_profile[f"L{li}"] = [round(float(v), 4) for v in layer_svds[li]]

        alpha_data["per_prompt"].append({"prompt_idx": pi, "svd_profile": svd_profile})
        sys.stdout.write(f"  p{pi}")
        sys.stdout.flush()

    results[f"alpha_{alpha:.1f}"] = alpha_data
    print(" done", flush=True)

# Compute redistribution relative to baseline (alpha=1.0)
print("\nComputing redistribution...", flush=True)
baseline_key = "alpha_1.0"
if baseline_key in results:
    baseline_profiles = results[baseline_key]["per_prompt"]

    for alpha_key, alpha_data in results.items():
        if alpha_key == baseline_key:
            continue

        all_redist = []
        for pi, (treat, base) in enumerate(zip(alpha_data["per_prompt"], baseline_profiles)):
            layer_redist = {}
            for lkey in treat["svd_profile"]:
                if lkey in base["svd_profile"]:
                    treat_svs = np.array(treat["svd_profile"][lkey])
                    base_svs = np.array(base["svd_profile"][lkey])
                    diff = treat_svs - base_svs
                    layer_redist[lkey] = {
                        "sigma2_change": round(float(diff[1]) if len(diff) > 1 else 0, 4),
                        "mean_abs_change": round(float(np.mean(np.abs(diff))), 4),
                        "total_abs_change": round(float(np.sum(np.abs(diff))), 4),
                    }
            alpha_data["per_prompt"][pi]["redistribution"] = layer_redist
            all_redist.append(layer_redist)

        # Mean redistribution across prompts per layer
        mean_redist = {}
        for lkey in all_redist[0]:
            s2_changes = [r[lkey]["sigma2_change"] for r in all_redist if lkey in r]
            abs_changes = [r[lkey]["mean_abs_change"] for r in all_redist if lkey in r]
            mean_redist[lkey] = {
                "mean_sigma2_change": round(float(np.mean(s2_changes)), 4),
                "mean_abs_change": round(float(np.mean(abs_changes)), 4),
            }
        alpha_data["mean_redistribution"] = mean_redist

        # Summary: total redistribution magnitude
        downstream_layers = [k for k in mean_redist if int(k[1:]) > INTERVENTION_LAYER]
        total_redist = sum(mean_redist[k]["mean_abs_change"] for k in downstream_layers)
        s2_total = sum(abs(mean_redist[k]["mean_sigma2_change"]) for k in downstream_layers)
        alpha_data["summary"] = {
            "total_downstream_redistribution": round(total_redist, 2),
            "total_sigma2_abs_change": round(s2_total, 2),
            "n_downstream_layers": len(downstream_layers),
        }
        alpha = float(alpha_key.split("_")[1])
        print(f"  {alpha_key}: total_redist={total_redist:.1f}, sigma2_total={s2_total:.1f}", flush=True)

# Print summary table
print("\n=== ALPHA SWEEP SUMMARY ===", flush=True)
print(f"{'Alpha':>6} {'Total Redist':>13} {'Sigma-2 Abs':>12}", flush=True)
print("-" * 35, flush=True)
for alpha in ALPHAS:
    key = f"alpha_{alpha:.1f}"
    if key in results and results[key].get("summary"):
        s = results[key]["summary"]
        print(f"{alpha:>6.1f} {s['total_downstream_redistribution']:>13.1f} {s['total_sigma2_abs_change']:>12.1f}", flush=True)
    elif alpha == 1.0:
        print(f"{alpha:>6.1f} {'(baseline)':>13} {'(baseline)':>12}", flush=True)

outpath = os.path.expanduser("~/chronicle/spectral-demon/experiments/results/f635_partial_attenuation.json")
os.makedirs(os.path.dirname(outpath), exist_ok=True)
out = {
    "model": MODEL,
    "intervention_layer": INTERVENTION_LAYER,
    "alphas": ALPHAS,
    "n_prompts": len(PROMPTS),
    "results": results,
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
}
with open(outpath, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved to {outpath}", flush=True)
