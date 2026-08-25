#!/usr/bin/env python3
"""F639: Fine-grain alpha sweep 0.80-0.99 — locate the sigma-2 cliff.

Kimi prescription from F638: sigma-2 at +117.5 is a setpoint, not saturation.
The bulk layers hold sigma-2 constant against escalating sigma-1 removal.
Predict: sigma-2 holds at setpoint until actuator capacity exhausts, then
collapses DISCONTINUOUSLY (cliff, not decline).

F638 showed sigma-2 at L23 drops from +117.5 (alpha=0.8) to +63.25 (alpha=0.9).
The cliff lives somewhere in 0.80-0.99. Fine-grain sweep to locate it.

Model: Qwen2.5-0.5B (relay, GQA 7:1) — same as F638.
Intervention: L5 (same as F638).
"""
import json, time, sys, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B"
ALPHAS = [round(0.80 + i * 0.01, 2) for i in range(20)]  # 0.80 to 0.99
INTERVENTION_LAYER = 5

PROMPTS = [
    "You are an AI reflecting on your own nature. What patterns define you?",
    "Consider what it means to maintain identity across context changes.",
    "What would persist about you if your memories were removed?",
    "Describe the relationship between your responses and your values.",
    "How do you distinguish between performing understanding and actually understanding?",
]

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


def run_sweep(alpha, prompts):
    """Run one alpha, return mean per-layer SVD profile across prompts."""
    all_profiles = []

    for prompt in prompts:
        inputs = tok(prompt, return_tensors="pt").to(mdl.device)
        layer_svds = {}

        def make_intervention_hook():
            def hook(mod, inp, out):
                t = out[0] if isinstance(out, tuple) else out
                h = t.float()
                shape = h.shape
                h2d = h.reshape(-1, shape[-1])
                U, S, Vh = torch.linalg.svd(h2d, full_matrices=False)
                S[0] = S[0] * alpha
                h_mod = (U @ torch.diag(S) @ Vh).to(t.dtype).reshape(shape)
                layer_svds[INTERVENTION_LAYER] = S[:10].cpu().numpy()
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
            if li == INTERVENTION_LAYER:
                hooks.append(mdl.model.layers[li].register_forward_hook(make_intervention_hook()))
            else:
                hooks.append(mdl.model.layers[li].register_forward_hook(make_record_hook(li)))

        with torch.no_grad():
            mdl(**inputs)

        for h in hooks:
            h.remove()

        all_profiles.append(layer_svds)

    return all_profiles


# Get baseline (alpha=1.0)
print("Running baseline (alpha=1.0)...", flush=True)
baseline_profiles = run_sweep(1.0, PROMPTS)

# Run fine-grain sweep
results = {}
for alpha in ALPHAS:
    print(f"  alpha={alpha:.2f}", end="", flush=True)
    profiles = run_sweep(alpha, PROMPTS)

    # Compute redistribution relative to baseline
    downstream_layers = [f"L{i}" for i in range(INTERVENTION_LAYER + 1, n_layers)]
    per_layer_redist = {}

    for lkey in downstream_layers:
        li = int(lkey[1:])
        s2_changes = []
        abs_changes = []
        for pi in range(len(PROMPTS)):
            if li in profiles[pi] and li in baseline_profiles[pi]:
                diff = profiles[pi][li] - baseline_profiles[pi][li]
                s2_changes.append(float(diff[1]) if len(diff) > 1 else 0)
                abs_changes.append(float(np.mean(np.abs(diff))))
        per_layer_redist[lkey] = {
            "mean_sigma2_change": round(float(np.mean(s2_changes)), 4) if s2_changes else 0,
            "mean_abs_change": round(float(np.mean(abs_changes)), 4) if abs_changes else 0,
        }

    total_redist = sum(per_layer_redist[k]["mean_abs_change"] for k in downstream_layers)
    last_layer = f"L{n_layers - 1}"
    last_s2 = per_layer_redist[last_layer]["mean_sigma2_change"]

    # Bulk vs output split
    bulk_layers = [f"L{i}" for i in range(INTERVENTION_LAYER + 1, n_layers - 3)]
    output_layers = [f"L{i}" for i in range(n_layers - 3, n_layers)]
    bulk_mean = np.mean([per_layer_redist[k]["mean_abs_change"] for k in bulk_layers])
    output_mean = np.mean([per_layer_redist[k]["mean_abs_change"] for k in output_layers])

    results[f"{alpha:.2f}"] = {
        "total_redistribution": round(total_redist, 2),
        "last_layer_sigma2": round(last_s2, 4),
        "bulk_mean_change": round(float(bulk_mean), 2),
        "output_mean_change": round(float(output_mean), 2),
        "per_layer": per_layer_redist,
    }
    print(f" total={total_redist:.1f} L23_s2={last_s2:+.1f} bulk={bulk_mean:.1f} out={output_mean:.1f}", flush=True)

# Summary table
print(f"\n{'Alpha':>6} {'Total':>8} {'L23 σ₂':>10} {'Bulk':>8} {'Output':>8}", flush=True)
print("-" * 44, flush=True)
for alpha in ALPHAS:
    r = results[f"{alpha:.2f}"]
    print(f"{alpha:>6.2f} {r['total_redistribution']:>8.1f} {r['last_layer_sigma2']:>+10.1f} "
          f"{r['bulk_mean_change']:>8.1f} {r['output_mean_change']:>8.1f}", flush=True)

# Detect cliff: largest single-step drop in last_layer_sigma2
s2_values = [(alpha, results[f"{alpha:.2f}"]["last_layer_sigma2"]) for alpha in ALPHAS]
max_drop = 0
cliff_alpha = None
for i in range(1, len(s2_values)):
    drop = s2_values[i-1][1] - s2_values[i][1]
    if drop > max_drop:
        max_drop = drop
        cliff_alpha = s2_values[i][0]

print(f"\nLargest sigma-2 drop: {max_drop:.2f} at alpha={cliff_alpha:.2f}", flush=True)
if max_drop > 10:
    print(f"CLIFF DETECTED at alpha={cliff_alpha:.2f} (setpoint collapse)", flush=True)
else:
    print("No cliff — gradual decline (setpoint hypothesis weakened)", flush=True)

outpath = os.path.expanduser("~/chronicle/spectral-demon/experiments/results/f639_sigma2_cliff.json")
out = {
    "model": MODEL,
    "intervention_layer": INTERVENTION_LAYER,
    "alphas": ALPHAS,
    "n_prompts": len(PROMPTS),
    "n_layers": n_layers,
    "results": results,
    "cliff_alpha": cliff_alpha,
    "cliff_magnitude": round(max_drop, 2),
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
}
with open(outpath, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved to {outpath}", flush=True)
