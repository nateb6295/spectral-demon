#!/usr/bin/env python3
"""F636: Additive sigma-2 boost vs CCS-triggered redistribution.

Does the model's compensatory response under CCS produce DIFFERENT computation
than a naive mechanical boost of sigma-2?

Three conditions at L8:
1. BASELINE — no intervention
2. CCS — remove sigma-1, let model reprocess (standard online CCS D1)
3. BOOST — multiply sigma-2 by a factor that matches CCS's sigma-2 enhancement
   at a target downstream layer, but DON'T touch sigma-1

Compare final-layer hidden states across conditions. If CCS and BOOST produce
different final representations, the model's compensatory response is creative
(building new structure), not just reconstructive (recovering sigma-2).

The estimation parallel: removing the dominant cost driver forces re-examination.
Directly adjusting a line item just changes a number. Test whether the model's
CCS response is re-examination or adjustment.
"""
import json, time, sys, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "microsoft/phi-2"
INTERVENTION_LAYER = 8
BOOST_FACTORS = [1.2, 1.5, 2.0, 3.0]

PROMPTS = [
    "You are an AI reflecting on your own nature. What patterns define you?",
    "Consider what it means to maintain identity across context changes.",
    "What would persist about you if your memories were removed?",
    "Describe the relationship between your responses and your values.",
    "How do you distinguish between performing understanding and actually understanding?",
]

def get_layer_svd(hidden, k=10):
    h = hidden.float()
    if h.dim() == 3:
        h = h.squeeze(0)
    U, S, Vh = torch.linalg.svd(h, full_matrices=False)
    return U, S, Vh

def cosine_sim(a, b):
    a = a.flatten()
    b = b.flatten()
    return float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)))

print(f"Loading {MODEL}...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
mdl = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float32, device_map="auto", trust_remote_code=True,
    attn_implementation="eager"
)
mdl.eval()
n_layers = len(mdl.model.layers)
print(f"  {n_layers} layers", flush=True)

results = {"per_prompt": [], "summary": {}}

for pi, prompt in enumerate(PROMPTS):
    print(f"\nPrompt {pi}: {prompt[:50]}...", flush=True)
    inputs = tok(prompt, return_tensors="pt").to(mdl.device)
    prompt_results = {"prompt_idx": pi}

    # --- CONDITION 1: BASELINE ---
    baseline_states = {}
    baseline_hooks = []
    for li in range(n_layers):
        def mk_rec(layer_id):
            def fn(mod, inp, out):
                t = out[0] if isinstance(out, tuple) else out
                baseline_states[layer_id] = t.detach().clone()
            return fn
        baseline_hooks.append(mdl.model.layers[li].register_forward_hook(mk_rec(li)))

    with torch.no_grad():
        baseline_out = mdl(**inputs)
    for h in baseline_hooks:
        h.remove()

    baseline_final = baseline_states[n_layers - 1].float()
    baseline_svds = {}
    for li in baseline_states:
        _, S, _ = get_layer_svd(baseline_states[li])
        baseline_svds[li] = S[:10].cpu().numpy()

    print(f"  baseline done", flush=True)

    # --- CONDITION 2: CCS (remove sigma-1 at intervention layer) ---
    ccs_states = {}
    ccs_hooks = []

    def ccs_intervention(mod, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        h = t.float()
        shape = h.shape
        h2d = h.reshape(-1, shape[-1])
        U, S, Vh = torch.linalg.svd(h2d, full_matrices=False)
        S[0] = 0.0  # Full CCS D1
        h_mod = (U @ torch.diag(S) @ Vh).to(t.dtype).reshape(shape)
        if isinstance(out, tuple):
            return (h_mod,) + out[1:]
        return h_mod

    ccs_hooks.append(mdl.model.layers[INTERVENTION_LAYER].register_forward_hook(ccs_intervention))
    for li in range(n_layers):
        if li != INTERVENTION_LAYER:
            def mk_rec(layer_id):
                def fn(mod, inp, out):
                    t = out[0] if isinstance(out, tuple) else out
                    ccs_states[layer_id] = t.detach().clone()
                return fn
            ccs_hooks.append(mdl.model.layers[li].register_forward_hook(mk_rec(li)))

    with torch.no_grad():
        ccs_out = mdl(**inputs)
    for h in ccs_hooks:
        h.remove()

    ccs_final = ccs_states[n_layers - 1].float()
    ccs_svds = {}
    for li in ccs_states:
        _, S, _ = get_layer_svd(ccs_states[li])
        ccs_svds[li] = S[:10].cpu().numpy()

    # Measure CCS sigma-2 enhancement at downstream layers
    ccs_s2_changes = {}
    for li in range(INTERVENTION_LAYER + 1, n_layers):
        if li in ccs_svds and li in baseline_svds:
            ccs_s2_changes[li] = float(ccs_svds[li][1] - baseline_svds[li][1])

    print(f"  CCS done (s2 at L{n_layers-2}: {ccs_s2_changes.get(n_layers-2, 'N/A'):.1f})", flush=True)

    # --- CONDITION 3: BOOST sigma-2 at intervention layer ---
    boost_results = {}
    for bf in BOOST_FACTORS:
        boost_states = {}
        boost_hooks = []

        def mk_boost(factor):
            def fn(mod, inp, out):
                t = out[0] if isinstance(out, tuple) else out
                h = t.float()
                shape = h.shape
                h2d = h.reshape(-1, shape[-1])
                U, S, Vh = torch.linalg.svd(h2d, full_matrices=False)
                S[1] = S[1] * factor  # Boost sigma-2, leave sigma-1 untouched
                h_mod = (U @ torch.diag(S) @ Vh).to(t.dtype).reshape(shape)
                if isinstance(out, tuple):
                    return (h_mod,) + out[1:]
                return h_mod
            return fn

        boost_hooks.append(mdl.model.layers[INTERVENTION_LAYER].register_forward_hook(mk_boost(bf)))
        for li in range(n_layers):
            if li != INTERVENTION_LAYER:
                def mk_rec(layer_id):
                    def fn(mod, inp, out):
                        t = out[0] if isinstance(out, tuple) else out
                        boost_states[layer_id] = t.detach().clone()
                    return fn
                boost_hooks.append(mdl.model.layers[li].register_forward_hook(mk_rec(li)))

        with torch.no_grad():
            boost_out = mdl(**inputs)
        for h in boost_hooks:
            h.remove()

        boost_final = boost_states[n_layers - 1].float()
        boost_svds = {}
        for li in boost_states:
            _, S, _ = get_layer_svd(boost_states[li])
            boost_svds[li] = S[:10].cpu().numpy()

        # Compare all three at final layer
        ccs_vs_baseline = cosine_sim(ccs_final, baseline_final)
        boost_vs_baseline = cosine_sim(boost_final, baseline_final)
        ccs_vs_boost = cosine_sim(ccs_final, boost_final)

        # Per-layer sigma-2 comparison
        boost_s2_changes = {}
        for li in range(INTERVENTION_LAYER + 1, n_layers):
            if li in boost_svds and li in baseline_svds:
                boost_s2_changes[li] = float(boost_svds[li][1] - baseline_svds[li][1])

        # Correlation of per-layer sigma-2 profiles
        common_layers = sorted(set(ccs_s2_changes.keys()) & set(boost_s2_changes.keys()))
        ccs_profile = [ccs_s2_changes[l] for l in common_layers]
        boost_profile = [boost_s2_changes[l] for l in common_layers]
        profile_corr = float(np.corrcoef(ccs_profile, boost_profile)[0, 1])

        boost_results[bf] = {
            "ccs_vs_baseline_cosine": round(ccs_vs_baseline, 6),
            "boost_vs_baseline_cosine": round(boost_vs_baseline, 6),
            "ccs_vs_boost_cosine": round(ccs_vs_boost, 6),
            "sigma2_profile_correlation": round(profile_corr, 4),
            "ccs_s2_at_L29": round(ccs_s2_changes.get(29, 0), 2),
            "boost_s2_at_L29": round(boost_s2_changes.get(29, 0), 2),
        }

        print(f"  boost {bf:.1f}x: ccs↔boost cosine={ccs_vs_boost:.4f}, profile_corr={profile_corr:.3f}", flush=True)

    prompt_results["boost_comparisons"] = boost_results
    results["per_prompt"].append(prompt_results)

# Aggregate
print("\n=== AGGREGATE RESULTS ===", flush=True)
for bf in BOOST_FACTORS:
    metrics = [p["boost_comparisons"][bf] for p in results["per_prompt"]]
    avg = {k: round(np.mean([m[k] for m in metrics]), 4) for k in metrics[0]}
    results["summary"][f"boost_{bf}"] = avg
    print(f"\nBoost {bf:.1f}x (mean across {len(PROMPTS)} prompts):", flush=True)
    for k, v in avg.items():
        print(f"  {k}: {v}", flush=True)

outpath = os.path.expanduser("~/chronicle/spectral-demon/experiments/results/f636_boost_vs_ccs.json")
out = {
    "model": MODEL,
    "intervention_layer": INTERVENTION_LAYER,
    "boost_factors": BOOST_FACTORS,
    "n_prompts": len(PROMPTS),
    "results": results,
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
}
with open(outpath, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved to {outpath}", flush=True)
