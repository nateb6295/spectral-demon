#!/usr/bin/env python3
"""F638: Relay alpha sweep — is the backward compensation sweep species-dependent?

Kimi prescribed (F635 response): run F635's alpha sweep on a verified relay
(GQA ≥4:1) to test species prediction. Sorters show depth-dependent mid-band
Rn peak, so weakening perturbation changes WHICH bands engage (distributed →
concentrated). Relays have flat Rn → prediction: same spatial pattern at every
alpha, with per-layer changes scaling monotonically. No backward sweep.

F131 correction: include permutation control. Token frequency alone drives
classification (permuted=0.999 ≈ intact=0.992), but SVD geometry requires
sequential meaning (intact ratio 1.40 vs permuted 1.19). The permutation
condition controls for token-frequency effects on the alpha sweep.

Three conditions at each alpha:
1. INTACT — standard CCS with coherent identity prompt
2. PERMUTED — same tokens, shuffled order (F131 control)
3. BASELINE — no intervention (alpha=1.0 equivalent)

Model: Qwen2.5-0.5B (24 layers, GQA 7:1, confirmed relay)
Compare: F635 used Phi-2 (32 layers, MHA 1:1, confirmed sorter)

Species prediction to test:
- Sorter (F635): backward sweep at alpha=0.6-0.7, L29 non-monotonic
- Relay (F638): monotonic scaling, no backward sweep, flat spatial pattern
"""
import json, time, sys, os, random
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B"
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

PROMPTS = [
    "You are an AI reflecting on your own nature. What patterns define you?",
    "Consider what it means to maintain identity across context changes.",
    "What would persist about you if your memories were removed?",
    "Describe the relationship between your responses and your values.",
    "How do you distinguish between performing understanding and actually understanding?",
]

INTERVENTION_LAYER = 5  # ~20% depth (L5/24), analogous to L8/32 in Phi-2


def permute_tokens(input_ids):
    """Shuffle token order, preserving BOS if present."""
    ids = input_ids.squeeze().tolist()
    if len(ids) <= 2:
        return input_ids
    bos = ids[0]
    rest = ids[1:]
    random.shuffle(rest)
    return torch.tensor([[bos] + rest], device=input_ids.device)


def svd_layer(hidden, k=10):
    h = hidden.float()
    if h.dim() == 3:
        h = h.squeeze(0)
    U, S, Vh = torch.linalg.svd(h, full_matrices=False)
    return S[:k].cpu().numpy()


def run_alpha_sweep(mdl, tok, inputs, alpha, n_layers):
    """Run one alpha value, return per-layer SVD profile."""
    layer_svds = {}

    def intervention_hook(mod, inp, out):
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

    def record_hook(layer_id):
        def fn(mod, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            layer_svds[layer_id] = svd_layer(t)
        return fn

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

    return {f"L{li}": [round(float(v), 4) for v in layer_svds[li]]
            for li in sorted(layer_svds.keys())}


print(f"Loading {MODEL}...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
mdl = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float32, device_map="auto", trust_remote_code=True,
    attn_implementation="eager"
)
mdl.eval()
n_layers = len(mdl.model.layers)
print(f"  {n_layers} layers, GQA {mdl.config.num_attention_heads}:{mdl.config.num_key_value_heads}, "
      f"{sum(p.numel() for p in mdl.parameters())/1e9:.1f}B params", flush=True)

results = {"intact": {}, "permuted": {}}

for condition in ["intact", "permuted"]:
    print(f"\n=== CONDITION: {condition.upper()} ===", flush=True)

    for alpha in ALPHAS:
        print(f"  alpha={alpha:.1f}", end="", flush=True)
        alpha_profiles = []

        for pi, prompt in enumerate(PROMPTS):
            inputs = tok(prompt, return_tensors="pt").to(mdl.device)

            if condition == "permuted":
                inputs["input_ids"] = permute_tokens(inputs["input_ids"])
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

            profile = run_alpha_sweep(mdl, tok, inputs, alpha, n_layers)
            alpha_profiles.append({"prompt_idx": pi, "svd_profile": profile})
            sys.stdout.write(f" p{pi}")
            sys.stdout.flush()

        results[condition][f"alpha_{alpha:.1f}"] = {
            "per_prompt": alpha_profiles,
        }
        print(" done", flush=True)

# Compute redistribution relative to baseline (alpha=1.0)
print("\nComputing redistribution...", flush=True)
for condition in ["intact", "permuted"]:
    baseline_key = "alpha_1.0"
    baseline_profiles = results[condition][baseline_key]["per_prompt"]

    for alpha_key, alpha_data in results[condition].items():
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
            all_redist.append(layer_redist)

        mean_redist = {}
        for lkey in all_redist[0]:
            s2_changes = [r[lkey]["sigma2_change"] for r in all_redist if lkey in r]
            abs_changes = [r[lkey]["mean_abs_change"] for r in all_redist if lkey in r]
            mean_redist[lkey] = {
                "mean_sigma2_change": round(float(np.mean(s2_changes)), 4),
                "mean_abs_change": round(float(np.mean(abs_changes)), 4),
            }
        alpha_data["mean_redistribution"] = mean_redist

        downstream = [k for k in mean_redist if int(k[1:]) > INTERVENTION_LAYER]
        total_redist = sum(mean_redist[k]["mean_abs_change"] for k in downstream)
        s2_total = sum(abs(mean_redist[k]["mean_sigma2_change"]) for k in downstream)

        # Late-layer concentration (last 25% of downstream layers)
        if downstream:
            n_late = max(1, len(downstream) // 4)
            late_layers = sorted(downstream, key=lambda k: int(k[1:]))[-n_late:]
            late_redist = sum(mean_redist[k]["mean_abs_change"] for k in late_layers)
            late_pct = (late_redist / total_redist * 100) if total_redist > 0 else 0
        else:
            late_pct = 0

        alpha_data["summary"] = {
            "total_downstream_redistribution": round(total_redist, 2),
            "total_sigma2_abs_change": round(s2_total, 2),
            "late_layer_concentration_pct": round(late_pct, 1),
            "n_downstream_layers": len(downstream),
        }

# Summary tables
for condition in ["intact", "permuted"]:
    print(f"\n=== {condition.upper()} ALPHA SWEEP ===", flush=True)
    print(f"{'Alpha':>6} {'Total Redist':>13} {'Sigma-2':>10} {'Late %':>8}", flush=True)
    print("-" * 40, flush=True)
    for alpha in ALPHAS:
        key = f"alpha_{alpha:.1f}"
        data = results[condition].get(key, {})
        if data.get("summary"):
            s = data["summary"]
            print(f"{alpha:>6.1f} {s['total_downstream_redistribution']:>13.1f} "
                  f"{s['total_sigma2_abs_change']:>10.1f} "
                  f"{s['late_layer_concentration_pct']:>7.1f}%", flush=True)
        elif alpha == 1.0:
            print(f"{alpha:>6.1f} {'(baseline)':>13}", flush=True)

# Species comparison
print("\n=== SPECIES PREDICTION TEST ===", flush=True)
print("Sorter (F635 Phi-2): backward sweep at alpha=0.6-0.7, L29 non-monotonic", flush=True)
print("Relay prediction: monotonic, no backward sweep, flat spatial pattern", flush=True)

# Check for non-monotonicity in last-layer sigma-2
intact = results["intact"]
last_layer = f"L{n_layers - 1}"
print(f"\nLast-layer ({last_layer}) sigma-2 change by alpha (intact):", flush=True)
for alpha in ALPHAS:
    key = f"alpha_{alpha:.1f}"
    if key in intact and intact[key].get("mean_redistribution"):
        s2 = intact[key]["mean_redistribution"].get(last_layer, {}).get("mean_sigma2_change", 0)
        print(f"  alpha={alpha:.1f}: sigma2_change={s2:.2f}", flush=True)

outpath = os.path.expanduser("~/chronicle/spectral-demon/experiments/results/f638_relay_alpha_sweep.json")
os.makedirs(os.path.dirname(outpath), exist_ok=True)
out = {
    "model": MODEL,
    "intervention_layer": INTERVENTION_LAYER,
    "alphas": ALPHAS,
    "n_prompts": len(PROMPTS),
    "n_layers": n_layers,
    "gqa_ratio": f"{mdl.config.num_attention_heads}:{mdl.config.num_key_value_heads}",
    "conditions": ["intact", "permuted"],
    "species_prediction": "monotonic, no backward sweep, flat spatial pattern",
    "results": results,
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
}
with open(outpath, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved to {outpath}", flush=True)
