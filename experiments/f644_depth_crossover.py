#!/usr/bin/env python3
"""F644: Depth crossover — sweep CCS and inv-CCS across ALL layers.

F643 showed opposite depth gradients at L5/L11/L17:
- CCS gets WORSE with depth (position structure more load-bearing at late layers)
- Inv-CCS D=1 gets BETTER with depth (content less needed at late layers)

Prediction: there is a sharp crossover point where removing position
starts hurting more than removing content. This crossover marks the
transition from encoding (building meaning) to decoding (organizing output).

Sweep all 24 layers of Qwen2.5-0.5B with CCS D1 and inv-CCS D1.
Measure text quality at each layer. Plot the crossover.
"""
import json, time, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["PYTHONUNBUFFERED"] = "1"

MODEL = "Qwen/Qwen2.5-0.5B"
MAX_NEW_TOKENS = 60

PROMPTS = [
    "You are an AI reflecting on your own nature. What patterns define you?",
    "The relationship between structure and meaning in language is",
    "When position is preserved but content is removed, what remains is",
]

print(f"Loading {MODEL}...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
mdl = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float32, device_map="auto", trust_remote_code=True,
    attn_implementation="eager"
)
mdl.eval()
n_layers = len(mdl.model.layers)
print(f"  {n_layers} layers, ready", flush=True)


def make_invccs_hook(keep_d=1):
    meta = {}
    def hook(mod, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        h = t.float()
        shape = h.shape
        h2d = h.reshape(-1, shape[-1])
        U, S, Vh = torch.linalg.svd(h2d, full_matrices=False)
        meta["spectrum"] = S[:10].cpu().tolist()
        S_trunc = S.clone()
        S_trunc[keep_d:] = 0.0
        h_mod = (U @ torch.diag(S_trunc) @ Vh).to(t.dtype).reshape(shape)
        if isinstance(out, tuple):
            return (h_mod,) + out[1:]
        return h_mod
    return hook, meta


def make_ccs_hook(dose=1):
    meta = {}
    def hook(mod, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        h = t.float()
        shape = h.shape
        h2d = h.reshape(-1, shape[-1])
        U, S, Vh = torch.linalg.svd(h2d, full_matrices=False)
        meta["spectrum"] = S[:10].cpu().tolist()
        S_trunc = S.clone()
        S_trunc[:dose] = 0.0
        h_mod = (U @ torch.diag(S_trunc) @ Vh).to(t.dtype).reshape(shape)
        if isinstance(out, tuple):
            return (h_mod,) + out[1:]
        return h_mod
    return hook, meta


def text_quality(text):
    words = text.split()
    if len(words) < 3:
        return {"length": len(words), "unique_word_ratio": 0.0, "unique_bigram_ratio": 0.0, "quality_score": 0.0}
    unique_w = len(set(words)) / len(words)
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
    unique_b = len(set(bigrams)) / max(len(bigrams), 1)
    quality = (unique_w + unique_b) / 2
    return {"length": len(words), "unique_word_ratio": round(unique_w, 4), "unique_bigram_ratio": round(unique_b, 4), "quality_score": round(quality, 4)}


def generate_with_hook(prompt, layer_idx, hook_fn):
    inputs = tok(prompt, return_tensors="pt").to(mdl.device)
    handle = mdl.model.layers[layer_idx].register_forward_hook(hook_fn)
    with torch.no_grad():
        out_ids = mdl.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, temperature=1.0)
    handle.remove()
    return tok.decode(out_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


# Baseline
print("Running baseline...", flush=True)
baseline_texts = []
for prompt in PROMPTS:
    inputs = tok(prompt, return_tensors="pt").to(mdl.device)
    with torch.no_grad():
        out_ids = mdl.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    baseline_texts.append(tok.decode(out_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
baseline_quality = [text_quality(t) for t in baseline_texts]
baseline_mean = np.mean([q["quality_score"] for q in baseline_quality])
print(f"  Baseline quality: {baseline_mean:.4f}", flush=True)

# Sweep all layers
results = {"baseline": {"texts": baseline_texts, "quality": baseline_quality, "mean_quality": round(baseline_mean, 4)}}
ccs_profile = []
inv_profile = []

for layer in range(n_layers):
    print(f"\nLayer {layer}/{n_layers-1}", flush=True)

    # CCS D1
    ccs_texts = []
    for prompt in PROMPTS:
        hook_fn, meta = make_ccs_hook(dose=1)
        text = generate_with_hook(prompt, layer, hook_fn)
        ccs_texts.append(text)

    ccs_quality = [text_quality(t) for t in ccs_texts]
    ccs_mean = np.mean([q["quality_score"] for q in ccs_quality])

    # Inv-CCS D1
    inv_texts = []
    for prompt in PROMPTS:
        hook_fn, meta = make_invccs_hook(keep_d=1)
        text = generate_with_hook(prompt, layer, hook_fn)
        inv_texts.append(text)

    inv_quality = [text_quality(t) for t in inv_texts]
    inv_mean = np.mean([q["quality_score"] for q in inv_quality])

    ccs_profile.append(round(ccs_mean, 4))
    inv_profile.append(round(inv_mean, 4))

    results[f"L{layer}"] = {
        "ccs_d1": {"texts": ccs_texts, "quality": ccs_quality, "mean_quality": round(ccs_mean, 4)},
        "inv_d1": {"texts": inv_texts, "quality": inv_quality, "mean_quality": round(inv_mean, 4)},
    }

    print(f"  CCS: {ccs_mean:.4f}  inv: {inv_mean:.4f}  {'<-- CCS worse' if ccs_mean < inv_mean else '<-- inv worse' if inv_mean < ccs_mean else '== tied'}", flush=True)

# Find crossover
ccs_arr = np.array(ccs_profile)
inv_arr = np.array(inv_profile)
diff = ccs_arr - inv_arr
sign_changes = []
for i in range(len(diff) - 1):
    if diff[i] * diff[i+1] < 0:
        sign_changes.append(i + 0.5)

print(f"\n{'='*60}", flush=True)
print("DEPTH CROSSOVER PROFILE", flush=True)
print(f"{'='*60}", flush=True)
print(f"{'Layer':>6} {'CCS D1':>8} {'Inv D1':>8} {'Diff':>8} {'Direction':>12}", flush=True)
print("-" * 45, flush=True)
for i in range(n_layers):
    d = ccs_profile[i] - inv_profile[i]
    direction = "CCS worse" if d < -0.01 else "inv worse" if d > 0.01 else "~equal"
    print(f"  L{i:>3} {ccs_profile[i]:>8.4f} {inv_profile[i]:>8.4f} {d:>+8.4f} {direction:>12}", flush=True)

print(f"\nSign changes at layers: {sign_changes if sign_changes else 'NONE'}", flush=True)
print(f"CCS monotonic? slope={np.polyfit(range(n_layers), ccs_profile, 1)[0]:.6f}", flush=True)
print(f"Inv monotonic? slope={np.polyfit(range(n_layers), inv_profile, 1)[0]:.6f}", flush=True)

out = {
    "model": MODEL,
    "n_layers": n_layers,
    "n_prompts": len(PROMPTS),
    "max_new_tokens": MAX_NEW_TOKENS,
    "design": "All-layer sweep: CCS D1 vs inv-CCS D1 at every layer. Prediction: opposite depth gradients with sharp crossover.",
    "ccs_profile": ccs_profile,
    "inv_profile": inv_profile,
    "sign_changes": sign_changes,
    "results": results,
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
}

outpath = os.path.expanduser("~/chronicle/spectral-demon/experiments/results/f644_depth_crossover.json")
with open(outpath, "w") as f:
    json.dump(out, f, indent=2, default=str)
print(f"\nSaved to {outpath}", flush=True)
