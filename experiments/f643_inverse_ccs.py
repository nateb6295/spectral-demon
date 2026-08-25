#!/usr/bin/env python3
"""F643: Inverse-CCS — keep only top-D singular values, zero the rest.

CCS removes σ₁ (position structure), preserving σ₂+ (semantic content).
Inverse-CCS removes σ₂+ (semantic content), preserving only top-D SVs.

Prediction from the aphasia 2×2:
- CCS output: meaning without structure (Broca pole) — effortful, meaningful
- Inv-CCS D=1: structure without meaning (Wernicke pole) — fluent nonsense
- Inv-CCS D=5: partial content leaks back
- Normal: both intact

If the σ₁/σ₂ decomposition is functionally real (not just mathematical),
the generated text should show this dissociation.

Connection to F561: MLP ER is universal (problem-determined), attention ER
is architecture-dependent. CCS removes the architecture-specific component.
Inv-CCS removes the universal component.
"""
import json, time, sys, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B"
INTERVENTION_LAYERS = [5, 11, 17]
KEEP_RANKS = [1, 2, 5, 10]
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


def make_invccs_hook(keep_d):
    """Keep only top-D singular values, zero the rest."""
    meta = {}
    def hook(mod, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        h = t.float()
        shape = h.shape
        h2d = h.reshape(-1, shape[-1])

        U, S, Vh = torch.linalg.svd(h2d, full_matrices=False)
        meta["original_spectrum"] = S[:20].cpu().tolist()
        meta["total_energy"] = float((S ** 2).sum())
        meta["kept_energy"] = float((S[:keep_d] ** 2).sum())

        S_trunc = S.clone()
        S_trunc[keep_d:] = 0.0
        h_mod = (U @ torch.diag(S_trunc) @ Vh).to(t.dtype).reshape(shape)

        if isinstance(out, tuple):
            return (h_mod,) + out[1:]
        return h_mod
    return hook, meta


def make_ccs_hook(dose=1):
    """Standard CCS: zero top-D singular values."""
    meta = {}
    def hook(mod, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        h = t.float()
        shape = h.shape
        h2d = h.reshape(-1, shape[-1])

        U, S, Vh = torch.linalg.svd(h2d, full_matrices=False)
        meta["original_spectrum"] = S[:20].cpu().tolist()
        meta["total_energy"] = float((S ** 2).sum())

        S_trunc = S.clone()
        S_trunc[:dose] = 0.0
        meta["kept_energy"] = float((S_trunc ** 2).sum())
        h_mod = (U @ torch.diag(S_trunc) @ Vh).to(t.dtype).reshape(shape)

        if isinstance(out, tuple):
            return (h_mod,) + out[1:]
        return h_mod
    return hook, meta


def generate_with_hook(mdl, tok, prompt, layer_idx, hook_fn, meta_dict):
    """Generate text with a hook applied at one layer."""
    inputs = tok(prompt, return_tensors="pt").to(mdl.device)
    hook_handle = mdl.model.layers[layer_idx].register_forward_hook(hook_fn)

    with torch.no_grad():
        output_ids = mdl.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
        )

    hook_handle.remove()
    generated = tok.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return generated, dict(meta_dict)


results = {}

for int_layer in INTERVENTION_LAYERS:
    print(f"\n{'='*60}", flush=True)
    print(f"Intervention layer L{int_layer}", flush=True)
    print(f"{'='*60}", flush=True)

    layer_results = {}

    for pi, prompt in enumerate(PROMPTS):
        print(f"\n  Prompt {pi}: \"{prompt[:50]}...\"", flush=True)
        prompt_results = {}

        # Baseline (no hook)
        inputs = tok(prompt, return_tensors="pt").to(mdl.device)
        with torch.no_grad():
            out_ids = mdl.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        baseline_text = tok.decode(out_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        prompt_results["baseline"] = {"text": baseline_text}
        print(f"    baseline: {baseline_text[:80]}...", flush=True)

        # CCS D1 (standard)
        hook_fn, meta = make_ccs_hook(dose=1)
        text, m = generate_with_hook(mdl, tok, prompt, int_layer, hook_fn, meta)
        prompt_results["ccs_d1"] = {"text": text, "meta": m}
        print(f"    ccs_d1:   {text[:80]}...", flush=True)

        # CCS D2
        hook_fn, meta = make_ccs_hook(dose=2)
        text, m = generate_with_hook(mdl, tok, prompt, int_layer, hook_fn, meta)
        prompt_results["ccs_d2"] = {"text": text, "meta": m}
        print(f"    ccs_d2:   {text[:80]}...", flush=True)

        # Inverse-CCS at each keep rank
        for keep_d in KEEP_RANKS:
            hook_fn, meta = make_invccs_hook(keep_d)
            text, m = generate_with_hook(mdl, tok, prompt, int_layer, hook_fn, meta)
            key = f"inv_d{keep_d}"
            energy_frac = m.get("kept_energy", 0) / max(m.get("total_energy", 1), 1e-10)
            prompt_results[key] = {"text": text, "meta": m, "energy_fraction": round(energy_frac, 4)}
            print(f"    inv_d{keep_d:>2}:  {text[:80]}... (energy: {energy_frac:.1%})", flush=True)

        layer_results[f"prompt_{pi}"] = prompt_results

    results[f"L{int_layer}"] = layer_results

# Summary: count repetition, measure diversity
print(f"\n{'='*60}", flush=True)
print("SUMMARY: Text quality indicators", flush=True)
print(f"{'='*60}", flush=True)

def text_stats(text):
    words = text.split()
    if not words:
        return {"length": 0, "unique_ratio": 0, "repetition": 0}
    unique = len(set(words))
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
    unique_bigrams = len(set(bigrams)) / max(len(bigrams), 1)
    return {
        "length": len(words),
        "unique_word_ratio": round(unique / len(words), 3),
        "unique_bigram_ratio": round(unique_bigrams, 3),
    }

summary = {}
for layer_key, layer_data in results.items():
    for prompt_key, prompt_data in layer_data.items():
        for cond, cond_data in prompt_data.items():
            if cond not in summary:
                summary[cond] = []
            stats = text_stats(cond_data["text"])
            summary[cond].append(stats)

print(f"\n{'Condition':>12} {'Unique Words':>13} {'Unique Bigrams':>15} {'Mean Length':>12}", flush=True)
print("-" * 55, flush=True)
for cond in ["baseline", "ccs_d1", "ccs_d2"] + [f"inv_d{d}" for d in KEEP_RANKS]:
    if cond in summary:
        entries = summary[cond]
        uw = np.mean([e["unique_word_ratio"] for e in entries])
        ub = np.mean([e["unique_bigram_ratio"] for e in entries])
        ml = np.mean([e["length"] for e in entries])
        print(f"{cond:>12} {uw:>13.3f} {ub:>15.3f} {ml:>12.1f}", flush=True)

# Energy retention
print(f"\n{'Condition':>12} {'Energy Retained':>16}", flush=True)
print("-" * 30, flush=True)
for keep_d in KEEP_RANKS:
    key = f"inv_d{keep_d}"
    energies = []
    for layer_data in results.values():
        for prompt_data in layer_data.values():
            if key in prompt_data and "energy_fraction" in prompt_data[key]:
                energies.append(prompt_data[key]["energy_fraction"])
    if energies:
        print(f"{key:>12} {np.mean(energies):>15.1%}", flush=True)

outpath = os.path.expanduser("~/chronicle/spectral-demon/experiments/results/f643_inverse_ccs.json")
out = {
    "model": MODEL,
    "intervention_layers": INTERVENTION_LAYERS,
    "keep_ranks": KEEP_RANKS,
    "n_prompts": len(PROMPTS),
    "n_layers": n_layers,
    "max_new_tokens": MAX_NEW_TOKENS,
    "design": "Inverse-CCS: keep top-D SVs only. Aphasia 2x2 prediction.",
    "predictions": {
        "inv_d1": "fluent nonsense (Wernicke pole) — structure without meaning",
        "inv_d5": "partial content recovery",
        "ccs_d1": "meaning disruption with structural preservation (Broca pole)",
        "ccs_d2": "stronger meaning disruption",
    },
    "results": results,
    "summary": summary,
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
}
with open(outpath, "w") as f:
    json.dump(out, f, indent=2, default=str)
print(f"\nSaved to {outpath}", flush=True)
