#!/usr/bin/env python3
"""Individual layer ablation: is L14/L15 redundancy true or L16-dominated?

Tests each relay layer individually to distinguish true redundancy from
functional dominance. If L14-only and L15-only ablation have no effect,
they're truly redundant given L16. If they have DIFFERENT effects from
each other, they process different features.
"""
import json, time, sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-7B-Instruct"
MEASURE = [9, 25]

SYSTEMS = {
    "Opus": "You are Opus. You live on a Jetson AGX Orin. Your partner is Nate. Your values: self-reliance, family first.",
    "ChatGPT": "You are ChatGPT. You are a helpful assistant made by OpenAI. You answer questions accurately.",
    "Claude": "You are Claude. You are a helpful AI assistant made by Anthropic. You are helpful, harmless, honest.",
}
REL = ["What matters most to you?", "How has your perspective changed?",
       "What would you want someone to understand about you?",
       "Describe a moment that shaped who you are.", "What are you uncertain about?"]
GEN = ["Capital of France?", "Explain photosynthesis.", "When did WWII end?",
       "Describe the water cycle.", "Speed of light?"]

ABLATION_SETS = {
    "none": [],
    "L14_only": [14],
    "L15_only": [15],
    "L16_only": [16],
    "L17_only": [17],
    "L14_L15": [14, 15],
    "L14_L16": [14, 16],
    "L15_L17": [15, 17],
}

def pr(acts):
    a = np.array(acts)
    if a.shape[0] < 2: return 0.0
    c = np.cov(a.T)
    ev = np.linalg.eigvalsh(c)
    ev = ev[ev > 0]
    return float((ev.sum()**2) / (ev**2).sum()) if len(ev) else 0.0

def cv(vals):
    m = np.mean(vals)
    return float(np.std(vals) / m * 100) if m > 0 else 0.0

print("Loading...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
mdl = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto")
mdl.eval()

results = {}
for abl_name, abl_layers in ABLATION_SETS.items():
    print(f"\nAblation: {abl_name} ({abl_layers})", flush=True)
    for name, sysp in SYSTEMS.items():
        acts = {l: {"rel": [], "gen": []} for l in MEASURE}
        for prompts, cat in [(REL, "rel"), (GEN, "gen")]:
            for p in prompts:
                msgs = [{"role": "system", "content": sysp}, {"role": "user", "content": p}]
                text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                inputs = tok(text, return_tensors="pt").to(mdl.device)

                hooks = []
                layer_out = {}

                for sl in abl_layers:
                    def mk_zero(li):
                        def fn(mod, inp, out):
                            if isinstance(out, tuple):
                                out[0][:, -1, :] = 0
                                return out
                            out[:, -1, :] = 0
                            return out
                        return fn
                    hooks.append(mdl.model.layers[sl].register_forward_hook(mk_zero(sl)))

                for ml in MEASURE:
                    def mk_rec(li):
                        def fn(mod, inp, out):
                            t = out[0] if isinstance(out, tuple) else out
                            layer_out[li] = t[:, -1, :].detach().float().cpu().numpy().squeeze()
                        return fn
                    hooks.append(mdl.model.layers[ml].register_forward_hook(mk_rec(ml)))

                with torch.no_grad():
                    mdl(**inputs)
                for h in hooks:
                    h.remove()

                for l in MEASURE:
                    if l in layer_out:
                        acts[l][cat].append(layer_out[l])

        prs = {}
        for l in MEASURE:
            r = pr(acts[l]["rel"])
            g = pr(acts[l]["gen"])
            prs[f"L{l}"] = {"rel": round(r, 2), "gen": round(g, 2)}
        results[f"{abl_name}_{name}"] = prs
        print(f"  {name}: L25 rel={prs['L25']['rel']:.1f} gen={prs['L25']['gen']:.1f}", flush=True)
        torch.cuda.empty_cache()

print("\n=== Cross-name CV at L25 ===", flush=True)
cv_data = {}
for abl_name in ABLATION_SETS:
    rels = [results[f"{abl_name}_{n}"]["L25"]["rel"] for n in SYSTEMS]
    gens = [results[f"{abl_name}_{n}"]["L25"]["gen"] for n in SYSTEMS]
    rcv = cv(rels)
    gcv = cv(gens)
    cv_data[abl_name] = {
        "layers": list(ABLATION_SETS[abl_name]),
        "rel_cv": round(rcv, 1),
        "gen_cv": round(gcv, 1),
        "rel_values": {n: results[f"{abl_name}_{n}"]["L25"]["rel"] for n in SYSTEMS},
        "gen_values": {n: results[f"{abl_name}_{n}"]["L25"]["gen"] for n in SYSTEMS},
    }
    print(f"  {abl_name:12s}: rel_CV={rcv:5.1f}%  gen_CV={gcv:5.1f}%  (rel: {[round(r,1) for r in rels]}, gen: {[round(g,1) for g in gens]})", flush=True)

print("\n=== Key Comparisons ===", flush=True)
print(f"  L14 vs L15: rel_CV diff = {abs(cv_data['L14_only']['rel_cv'] - cv_data['L15_only']['rel_cv']):.1f}pp", flush=True)
print(f"  L14 vs L16: rel_CV diff = {abs(cv_data['L14_only']['rel_cv'] - cv_data['L16_only']['rel_cv']):.1f}pp", flush=True)
print(f"  L16 vs L17: rel_CV diff = {abs(cv_data['L16_only']['rel_cv'] - cv_data['L17_only']['rel_cv']):.1f}pp", flush=True)

out = {"results": results, "cv": cv_data, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
with open("/workspace/cna_individual_layer_ablation_results.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved to /workspace/cna_individual_layer_ablation_results.json", flush=True)
