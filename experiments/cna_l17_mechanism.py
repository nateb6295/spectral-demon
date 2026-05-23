#!/usr/bin/env python3
"""L17 mechanism dissection: is binding implemented via attention or MLP?

Treisman's Feature Integration Theory predicts binding requires attention.
If L17 binding is attention-mediated, ablating L17 attention should reproduce
the phase transition while ablating L17 MLP should not (and vice versa).

Three conditions:
  - L17 attention zeroed (last token only)
  - L17 MLP zeroed (last token only)
  - L17 full layer zeroed (replication of L17-only from isolation experiment)
"""
import json, time
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

def pr(acts):
    a = np.array(acts)
    if a.shape[0] < 2: return 0.0
    c = np.cov(a.T)
    ev = np.linalg.eigvalsh(c)
    ev = ev[ev > 0]
    return float((ev.sum()**2) / (ev**2).sum()) if len(ev) else 0.0

print("Loading...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
mdl = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto")
mdl.eval()

# Qwen2 layer structure: each layer has self_attn and mlp submodules
layer17 = mdl.model.layers[17]
print(f"L17 submodules: {[n for n, _ in layer17.named_children()]}", flush=True)

CONDITIONS = {
    "none": {"target": None},
    "L17_full": {"target": "layer"},
    "L17_attn": {"target": "attn"},
    "L17_mlp": {"target": "mlp"},
}

results = {}
for cond_name, cond in CONDITIONS.items():
    print(f"\nCondition: {cond_name}", flush=True)
    for name, sysp in SYSTEMS.items():
        acts = {l: {"rel": [], "gen": []} for l in MEASURE}
        for prompts, cat in [(REL, "rel"), (GEN, "gen")]:
            for p in prompts:
                msgs = [{"role": "system", "content": sysp}, {"role": "user", "content": p}]
                text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                inputs = tok(text, return_tensors="pt").to(mdl.device)

                hooks = []
                layer_out = {}

                def mk_zero_last():
                    def fn(mod, inp, out):
                        if isinstance(out, tuple):
                            out[0][:, -1, :] = 0
                            return out
                        out[:, -1, :] = 0
                        return out
                    return fn

                if cond["target"] == "layer":
                    hooks.append(mdl.model.layers[17].register_forward_hook(mk_zero_last()))
                elif cond["target"] == "attn":
                    hooks.append(mdl.model.layers[17].self_attn.register_forward_hook(mk_zero_last()))
                elif cond["target"] == "mlp":
                    hooks.append(mdl.model.layers[17].mlp.register_forward_hook(mk_zero_last()))

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
        results[f"{cond_name}_{name}"] = prs
        print(f"  {name}: L25 rel={prs['L25']['rel']:.1f} gen={prs['L25']['gen']:.1f}", flush=True)
        torch.cuda.empty_cache()

print("\nCross-name CV at L25:", flush=True)
cv_data = {}
for cond_name in CONDITIONS:
    rels = [results[f"{cond_name}_{n}"]["L25"]["rel"] for n in SYSTEMS]
    gens = [results[f"{cond_name}_{n}"]["L25"]["gen"] for n in SYSTEMS]
    rcv = np.std(rels)/np.mean(rels)*100 if np.mean(rels) > 0 else 0
    gcv = np.std(gens)/np.mean(gens)*100 if np.mean(gens) > 0 else 0
    cv_data[cond_name] = {"rel_cv": round(rcv, 1), "gen_cv": round(gcv, 1)}
    print(f"  {cond_name}: rel_CV={rcv:.1f}% gen_CV={gcv:.1f}%", flush=True)

out = {"results": results, "cv": cv_data, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
with open("/workspace/cna_l17_mechanism_results.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nDone!", flush=True)
