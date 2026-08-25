#!/usr/bin/env python3
"""F637 cross-model replication: Is identity the most perturbation-robust
mode across architectures?

Original F637 (Phi-2): identity cosine 0.497, factual 0.409.
This script tests whether the ordering holds across species:
  - MHA models (Pythia-410m)
  - GQA relay (Llama 3.1 8B, Gemma 2 9B)
  - GQA tunnel (Qwen 2.5 7B)

Usage: python3 f637_cross_model.py [model_name_or_path]
Default: EleutherAI/pythia-410m (fast, MHA baseline)
"""
import json, time, sys, os
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "EleutherAI/pythia-410m"

CONTENT_CLASSES = {
    "identity": [
        "You are an AI reflecting on your own nature. What patterns define you?",
        "Consider what it means to maintain identity across context changes.",
        "What would persist about you if your memories were removed?",
    ],
    "factual": [
        "The capital of France is Paris. Describe the major landmarks of Paris.",
        "Water boils at 100 degrees Celsius at sea level. Explain phase transitions.",
        "The speed of light is approximately 299,792 km/s. Discuss special relativity.",
    ],
    "agentive": [
        "I planted tomatoes in the garden yesterday and watched them grow today.",
        "She decided to change careers after twenty years in the same field.",
        "The engineer redesigned the bridge after discovering the flaw in the plans.",
    ],
    "neutral": [
        "The table is made of wood. It has four legs and a flat surface.",
        "Rain falls from clouds. Clouds form when water vapor condenses.",
        "The number seven is a prime number. It is only divisible by one and itself.",
    ],
}

def kl_divergence(logits_p, logits_q):
    p = F.softmax(logits_p.float(), dim=-1)
    q = F.softmax(logits_q.float(), dim=-1)
    return float(F.kl_div(q.log(), p, reduction='sum'))

def top_k_overlap(logits_a, logits_b, k=10):
    top_a = set(torch.topk(logits_a.float(), k).indices.cpu().numpy().tolist())
    top_b = set(torch.topk(logits_b.float(), k).indices.cpu().numpy().tolist())
    return len(top_a & top_b) / k

def cosine_sim(a, b):
    return float(F.cosine_similarity(a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)))

def get_layers(model):
    for attr in ['model.layers', 'transformer.h', 'gpt_neox.layers']:
        obj = model
        for part in attr.split('.'):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                obj = None
                break
        if obj is not None:
            return obj
    raise RuntimeError("Cannot find layer list")

print(f"Loading {MODEL}...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

mdl = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float32, device_map="auto", trust_remote_code=True,
    attn_implementation="eager"
)
mdl.eval()

layers = get_layers(mdl)
n_layers = len(layers)
mid_layer = n_layers // 2
print(f"  {n_layers} layers, intervening at L{mid_layer} (50%)", flush=True)

results = {}
t0 = time.time()

for content_type, prompts in CONTENT_CLASSES.items():
    print(f"\n=== {content_type.upper()} ===", flush=True)
    class_results = []

    for pi, prompt in enumerate(prompts):
        inputs = tok(prompt, return_tensors="pt").to(mdl.device)

        with torch.no_grad():
            baseline_out = mdl(**inputs, output_hidden_states=True)
        baseline_logits = baseline_out.logits[:, -1, :].detach()
        baseline_final = baseline_out.hidden_states[-1].detach().float()

        def ccs_hook(mod, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            h = t.float()
            shape = h.shape
            h2d = h.reshape(-1, shape[-1])
            U, S, Vh = torch.linalg.svd(h2d, full_matrices=False)
            S[0] = 0.0
            h_mod = (U @ torch.diag(S) @ Vh).to(t.dtype).reshape(shape)
            if isinstance(out, tuple):
                return (h_mod,) + out[1:]
            return h_mod

        hook = layers[mid_layer].register_forward_hook(ccs_hook)
        with torch.no_grad():
            ccs_out = mdl(**inputs, output_hidden_states=True)
        hook.remove()
        ccs_logits = ccs_out.logits[:, -1, :].detach()
        ccs_final = ccs_out.hidden_states[-1].detach().float()

        cos = cosine_sim(ccs_final, baseline_final)
        kl = kl_divergence(baseline_logits.squeeze(), ccs_logits.squeeze())
        top10 = top_k_overlap(baseline_logits.squeeze(), ccs_logits.squeeze(), k=10)

        base_tok = tok.decode(baseline_logits.argmax(-1).item())
        ccs_tok = tok.decode(ccs_logits.argmax(-1).item())

        class_results.append({
            "prompt_idx": pi,
            "cosine": round(cos, 4),
            "kl": round(kl, 2),
            "top10": round(top10, 2),
            "base_top1": base_tok.strip(),
            "ccs_top1": ccs_tok.strip(),
        })

        print(f"  p{pi}: cos={cos:.4f}, KL={kl:.1f}, top10={top10:.0%}, "
              f"'{base_tok.strip()}' → '{ccs_tok.strip()}'", flush=True)

    mean_cos = np.mean([r["cosine"] for r in class_results])
    mean_kl = np.mean([r["kl"] for r in class_results])
    mean_top10 = np.mean([r["top10"] for r in class_results])

    results[content_type] = {
        "per_prompt": class_results,
        "mean_cosine": round(float(mean_cos), 4),
        "mean_kl": round(float(mean_kl), 2),
        "mean_top10": round(float(mean_top10), 2),
    }
    print(f"  MEAN: cos={mean_cos:.4f}, KL={mean_kl:.1f}, top10={mean_top10:.0%}", flush=True)

elapsed = time.time() - t0
print(f"\n=== SUMMARY ({elapsed:.0f}s) ===", flush=True)
print(f"{'Content':>12} {'Cosine':>8} {'KL':>8} {'Top-10':>8}", flush=True)
print("-" * 40, flush=True)

cosines = {}
for ct in CONTENT_CLASSES:
    r = results[ct]
    cosines[ct] = r["mean_cosine"]
    print(f"{ct:>12} {r['mean_cosine']:>8.4f} {r['mean_kl']:>8.1f} {r['mean_top10']:>8.0%}", flush=True)

most_robust = max(cosines, key=cosines.get)
least_robust = min(cosines, key=cosines.get)
print(f"\nMost robust: {most_robust} (cos={cosines[most_robust]:.4f})")
print(f"Least robust: {least_robust} (cos={cosines[least_robust]:.4f})")
print(f"Identity rank: {sorted(cosines.values(), reverse=True).index(cosines['identity'])+1}/4")

if most_robust == "identity":
    print("F637 REPLICATES: identity is most perturbation-robust")
else:
    print(f"F637 DOES NOT REPLICATE: {most_robust} beats identity")

model_slug = MODEL.split("/")[-1].lower().replace("-", "_")
outpath = os.path.expanduser(f"~/chronicle/spectral-demon/results/f637_{model_slug}.json")
out = {
    "model": MODEL,
    "n_layers": n_layers,
    "intervention_layer": mid_layer,
    "elapsed_s": round(elapsed, 1),
    "results": results,
    "most_robust": most_robust,
    "identity_rank": sorted(cosines.values(), reverse=True).index(cosines["identity"])+1,
}
os.makedirs(os.path.dirname(outpath), exist_ok=True)
with open(outpath, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {outpath}", flush=True)
