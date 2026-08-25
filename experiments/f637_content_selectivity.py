#!/usr/bin/env python3
"""F637: Content selectivity of CCS departure + behavioral validation.

Two questions:
1. Is CCS's creative departure (cosine ~0.48) content-selective?
   Compare identity-relevant vs factual vs neutral prompts.
2. Does spectral departure translate to behavioral divergence?
   Measure logit KL divergence and top-k token overlap.

If cosine departure is similar across content types, CCS is a generic
perturbation tool. If it's larger for identity prompts, there's
content-selective creative response.
"""
import json, time, sys, os
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "microsoft/phi-2"
INTERVENTION_LAYER = 8

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
    """KL(P || Q) where P and Q are logit vectors."""
    p = F.softmax(logits_p.float(), dim=-1)
    q = F.softmax(logits_q.float(), dim=-1)
    return float(F.kl_div(q.log(), p, reduction='sum'))

def top_k_overlap(logits_a, logits_b, k=10):
    """Fraction of top-k tokens shared between two logit distributions."""
    top_a = set(torch.topk(logits_a.float(), k).indices.cpu().numpy().tolist())
    top_b = set(torch.topk(logits_b.float(), k).indices.cpu().numpy().tolist())
    return len(top_a & top_b) / k

def cosine_sim(a, b):
    return float(F.cosine_similarity(a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)))

print(f"Loading {MODEL}...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
mdl = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float32, device_map="auto", trust_remote_code=True,
    attn_implementation="eager"
)
mdl.eval()
n_layers = len(mdl.model.layers)
print(f"  {n_layers} layers", flush=True)

results = {}

for content_type, prompts in CONTENT_CLASSES.items():
    print(f"\n=== {content_type.upper()} ===", flush=True)
    class_results = []

    for pi, prompt in enumerate(prompts):
        inputs = tok(prompt, return_tensors="pt").to(mdl.device)

        # BASELINE
        with torch.no_grad():
            baseline_out = mdl(**inputs, output_hidden_states=True)
        baseline_logits = baseline_out.logits[:, -1, :].detach()
        baseline_final = baseline_out.hidden_states[-1].detach().float()

        # CCS (remove sigma-1 at intervention layer)
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

        hook = mdl.model.layers[INTERVENTION_LAYER].register_forward_hook(ccs_hook)
        with torch.no_grad():
            ccs_out = mdl(**inputs, output_hidden_states=True)
        hook.remove()
        ccs_logits = ccs_out.logits[:, -1, :].detach()
        ccs_final = ccs_out.hidden_states[-1].detach().float()

        # Metrics
        cos = cosine_sim(ccs_final, baseline_final)
        kl = kl_divergence(baseline_logits.squeeze(), ccs_logits.squeeze())
        top10 = top_k_overlap(baseline_logits.squeeze(), ccs_logits.squeeze(), k=10)
        top50 = top_k_overlap(baseline_logits.squeeze(), ccs_logits.squeeze(), k=50)

        # Most likely token under each
        base_tok = tok.decode(baseline_logits.argmax(-1).item())
        ccs_tok = tok.decode(ccs_logits.argmax(-1).item())
        same_top1 = base_tok == ccs_tok

        class_results.append({
            "prompt_idx": pi,
            "cosine_departure": round(cos, 4),
            "kl_divergence": round(kl, 2),
            "top10_overlap": round(top10, 2),
            "top50_overlap": round(top50, 2),
            "same_top1": same_top1,
            "baseline_top1": base_tok.strip(),
            "ccs_top1": ccs_tok.strip(),
        })

        print(f"  p{pi}: cos={cos:.4f}, KL={kl:.1f}, top10={top10:.0%}, "
              f"top1: '{base_tok.strip()}' → '{ccs_tok.strip()}'", flush=True)

    # Class summary
    mean_cos = np.mean([r["cosine_departure"] for r in class_results])
    mean_kl = np.mean([r["kl_divergence"] for r in class_results])
    mean_top10 = np.mean([r["top10_overlap"] for r in class_results])
    top1_changed = sum(1 for r in class_results if not r["same_top1"])

    results[content_type] = {
        "per_prompt": class_results,
        "mean_cosine_departure": round(float(mean_cos), 4),
        "mean_kl_divergence": round(float(mean_kl), 2),
        "mean_top10_overlap": round(float(mean_top10), 2),
        "top1_changed": f"{top1_changed}/{len(prompts)}",
    }

    print(f"  MEAN: cos={mean_cos:.4f}, KL={mean_kl:.1f}, top10={mean_top10:.0%}, "
          f"top1 changed: {top1_changed}/{len(prompts)}", flush=True)

# Summary table
print("\n=== CONTENT SELECTIVITY SUMMARY ===", flush=True)
print(f"{'Content':>12} {'Cosine':>8} {'KL Div':>8} {'Top-10':>8} {'Top-1 Changed':>15}", flush=True)
print("-" * 55, flush=True)
for ct in CONTENT_CLASSES:
    r = results[ct]
    print(f"{ct:>12} {r['mean_cosine_departure']:>8.4f} {r['mean_kl_divergence']:>8.1f} "
          f"{r['mean_top10_overlap']:>8.0%} {r['top1_changed']:>15}", flush=True)

outpath = os.path.expanduser("~/chronicle/spectral-demon/experiments/results/f637_content_selectivity.json")
out = {
    "model": MODEL,
    "intervention_layer": INTERVENTION_LAYER,
    "content_classes": list(CONTENT_CLASSES.keys()),
    "results": results,
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
}
with open(outpath, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved to {outpath}", flush=True)
