#!/usr/bin/env python3
"""Is depth-invariance of sigma2/sigma1 learned, or the baseline?

PREDICTION #3, registered 2026-08-23 07:05 at conf 0.75 BEFORE this ran:
  random-init sigma2/sigma1 is near-perfectly FLAT across depth
  (std/mean < 0.01), and the TRAINED model of matched architecture is MORE
  variable across depth, not less.

  CORRECT  : random std/mean < 0.01 AND trained std/mean > random std/mean
  INCORRECT: trained flatter than random, or random not flat
  PARTIAL  : random flat but trained comparable rather than clearly more variable

WHY IT MATTERS. The papers' non-trivial claim (journal 2026-06-28, answering
Kimi's rank-1 argument) is not that the ratio is low — that follows from rank-1
token trajectories almost by construction — but that the population geometry is
INVARIANT ACROSS DEPTH despite nonlinear attention+MLP maps. If a random-init
network is already perfectly depth-invariant, invariance is free and cannot be
the evidence. The claim would need restating as "holds a specific NON-RANDOM
value across depth," which the magnitude nulls of 2026-08-23 do support.

Basis for the prediction: cna_random_init_criticality.json (Qwen2.5-7B,
power-law exponent, 8 layers) gave random attention std 0.0000, mlp 0.0006 vs
trained 0.0329 and 0.0031. DIFFERENT QUANTITY, hence 0.75 and not higher.

One model at a time (reflex 5). No logit lens — hidden states and SVD only, so
the double-norm bug cannot apply.
"""
import gc, json, os, sys
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

os.environ.setdefault("HF_HOME", "/mnt/hdd/huggingface")
MODEL = os.environ.get("DI_MODEL", "EleutherAI/pythia-410m")
OUT = os.path.expanduser("~/chronicle/spectral-demon/results/depth_invariance_null_%s.json" % MODEL.split("/")[-1] + "")

PROMPTS = [
    "The capital of France is Paris, a city known for its architecture and history.",
    "In 1687 Newton published the Principia, which set out the laws of motion.",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr",
    "The mitochondrion is often described as the powerhouse of the cell.",
    "She walked down to the river and sat on the cold stones until dusk.",
    "Consider a compact Riemannian manifold with strictly positive curvature.",
    "Interest rates rose sharply last quarter, tightening credit conditions.",
    "The elk stood at the treeline, unbothered by the tram passing below.",
]


def ratios(model, tok, dev):
    """sigma2/sigma1 per layer, pooled over prompts."""
    per_layer = None
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = model(**ids, output_hidden_states=True)
        hs = out.hidden_states                      # tuple(L+1) of (1,S,D)
        if per_layer is None:
            per_layer = [[] for _ in hs]
        for i, h in enumerate(hs):
            m = h[0].float().cpu().numpy()          # (S, D)
            if m.shape[0] < 2:
                continue
            s = np.linalg.svd(m, compute_uv=False)
            if len(s) > 1 and s[0] > 0:
                per_layer[i].append(float(s[1] / s[0]))
        del out, hs
    return [float(np.mean(v)) if v else float("nan") for v in per_layer]


def run(arm, dev):
    tok = AutoTokenizer.from_pretrained(MODEL)
    if arm == "trained":
        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
    else:
        cfg = AutoConfig.from_pretrained(MODEL)
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(cfg)   # random init, same arch
    model.eval().to(dev)
    r = ratios(model, tok, dev)
    del model, tok
    gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()
    return r


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}  model={MODEL}", flush=True)
    res = {}
    for arm in ("trained", "random"):        # one at a time, reflex 5
        print(f"[{arm}] loading…", flush=True)
        res[arm] = run(arm, dev)
        v = np.array([x for x in res[arm] if np.isfinite(x)])
        res[arm + "_std_over_mean"] = float(v.std() / v.mean())
        print(f"[{arm}] layers={len(v)} mean={v.mean():.4f} "
              f"std={v.std():.4f} std/mean={v.std()/v.mean():.4f}", flush=True)

    r_sm, t_sm = res["random_std_over_mean"], res["trained_std_over_mean"]
    if r_sm < 0.01 and t_sm > r_sm:
        verdict = "CORRECT"
    elif t_sm < r_sm or r_sm >= 0.01:
        verdict = "INCORRECT"
    else:
        verdict = "PARTIAL"
    res["prediction_3_verdict"] = verdict
    res["criteria"] = "CORRECT iff random std/mean < 0.01 AND trained > random"
    print(f"\nrandom std/mean  = {r_sm:.5f}")
    print(f"trained std/mean = {t_sm:.5f}")
    print(f"PREDICTION #3 (conf 0.75): {verdict}")
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
