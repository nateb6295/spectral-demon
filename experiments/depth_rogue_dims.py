#!/usr/bin/env python3
"""Is trained depth non-uniformity layer personality, or a few massive activations?

KIMI: "trained depth-variance could be rogue dimensions, not layer personality.
Timkey & van Schijndel (2021): anisotropy statistics are dominated by a few
rogue dims. Sun et al. (2024): massive activations in specific layers. Either
crushes sigma2/sigma1 locally and fakes specialization. Discriminator:
recompute the trained profile with top-k dims clipped. If it survives clipping,
it's broad per-layer reshaping and the species test stands. If it collapses,
the non-uniformity is a massive-activation footprint — and my June rank-1
argument is half-revived."

EXPECTATION, written before running (reflex 11):
  BROAD RESHAPING  -> trained CV stays high as k rises. Clipping a handful of
                      feature dims out of 768 should barely matter.
  ROGUE-DIM ARTIFACT -> trained CV collapses toward the random baseline (~0.11)
                      by k=3 or so, because the thing driving it was a couple
                      of outlier channels.

I do not know which. gpt2 is a known massive-activation model, so a collapse
would not surprise me. Baseline to beat: trained CV 1.0293, random CV 0.1087,
both excluding endpoints.

Clipping = zero the k feature dimensions with the largest mean |activation|
within each layer, then recompute the SVD. Per-layer, because the rogue dims
are not the same at every depth.
"""
import gc, json, os
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

os.environ.setdefault("HF_HOME", "/mnt/hdd/huggingface")
MODEL = os.environ.get("DR_MODEL", "gpt2")
OUT = os.path.expanduser(
    f"~/chronicle/spectral-demon/results/depth_rogue2_{MODEL.split('/')[-1]}.json")
KS = [0, 1, 3, 10, 30]

PROMPTS = [
    "The capital of France is Paris, a city known for its architecture and history.",
    "In 1687 Newton published the Principia, which set out the laws of motion.",
    "The mitochondrion is often described as the powerhouse of the cell.",
    "She walked down to the river and sat on the cold stones until dusk.",
    "Consider a compact Riemannian manifold with strictly positive curvature.",
    "Interest rates rose sharply last quarter, tightening credit conditions.",
    "The elk stood at the treeline, unbothered by the tram passing below.",
    "Proof takes a formal statement and shows how it follows from others.",
]


def cv(v):
    v = np.asarray([x for x in v if np.isfinite(x)])
    return float(v[1:-1].std() / v[1:-1].mean())


def profile(model, tok, dev, k, mode="top"):
    """mode='top' clips the k largest-magnitude dims; mode='rand' clips k
    randomly chosen dims.

    The matched control, added after the top-k sweep showed a SIGN FLIP rather
    than a collapse: trained goes from 9.5x more depth-variable than random at
    k=0 to 0.68x at k=30. Clipping top-k is differentially destructive — it
    removes structured massive activations from the trained model and merely
    the largest noise draws from the random one. If the flip survives clipping
    RANDOM dims at matched k, it is real. If it vanishes, it was the asymmetry.
    """
    per = None
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").to(dev)
        with torch.no_grad():
            hs = model(**ids, output_hidden_states=True).hidden_states
        if per is None:
            per = [[] for _ in hs]
        for i, h in enumerate(hs):
            m = h[0].float().cpu().numpy()          # (S, D)
            if m.shape[0] < 2:
                continue
            if k:
                if mode == "top":
                    idx = np.argsort(np.abs(m).mean(axis=0))[-k:]
                else:
                    idx = np.random.default_rng(i).choice(
                        m.shape[1], size=k, replace=False)
                m = m.copy(); m[:, idx] = 0.0
            s = np.linalg.svd(m, compute_uv=False)
            if len(s) > 1 and s[0] > 0:
                per[i].append(float(s[1] / s[0]))
        del hs
    return [float(np.mean(v)) if v else float("nan") for v in per]


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev} model={MODEL}", flush=True)
    res = {}
    for arm in ("trained", "random"):
        tok = AutoTokenizer.from_pretrained(MODEL)
        if arm == "trained":
            model = AutoModelForCausalLM.from_pretrained(MODEL)
        else:
            torch.manual_seed(0)
            model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(MODEL))
        model.eval().to(dev)
        for mode in ("top", "rand"):
            for k in KS:
                c = cv(profile(model, tok, dev, k, mode))
                res[f"{arm}_{mode}_k{k}"] = c
                print(f"  {arm:8} clip {mode}-{k:<3} CV={c:.4f}", flush=True)
        del model, tok; gc.collect()
        if dev == "cuda":
            torch.cuda.empty_cache()

    t0, r0 = res["trained_top_k0"], res["random_top_k0"]
    flip_top = res["trained_top_k30"] < res["random_top_k30"]
    flip_rand = res["trained_rand_k30"] < res["random_rand_k30"]
    res["sign_flip_top_k"] = flip_top
    res["sign_flip_random_dims"] = flip_rand
    res["flip_is_real"] = bool(flip_top and flip_rand)
    print(f"\n  sign flip with top-k clipping   : {flip_top}")
    print(f"  sign flip with RANDOM-dim clipping: {flip_rand}   <- the control")
    survives = all(res[f"trained_top_k{k}"] > (t0 + r0) / 2 for k in KS)
    res["verdict"] = ("BROAD RESHAPING — trained CV stays well above the "
                      "trained/random midpoint at every k"
                      if survives else
                      "ROGUE-DIM FOOTPRINT — clipping collapses trained CV "
                      "toward random; Kimi's rank-1 argument half-revived")
    print(f"\n  midpoint to beat: {(t0+r0)/2:.4f}")
    print(" ", res["verdict"])
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
