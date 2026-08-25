#!/usr/bin/env python3
"""Does the rank-1 basin survive top-k clipping, or IS it the rogue channels?

PREDICTION #5, registered at conf 0.6 BEFORE this ran: the basin largely
flattens — cliff ratio drops below 2x from 4.2x — i.e. basin and rogue dims are
one phenomenon.

Confidence is 0.6 not 0.8 because this is an ABSENCE prediction and my scored
record is 0-for-3 on absence, 1-for-1 on presence. The discount is explicit.

Prints the FULL per-layer profile at every k, because the lesson of the morning
is that CV hides where the contractive layers sit (Ox), and because I read the
same table three times and got three different amounts out of it.
"""
import gc, json, os
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ.setdefault("HF_HOME", "/mnt/hdd/huggingface")
MODEL = os.environ.get("BS_MODEL", "gpt2")
OUT = os.path.expanduser(
    f"~/chronicle/spectral-demon/results/basin_clip_{MODEL.split('/')[-1]}.json")
KS = [0, 3, 10, 30]
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


def profile(model, tok, dev, k):
    per = None
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").to(dev)
        with torch.no_grad():
            hs = model(**ids, output_hidden_states=True).hidden_states
        if per is None:
            per = [[] for _ in hs]
        for i, h in enumerate(hs):
            m = h[0].float().cpu().numpy()
            if m.shape[0] < 2:
                continue
            if k:
                idx = np.argsort(np.abs(m).mean(axis=0))[-k:]
                m = m.copy(); m[:, idx] = 0.0
            s = np.linalg.svd(m, compute_uv=False)
            if len(s) > 1 and s[0] > 0:
                per[i].append(float(s[1] / s[0]))
        del hs
    return [float(np.mean(v)) if v else float("nan") for v in per]


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL).eval().to(dev)
    res = {}
    for k in KS:
        res[f"k{k}"] = profile(model, tok, dev, k)
    del model, tok; gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()

    n = len(res["k0"])
    print(f"{MODEL} — full per-layer profile at each clip level\n")
    print("layer " + "".join(f"{'k='+str(k):>10}" for k in KS))
    for i in range(n):
        print(f"L{i:<4} " + "".join(f"{res['k'+str(k)][i]:>10.4f}" for k in KS))
    print()
    for k in KS:
        v = np.array(res[f"k{k}"])
        cliff = v[2] / v[3] if v[3] > 0 else float("inf")
        body = v[1:-1]
        res[f"k{k}_cliff_L2_L3"] = float(cliff)
        res[f"k{k}_body_min"] = float(body.min())
        res[f"k{k}_body_max"] = float(body.max())
        print(f"  k={k:<3} L2/L3 cliff {cliff:6.2f}x   body min {body.min():.4f} "
              f"max {body.max():.4f}  range {body.max()-body.min():.4f}")

    c0, c30 = res["k0_cliff_L2_L3"], res["k30_cliff_L2_L3"]
    verdict = ("CORRECT — basin flattens, cliff below 2x" if c30 < 2.0 else
               "INCORRECT — cliff survives above 3x" if c30 > 3.0 else
               "PARTIAL — cliff attenuates to 2-3x")
    res["prediction_5"] = verdict
    print(f"\n  cliff {c0:.2f}x -> {c30:.2f}x")
    print(f"  PREDICTION #5 (conf 0.6): {verdict}")
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
