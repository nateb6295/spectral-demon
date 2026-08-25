#!/usr/bin/env python3
"""Depth or width? Content sensitivity above the token floor, on the pythia ladder.

Nate caught the confound: "scale" in the Qwen result meant parameters, depth AND
width moving together (0.5B = 24L/896, 3B = 36L/2048). Pythia separates them:

  410m  24L / 1024  ┐ same depth, 2x width
  1.4b  24L / 2048  ┘
  2.8b  32L / 2560  <- depth step

PREDICTION #7, registered at conf 0.6 BEFORE this ran: the width-only step
produces a SMALLER increase in normalised sensitivity than the depth step.

TOKENISER BASELINE IS RECOMPUTED HERE. The Qwen run normalised by a 4.24x
token-change ratio measured with the Qwen tokeniser. Pythia tokenises
differently, so reusing that number would silently import another model's
baseline — the same class of error as comparing to a random matrix of the wrong
shape. Measured per-model below.
"""
import gc, json, os
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ.setdefault("HF_HOME", "/mnt/hdd/huggingface")
MODELS = ["EleutherAI/pythia-410m", "EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b"]
OUT = os.path.expanduser("~/chronicle/spectral-demon/results/pythia_depth_width.json")

# Homogeneous 12-frame set. The old 4 hand-written pairs had 22x within-set
# variance on the irrelevant edits (three different KINDS of edit), so the mean
# reported which edit was disruptive rather than which model was sensitive.
# New set: one uniform operation, 1.5x spread, stable across tokenisers.
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from stimulus_set import build as _build
ANCHOR, RELEVANT, IRRELEVANT = _build()


def tok_ratio(tok):
    """Token-change ratio for THIS tokeniser — the no-semantics floor."""
    def jac(a, b):
        A, B = set(tok(a)["input_ids"]), set(tok(b)["input_ids"])
        return 1 - len(A & B) / len(A | B)
    r = np.mean([jac(a, b) for a, b in zip(ANCHOR, RELEVANT)])
    i = np.mean([jac(a, b) for a, b in zip(ANCHOR, IRRELEVANT)])
    return float(r), float(i), float(r / i) if i else float("nan")


def prof(model, tok, dev, text):
    ids = tok(text, return_tensors="pt").to(dev)
    with torch.no_grad():
        hs = model(**ids, output_hidden_states=True).hidden_states
    out = []
    for h in hs:
        m = h[0].float().cpu().numpy()
        s = np.linalg.svd(m, compute_uv=False) if m.shape[0] > 1 else None
        out.append(float(s[1] / s[0]) if s is not None and len(s) > 1 and s[0] > 0
                   else np.nan)
    del hs
    return np.array(out[1:-1])          # endpoints dropped


def delta(model, tok, dev, a_list, b_list):
    ds = []
    for a, b in zip(a_list, b_list):
        pa, pb = prof(model, tok, dev, a), prof(model, tok, dev, b)
        n = min(len(pa), len(pb))
        ds.append(np.nanmean(np.abs(pa[:n] - pb[:n])))
    return float(np.mean(ds))


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = {}
    for mid in MODELS:
        name = mid.split("/")[-1]
        tok = AutoTokenizer.from_pretrained(mid)
        tr, ti, ratio = tok_ratio(tok)
        model = AutoModelForCausalLM.from_pretrained(
            mid, dtype=torch.float32).eval().to(dev)     # pythia is MHA, SDPA ok
        r = delta(model, tok, dev, ANCHOR, RELEVANT)
        i = delta(model, tok, dev, ANCHOR, IRRELEVANT)
        norm = (r / i) / ratio if i and ratio else float("nan")
        res[name] = {"layers": model.config.num_hidden_layers,
                     "hidden": model.config.hidden_size,
                     "relevant": r, "irrelevant": i,
                     "resp_ratio": r / i if i else None,
                     "token_ratio": ratio, "normalised": norm}
        print(f"  {name:14} {model.config.num_hidden_layers}L/"
              f"{model.config.hidden_size:<5} resp {r/i:6.2f}  tok {ratio:5.2f}  "
              f"norm {norm:6.2f}", flush=True)
        del model, tok; gc.collect()
        if dev == "cuda":
            torch.cuda.empty_cache()

    a, b, c = (res["pythia-410m"]["normalised"], res["pythia-1.4b"]["normalised"],
               res["pythia-2.8b"]["normalised"])
    d_width, d_depth = b - a, c - b
    res["delta_width_only"] = d_width
    res["delta_depth_step"] = d_depth
    v = ("CORRECT — depth step larger" if d_width < d_depth else
         "INCORRECT — width step larger")
    if abs(d_width - d_depth) < 0.2 * max(abs(d_width), abs(d_depth), 1e-9):
        v = "PARTIAL — within 20%"
    res["prediction_7"] = v
    res["n_frames"] = len(ANCHOR)
    res["note"] = ("rerun with the homogeneous 12-frame set; the original run "
                   "used 4 hand-written pairs with 22x within-set variance")
    print(f"\n  width-only 410m->1.4b : {d_width:+.3f}")
    print(f"  depth step 1.4b->2.8b : {d_depth:+.3f}")
    print(f"  PREDICTION #7 (conf 0.6): {v}")
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
