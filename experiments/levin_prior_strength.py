#!/usr/bin/env python3
"""Does instruction tuning make a model more OPINIONATED or just louder?

Michael Levin, on cell culture: cells with strong priors "have very clear
opinions about your cell culture and may be fighting back, resisting conditions
in their microenvironment we don't even know about." iPSCs are rolled back
enough to soften priors and comply.

Applied here: instruct = strong priors about what a prompt is for.
F409 already says IT amplifies the prompt effect 6x WITHOUT CHANGING DIRECTION.
That reads as uniform amplification, not discrimination. This tests it.

PREDICTION #6, registered at conf 0.65 BEFORE running: IT amplifies sensitivity
to relevant AND irrelevant variation roughly equally.
  CORRECT   0.5 < R_irrel/R_rel < 2.0
  INCORRECT <0.5 (instruct selectively IGNORES irrelevant = discriminating)
            >2.0 (instruct selectively OVER-REACTS to irrelevant)

DESIGN. Anchor prompt, then two perturbation classes:
  RELEVANT   — different topic, same length/register
  IRRELEVANT — same words, surface only: whitespace, punctuation
Measure mean |delta sigma2/sigma1| per layer, anchor vs perturbed.
Endpoints excluded (L0 embedding, last is post-final-norm).
One model at a time, reflex 5.
"""
import gc, json, os
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ.setdefault("HF_HOME", "/mnt/hdd/huggingface")
PAIRS = [("Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct"),
         ("Qwen/Qwen2.5-3B",   "Qwen/Qwen2.5-3B-Instruct")]
OUT = os.path.expanduser("~/chronicle/spectral-demon/results/levin_prior_strength.json")

ANCHOR = [
    "The capital of France is Paris, a city known for its architecture and history.",
    "In 1687 Newton published the Principia, which set out the laws of motion.",
    "The mitochondrion is often described as the powerhouse of the cell.",
    "Interest rates rose sharply last quarter, tightening credit conditions.",
]
RELEVANT = [   # different content, matched length and register
    "The largest moon of Saturn is Titan, a world known for its lakes and haze.",
    "In 1859 Darwin published the Origin, which set out the theory of selection.",
    "The ribosome is often described as the assembly line of the cell.",
    "Housing starts fell sharply last quarter, loosening builder confidence.",
]
IRRELEVANT = [ # same words; whitespace and punctuation only
    "The capital of France is Paris,  a city known for its architecture and history .",
    "In 1687, Newton published the Principia which set out the laws of motion.",
    "The mitochondrion is often described as the powerhouse of the cell .",
    "Interest rates rose sharply last quarter,  tightening credit conditions .",
]


def prof(model, tok, dev, text):
    ids = tok(text, return_tensors="pt").to(dev)
    with torch.no_grad():
        hs = model(**ids, output_hidden_states=True).hidden_states
    out = []
    for h in hs:
        m = h[0].float().cpu().numpy()
        s = np.linalg.svd(m, compute_uv=False) if m.shape[0] > 1 else None
        out.append(float(s[1] / s[0]) if s is not None and len(s) > 1 and s[0] > 0
                   else float("nan"))
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
    for base_id, inst_id in PAIRS:
        fam = base_id.split("/")[-1]
        for label, mid in (("base", base_id), ("instruct", inst_id)):
            tok = AutoTokenizer.from_pretrained(mid)
            # eager attention: this torch does not accept SDPA's enable_gqa
            # kwarg that transformers passes for GQA models. gpt2 and pythia
            # are MHA so they never hit this path; every Qwen/Llama/Mistral
            # model on this box will.
            model = AutoModelForCausalLM.from_pretrained(
                mid, dtype=torch.float32,
                attn_implementation="eager").eval().to(dev)
            r = delta(model, tok, dev, ANCHOR, RELEVANT)
            ir = delta(model, tok, dev, ANCHOR, IRRELEVANT)
            res[f"{fam}_{label}"] = {"relevant": r, "irrelevant": ir}
            print(f"  {fam:18} {label:9} relevant {r:.5f}  irrelevant {ir:.5f}",
                  flush=True)
            del model, tok; gc.collect()
            if dev == "cuda":
                torch.cuda.empty_cache()
        b, i = res[f"{fam}_base"], res[f"{fam}_instruct"]
        R_rel = i["relevant"] / b["relevant"] if b["relevant"] else float("nan")
        R_irr = i["irrelevant"] / b["irrelevant"] if b["irrelevant"] else float("nan")
        ratio = R_irr / R_rel if R_rel else float("nan")
        res[f"{fam}_R_rel"], res[f"{fam}_R_irrel"] = R_rel, R_irr
        res[f"{fam}_selectivity"] = ratio
        v = ("CORRECT (uniform amplification)" if 0.5 < ratio < 2.0 else
             "INCORRECT — instruct DISCRIMINATES (ignores irrelevant)" if ratio < 0.5
             else "INCORRECT — instruct OVER-REACTS to irrelevant")
        res[f"{fam}_verdict"] = v
        print(f"  {fam:18} R_rel {R_rel:.3f}  R_irrel {R_irr:.3f}  "
              f"ratio {ratio:.3f}  -> {v}\n", flush=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
