#!/usr/bin/env python3
"""Is trained depth non-uniformity LEARNED, or driven by input statistics?

Two mesh breaks, run together because they are the same code with different
inputs.

OX: "run the trained model on token-shuffled text. If trained CV falls toward
random levels, most of the trained profile is input-statistics-driven, not
learned specialization — and the 'trained stays variable throughout' mechanism
claim dies."

KIMI: "feed random token IDs to the random twin — if non-flatness survives,
it's architecture (position/sink), not text propagation."

EXPECTATIONS, written before running (reflex 11):
  A trained + real text      baseline, CV ~0.87 (pythia) / ~0.98 (gpt2)
  B trained + shuffled text  if LEARNED, stays near A. If INPUT-DRIVEN, falls
                             toward D. This is the one that can kill the claim.
  C random  + real text      baseline, CV ~0.21 / ~0.10
  D random  + random tokens  if the residual non-flatness is TEXT PROPAGATION,
                             D < C. If ARCHITECTURE, D ~= C.

All CVs computed excluding BOTH endpoints: L0 is the embedding and
hidden_states[-1] is post-final-norm, so neither shares a normalisation with
the body. (Ox caught the second one; I had only excluded the first.)
"""
import gc, json, os, random
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

os.environ.setdefault("HF_HOME", "/mnt/hdd/huggingface")
MODEL = os.environ.get("DM_MODEL", "gpt2")
OUT = os.path.expanduser(
    f"~/chronicle/spectral-demon/results/depth_mechanism_{MODEL.split('/')[-1]}.json")

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
    return float(v[1:-1].std() / v[1:-1].mean())   # both endpoints dropped


def profile(model, tok, dev, mode, seed=0):
    rng = random.Random(seed)
    per = None
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt")["input_ids"]
        if mode == "shuffled":
            flat = ids[0].tolist(); rng.shuffle(flat)
            ids = torch.tensor([flat])
        elif mode == "randtok":
            n = ids.shape[1]
            ids = torch.tensor([[rng.randrange(tok.vocab_size) for _ in range(n)]])
        ids = ids.to(dev)
        with torch.no_grad():
            hs = model(input_ids=ids, output_hidden_states=True).hidden_states
        if per is None:
            per = [[] for _ in hs]
        for i, h in enumerate(hs):
            m = h[0].float().cpu().numpy()
            if m.shape[0] < 2:
                continue
            s = np.linalg.svd(m, compute_uv=False)
            if len(s) > 1 and s[0] > 0:
                per[i].append(float(s[1] / s[0]))
        del hs
    return [float(np.mean(v)) if v else float("nan") for v in per]


def build(arm, dev):
    tok = AutoTokenizer.from_pretrained(MODEL)
    if arm == "trained":
        m = AutoModelForCausalLM.from_pretrained(MODEL)
    else:
        torch.manual_seed(0)
        m = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(MODEL))
    return m.eval().to(dev), tok


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev} model={MODEL}", flush=True)
    res = {}
    for arm, modes in (("trained", ["real", "shuffled"]),
                       ("random", ["real", "randtok"])):
        model, tok = build(arm, dev)            # one model at a time, reflex 5
        for mode in modes:
            k = f"{arm}_{mode}"
            res[k] = profile(model, tok, dev, mode)
            res[k + "_cv"] = cv(res[k])
            print(f"  {k:18} CV={res[k+'_cv']:.4f}", flush=True)
        del model, tok; gc.collect()
        if dev == "cuda":
            torch.cuda.empty_cache()

    A, B = res["trained_real_cv"], res["trained_shuffled_cv"]
    C, D = res["random_real_cv"], res["random_randtok_cv"]
    print(f"\n  A trained+real     {A:.4f}")
    print(f"  B trained+shuffled {B:.4f}   ({100*(B-A)/A:+.0f}% vs A)")
    print(f"  C random+real      {C:.4f}")
    print(f"  D random+randtok   {D:.4f}   ({100*(D-C)/C:+.0f}% vs C)")
    res["verdict_learned_vs_input"] = (
        "LEARNED — shuffling text barely moves the trained profile"
        if abs(B - A) / A < 0.30 else
        "INPUT-DRIVEN — shuffling collapses the trained profile toward random"
        if B < (A + C) / 2 else "AMBIGUOUS")
    res["verdict_random_residual"] = (
        "ARCHITECTURE — random tokens keep the residual non-flatness"
        if abs(D - C) / C < 0.30 else
        "TEXT PROPAGATION — random tokens flatten it further")
    print("\n ", res["verdict_learned_vs_input"])
    print(" ", res["verdict_random_residual"])
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
