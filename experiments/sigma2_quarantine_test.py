#!/usr/bin/env python3
"""Is sigma_2 clean, or does it inherit the SECOND massive activation?

All afternoon I have leaned on Kimi's quarantine argument: SVD sorts by
variance, the sink dominates component one, therefore sigma_2 lives in a
subspace the sink does not occupy, therefore every sigma_2 result survives the
F114(i) retraction and is in fact strengthened by it.

Kimi has now pointed out the hole: **SVD guarantees v2 orthogonal to v1. It does
NOT guarantee v2 orthogonal to THE SINK.** If the sink subspace has rank > 1,
sigma_2 inherits the leftover.

And I already measured that it does, this morning, without noticing. Sun et al.
2402.17762 sec 2.2 category (b) is "starting token AND the first strong
delimiter." pythia-2.8b is category (b): first token 14.2x median norm, first
delimiter 19.9x. TWO high-norm rows. So in that model the sink is demonstrably
rank >= 2, and the quarantine argument is not safe there.

BUILT-IN CONTRAST, which is why this is worth running rather than assuming:
  gpt2, pythia-410m   -> Sun category (a), starting token ONLY, sink rank 1
  pythia-2.8b         -> Sun category (b), token + first delimiter, rank >= 2

EXPECTATION, written before running (reflex 9):
  QUARANTINE HOLDS (cat a) -> in gpt2 and pythia-410m, |cos(v2, h_top1)| low and
      |cos(v2, h_2nd)| low, because there is no second sink to leak into v2.
  QUARANTINE BREAKS (cat b) -> in pythia-2.8b, |cos(v2, h_delim)| HIGH in the
      layers where both sites are active. v1 takes the bigger site, v2 takes
      the other one, and "sigma_2 carries individual signal" is contaminated in
      exactly the models Sun calls category (b).
  KILL -> if 2.8b's |cos(v2, h_2nd)| stays below ~0.5 while both sites are hot,
      the quarantine survives even at rank 2 and I say so.

I expect the break, ~0.7. This is a prediction AGAINST the position I have
been publicly defending since noon, which is the direction I am least likely to
be motivated-wrong in — and also the direction where a clean confirmation feels
righteous. Checked the arithmetic, not the feeling: if two rows carry 14x and
20x the norm of everything else, the top TWO singular directions are those two
rows, and that is forced, not empirical.

Method: no intervention. v1, v2 from SVD of the (seq x d) hidden states. h_top1
and h_top2 are the unit residuals of the two highest-norm token positions.
bf16 -- fp16 overflows on these activations.
"""
import gc, json, os

os.environ.setdefault("HF_HOME", "/mnt/hdd/huggingface")
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from stimulus_set import build

MODELS = [("gpt2", "a"), ("EleutherAI/pythia-410m", "a"), ("EleutherAI/pythia-2.8b", "b")]


def analyse(model_id, device):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, attn_implementation="eager").to(device).eval()
    anchors, _, _ = build()
    out = {k: [] for k in ("nr1", "nr2", "c1_1", "c2_1", "c2_2", "tok2")}
    for text in anchors:
        ids = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            o = model(**ids, output_hidden_states=True)
        hs = torch.stack([h[0].float() for h in o.hidden_states])
        assert torch.isfinite(hs).all()
        toks = [tok.decode([i]) for i in ids["input_ids"][0].tolist()]
        rows = {k: [] for k in out}
        for l in range(hs.shape[0]):
            H = hs[l]
            norms = H.norm(dim=-1)
            med = norms[1:-1].median().clamp_min(1e-6)
            order = torch.argsort(norms, descending=True)
            t1, t2 = int(order[0]), int(order[1])
            _, _, Vh = torch.linalg.svd(H, full_matrices=False)
            v1, v2 = Vh[0], Vh[1]
            h1 = H[t1] / H[t1].norm().clamp_min(1e-6)
            h2 = H[t2] / H[t2].norm().clamp_min(1e-6)
            rows["nr1"].append(float(norms[t1] / med))
            rows["nr2"].append(float(norms[t2] / med))
            rows["c1_1"].append(abs(float(v1 @ h1)))
            rows["c2_1"].append(abs(float(v2 @ h1)))
            rows["c2_2"].append(abs(float(v2 @ h2)))
            rows["tok2"].append(repr(toks[t2])[:8])
        for k in out:
            out[k].append(rows[k])
    del model; gc.collect(); torch.cuda.empty_cache()
    return {k: (np.array(v) if k != "tok2" else v) for k, v in out.items()}


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    summary = {}
    for mid, cat in MODELS:
        print(f"\n=== {mid}   (Sun category {cat})", flush=True)
        r = analyse(mid, dev)
        L = r["nr1"].shape[1]
        print(f"  {'layer':>5} {'top1/med':>9} {'top2/med':>9} {'|cos v1,h1|':>12} "
              f"{'|cos v2,h1|':>12} {'|cos v2,h2|':>12}")
        for l in range(L):
            print(f"  {l:>5} {r['nr1'][:,l].mean():>9.2f} {r['nr2'][:,l].mean():>9.2f} "
                  f"{r['c1_1'][:,l].mean():>12.3f} {r['c2_1'][:,l].mean():>12.3f} "
                  f"{r['c2_2'][:,l].mean():>12.3f}")
        # "both sites hot" = second-highest row also well above median
        hot = r["nr2"].mean(0) > 3
        band = np.where(hot)[0]
        if len(band):
            c22 = r["c2_2"][:, band].mean()
            print(f"  layers with a SECOND high-norm row (>3x median): {band.tolist()}")
            print(f"  mean |cos(v2, h_2nd)| in that band = {c22:.3f}")
        else:
            c22 = float(r["c2_2"].mean())
            print(f"  NO second high-norm row anywhere. mean |cos(v2,h_2nd)| = {c22:.3f}")
        summary[mid] = {"cat": cat, "n_hot": int(len(band)), "c2_2_hot": float(c22),
                        "max_nr2": float(r["nr2"].mean(0).max())}

    print("\n" + "=" * 70)
    print("VERDICT — does sigma_2 inherit a second sink?")
    for mid, s in summary.items():
        v = ("CONTAMINATED" if s["c2_2_hot"] > 0.9 else
             "clean" if s["c2_2_hot"] < 0.5 else "PARTIAL")
        print(f"  {mid:>26} cat({s['cat']})  2nd-row peak {s['max_nr2']:>6.2f}x  "
              f"hot layers {s['n_hot']:>3}  |cos(v2,h2)| {s['c2_2_hot']:.3f}  {v}")
    print("\n  If cat(a) is clean and cat(b) is contaminated, the quarantine")
    print("  argument holds ONLY for category-(a) models and every sigma_2")
    print("  claim must be stratified by massive-activation category.")
    out = os.path.join(os.path.dirname(__file__), "../results/sigma2_quarantine.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(summary, open(out, "w"), indent=2)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
