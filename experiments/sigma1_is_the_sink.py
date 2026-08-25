#!/usr/bin/env python3
"""Does sigma_1 point at the massive activation? The one link exposure 3 is missing.

CHAIN, three links, two already published:
  1. A massive activation sits on BoS (and, per Sun et al. cat (b), the first
     strong delimiter). MEASURED here this morning: pythia-410m first token
     46.4x median norm from index 6; pythia-2.8b first delimiter 19.9x.
  2. The BoS residual stream content is INPUT-INDEPENDENT. Cancedda 2024
     (2402.09221) sec 5 states this flatly, as a premise for another experiment.
  3. sigma_1 is dominated by that direction.   <-- ONLY THIS IS UNTESTED

If link 3 holds, F114 clause (i) -- "sigma_1 is content-independent, angle std
0.01-0.06 deg across 8 prompt types" -- is a CONSEQUENCE of two documented facts
rather than a finding about identity. That is exposure 3, closed against us.

EXPECTATION, written before running (reflex 9):
  SINK ACCOUNT  -> |cos(v1, h_bos_hat)| high (>0.9) at every layer AFTER the
      massive-activation onset, and markedly lower BEFORE it. The jump should
      coincide with the norm-ratio onset I already measured: index 6 in
      pythia-410m. Coincidence of the two onsets is the fingerprint.
  IDENTITY ACCOUNT -> |cos| stays moderate or low post-onset; sigma_1 points
      somewhere the BoS vector does not dominate.
  KILL -> post-onset mean |cos| < 0.5 means the sink does NOT explain sigma_1,
      exposure 3 weakens a lot, and I say so plainly.

I expect the sink account, ~0.75 confidence. Noting that I have been 3/3 on
presence predictions and 1/4 on absence ones today, and this is a presence
prediction AGAINST my own programme -- which is the direction I am least likely
to be motivated-wrong in, but also the direction where a tidy confirmation
would feel virtuous. See feedback_legibility_vs_correctness.

WHY NOT ABLATION: Kimi, correctly -- zeroing the sink collapses attention
entropy (StreamingLLM: sink KV is load-bearing), so a post-ablation change
cannot separate "sigma_1 was the sink" from "the forward pass degenerated."
This probe intervenes on NOTHING. It only measures an angle in a clean pass.

Convention: H is (seq x d); H = U S V^T; v1 = V[:,0] is the top FEATURE-space
direction, comparable across prompts of different length. h_bos = H[0], the
first token's residual. bf16 -- fp16 overflows to NaN on these activations.
"""
import gc, json, os

os.environ.setdefault("HF_HOME", "/mnt/hdd/huggingface")
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from stimulus_set import build

MODELS = ["EleutherAI/pythia-410m", "EleutherAI/pythia-2.8b", "gpt2"]


def analyse(model_id, device):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, attn_implementation="eager").to(device).eval()
    anchors, _, _ = build()

    cos_bos, cos_top, normratio, v1s = [], [], [], []
    for text in anchors:
        ids = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**ids, output_hidden_states=True)
        hs = torch.stack([h[0].float() for h in out.hidden_states])  # (L+1, seq, d)
        assert torch.isfinite(hs).all(), f"{model_id}: non-finite states"

        cb, ct, nr, vv = [], [], [], []
        for l in range(hs.shape[0]):
            H = hs[l]                                   # (seq, d)
            norms = H.norm(dim=-1)
            med = norms[1:-1].median().clamp_min(1e-6)
            top = int(norms.argmax())
            _, _, Vh = torch.linalg.svd(H, full_matrices=False)
            v1 = Vh[0]                                  # (d,)
            hb = H[0] / H[0].norm().clamp_min(1e-6)
            ht = H[top] / H[top].norm().clamp_min(1e-6)
            cb.append(abs(float(v1 @ hb)))
            ct.append(abs(float(v1 @ ht)))
            nr.append(float(norms.max() / med))
            vv.append((v1 / v1.norm()).cpu().numpy())
        cos_bos.append(cb); cos_top.append(ct); normratio.append(nr); v1s.append(vv)

    del model; gc.collect(); torch.cuda.empty_cache()
    return (np.array(cos_bos), np.array(cos_top), np.array(normratio),
            np.array(v1s))


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}
    for mid in MODELS:
        print(f"\n=== {mid}", flush=True)
        cb, ct, nr, v1s = analyse(mid, dev)
        L = cb.shape[1]
        # onset = first layer where the max-norm token exceeds 5x the median
        m_nr = nr.mean(0)
        above = np.where(m_nr > 5)[0]
        onset = int(above[0]) if len(above) else None

        # F114's own measurement: cross-prompt spread of the v1 direction
        spread = []
        for l in range(L):
            V = v1s[:, l, :]
            V = V / np.linalg.norm(V, axis=1, keepdims=True)
            C = np.abs(V @ V.T)
            iu = np.triu_indices(len(V), 1)
            spread.append(np.degrees(np.arccos(np.clip(C[iu], -1, 1))).std())

        print(f"  {'layer':>5} {'maxnorm/med':>12} {'|cos(v1,bos)|':>14} "
              f"{'|cos(v1,topnorm)|':>18} {'v1 spread deg':>14}")
        for l in range(L):
            mark = " <-ONSET" if l == onset else ""
            print(f"  {l:>5} {m_nr[l]:>12.2f} {cb[:,l].mean():>14.3f} "
                  f"{ct[:,l].mean():>18.3f} {spread[l]:>14.3f}{mark}")

        post = slice(onset, L - 1) if onset is not None else slice(0, L)
        pre = slice(0, onset) if onset else slice(0, 1)
        results[mid] = {
            "onset": onset,
            "cos_bos_post": float(cb[:, post].mean()),
            "cos_bos_pre": float(cb[:, pre].mean()),
            "cos_top_post": float(ct[:, post].mean()),
            "spread_post": float(np.mean(spread[post])),
            "spread_pre": float(np.mean(spread[pre])),
        }
        r = results[mid]
        print(f"  onset={onset}  |cos(v1,bos)| pre={r['cos_bos_pre']:.3f} "
              f"post={r['cos_bos_post']:.3f}   v1 spread pre={r['spread_pre']:.2f} "
              f"post={r['spread_post']:.2f} deg")

    print("\n" + "=" * 66)
    print("VERDICT — is sigma_1 the sink?")
    for mid, r in results.items():
        c = r["cos_bos_post"]
        v = ("SINK (>0.9)" if c > 0.9 else
             "KILLED (<0.5) — sink does not explain sigma_1" if c < 0.5 else
             "PARTIAL (0.5-0.9) — say exactly that, pick neither")
        print(f"  {mid:>28}  post-onset |cos(v1,bos)| = {c:.3f}   {v}")
        print(f"  {'':>28}  v1 cross-prompt spread {r['spread_pre']:.2f} deg pre "
              f"-> {r['spread_post']:.2f} deg post-onset")
    out = os.path.join(os.path.dirname(__file__), "../results/sigma1_is_the_sink.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
