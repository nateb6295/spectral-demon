"""Ox's floor control. Is the effective-rank reshaping SINK-specific, or does ANY
deletion do it?

My earlier run (f499c_sink_envelope.py) removed the top-2 highest-norm token
positions, saw the depth profile reshape, and concluded the effective-rank
profile is sink-dominated. Ox, correctly:

  "effective rank is exp(SV entropy) computed on n~15 samples (max entropy
   ln 15 = 2.71 nats), and entropy at that n is fragile to *any* row deletion.
   Run corr(FULL, RANDOM-2-MASKED). If that also lands near +0.2 / -0.45,
   columns 2-3 measure small-sample fragility, not sink specificity...
   no floor, no measurement."

He also caught that fixed k=2 is wrong: the sink is not present at every depth.
In pythia-410m it dissipates by the final layer (max-norm/median 8.15 -> 1.03),
so there my mask was deleting two ORDINARY CONTENT tokens and calling it sink
removal. Mask must be NORM-ADAPTIVE per layer.

And: correlations do not answer the F499c question. He wants the masked profile
SHAPE — is there a mid-band feature at L12-19 once the sink is out.

THREE FIXES, all his:
  1. ADAPTIVE mask: drop positions with norm > 3x layer median. Drops the sink
     where it exists, drops nothing where it does not.
  2. RANDOM FLOOR: drop the SAME NUMBER of positions, chosen at random, 25 draws.
     This is the control I did not have.
  3. Report the PROFILES, not just correlations.

EXPECTATION, written before running (reflex 9):
  SINK-SPECIFIC -> corr(FULL, ADAPTIVE) low, and corr(FULL, RANDOM) HIGH (>0.8).
      Removing random tokens should barely move a profile; removing the sink
      should wreck it. The gap between those two numbers is the whole result.
  SMALL-SAMPLE FRAGILITY (Ox is right, I was wrong) -> corr(FULL, RANDOM) also
      lands low, near the adaptive number. Then this afternoon's conclusion was
      noise and F499c is untouched by it.

I genuinely do not know. Deleting the single highest-norm row from a 15-row
matrix is a large perturbation to an entropy estimate no matter what that row
is. Ox's objection is not a technicality and I did not think of it.
"""
import gc, json, os
os.environ.setdefault("HF_HOME", "/mnt/hdd/huggingface")
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from stimulus_set import build

MODELS = ["EleutherAI/pythia-410m", "EleutherAI/pythia-2.8b"]
THRESH, NDRAW, SEED = 3.0, 25, 0


def eff_rank(S):
    p = S / S.sum().clamp_min(1e-9)
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def run(mid, dev):
    rng = np.random.default_rng(SEED)
    tok = AutoTokenizer.from_pretrained(mid)
    m = AutoModelForCausalLM.from_pretrained(
        mid, dtype=torch.bfloat16, attn_implementation="eager").to(dev).eval()
    anchors, _, _ = build()
    full, adapt, rand, sink, ndrop = [], [], [], [], []
    for text in anchors:
        ids = tok(text, return_tensors="pt").to(dev)
        with torch.no_grad():
            o = m(**ids, output_hidden_states=True)
        hs = torch.stack([h[0].float() for h in o.hidden_states])
        a, b, c, s, nd = [], [], [], [], []
        for l in range(hs.shape[0]):
            H = hs[l]; n = H.norm(dim=-1)
            med = n[1:-1].median().clamp_min(1e-6)
            s.append(float(n.max() / med))
            a.append(eff_rank(torch.linalg.svdvals(H)))
            hot = (n / med) > THRESH
            k = int(hot.sum())
            nd.append(k)
            if k == 0:
                b.append(a[-1])           # nothing to remove: adaptive == full
                c.append(a[-1])
                continue
            keep = ~hot
            b.append(eff_rank(torch.linalg.svdvals(H[keep])))
            # FLOOR: remove k RANDOM positions, same count, many draws
            draws = []
            for _ in range(NDRAW):
                idx = rng.choice(H.shape[0], size=k, replace=False)
                kp = torch.ones(H.shape[0], dtype=torch.bool, device=H.device)
                kp[torch.as_tensor(idx, device=H.device)] = False
                draws.append(eff_rank(torch.linalg.svdvals(H[kp])))
            c.append(float(np.mean(draws)))
        full.append(a); adapt.append(b); rand.append(c); sink.append(s); ndrop.append(nd)
    del m; gc.collect(); torch.cuda.empty_cache()
    return (np.array(full).mean(0), np.array(adapt).mean(0),
            np.array(rand).mean(0), np.array(sink).mean(0), np.array(ndrop).mean(0))


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {}
    for mid in MODELS:
        f, a, r, s, nd = run(mid, dev)
        L = len(f); sl = slice(1, L - 1)
        print(f"\n=== {mid}   adaptive mask: norm > {THRESH}x median | floor: {NDRAW} random draws")
        print(f"  {'lyr':>4} {'sink':>7} {'#drop':>6} {'FULL':>8} {'ADAPTIVE':>9} {'RANDOM':>8}")
        for l in range(L):
            band = " <-mid" if 0.40 <= l / (L - 1) <= 0.70 else ""
            print(f"  {l:>4} {s[l]:>7.2f} {nd[l]:>6.1f} {f[l]:>8.3f} {a[l]:>9.3f} {r[l]:>8.3f}{band}")
        ca = np.corrcoef(f[sl], a[sl])[0, 1]
        cr = np.corrcoef(f[sl], r[sl])[0, 1]
        print(f"\n  corr(FULL, ADAPTIVE-masked) = {ca:+.3f}")
        print(f"  corr(FULL, RANDOM-masked)   = {cr:+.3f}   <- THE FLOOR")
        print(f"  separation = {cr - ca:+.3f}")
        out[mid] = {"corr_adaptive": ca, "corr_random": cr, "sep": cr - ca}

    print("\n" + "=" * 66)
    print("VERDICT — was the reshaping SINK-specific, or any-deletion fragility?")
    for mid, v in out.items():
        d = ("SINK-SPECIFIC — random deletion leaves the profile intact"
             if v["corr_random"] > 0.8 and v["sep"] > 0.4 else
             "FRAGILITY — Ox is right, my earlier conclusion was noise"
             if v["corr_random"] < 0.5 else "PARTIAL")
        print(f"  {mid:>26}  adaptive {v['corr_adaptive']:+.3f} | floor "
              f"{v['corr_random']:+.3f} | sep {v['sep']:+.3f}   {d}")
    p = os.path.join(os.path.dirname(__file__), "../results/f499c_floor_control.json")
    json.dump(out, open(p, "w"), indent=2)
    print(f"\nsaved {p}")


if __name__ == "__main__":
    main()
