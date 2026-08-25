"""Is the F499c mid-band window real geometry, or the sink's envelope?

F499c places a "mid-band regulatory window" at L12-19, measured via per-layer
effective rank (capsule #89345: per-layer Delta-effective-rank, baseline-confound
tested, L10-12 residuals -19% to -26% running AGAINST the confound direction).

Kimi's objection, Aug 23, after the F114(i) retraction: effective rank is
computed from the full singular spectrum INCLUDING sigma_1, and sigma_1 is now
demonstrated to BE the attention sink (|cos(v1,h_BoS)| = 0.99-1.00 wherever a
massive activation exists). Worse, this is not a loose worry -- 2510.06477
Theorem 1 proves the massive activation LOWER-BOUNDS sigma_1 and bounds
singular-value entropy as a corollary. Effective rank is precisely the quantity
the sink provably controls.

So a per-layer effective-rank profile may inherit the sink's onset -> dissipation
envelope rather than any enrichment geometry. In pythia-410m that envelope runs
1.3 at L5, 46 at L9, back to 1.03 at L24 -- a big smooth arc straight through
the mid-band.

THE TEST, no CCS framing needed: does mid-band structure in the effective-rank
profile SURVIVE removing the sink positions?
  masked = recompute the spectrum with the top-k highest-norm token positions
  EXCLUDED (not zeroed -- Kimi, correctly: zeroing collapses attention entropy,
  StreamingLLM, and makes a negative uninterpretable).

EXPECTATION, written before running (reflex 9):
  GEOMETRY REAL -> the masked profile keeps a distinguishable mid-band feature.
      Shape correlation between full and masked profiles stays high, and the
      mid-band does not simply track the sink envelope.
  SINK ENVELOPE -> masked profile flattens or loses the mid-band, and the FULL
      profile correlates strongly with the sink norm-ratio curve while the
      masked one does not.
  The decisive number is corr(full effective-rank profile, sink norm-ratio
  profile). If that is high and the masked profile decorrelates, F499c's window
  was the envelope.

I hold no strong prior. F499c has survived one confound test already, which is
more than F114(i) ever did. But it was characterised before we knew sigma_1 was
the sink, and this is a metric Theorem 1 speaks to directly.

CAVEAT STATED UP FRONT: this uses my 12-frame stimulus set, not the CCS framing
pairs F499c was actually measured on. So it can show the metric is
sink-dominated in general; it CANNOT by itself retract F499c's delta result.
Positive here means "go re-run F499c properly", not "F499c is dead."
"""
import gc, json, os
os.environ.setdefault("HF_HOME", "/mnt/hdd/huggingface")
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from stimulus_set import build

MODELS = ["EleutherAI/pythia-410m", "EleutherAI/pythia-2.8b"]
TOPK = 2   # exclude the two highest-norm positions: cat-(b) models have two sites


def eff_rank(S):
    p = S / S.sum().clamp_min(1e-9)
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def run(mid, dev):
    tok = AutoTokenizer.from_pretrained(mid)
    m = AutoModelForCausalLM.from_pretrained(
        mid, dtype=torch.bfloat16, attn_implementation="eager").to(dev).eval()
    anchors, _, _ = build()
    full, mask, sink = [], [], []
    for text in anchors:
        ids = tok(text, return_tensors="pt").to(dev)
        with torch.no_grad():
            o = m(**ids, output_hidden_states=True)
        hs = torch.stack([h[0].float() for h in o.hidden_states])
        a, b, c = [], [], []
        for l in range(hs.shape[0]):
            H = hs[l]
            n = H.norm(dim=-1)
            med = n[1:-1].median().clamp_min(1e-6)
            c.append(float(n.max() / med))
            a.append(eff_rank(torch.linalg.svdvals(H)))
            keep = torch.ones(H.shape[0], dtype=torch.bool, device=H.device)
            keep[torch.argsort(n, descending=True)[:TOPK]] = False
            b.append(eff_rank(torch.linalg.svdvals(H[keep])))
        full.append(a); mask.append(b); sink.append(c)
    del m; gc.collect(); torch.cuda.empty_cache()
    return np.array(full).mean(0), np.array(mask).mean(0), np.array(sink).mean(0)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {}
    for mid in MODELS:
        f, k, s = run(mid, dev)
        L = len(f)
        print(f"\n=== {mid}   (top-{TOPK} positions excluded for 'masked')")
        print(f"  {'layer':>5} {'sink nr':>9} {'effrank FULL':>13} {'effrank MASKED':>15}")
        for l in range(L):
            band = " <-mid-band" if 0.40 <= l / (L - 1) <= 0.70 else ""
            print(f"  {l:>5} {s[l]:>9.2f} {f[l]:>13.3f} {k[l]:>15.3f}{band}")
        # correlations against the sink envelope, endpoints dropped
        sl = slice(1, L - 1)
        cf = np.corrcoef(f[sl], s[sl])[0, 1]
        cm = np.corrcoef(k[sl], s[sl])[0, 1]
        cfm = np.corrcoef(f[sl], k[sl])[0, 1]
        print(f"\n  corr(FULL effrank, sink envelope)   = {cf:+.3f}")
        print(f"  corr(MASKED effrank, sink envelope) = {cm:+.3f}")
        print(f"  corr(FULL, MASKED)                  = {cfm:+.3f}")
        out[mid] = {"corr_full_sink": cf, "corr_masked_sink": cm, "corr_full_masked": cfm}

    print("\n" + "=" * 68)
    print("VERDICT — is the effective-rank depth profile sink-driven?")
    for mid, r in out.items():
        # SIGN MATTERS. First version of this compared |corr| and called
        # pythia-2.8b "SURVIVES" on 0.869 -> 0.814 -- while the sign FLIPPED
        # from -0.869 to +0.814 and the full/masked profiles came out
        # ANTI-correlated at -0.449. That is a total inversion, not survival.
        # Caught by reading the rows, not the verdict line. Fourth classifier
        # today that could not handle its own case.
        cf, cm, cfm = r["corr_full_sink"], r["corr_masked_sink"], r["corr_full_masked"]
        flipped = cf * cm < 0
        v = ("SINK-DRIVEN — masking INVERTS the relationship" if flipped and abs(cf) > 0.7
             else "SINK-DRIVEN — masking decorrelates it"
             if abs(cf) > 0.7 and abs(cf) - abs(cm) > 0.3
             else "SURVIVES — masked profile tracks the full one"
             if cfm > 0.7 else "PARTIAL")
        drop = abs(cf) - abs(cm)
        print(f"  {mid:>26}  full {r['corr_full_sink']:+.3f} -> masked "
              f"{r['corr_masked_sink']:+.3f}  (|drop| {drop:+.3f})  {v}")
    print("\n  NOTE: 12-frame neutral stimuli, NOT the CCS framing pairs F499c")
    print("  used. A positive here means GO RE-RUN F499c, not F499c is dead.")
    p = os.path.join(os.path.dirname(__file__), "../results/f499c_sink_envelope.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(out, open(p, "w"), indent=2)
    print(f"\nsaved {p}")


if __name__ == "__main__":
    main()
