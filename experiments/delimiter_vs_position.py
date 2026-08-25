#!/usr/bin/env python3
"""Is the pythia-2.8b terminal massive activation bound to the DELIMITER or to the POSITION?

delimiter_massive_activation.py established: pythia-2.8b puts a ~20x norm spike
on the final token of every one of 12 stimuli; pythia-410m puts 1.39x. That
explains the 50x whitespace asymmetry -- my "irrelevant" edit retokenises a
massive-activation site.

But all 12 stimuli end in a period, so "it's the delimiter" (Sun et al. 2024)
and "it's the last position" are perfectly confounded in that result. I stated
the delimiter reading in my writeup. I have not earned it.

EXPECTATION, written before running (reflex 9):
  DELIMITER-BOUND -> strip the final period and the spike goes away or drops
      sharply; the new final token is a content word, which per Sun et al.
      should not carry a massive activation.
  POSITION-BOUND  -> the spike stays at ~20x on whatever token is last.
  I genuinely do not know. Sun et al. say delimiter, so that is the literature's
  bet and mine, but attention sinks are famously positional and the last
  position is privileged in a causal LM (it is the only one that must carry the
  prediction). If it is positional, the "delimiter" language in my previous post
  is wrong and I retract it rather than soften it.

Third condition, to catch the boring answer: the period MOVED INTERIOR. If the
period carries the activation wherever it sits, that is delimiter-bound and
position-independent -- the strongest version. If a mid-sentence period is
ordinary, then it is the conjunction of the two, which is a different claim
again and I should say so.
"""
import os, re

os.environ.setdefault("HF_HOME", "/mnt/hdd/huggingface")
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np, torch, gc
from transformers import AutoTokenizer, AutoModelForCausalLM
from stimulus_set import build
from delimiter_massive_activation import per_token_norms

MODEL = "EleutherAI/pythia-2.8b"


def variants(s):
    """A: as-is '...word.'  B: period stripped  C: period interior, sentence continues."""
    base = re.sub(r"\.$", "", s)
    return {
        "A_period_final": base + ".",
        "B_noperiod_final": base,
        "C_period_interior": base + ". It was noted.",
    }


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="eager").to(dev).eval()

    anchors, _, _ = build()
    acc = {}
    interior_period = []
    for s in anchors:
        for name, text in variants(s).items():
            norms, ids = per_token_norms(model, tok, text, dev)
            assert np.isfinite(norms).all(), "non-finite norms"
            med = np.maximum(np.median(norms[:, 1:-1], axis=1), 1e-6)
            acc.setdefault(name, []).append(norms[:, -1] / med)
            if name == "C_period_interior":
                # locate the interior '.' -- the one that used to be terminal
                toks = [tok.decode([i]) for i in ids]
                idx = [i for i, t in enumerate(toks) if t.strip() == "."]
                if len(idx) >= 2:
                    interior_period.append(norms[:, idx[0]] / med)

    print(f"{MODEL}   n={len(anchors)} stimuli\n")
    print(f"{'condition':>22} {'peak final/med':>16} {'at idx':>7}")
    peaks = {}
    for name in ("A_period_final", "B_noperiod_final", "C_period_interior"):
        m = np.stack(acc[name]).mean(0)
        peaks[name] = m.max()
        print(f"{name:>22} {m.max():>16.2f} {int(m.argmax()):>7}")

    if interior_period:
        ip = np.stack(interior_period).mean(0)
        print(f"{'  (the interior period)':>22} {ip.max():>16.2f} {int(ip.argmax()):>7}")

    print("\nVERDICT")
    a, b = peaks["A_period_final"], peaks["B_noperiod_final"]
    print(f"  final-position spike with period {a:.2f}, without {b:.2f}  ({a/max(b,1e-6):.2f}x)")
    if b < 0.4 * a:
        print("  -> DELIMITER matters: stripping the period removes most of the spike.")
    elif b > 0.7 * a:
        print("  -> POSITIONAL: the last token spikes regardless of what it is.")
        print("     'delimiter massive activation' was the wrong name. Retract it.")
    else:
        print("  -> PARTIAL. Both contribute. Say exactly that, do not pick one.")
    if interior_period:
        print(f"  a period sitting INTERIOR reaches {ip.max():.2f}x "
              f"(vs {a:.2f}x when final)")
        if ip.max() > 0.6 * a:
            print("     -> the period carries it wherever it sits: delimiter-bound, position-free.")
        else:
            print("     -> needs BOTH delimiter and final position. Conjunction, not either.")

    del model; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
