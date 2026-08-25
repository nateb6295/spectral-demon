#!/usr/bin/env python3
"""Which period gets the massive activation -- the FIRST one, or one at a given index?

delimiter_vs_position.py showed pythia-2.8b puts ~20x on a sentence's period,
that it is not positional (strip the period, the last token is ordinary at
1.09x), and that the period keeps its 20x when text is appended after it while
a SECOND period added later gets only 1.14x. So: one anchor per sequence.

Unresolved, and I said so before claiming otherwise: in that test the anchored
period sat at the same token index it had in the original. 'the first period'
and 'a period around index 12' were still confounded.

This separates them. Prepend a short clause so a period appears EARLY, then
measure both periods in the same forward pass.

    "Yes. The capital of France is Paris, a city known for its architecture."
          ^ early period (index ~2)                                        ^ late period

EXPECTATION, written before running (reflex 9):
  FIRST-PERIOD-WINS -> early period ~20x, late period ~1x. The anchor is claimed
      by whichever delimiter appears first and later ones get nothing.
  INDEX-BOUND       -> early period ~1x, late period ~20x. Position in the
      sequence is what matters and 'first' was an accident of my stimuli.
  SPLIT/BOTH        -> both elevated. Then it is not exclusive at all and the
      'one anchor per sequence' language from the last post is wrong; the second
      period in the previous test failed for some other reason and I go find it.

I expect FIRST-PERIOD-WINS, at maybe 0.7 confidence, because sink exclusivity
is the pattern in the literature and because the added ". It was noted." period
got nothing. Logging the confidence so the calibration record stays honest --
this is a presence prediction and I am 3/3 on those, which is exactly the
streak that should make me suspicious of it.
"""
import os

os.environ.setdefault("HF_HOME", "/mnt/hdd/huggingface")
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np, torch, gc
from transformers import AutoTokenizer, AutoModelForCausalLM
from stimulus_set import build
from delimiter_massive_activation import per_token_norms

MODEL = "EleutherAI/pythia-2.8b"
PREFIX = "Yes. "


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="eager").to(dev).eval()

    anchors, _, _ = build()
    early, late, gap = [], [], []
    for s in anchors:
        text = PREFIX + s
        norms, ids = per_token_norms(model, tok, text, dev)
        assert np.isfinite(norms).all()
        toks = [tok.decode([i]) for i in ids]
        idx = [i for i, t in enumerate(toks) if t.strip() == "."]
        if len(idx) < 2:
            print(f"  skip (found {len(idx)} periods): {text[:50]}")
            continue
        med = np.maximum(np.median(norms[:, 1:-1], axis=1), 1e-6)
        early.append(norms[:, idx[0]] / med)
        late.append(norms[:, idx[-1]] / med)
        gap.append(idx[-1] - idx[0])

    e = np.stack(early).mean(0)
    l = np.stack(late).mean(0)
    print(f"\n{MODEL}   n={len(early)}   mean token gap between periods {np.mean(gap):.1f}\n")
    print(f"  EARLY period (index ~{2}): peak {e.max():.2f}x at layer {int(e.argmax())}")
    print(f"  LATE  period (terminal) : peak {l.max():.2f}x at layer {int(l.argmax())}")

    print("\nVERDICT")
    if e.max() > 5 and l.max() < 3:
        print("  -> FIRST-PERIOD-WINS. The anchor is claimed by the earliest delimiter")
        print("     and holds it; later periods get nothing. 'One anchor per sequence'")
        print("     survives, and it is first-come.")
    elif l.max() > 5 and e.max() < 3:
        print("  -> INDEX-BOUND. 'First period' was an artefact of my stimuli. Retract it.")
    elif e.max() > 5 and l.max() > 5:
        print("  -> BOTH elevated. Not exclusive. The 'one anchor per sequence' claim in")
        print("     the Aug 23 post is WRONG -- go find why the appended period got 1.14x.")
    else:
        print(f"  -> NEITHER spikes (early {e.max():.2f}, late {l.max():.2f}). The prefix")
        print("     killed the effect outright, which is its own finding. Do not")
        print("     interpret; rerun without the prefix to confirm the baseline holds.")

    del model; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
