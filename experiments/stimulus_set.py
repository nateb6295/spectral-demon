#!/usr/bin/env python3
"""A homogeneous perturbation set, with its own variance measured.

WHY. The Aug 23 depth/width run died on its stimuli, not its models. Four
"irrelevant" edits produced deltas of 0.049, 0.002, 0.047, 0.050 on
pythia-2.8b — a 22x spread — because they were four DIFFERENT KINDS of edit:
a moved comma, a space before a period, a doubled space. Heterogeneous by
construction. An n=4 mean over that is reporting which edit happened to be
disruptive, not which model is sensitive.

FIX, two parts:
  1. ONE kind of edit, applied uniformly. Every irrelevant perturbation is the
     same operation on a different sentence, so the set has a defined meaning.
  2. Enough sentences, and REPORT THE WITHIN-SET VARIANCE alongside any mean.
     A mean without its spread is what hid the problem for three hours.

The relevant set is matched on syntactic frame: same template, swapped content
words, so length and register are held and only the subject matter moves.
"""
import re

# 12 frames. Each yields an anchor and a content-swapped twin sharing the frame.
FRAMES = [
    ("The capital of {a} is {b}, a city known for its {c}.",
     ("France", "Paris", "architecture"), ("Japan", "Tokyo", "railways")),
    ("In {a} {b} published the {c}, which changed the field.",
     ("1687", "Newton", "Principia"), ("1859", "Darwin", "Origin")),
    ("The {a} is often described as the {b} of the {c}.",
     ("mitochondrion", "powerhouse", "cell"), ("ribosome", "factory", "cell")),
    ("{a} rates rose sharply last quarter, tightening {b} conditions.",
     ("Interest", "credit"), ("Vacancy", "housing")),
    ("She walked down to the {a} and sat on the cold {b} until dusk.",
     ("river", "stones"), ("harbour", "railings")),
    ("Consider a compact {a} manifold with strictly positive {b}.",
     ("Riemannian", "curvature"), ("symplectic", "volume")),
    ("The {a} stood at the treeline, unbothered by the {b} passing below.",
     ("elk", "tram"), ("heron", "barge")),
    ("Proof takes a formal {a} and shows how it follows from {b}.",
     ("statement", "axioms"), ("conjecture", "lemmas")),
    ("Most {a} in the region depend on seasonal {b} for their income.",
     ("families", "harvests"), ("workshops", "contracts")),
    ("The committee rejected the {a} after reviewing the {b} evidence.",
     ("proposal", "financial"), ("appeal", "forensic")),
    ("Every {a} in the archive was catalogued by hand before {b}.",
     ("letter", "1970"), ("specimen", "1930")),
    ("A {a} forms when cold air settles over the warmer {b} overnight.",
     ("fog", "water"), ("frost", "ground")),
]

# ONE irrelevant operation, applied uniformly: insert a single space before the
# terminal period. Chosen because it is the smallest edit that is guaranteed
# well-formed for every sentence and identical in kind across all of them.
def irrelevant(s):
    return re.sub(r"\.$", " .", s)


def build():
    anchors, relevants, irrelevants = [], [], []
    for tmpl, a_words, b_words in FRAMES:
        keys = re.findall(r"\{(\w)\}", tmpl)
        anchors.append(tmpl.format(**dict(zip(keys, a_words))))
        relevants.append(tmpl.format(**dict(zip(keys, b_words))))
        irrelevants.append(irrelevant(anchors[-1]))
    return anchors, relevants, irrelevants


if __name__ == "__main__":
    import os, numpy as np
    os.environ.setdefault("HF_HOME", "/mnt/hdd/huggingface")
    from transformers import AutoTokenizer
    A, R, I = build()
    print(f"{len(A)} frames\n")
    print("sample:")
    print("  anchor    ", A[0])
    print("  relevant  ", R[0])
    print("  irrelevant", I[0])
    print("\nWITHIN-SET VARIANCE — the thing the old set never reported:")
    print(f"{'tokeniser':>22} {'set':>11} {'mean':>7} {'std':>7} {'spread':>8}")
    for mid in ("EleutherAI/pythia-410m", "Qwen/Qwen2.5-0.5B"):
        tok = AutoTokenizer.from_pretrained(mid)
        def jac(a, b):
            X, Y = set(tok(a)["input_ids"]), set(tok(b)["input_ids"])
            return 1 - len(X & Y) / len(X | Y)
        for lbl, lst in (("relevant", R), ("irrelevant", I)):
            d = np.array([jac(a, b) for a, b in zip(A, lst)])
            spread = d.max() / d.min() if d.min() > 0 else float("inf")
            print(f"{mid.split('/')[-1]:>22} {lbl:>11} {d.mean():>7.3f} "
                  f"{d.std():>7.3f} {spread:>8.1f}x")
