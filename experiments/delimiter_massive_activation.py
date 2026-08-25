#!/usr/bin/env python3
"""Is the 50x terminal-whitespace asymmetry a massive activation on the delimiter?

BACKGROUND. Aug 23: pythia-2.8b was 50.1x more sensitive to a space inserted
before the terminal period than to the same whitespace mid-sentence. pythia-410m
showed 1.2x. Mid-sentence edits gave the sensible ordering (2.8b less sensitive
than 410m), so the asymmetry lives entirely at the terminal position. I had no
account of it and was about to spend a 6.9b run chasing it.

Then I read arXiv 2510.06477 (Queipo de Llano et al., "Attention Sinks and
Compression Valleys are Two Sides of the Same Coin"). It is about the bos token,
not the terminal one -- but its background section restates Sun et al. 2024:
massive activations "consistently appear on DELIMITER and special tokens." A
terminal period is a delimiter. stimulus_set.irrelevant() is
re.sub(r"\.$", " .", s) -- it retokenises exactly that delimiter.

So the deflating hypothesis: I was not measuring depth sensitivity. I was
poking the sink.

EXPECTATION, written before running (reflex 9):
  DELIMITER-MASSIVE-ACTIVATION -> pythia-2.8b carries a large norm spike on the
      final token relative to interior content tokens; pythia-410m does not, or
      carries a much smaller one. The gap between the two models' terminal
      ratios should be large -- I will not pretend to predict 50x from a norm
      ratio, but I expect the ORDERING to match and the gap to be visible
      without squinting (call it >=3x difference in terminal ratio).
  KILL CONDITION -> if 2.8b's terminal ratio is within 2x of 410m's, this
      explanation is dead. The 50x stays unexplained and I say so.

POSITIVE CONTROL, and this is the good part -- I did not have to design it.
The paper states a fact about a model I have on disk: in pythia-410m the
first-token norm spikes "consistently at layer 5 regardless of input." That is
a known answer for an instrument I am about to trust. If this script does not
show a layer-5 first-token spike in 410m, the script is wrong and every other
number it prints is void. Check that cell FIRST.

Design note: no perturbation here. Norms only, on the unmodified anchors. If
the terminal token is already a massive-activation site, that is visible in the
clean forward pass and no delta is needed.
"""
import gc, json, os, sys

os.environ.setdefault("HF_HOME", "/mnt/hdd/huggingface")
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from stimulus_set import build

MODELS = ["EleutherAI/pythia-410m", "EleutherAI/pythia-2.8b"]


def per_token_norms(model, tok, text, device):
    """(n_layers+1, seq) L2 norms of the residual stream."""
    ids = tok(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**ids, output_hidden_states=True)
    # drop the post-final-norm duplicate HF appends? No -- hidden_states[0] is
    # the embedding and [-1] is the final layer AFTER the final norm. Keep all,
    # label honestly: index 0 = embeddings, index L = block L output.
    hs = torch.stack([h[0].float() for h in out.hidden_states])  # (L+1, seq, d)
    return hs.norm(dim=-1).cpu().numpy(), ids["input_ids"][0].tolist()


def run(model_id, device):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, attn_implementation="eager"
    ).to(device).eval()

    anchors, _, _ = build()
    first_r, term_r, raw = [], [], []
    for text in anchors:
        norms, ids = per_token_norms(model, tok, text, device)
        # interior = everything except position 0 and the final token
        interior = norms[:, 1:-1]
        med = np.median(interior, axis=1)              # (L+1,) per layer
        med = np.maximum(med, 1e-6)
        first_r.append(norms[:, 0] / med)
        term_r.append(norms[:, -1] / med)
        raw.append({"text": text, "n_tok": len(ids),
                    "final_tok": repr(tok.decode([ids[-1]]))})

    first_r = np.stack(first_r)   # (n_stim, L+1)
    term_r = np.stack(term_r)
    if not (np.isfinite(first_r).all() and np.isfinite(term_r).all()):
        raise RuntimeError(
            f"{model_id}: non-finite norms. fp16 overflowed on this model once "
            "already -- the massive activations exceed the dtype range. "
            "Do not read past this line.")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return first_r, term_r, raw


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}
    for mid in MODELS:
        print(f"\n=== {mid}", flush=True)
        f, t, raw = run(mid, device)
        results[mid] = {"first": f.tolist(), "term": t.tolist(), "stim": raw}
        L = f.shape[1]
        print(f"  final token of frame 0: {raw[0]['final_tok']}")
        print(f"  {'layer':>5} {'first/med':>10} {'term/med':>10}")
        for l in range(L):
            fm, tm = f[:, l].mean(), t[:, l].mean()
            flag = ""
            if fm > 5:
                flag += " <-FIRST-SPIKE"
            if tm > 5:
                flag += " <-TERM-SPIKE"
            print(f"  {l:>5} {fm:>10.2f} {tm:>10.2f}{flag}")
        print(f"  PEAK first/med = {f.mean(0).max():.2f} at layer {int(f.mean(0).argmax())}")
        print(f"  PEAK  term/med = {t.mean(0).max():.2f} at layer {int(t.mean(0).argmax())}")

    print("\n" + "=" * 62)
    print("POSITIVE CONTROL (paper: pythia-410m first-token spikes at layer 5)")
    f410 = np.array(results["EleutherAI/pythia-410m"]["first"]).mean(0)
    above = np.where(f410 > 5)[0]
    onset = int(above[0]) if len(above) else -1
    pk = int(f410.argmax())
    print(f"  first index with ratio>5 = {onset}  (hidden_states index; "
          f"index k = output of block k-1)")
    print(f"  peak ratio {f410.max():.2f} at index {pk}")
    ok = onset in (5, 6) and f410.max() > 3
    print(f"  VERDICT: {'PASS -- instrument reproduces a known answer' if ok else 'FAIL -- instrument is not trustworthy, ignore everything below'}")

    print("\nTHE TEST")
    tr = {m: np.array(results[m]["term"]).mean(0).max() for m in MODELS}
    for m in MODELS:
        print(f"  {m:>28}  peak terminal/median = {tr[m]:.2f}")
    gap = tr[MODELS[1]] / max(tr[MODELS[0]], 1e-6)
    print(f"  2.8b / 410m terminal ratio gap = {gap:.2f}x")
    if gap >= 3:
        print("  -> CONSISTENT with delimiter massive activation.")
    elif gap <= 2:
        print("  -> KILLED. Terminal ratios comparable; 50x stays unexplained.")
    else:
        print("  -> AMBIGUOUS (2-3x). Not the clean answer either way.")

    out = os.path.join(os.path.dirname(__file__), "../results/delimiter_massive_activation.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(results, fh)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
