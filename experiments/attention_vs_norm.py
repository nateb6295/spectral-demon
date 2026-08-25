"""Are Cancedda's attention bars and Sun's category-(b) delimiter the SAME population?

THE TENSION. Cancedda 2402.09221 sec 7 selects non-BoS "attention bars" by
ATTENTION RECEIVED (high-mean, low-variance over subsequent tokens) and finds
their cosine similarity to the BoS residual is "close to zero" -- they "occupy a
different subspace within the dark subspace from the BoS token."

I measured pythia-2.8b's first strong delimiter, selected by RESIDUAL NORM (Sun
et al. category (b)), at |cos(h_BoS, h_delim)| = 0.987. Nearly parallel. Ruled
out a terminal-position artefact (0.986 with the delimiter at index 2).

Those two results contradict -- IF the populations are the same. They are
selected by DIFFERENT CRITERIA and need not be. A token can draw heavy attention
without carrying a massive activation, and vice versa. The literature tends to
use "attention sink" for both, which is exactly how two things acquire one name.

THE TEST. Measure attention RECEIVED by the norm-selected delimiter.
  SAME POPULATION  -> delimiter attention is elevated, comparable to BoS. Then
      the two results really do contradict and it is a family difference
      (pythia vs LLaMA2-13B) that needs explaining.
  DISSOCIABLE      -> delimiter carries a 20x residual norm while receiving
      ordinary attention. Then Cancedda and I measured different objects, both
      results stand, and "attention sink" is doing double duty in the field.

EXPECTATION, written before running (reflex 9):
  I expect the delimiter to receive ELEVATED attention but well below BoS --
  call it above the ordinary-token floor, below the BoS ceiling. ~0.6.
  Reasoning: the literature conflates them because they usually co-occur, so a
  total dissociation would surprise me; but Cancedda's near-zero cosine has to
  come from somewhere, and "high norm, moderate attention" is the shape that
  reconciles both.
  KILL FOR MY PREDICTION: delimiter attention indistinguishable from ordinary
  tokens (clean dissociation), or indistinguishable from BoS (clean identity).
  Either extreme means I split a difference that was not there.

Attention received by token j = mean over query positions i > j of attn[i, j],
averaged over heads. Causal model, so only later queries can attend to j.
bf16; eager attention so weights are returned.
"""
import gc, json, os
os.environ.setdefault("HF_HOME", "/mnt/hdd/huggingface")
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from stimulus_set import build

MID = "EleutherAI/pythia-2.8b"


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MID)
    m = AutoModelForCausalLM.from_pretrained(
        MID, dtype=torch.bfloat16, attn_implementation="eager").to(dev).eval()
    anchors, _, _ = build()
    # PREFIX. Without it the first period IS the final token, so no query
    # positions follow it and attention-received is undefined -- the first run
    # returned NaN for every layer. Same fix as delimiter_which_period.py:
    # put the first strong delimiter at index ~2 with the whole sentence after
    # it. Established earlier today that the anchor is first-come, not
    # positional (early period 19.64x, terminal 1.19x).
    anchors = ["Yes. " + a for a in anchors]
    A_bos, A_dlm, A_ord, N_bos, N_dlm = [], [], [], [], []
    for text in anchors:
        ids = tok(text, return_tensors="pt").to(dev)
        with torch.no_grad():
            o = m(**ids, output_hidden_states=True, output_attentions=True)
        toks = [tok.decode([i]) for i in ids["input_ids"][0].tolist()]
        per = [i for i, t in enumerate(toks) if t.strip() == "."]
        if not per:
            continue
        d = per[0]
        seq = len(toks)
        hs = torch.stack([h[0].float() for h in o.hidden_states])
        ab, ad, ao, nb, nd = [], [], [], [], []
        for l, att in enumerate(o.attentions):          # (1, heads, seq, seq)
            a = att[0].float().mean(0)                  # avg over heads -> (seq, seq)
            def recv(j):
                if j >= seq - 1:
                    return float("nan")
                return float(a[j + 1:, j].mean())       # only queries AFTER j
            ab.append(recv(0)); ad.append(recv(d))
            others = [recv(j) for j in range(1, seq - 1) if j != d]
            others = [x for x in others if x == x]
            ao.append(float(np.mean(others)) if others else float("nan"))
            H = hs[l + 1]; n = H.norm(dim=-1)
            med = n[1:-1].median().clamp_min(1e-6)
            nb.append(float(n[0] / med)); nd.append(float(n[d] / med))
        A_bos.append(ab); A_dlm.append(ad); A_ord.append(ao); N_bos.append(nb); N_dlm.append(nd)

    A_bos, A_dlm, A_ord = np.array(A_bos), np.array(A_dlm), np.array(A_ord)
    N_bos, N_dlm = np.array(N_bos), np.array(N_dlm)
    L = A_bos.shape[1]
    print(f"{MID}   n={len(A_bos)} stimuli   attention RECEIVED (mean over later queries, all heads)\n")
    print(f"  {'lyr':>4} {'BoS norm':>9} {'dlm norm':>9} | {'attn BoS':>9} {'attn dlm':>9} "
          f"{'attn ord':>9} | {'dlm/ord':>8}")
    for l in range(L):
        r = A_dlm[:, l].mean() / max(A_ord[:, l].mean(), 1e-9)
        print(f"  {l:>4} {N_bos[:,l].mean():>9.2f} {N_dlm[:,l].mean():>9.2f} | "
              f"{A_bos[:,l].mean():>9.4f} {A_dlm[:,l].mean():>9.4f} {A_ord[:,l].mean():>9.4f} | {r:>8.2f}x")

    hot = np.where(N_dlm.mean(0) > 3)[0]
    b, d_, o_ = A_bos[:, hot].mean(), A_dlm[:, hot].mean(), A_ord[:, hot].mean()
    print(f"\n  band where the delimiter carries a massive activation: L{hot[0]}..L{hot[-1]}")
    print(f"    attention received — BoS {b:.4f} | delimiter {d_:.4f} | ordinary {o_:.4f}")
    print(f"    delimiter / ordinary = {d_/max(o_,1e-9):.2f}x     delimiter / BoS = {d_/max(b,1e-9):.2f}x")
    print("\nVERDICT")
    # NaN GUARD. The first run printed "INTERMEDIATE -- my prediction" off NaN
    # input, because every comparison against NaN is False and it fell through
    # to the else. A void result defaulted to CONFIRMING MY HYPOTHESIS. Fifth
    # classifier today blind to its own edge case; the only one that flattered me.
    if not (d_ == d_ and b == b and o_ == o_):
        print("  VOID — non-finite attention. Do not read anything below.")
        return
    if d_ / max(o_, 1e-9) < 1.5:
        print("  DISSOCIABLE — 20x residual norm, ordinary attention. Cancedda and I")
        print("  measured different objects. Both results stand. 'Attention sink' is")
        print("  doing double duty for two distinct phenomena.")
    elif d_ / max(b, 1e-9) > 0.7:
        print("  SAME POPULATION — delimiter draws BoS-like attention. The contradiction")
        print("  with Cancedda sec 7 is real and needs a family-difference explanation.")
    else:
        print("  INTERMEDIATE — elevated over ordinary, well below BoS. My prediction.")
    json.dump({"attn_bos": float(b), "attn_dlm": float(d_), "attn_ord": float(o_),
               "hot": [int(hot[0]), int(hot[-1])]},
              open(os.path.join(os.path.dirname(__file__),
                                "../results/attention_vs_norm.json"), "w"), indent=2)
    del m; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
