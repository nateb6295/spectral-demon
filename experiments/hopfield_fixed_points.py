#!/usr/bin/env python3
"""Ox's BREAK: are attention heads Hopfield associative memories, or one sink well?

Ramsauer 2020: one attention step == one modern-Hopfield update
    xi <- X softmax(beta X^T xi)
If a head is a genuine associative memory, ITERATING to a fixed point should
land on STORED PATTERNS (content). F114 predicts it lands on the SINK.

Homoassociative form (X = keys) is the true Hopfield dynamics; attention's
heteroassociative X_v/X_k split is not what "retrieves a stored pattern" means.

Prereg: data/hopfield_attractor_prereg.md.  bfloat16 only.
Positive control runs FIRST and must find the sink where F114 says it lives.
"""
import json, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "EleutherAI/pythia-410m"
MAXIT, TOL = 200, 1e-5

PROMPTS = [
    "The capital of France is Paris, a city known for its museums.",
    "Water freezes at zero degrees Celsius under standard pressure.",
    "She opened the letter and read it twice before speaking.",
]

def fixed_point(K, xi, beta):
    """Iterate xi <- K^T softmax(beta K xi). K is (n,d). Returns (xi, iters, status)."""
    prev = None
    for t in range(MAXIT):
        p = torch.softmax(beta * (K @ xi), dim=0)          # (n,)
        nxt = K.T @ p                                       # (d,)
        if not torch.isfinite(nxt).all():
            return xi, t, "nonfinite"
        d = (nxt - xi).norm().item()
        if d < TOL:
            return nxt, t, "converged"
        if prev is not None and (nxt - prev).norm().item() < TOL:
            return nxt, t, "cycle2"
        prev, xi = xi, nxt
    return xi, MAXIT, "no_converge"

def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, output_hidden_states=True).eval().cuda()
    cfg = model.config
    nL, nH = cfg.num_hidden_layers, cfg.num_attention_heads
    dh = cfg.hidden_size // nH
    beta = 1.0 / (dh ** 0.5)
    print(f"{MODEL}  layers={nL} heads={nH} d_head={dh} beta={beta:.4f}", flush=True)

    rows = []
    for pi, prompt in enumerate(PROMPTS):
        ids = tok(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model(**ids)
        hs = out.hidden_states                      # tuple (nL+1) of (1,n,D)
        n = hs[0].shape[1]

        # where does the massive activation live? per-layer max residual norm at BoS
        bos_norm = [hs[L][0, 0].float().norm().item() for L in range(nL + 1)]

        for L in range(nL):
            attn = model.gpt_neox.layers[L].attention
            h_in = model.gpt_neox.layers[L].input_layernorm(hs[L])      # (1,n,D)
            qkv = attn.query_key_value(h_in)[0]                          # (n, 3D)
            qkv = qkv.view(n, nH, 3 * dh)
            for h in range(nH):
                K = qkv[:, h, dh:2*dh].float()                           # (n,d) keys
                # basin census: start from every stored pattern
                fps, statuses = [], []
                for i in range(n):
                    xi, it, st = fixed_point(K, K[i].clone(), beta)
                    fps.append(xi); statuses.append(st)
                F = torch.stack(fps)
                F = F / (F.norm(dim=1, keepdim=True) + 1e-9)
                # distinct fixed points (cos > 0.99 == same)
                distinct, reps = 0, []
                for v in F:
                    if all((v @ r).abs().item() < 0.99 for r in reps):
                        reps.append(v); distinct += 1
                Kn = K / (K.norm(dim=1, keepdim=True) + 1e-9)
                # what fraction of basins land on the BoS pattern?
                to_bos = (F @ Kn[0]).abs()
                frac_bos = (to_bos > 0.99).float().mean().item()
                rows.append(dict(prompt=pi, layer=L, head=h, n=n,
                                 distinct=distinct, frac_bos=frac_bos,
                                 bos_key_norm=K[0].norm().item(),
                                 mean_key_norm=K[1:].norm(dim=1).mean().item(),
                                 nonconv=sum(s != "converged" for s in statuses),
                                 cycles=sum(s == "cycle2" for s in statuses)))
            print(f"  p{pi} L{L} done", flush=True)
        rows.append(dict(prompt=pi, bos_resid_norm_by_layer=bos_norm))

    with open("/home/nate-agx/chronicle/data/hopfield_fixed_points.json", "w") as f:
        json.dump(rows, f, indent=1)
    print("WROTE data/hopfield_fixed_points.json", flush=True)

main()
