#!/usr/bin/env python3
"""Does basin collapse come from NORM dispersion or from collapsed DIRECTIONS?

Kimi: iterated softmax retrieval is norm-seeking; global attractor = argmax|k|^2.
      My floor matched MEANS, not DISPERSION. Decisive test: equalize norms,
      keep directions. -> 14 basins = pure norm artifact. -> ~2 = directions.
Ox:   I measured basin COUNT, never IDENTITY. Drain account predicts x* is a
      NEAR-ZERO vector (v_BoS ~ 0), not the massive activation.
"""
import json, torch, statistics as st
from transformers import AutoModelForCausalLM, AutoTokenizer
MODEL="EleutherAI/pythia-410m"; MAXIT,TOL=300,1e-7
PROMPT="The capital of France is Paris, a city known for its museums."

def fp(K,xi,beta):
    for t in range(MAXIT):
        nxt=K.T@torch.softmax(beta*(K@xi),0)
        if not torch.isfinite(nxt).all(): return None
        if (nxt-xi).norm()<TOL: return nxt
        xi=nxt
    return xi
def census(K,beta):
    F=[fp(K,K[i].clone(),beta) for i in range(K.shape[0])]
    F=[f for f in F if f is not None]
    if not F: return 0,None
    F=torch.stack(F); Fn=F/(F.norm(dim=1,keepdim=True)+1e-9)
    reps=[]
    for v in Fn:
        if all((v@r).abs()<0.99 for r in reps): reps.append(v)
    return len(reps),F

tok=AutoTokenizer.from_pretrained(MODEL)
m=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,output_hidden_states=True).eval().cuda()
cfg=m.config; nL,nH=cfg.num_hidden_layers,cfg.num_attention_heads; dh=cfg.hidden_size//nH
beta=1.0/dh**0.5
ids=tok(PROMPT,return_tensors="pt").to("cuda")
with torch.no_grad(): out=m(**ids)
hs=out.hidden_states; n=hs[0].shape[1]
print(f"{MODEL} n={n} beta={beta:.4f}\n")
rows=[]
for L in range(0,nL,2):
    a=m.gpt_neox.layers[L].attention
    qkv=a.query_key_value(m.gpt_neox.layers[L].input_layernorm(hs[L]))[0].view(n,nH,3*dh)
    for h in range(nH):
        K=qkv[:,h,dh:2*dh].float(); V=qkv[:,h,2*dh:].float()
        norms=K.norm(dim=1)
        cv=(norms.std()/norms.mean()).item()
        b_raw,F=census(K,beta)
        Keq=K/(norms[:,None]+1e-9)*norms.mean()          # KEEP directions, EQUALIZE norms
        b_eq,_=census(Keq,beta)
        g=torch.Generator().manual_seed(L*100+h); Korth=torch.linalg.qr(torch.randn(dh,n,generator=g))[0].T.to(K.device)
        Korth=Korth*norms[:,None]                          # KEEP norms, RANDOM directions
        b_orth,_=census(Korth,beta)
        d=dict(layer=L,head=h,cv=cv,b_raw=b_raw,b_eq=b_eq,b_orth=b_orth,
               argmax_norm=int(norms.argmax()), vbos_norm=V[0].norm().item(),
               vmean_norm=V[1:].norm(dim=1).mean().item())
        if F is not None and len(F):
            Fn=F/(F.norm(dim=1,keepdim=True)+1e-9)
            d["fp_norm_med"]=F.norm(dim=1).median().item()
            d["cos_fp_kargmax"]=(Fn@(K[norms.argmax()]/norms.max())).abs().median().item()
            d["cos_fp_kbos"]=(Fn@(K[0]/(norms[0]+1e-9))).abs().median().item()
        rows.append(d)
    print(f"  L{L} done",flush=True)
json.dump(rows,open("/home/nate-agx/chronicle/data/hopfield_norm_vs_direction.json","w"),indent=1)
print("\n=== RESULT")
print(f"  raw learned          : {st.mean([r['b_raw'] for r in rows]):.2f} basins")
print(f"  NORM-EQUALIZED (dirs): {st.mean([r['b_eq'] for r in rows]):.2f}   <- Kimi's decisive test")
print(f"  norms kept, dirs rand: {st.mean([r['b_orth'] for r in rows]):.2f}   <- reverse control")
print(f"  key-norm CV          : median {st.median([r['cv'] for r in rows]):.3f}")
fpk=[r['cos_fp_kargmax'] for r in rows if 'cos_fp_kargmax' in r]
print(f"  cos(x*, k_argmaxnorm): median {st.median(fpk):.3f}   <- is the attractor the biggest key?")
print(f"  v_BoS norm / v_mean  : {st.median([r['vbos_norm']/max(r['vmean_norm'],1e-9) for r in rows]):.3f}  <- Ox: drain has near-null value?")
