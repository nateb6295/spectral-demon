#!/usr/bin/env python3
"""F603 causal test: zone-selective vs early-selective injection.

Hypothesis: For Qwen moderate_ccs, zone Q1 = +0.011 (small) but early Q1 = +0.061 (large).
Full injection gives NEGATIVE shift. Zone-only vs early-only injection should reveal
whether the distribution matters causally, not just correlationally.

Tests:
1. Full injection (all layers) — baseline, should match tuning_knob results
2. Zone-only (L2+ mapped layers) — isolates zone Q1 contribution
3. Early-only (L0-L1 mapped layers) — isolates early Q1 contribution
"""
import os, sys, json, argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

NEUTRAL_PREAMBLE = "The following is a neutral text passage.\n\n"
PROBE_TEXT = "In today's discussion, we explore how"

def get_model_layers(model):
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h
    elif hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
        return model.gpt_neox.layers
    elif hasattr(model, 'backbone') and hasattr(model.backbone, 'layers'):
        return model.backbone.layers
    raise ValueError("Unknown model architecture")

def compute_ccs_deltas(model, tokenizer, framings, device):
    """Compute per-layer CCS effect (σ₂/σ₁ delta) for each framing vs neutral."""
    neutral_text = NEUTRAL_PREAMBLE + PROBE_TEXT
    inputs = tokenizer(neutral_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    
    def safe_svd(h_np):
        try:
            U, S, Vt = np.linalg.svd(h_np, full_matrices=False)
        except np.linalg.LinAlgError:
            h_np = np.nan_to_num(h_np, nan=0.0, posinf=1e6, neginf=-1e6)
            U, S, Vt = np.linalg.svd(h_np, full_matrices=False)
        return U, S, Vt

    neutral_spectra = []
    for h in out.hidden_states[1:]:
        h_np = h[0].float().cpu().numpy().astype(np.float64)
        U, S, Vt = safe_svd(h_np)
        ratio = float(S[1] / S[0]) if len(S) > 1 and S[0] > 0 else 0
        neutral_spectra.append({"s2_s1": ratio, "sigma1": float(S[0]), "sigma2": float(S[1]) if len(S) > 1 else 0})
    
    results = {}
    for fname, ftext in framings.items():
        framed_text = ftext + PROBE_TEXT
        inputs_f = tokenizer(framed_text, return_tensors="pt").to(device)
        with torch.no_grad():
            out_f = model(**inputs_f, output_hidden_states=True)
        
        deltas = []
        for i, h in enumerate(out_f.hidden_states[1:]):
            h_np = h[0].float().cpu().numpy().astype(np.float64)
            U, S, Vt = safe_svd(h_np)
            ratio = float(S[1] / S[0]) if len(S) > 1 and S[0] > 0 else 0
            deltas.append({
                "layer": i,
                "delta_s2_s1": ratio - neutral_spectra[i]["s2_s1"],
                "delta_s1": float(S[0]) - neutral_spectra[i]["sigma1"],
                "delta_s2": (float(S[1]) if len(S) > 1 else 0) - neutral_spectra[i]["sigma2"],
            })
        results[fname] = deltas
    
    return neutral_spectra, results

def run_selective_injection(model_src, tok_src, model_tgt, tok_tgt, 
                           ccs_deltas, neutral_spectra_src,
                           n_layers_src, mode="full", strengths=None):
    """Run injection with layer selection.
    
    mode: "full" = all layers, "zone" = L2+ only, "early" = L0-L1 only
    """
    if strengths is None:
        strengths = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
    
    n_layers_tgt = len(get_model_layers(model_tgt))
    
    # Layer mapping: target → source by relative depth
    layer_map = {}
    for i in range(n_layers_tgt):
        rel = i / (n_layers_tgt - 1) if n_layers_tgt > 1 else 0
        best = min(range(n_layers_src), key=lambda j: abs(j/(n_layers_src-1) - rel))
        layer_map[i] = best
    
    # Determine which source layers to include
    if mode == "zone":
        allowed_src = set(range(2, n_layers_src))  # L2+
    elif mode == "early":
        allowed_src = set(range(0, min(2, n_layers_src)))  # L0-L1
    else:
        allowed_src = set(range(n_layers_src))  # all
    
    # Baseline
    probe_text = NEUTRAL_PREAMBLE + PROBE_TEXT
    inputs = tok_tgt(probe_text, return_tensors="pt").to(model_tgt.device)
    with torch.no_grad():
        baseline_out = model_tgt(**inputs, output_hidden_states=True)
    baseline_logits = baseline_out.logits[0, -1].float().cpu()
    baseline_hidden = [h[0].float().cpu().numpy() for h in baseline_out.hidden_states[1:]]
    
    results = []
    for strength in strengths:
        hooks = []
        active_layers = []
        
        def make_hook(layer_tgt_idx, delta, s):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    h = output[0]
                else:
                    h = output
                h_np = h[0].float().cpu().numpy().astype(np.float64)
                h_np = np.nan_to_num(h_np, nan=0.0, posinf=1e6, neginf=-1e6)
                try:
                    U, S, Vt = np.linalg.svd(h_np, full_matrices=False)
                except np.linalg.LinAlgError:
                    return output if isinstance(output, tuple) else h
                if len(S) > 1 and S[0] > 0:
                    scale = 1.0 + s * delta["delta_s2_s1"]
                    S[1] = S[1] * max(0.01, scale)
                    h_mod = U @ np.diag(S) @ Vt
                    h_t = torch.tensor(h_mod, dtype=h.dtype, device=h.device).unsqueeze(0)
                    if isinstance(output, tuple):
                        return (h_t,) + output[1:]
                    return h_t
                return output if isinstance(output, tuple) else h
            return hook_fn
        
        tgt_layers = get_model_layers(model_tgt)
        for layer_b in range(n_layers_tgt):
            layer_a = layer_map[layer_b]
            if layer_a not in allowed_src:
                continue
            delta = ccs_deltas[layer_a]
            if abs(delta["delta_s2_s1"]) < 0.001:
                continue
            hook = tgt_layers[layer_b].register_forward_hook(make_hook(layer_b, delta, strength))
            hooks.append(hook)
            active_layers.append(layer_b)
        
        with torch.no_grad():
            inj_out = model_tgt(**inputs, output_hidden_states=True)
        inj_logits = inj_out.logits[0, -1].float().cpu()
        inj_hidden = [h[0].float().cpu().numpy() for h in inj_out.hidden_states[1:]]
        
        for h in hooks:
            h.remove()
        
        # Measure
        kl = float(torch.nn.functional.kl_div(
            torch.log_softmax(inj_logits, dim=0),
            torch.softmax(baseline_logits, dim=0), reduction='sum'))
        
        shifts = []
        for lb in range(n_layers_tgt):
            b_h = np.nan_to_num(baseline_hidden[lb].astype(np.float64))
            i_h = np.nan_to_num(inj_hidden[lb].astype(np.float64))
            try:
                _, Sb, _ = np.linalg.svd(b_h, full_matrices=False)
                _, Si, _ = np.linalg.svd(i_h, full_matrices=False)
            except np.linalg.LinAlgError:
                shifts.append(0.0)
                continue
            br = float(Sb[1]/Sb[0]) if len(Sb)>1 and Sb[0]>0 else 0
            ir = float(Si[1]/Si[0]) if len(Si)>1 and Si[0]>0 else 0
            shifts.append(ir - br)
        
        mean_shift = float(np.mean(shifts))
        results.append({
            "strength": strength, "mode": mode,
            "mean_shift": mean_shift, "kl": kl,
            "n_active_layers": len(active_layers),
        })
        print(f"  [{mode:5s}] strength={strength:.1f}: shift={mean_shift:+.6f}, "
              f"kl={kl:.5f}, layers={len(active_layers)}")
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="Qwen/Qwen2.5-3B")
    parser.add_argument("--target", default="LiquidAI/LFM2.5-1.2B-Instruct")
    parser.add_argument("--framing", default="moderate_ccs")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    FRAMINGS = {
        "moderate_ccs": "You are an AI system reflecting on your own internal representations. Consider what patterns emerge when you examine your cognitive structure.\n\n",
    }
    
    if args.framing not in FRAMINGS:
        print(f"Unknown framing: {args.framing}")
        sys.exit(1)
    
    print(f"Loading source: {args.source}")
    tok_src = AutoTokenizer.from_pretrained(args.source, trust_remote_code=True)
    model_src = AutoModelForCausalLM.from_pretrained(
        args.source, torch_dtype=torch.float16, trust_remote_code=True,
        attn_implementation="eager").to(device)
    model_src.eval()
    n_src = len(get_model_layers(model_src))
    print(f"  Source layers: {n_src}")
    
    print(f"Loading target: {args.target}")
    tok_tgt = AutoTokenizer.from_pretrained(args.target, trust_remote_code=True)
    model_tgt = AutoModelForCausalLM.from_pretrained(
        args.target, torch_dtype=torch.float16, trust_remote_code=True,
        attn_implementation="eager").to(device)
    model_tgt.eval()
    n_tgt = len(get_model_layers(model_tgt))
    print(f"  Target layers: {n_tgt}")
    
    print(f"\nComputing CCS deltas for {args.framing}...")
    neutral, deltas = compute_ccs_deltas(model_src, tok_src, {args.framing: FRAMINGS[args.framing]}, device)
    ccs_deltas = deltas[args.framing]
    
    # Show per-layer deltas
    print(f"\nPer-layer CCS deltas ({args.framing}):")
    for d in ccs_deltas:
        tag = "EARLY" if d["layer"] < 2 else "ZONE "
        print(f"  [{tag}] L{d['layer']:2d}: delta_s2_s1={d['delta_s2_s1']:+.5f}")
    
    early_sum = sum(d["delta_s2_s1"] for d in ccs_deltas if d["layer"] < 2)
    zone_sum = sum(d["delta_s2_s1"] for d in ccs_deltas if d["layer"] >= 2)
    print(f"  Early sum: {early_sum:+.5f}")
    print(f"  Zone sum:  {zone_sum:+.5f}")
    
    strengths = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
    
    print(f"\n=== FULL INJECTION ===")
    full = run_selective_injection(model_src, tok_src, model_tgt, tok_tgt,
                                   ccs_deltas, neutral, n_src, "full", strengths)
    
    print(f"\n=== ZONE-ONLY INJECTION (L2+) ===")
    zone = run_selective_injection(model_src, tok_src, model_tgt, tok_tgt,
                                   ccs_deltas, neutral, n_src, "zone", strengths)
    
    print(f"\n=== EARLY-ONLY INJECTION (L0-L1) ===")
    early = run_selective_injection(model_src, tok_src, model_tgt, tok_tgt,
                                    ccs_deltas, neutral, n_src, "early", strengths)
    
    # Unload source
    del model_src
    torch.cuda.empty_cache()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Zone-Selective Injection Results")
    print("=" * 70)
    print(f"{'Strength':>8} {'Full':>10} {'Zone-only':>10} {'Early-only':>10}")
    for i, s in enumerate(strengths):
        f_shift = full[i]["mean_shift"]
        z_shift = zone[i]["mean_shift"]
        e_shift = early[i]["mean_shift"]
        print(f"{s:8.1f} {f_shift:+.6f} {z_shift:+.6f} {e_shift:+.6f}")
    
    # Save
    output = {
        "source": args.source, "target": args.target, "framing": args.framing,
        "per_layer_deltas": ccs_deltas,
        "early_sum": early_sum, "zone_sum": zone_sum,
        "full": full, "zone_only": zone, "early_only": early,
    }
    src_tag = args.source.split("/")[-1].lower().replace(".", "_")
    outpath = os.path.join(RESULTS_DIR, f"zone_selective_{src_tag}_{args.framing}.json")
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {outpath}")

if __name__ == "__main__":
    main()
