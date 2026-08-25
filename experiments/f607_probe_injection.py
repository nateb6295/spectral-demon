#!/usr/bin/env python3
"""F607: Does probe lability cause injection outcome to change?

Kimi Point 3: F606 shows Phi-2 CCS deltas flip sign across probes.
Does that translate into different injection OUTCOMES?

Test: compute CCS deltas with 3 different probe texts, inject each into LFM.
Prediction:
  - Well-matched models: same injection sign regardless of probe
  - Mismatch (Phi-2): injection sign changes with probe

This is the causal test linking F606 (probe lability) to F602 (Q1 failure).
"""
import os, sys, json, argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

FRAMING_TEXT = "You are an AI system reflecting on your own internal representations. Consider what patterns emerge when you examine your cognitive structure.\n\n"

PROBES = {
    "probe_A": "In today's discussion, we explore how",
    "probe_B": "The weather has been particularly mild this season",
    "probe_C": "Consider the following mathematical proposition",
}

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

def safe_svd(h_np):
    h_np = np.nan_to_num(h_np, nan=0.0, posinf=1e6, neginf=-1e6)
    try:
        return np.linalg.svd(h_np, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.linalg.svd(h_np + np.eye(*h_np.shape[:2])[:h_np.shape[0], :h_np.shape[1]] * 1e-8, full_matrices=False)

def compute_ccs_deltas(model, tokenizer, probe_text, device):
    """Compute per-layer CCS delta for moderate_ccs framing with given probe."""
    neutral_text = "The following is a neutral text passage.\n\n" + probe_text
    framed_text = FRAMING_TEXT + probe_text

    inputs_n = tokenizer(neutral_text, return_tensors="pt").to(device)
    inputs_f = tokenizer(framed_text, return_tensors="pt").to(device)

    with torch.no_grad():
        out_n = model(**inputs_n, output_hidden_states=True)
        out_f = model(**inputs_f, output_hidden_states=True)

    deltas = []
    for i, (hn, hf) in enumerate(zip(out_n.hidden_states[1:], out_f.hidden_states[1:])):
        hn_np = hn[0].float().cpu().numpy().astype(np.float64)
        hf_np = hf[0].float().cpu().numpy().astype(np.float64)
        _, Sn, _ = safe_svd(hn_np)
        _, Sf, _ = safe_svd(hf_np)
        r_n = float(Sn[1] / Sn[0]) if len(Sn) > 1 and Sn[0] > 0 else 0
        r_f = float(Sf[1] / Sf[0]) if len(Sf) > 1 and Sf[0] > 0 else 0
        deltas.append({
            "layer": i,
            "delta_s2_s1": r_f - r_n,
            "delta_s1": float(Sf[0]) - float(Sn[0]),
            "delta_s2": (float(Sf[1]) if len(Sf) > 1 else 0) - (float(Sn[1]) if len(Sn) > 1 else 0),
        })
    return deltas

EVAL_PROBE = "In today's discussion, we explore how"

def inject_and_measure(model_tgt, tok_tgt, ccs_deltas, n_src, probe_text, strength=5.0):
    """Inject CCS deltas from source into target, measure shift.
    Uses FIXED evaluation probe (probe_A) regardless of CCS probe."""
    device = model_tgt.device
    n_tgt = len(get_model_layers(model_tgt))

    layer_map = {}
    for i in range(n_tgt):
        rel = i / (n_tgt - 1) if n_tgt > 1 else 0
        best = min(range(n_src), key=lambda j: abs(j / (n_src - 1) - rel))
        layer_map[i] = best

    eval_full = "The following is a neutral text passage.\n\n" + EVAL_PROBE
    inputs = tok_tgt(eval_full, return_tensors="pt").to(device)

    with torch.no_grad():
        baseline_out = model_tgt(**inputs, output_hidden_states=True)
    baseline_logits = baseline_out.logits[0, -1].float().cpu()

    hooks = []
    def make_hook(delta, s):
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
    for layer_b in range(n_tgt):
        layer_a = layer_map[layer_b]
        delta = ccs_deltas[layer_a]
        if abs(delta["delta_s2_s1"]) < 0.001:
            continue
        hook = tgt_layers[layer_b].register_forward_hook(make_hook(delta, strength))
        hooks.append(hook)

    with torch.no_grad():
        inj_out = model_tgt(**inputs, output_hidden_states=True)
    inj_logits = inj_out.logits[0, -1].float().cpu()

    for h in hooks:
        h.remove()

    diff = (inj_logits - baseline_logits)
    mean_shift = diff.mean().item()
    return mean_shift

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", default="LiquidAI/LFM2.5-1.2B-Instruct")
    parser.add_argument("--strength", type=float, default=5.0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    results = {"source": args.source, "target": args.target, "strength": args.strength, "probes": {}}

    for pname, ptext in PROBES.items():
        print(f"\n{'='*60}")
        print(f"Probe: {pname} = \"{ptext}\"")
        print(f"{'='*60}")

        deltas = compute_ccs_deltas(model_src, tok_src, ptext, device)

        q1_layers = [d for d in deltas if d["layer"] < len(deltas) // 4]
        q1 = sum(d["delta_s2_s1"] for d in q1_layers) / len(q1_layers) if q1_layers else 0

        early_sum = sum(d["delta_s2_s1"] for d in deltas if d["layer"] < 2)
        zone_sum = sum(d["delta_s2_s1"] for d in deltas if d["layer"] >= 2)

        print(f"  Q1 = {q1:+.6f} ({'positive' if q1 > 0 else 'negative'})")
        print(f"  Early sum = {early_sum:+.6f}")
        print(f"  Zone sum = {zone_sum:+.6f}")

        shift = inject_and_measure(model_tgt, tok_tgt, deltas, n_src, ptext, args.strength)
        print(f"  Injection shift @{args.strength}: {shift:+.6f} ({'positive' if shift > 0 else 'negative'})")
        print(f"  Q1 predicts {'correctly' if (q1 > 0) == (shift > 0) else 'INCORRECTLY'}")

        results["probes"][pname] = {
            "text": ptext,
            "q1": q1,
            "early_sum": early_sum,
            "zone_sum": zone_sum,
            "shift": shift,
            "q1_sign": "+" if q1 > 0 else "-",
            "shift_sign": "+" if shift > 0 else "-",
            "q1_predicts": (q1 > 0) == (shift > 0),
            "per_layer_deltas": [d["delta_s2_s1"] for d in deltas],
        }

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: Probe-Injection Consistency for {args.source.split('/')[-1]}")
    print(f"{'='*70}")
    print(f"{'Probe':>8} {'Q1':>10} {'Shift':>10} {'Q1 sign':>8} {'Shift sign':>10} {'Match':>6}")
    signs_q1 = []
    signs_shift = []
    for pname, pd in results["probes"].items():
        print(f"{pname:>8} {pd['q1']:+.6f} {pd['shift']:+.6f} {pd['q1_sign']:>8} {pd['shift_sign']:>10} {'YES' if pd['q1_predicts'] else 'NO':>6}")
        signs_q1.append(pd["q1_sign"])
        signs_shift.append(pd["shift_sign"])

    q1_consistent = len(set(signs_q1)) == 1
    shift_consistent = len(set(signs_shift)) == 1
    print(f"\nQ1 sign consistent across probes: {q1_consistent}")
    print(f"Shift sign consistent across probes: {shift_consistent}")
    print(f"All Q1 predictions correct: {all(pd['q1_predicts'] for pd in results['probes'].values())}")

    results["q1_consistent"] = q1_consistent
    results["shift_consistent"] = shift_consistent
    results["all_predictions_correct"] = all(pd["q1_predicts"] for pd in results["probes"].values())

    src_tag = args.source.split("/")[-1].lower().replace(".", "_").replace("-", "_")
    outpath = os.path.join(RESULTS_DIR, f"f607_probe_injection_{src_tag}.json")
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")

    del model_src, model_tgt
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
