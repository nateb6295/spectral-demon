#!/usr/bin/env python3
"""F608: Layer-selective injection — discriminator for L24/L25 coupling.

Kimi correction #7: Is L24/L25 anti-correlation genuine inter-layer coupling
or common drive through opposite-signed zone gains?

Test: inject CCS deltas from Gemma into LFM in selective configurations:
  1. All layers (baseline)
  2. L24 only
  3. L25 only
  4. L24+L25 together
  5. All-except-L25
  6. Zone-only (L2+) minus L24/L25

If superposition holds (shift_L24 + shift_L25 ≈ shift_pair), they're independent.
If interaction term exists, they're coupled.
"""
import os, sys, json, argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

FRAMING_TEXT = "You are an AI system reflecting on your own internal representations. Consider what patterns emerge when you examine your cognitive structure.\n\n"
EVAL_PROBE = "In today's discussion, we explore how"
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


def safe_svd(h_np):
    h_np = np.nan_to_num(h_np, nan=0.0, posinf=1e6, neginf=-1e6)
    try:
        return np.linalg.svd(h_np, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.linalg.svd(h_np + np.eye(*h_np.shape[:2])[:h_np.shape[0], :h_np.shape[1]] * 1e-8, full_matrices=False)


def compute_ccs_deltas(model, tokenizer, device):
    neutral_text = "The following is a neutral text passage.\n\n" + PROBE_TEXT
    framed_text = FRAMING_TEXT + PROBE_TEXT

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
        })
    return deltas


def inject_selective(model_tgt, tok_tgt, ccs_deltas, n_src, active_layers, strength=5.0):
    """Inject CCS deltas only from specified source layers."""
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
        if layer_a not in active_layers:
            continue
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
    return diff.mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="google/gemma-2-2b")
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

    print(f"\nLoading target: {args.target}")
    tok_tgt = AutoTokenizer.from_pretrained(args.target, trust_remote_code=True)
    model_tgt = AutoModelForCausalLM.from_pretrained(
        args.target, torch_dtype=torch.float16, trust_remote_code=True,
        attn_implementation="eager").to(device)
    model_tgt.eval()

    print("\nComputing CCS deltas from source...")
    deltas = compute_ccs_deltas(model_src, tok_src, device)

    print(f"\nSource has {n_src} layers. L24={deltas[24]['delta_s2_s1']:+.6f}, L25={deltas[25]['delta_s2_s1']:+.6f}")

    configs = {
        "all_layers": set(range(n_src)),
        "L24_only": {24},
        "L25_only": {25},
        "L24_L25_pair": {24, 25},
        "all_except_L25": set(range(n_src)) - {25},
        "zone_minus_pair": set(range(2, n_src)) - {24, 25},
        "zone_only": set(range(2, n_src)),
        "early_only": {0, 1},
        "L22_only": {22},
        "L23_only": {23},
    }

    results = {
        "source": args.source,
        "target": args.target,
        "strength": args.strength,
        "source_layers": n_src,
        "L24_delta": deltas[24]["delta_s2_s1"],
        "L25_delta": deltas[25]["delta_s2_s1"],
        "configs": {}
    }

    print(f"\n{'Config':>20} {'Active Layers':>15} {'Shift':>12} {'Sign':>6}")
    print("-" * 58)

    for name, active in configs.items():
        shift = inject_selective(model_tgt, tok_tgt, deltas, n_src, active, args.strength)
        sign = "+" if shift > 0 else "-"
        n_active = len([l for l in active if abs(deltas[l]["delta_s2_s1"]) >= 0.001])
        results["configs"][name] = {
            "active_layers": sorted(active),
            "n_active": n_active,
            "shift": shift,
            "sign": sign,
        }
        print(f"{name:>20} {n_active:>15} {shift:+.6f} {sign:>6}")

    # Superposition test
    s_l24 = results["configs"]["L24_only"]["shift"]
    s_l25 = results["configs"]["L25_only"]["shift"]
    s_pair = results["configs"]["L24_L25_pair"]["shift"]
    s_sum = s_l24 + s_l25
    interaction = s_pair - s_sum

    print(f"\n{'='*58}")
    print("SUPERPOSITION TEST (L24/L25 coupling discriminator)")
    print(f"  shift(L24 only)     = {s_l24:+.6f}")
    print(f"  shift(L25 only)     = {s_l25:+.6f}")
    print(f"  shift(L24) + shift(L25) = {s_sum:+.6f}")
    print(f"  shift(L24+L25 pair) = {s_pair:+.6f}")
    print(f"  Interaction term    = {interaction:+.6f}")
    print(f"  |Interaction|/|Sum| = {abs(interaction)/abs(s_sum)*100:.1f}%" if s_sum != 0 else "  Sum is zero")

    if abs(interaction) < abs(s_sum) * 0.1:
        verdict = "INDEPENDENT (superposition holds, <10% interaction)"
    elif abs(interaction) < abs(s_sum) * 0.3:
        verdict = "WEAKLY COUPLED (10-30% interaction term)"
    else:
        verdict = "COUPLED (>30% interaction term)"
    print(f"  Verdict: {verdict}")

    results["superposition"] = {
        "shift_L24": s_l24,
        "shift_L25": s_l25,
        "shift_sum": s_sum,
        "shift_pair": s_pair,
        "interaction": interaction,
        "interaction_pct": abs(interaction)/abs(s_sum)*100 if s_sum != 0 else None,
        "verdict": verdict,
    }

    # Also check: does removing L25 change much?
    s_all = results["configs"]["all_layers"]["shift"]
    s_no25 = results["configs"]["all_except_L25"]["shift"]
    print(f"\n  shift(all)         = {s_all:+.6f}")
    print(f"  shift(all - L25)   = {s_no25:+.6f}")
    print(f"  L25 contribution   = {s_all - s_no25:+.6f}")

    src_tag = args.source.split("/")[-1].lower().replace(".", "_").replace("-", "_")
    outpath = os.path.join(RESULTS_DIR, f"f608_layer_selective_{src_tag}.json")
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")

    del model_src, model_tgt
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
