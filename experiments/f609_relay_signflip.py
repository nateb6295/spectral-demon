#!/usr/bin/env python3
"""F609: Relay sign-flip test — is relay behavior sign-dependent or magnitude-invariant?

Test: Take Qwen (relay), compute CCS deltas, then inject with:
1. Original deltas (baseline)
2. NEGATED zone deltas (flip signs, keep magnitudes)
3. NEGATED early deltas
4. ALL negated

If relay is geometry-invariant (F607 showed 2.6% variation despite 36% zone variation),
then negating signs should change injection direction.
If it's magnitude-invariant, sign flip should have no effect.
"""
import os, sys, json
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
        deltas.append({"layer": i, "delta_s2_s1": r_f - r_n})
    return deltas


def inject_modified(model_tgt, tok_tgt, ccs_deltas, n_src, modifier_fn, strength=5.0):
    """Inject with modified deltas."""
    device = model_tgt.device
    n_tgt = len(get_model_layers(model_tgt))
    layer_map = {}
    for i in range(n_tgt):
        rel = i / (n_tgt - 1) if n_tgt > 1 else 0
        best = min(range(n_src), key=lambda j: abs(j / (n_src - 1) - rel))
        layer_map[i] = best

    modified_deltas = modifier_fn(ccs_deltas)

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
        delta = modified_deltas[layer_a]
        if abs(delta["delta_s2_s1"]) < 0.001:
            continue
        hook = tgt_layers[layer_b].register_forward_hook(make_hook(delta, strength))
        hooks.append(hook)

    with torch.no_grad():
        inj_out = model_tgt(**inputs, output_hidden_states=True)
    inj_logits = inj_out.logits[0, -1].float().cpu()
    for h in hooks:
        h.remove()
    return (inj_logits - baseline_logits).mean().item()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sources = [
        ("Qwen/Qwen2.5-3B", "Relay"),
        ("EleutherAI/pythia-2.8b", "Tunnel"),
        ("google/gemma-2-2b", "Sorter"),
        ("microsoft/phi-2", "Mismatch"),
    ]

    target_name = "LiquidAI/LFM2.5-1.2B-Instruct"
    print(f"Loading target: {target_name}")
    tok_tgt = AutoTokenizer.from_pretrained(target_name, trust_remote_code=True)
    model_tgt = AutoModelForCausalLM.from_pretrained(
        target_name, torch_dtype=torch.float16, trust_remote_code=True,
        attn_implementation="eager").to(device)
    model_tgt.eval()

    results = {"target": target_name, "strength": 5.0, "species": {}}

    for src_name, species in sources:
        print(f"\n{'='*60}")
        print(f"Source: {src_name} ({species})")
        print(f"{'='*60}")

        tok_src = AutoTokenizer.from_pretrained(src_name, trust_remote_code=True)
        model_src = AutoModelForCausalLM.from_pretrained(
            src_name, torch_dtype=torch.float16, trust_remote_code=True,
            attn_implementation="eager").to(device)
        model_src.eval()
        n_src = len(get_model_layers(model_src))

        deltas = compute_ccs_deltas(model_src, tok_src, device)

        modifiers = {
            "original": lambda d: d,
            "negate_zone": lambda d: [
                {"layer": x["layer"], "delta_s2_s1": -x["delta_s2_s1"] if x["layer"] >= 2 else x["delta_s2_s1"]}
                for x in d
            ],
            "negate_early": lambda d: [
                {"layer": x["layer"], "delta_s2_s1": -x["delta_s2_s1"] if x["layer"] < 2 else x["delta_s2_s1"]}
                for x in d
            ],
            "negate_all": lambda d: [
                {"layer": x["layer"], "delta_s2_s1": -x["delta_s2_s1"]}
                for x in d
            ],
        }

        species_result = {"source": src_name, "n_layers": n_src, "configs": {}}

        print(f"{'Config':>15} {'Shift':>12} {'Sign':>6}")
        print("-" * 38)
        for name, mod_fn in modifiers.items():
            shift = inject_modified(model_tgt, tok_tgt, deltas, n_src, mod_fn, 5.0)
            sign = "+" if shift > 0 else "-"
            species_result["configs"][name] = {"shift": shift, "sign": sign}
            print(f"{name:>15} {shift:+.6f} {sign:>6}")

        # Analysis
        orig = species_result["configs"]["original"]["shift"]
        neg_zone = species_result["configs"]["negate_zone"]["shift"]
        neg_all = species_result["configs"]["negate_all"]["shift"]

        flipped = (orig > 0) != (neg_zone > 0)
        ratio = neg_zone / orig if orig != 0 else float('inf')
        species_result["zone_flip"] = flipped
        species_result["zone_ratio"] = ratio

        print(f"\n  Zone negation {'FLIPS' if flipped else 'preserves'} injection sign")
        print(f"  Ratio (neg_zone/original): {ratio:.3f}")

        results["species"][species] = species_result
        del model_src
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: Sign-Flip Response")
    print(f"{'='*60}")
    print(f"{'Species':>12} {'Original':>10} {'Neg Zone':>10} {'Neg All':>10} {'Flips?':>8} {'Ratio':>8}")
    for species, sr in results["species"].items():
        c = sr["configs"]
        print(f"{species:>12} {c['original']['shift']:+.4f} {c['negate_zone']['shift']:+.4f} {c['negate_all']['shift']:+.4f} {'YES' if sr['zone_flip'] else 'NO':>8} {sr['zone_ratio']:+.3f}")

    outpath = os.path.join(RESULTS_DIR, "f609_relay_signflip.json")
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")

    del model_tgt
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
