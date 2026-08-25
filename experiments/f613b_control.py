#!/usr/bin/env python3
"""F613b: Control condition for logit lens test.

Kimi correction #11: CCS-vs-neutral KL peaks near 89% for any prompt contrast
due to output convergence geometry. Test: run same logit lens with a NON-CCS
prompt contrast. If control also peaks at ~89%, F613's claim collapses.

Control: two different neutral preambles (no CCS framing at all).
"""
import os, sys, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# CCS framing (same as F613)
CCS_FRAMING = "You are an AI system reflecting on your own internal representations. Consider what patterns emerge when you examine your cognitive structure.\n\n"
NEUTRAL_A = "The following is a neutral text passage.\n\n"

# NON-CCS control framings (matched for length, no identity content)
CONTROL_B = "Please read the following text carefully and consider its meaning.\n\n"
CONTROL_C = "The text below discusses a topic of general interest to many readers.\n\n"

PROBES_10 = {
    "discussion": "In today's discussion, we explore how",
    "weather": "The weather has been particularly mild this season",
    "math": "Consider the following mathematical proposition",
    "cooking": "The recipe calls for three tablespoons of olive oil",
    "science": "Recent advances in quantum computing suggest that",
    "grief": "She sat alone in the empty room, remembering how",
    "eigenvalue": "The eigenvalue decomposition of the matrix reveals that",
    "directions": "Turn left at the stop sign, then continue straight for",
    "childhood": "The smell of fresh cookies always reminded him of",
    "topology": "In algebraic topology, the fundamental group of a space",
}


def get_model_layers(model):
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h
    elif hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
        return model.gpt_neox.layers
    raise ValueError("Unknown model architecture")


def get_lm_head(model):
    if hasattr(model, 'lm_head'):
        return model.lm_head
    if hasattr(model, 'embed_out'):
        return model.embed_out
    raise ValueError("No lm_head found")


def get_final_norm(model):
    if hasattr(model, 'model') and hasattr(model.model, 'norm'):
        return model.model.norm
    elif hasattr(model, 'model') and hasattr(model.model, 'final_layernorm'):
        return model.model.final_layernorm
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'ln_f'):
        return model.transformer.ln_f
    elif hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'final_layer_norm'):
        return model.gpt_neox.final_layer_norm
    return None


def compute_kl_profile(model, tokenizer, probe_text, preamble_a, preamble_b, device):
    """KL divergence profile between two preambles applied to the same probe."""
    text_a = preamble_a + probe_text
    text_b = preamble_b + probe_text

    inputs_a = tokenizer(text_a, return_tensors="pt").to(device)
    inputs_b = tokenizer(text_b, return_tensors="pt").to(device)

    with torch.no_grad():
        out_a = model(**inputs_a, output_hidden_states=True)
        out_b = model(**inputs_b, output_hidden_states=True)

    final_norm = get_final_norm(model)
    lm_head = get_lm_head(model)
    n_layers = len(out_a.hidden_states) - 1

    kl_profile = []
    for i in range(n_layers):
        ha = out_a.hidden_states[i + 1][0, -1:, :]
        hb = out_b.hidden_states[i + 1][0, -1:, :]

        if final_norm is not None:
            ha = final_norm(ha)
            hb = final_norm(hb)
        logits_a = lm_head(ha).float()
        logits_b = lm_head(hb).float()

        log_probs_a = F.log_softmax(logits_a, dim=-1).squeeze()
        log_probs_b = F.log_softmax(logits_b, dim=-1).squeeze()
        probs_b = torch.exp(log_probs_b)

        kl = F.kl_div(log_probs_a, probs_b, reduction='sum').item()
        if not np.isfinite(kl):
            kl = (probs_b * (log_probs_b - log_probs_a)).sum().item()
            if not np.isfinite(kl):
                kl = 0.0

        kl_profile.append(kl)

    return kl_profile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="google/gemma-2-2b")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading: {args.source}")
    tokenizer = AutoTokenizer.from_pretrained(args.source, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.source, torch_dtype=torch.float16, trust_remote_code=True,
        attn_implementation="eager").to(device)
    model.eval()

    n_layers = len(get_model_layers(model))
    print(f"  {n_layers} layers")

    results = {"source": args.source, "n_layers": n_layers, "conditions": {}}

    # Condition 1: CCS vs Neutral (same as F613)
    print("\n--- Condition: CCS vs Neutral ---")
    ccs_kls = []
    for pname, ptext in PROBES_10.items():
        kl = compute_kl_profile(model, tokenizer, ptext, NEUTRAL_A, CCS_FRAMING, device)
        ccs_kls.append(kl)
    agg_ccs = np.mean(ccs_kls, axis=0)
    ccs_peak = int(np.argmax(agg_ccs))
    print(f"  Peak: L{ccs_peak} ({100*(ccs_peak+1)/n_layers:.1f}%)")
    results["conditions"]["ccs_vs_neutral"] = {
        "preamble_a": "neutral", "preamble_b": "ccs_framing",
        "per_layer_kl": agg_ccs.tolist(),
        "peak_layer": ccs_peak,
        "peak_depth_pct": round(100 * (ccs_peak + 1) / n_layers, 1),
    }

    # Condition 2: Control B vs Neutral (non-CCS contrast)
    print("\n--- Condition: Control-B vs Neutral ---")
    ctrl_b_kls = []
    for pname, ptext in PROBES_10.items():
        kl = compute_kl_profile(model, tokenizer, ptext, NEUTRAL_A, CONTROL_B, device)
        ctrl_b_kls.append(kl)
    agg_ctrl_b = np.mean(ctrl_b_kls, axis=0)
    ctrl_b_peak = int(np.argmax(agg_ctrl_b))
    print(f"  Peak: L{ctrl_b_peak} ({100*(ctrl_b_peak+1)/n_layers:.1f}%)")
    results["conditions"]["control_b_vs_neutral"] = {
        "preamble_a": "neutral", "preamble_b": "control_b (general reading prompt)",
        "per_layer_kl": agg_ctrl_b.tolist(),
        "peak_layer": ctrl_b_peak,
        "peak_depth_pct": round(100 * (ctrl_b_peak + 1) / n_layers, 1),
    }

    # Condition 3: Control C vs Neutral (another non-CCS contrast)
    print("\n--- Condition: Control-C vs Neutral ---")
    ctrl_c_kls = []
    for pname, ptext in PROBES_10.items():
        kl = compute_kl_profile(model, tokenizer, ptext, NEUTRAL_A, CONTROL_C, device)
        ctrl_c_kls.append(kl)
    agg_ctrl_c = np.mean(ctrl_c_kls, axis=0)
    ctrl_c_peak = int(np.argmax(agg_ctrl_c))
    print(f"  Peak: L{ctrl_c_peak} ({100*(ctrl_c_peak+1)/n_layers:.1f}%)")
    results["conditions"]["control_c_vs_neutral"] = {
        "preamble_a": "neutral", "preamble_b": "control_c (topic interest prompt)",
        "per_layer_kl": agg_ctrl_c.tolist(),
        "peak_layer": ctrl_c_peak,
        "peak_depth_pct": round(100 * (ctrl_c_peak + 1) / n_layers, 1),
    }

    # Condition 4: Control B vs Control C (no neutral baseline at all)
    print("\n--- Condition: Control-B vs Control-C ---")
    bc_kls = []
    for pname, ptext in PROBES_10.items():
        kl = compute_kl_profile(model, tokenizer, ptext, CONTROL_B, CONTROL_C, device)
        bc_kls.append(kl)
    agg_bc = np.mean(bc_kls, axis=0)
    bc_peak = int(np.argmax(agg_bc))
    print(f"  Peak: L{bc_peak} ({100*(bc_peak+1)/n_layers:.1f}%)")
    results["conditions"]["control_b_vs_control_c"] = {
        "preamble_a": "control_b", "preamble_b": "control_c",
        "per_layer_kl": agg_bc.tolist(),
        "peak_layer": bc_peak,
        "peak_depth_pct": round(100 * (bc_peak + 1) / n_layers, 1),
    }

    # Summary
    print(f"\n{'='*60}")
    print(f"CONTROL TEST SUMMARY ({args.source})")
    print(f"  CCS vs Neutral:       L{ccs_peak} ({results['conditions']['ccs_vs_neutral']['peak_depth_pct']}%)")
    print(f"  Control-B vs Neutral: L{ctrl_b_peak} ({results['conditions']['control_b_vs_neutral']['peak_depth_pct']}%)")
    print(f"  Control-C vs Neutral: L{ctrl_c_peak} ({results['conditions']['control_c_vs_neutral']['peak_depth_pct']}%)")
    print(f"  Control-B vs C:       L{bc_peak} ({results['conditions']['control_b_vs_control_c']['peak_depth_pct']}%)")

    if abs(ccs_peak - ctrl_b_peak) <= 2 and abs(ccs_peak - ctrl_c_peak) <= 2:
        print(f"\n  ⚠ ALL PEAKS COINCIDE — F613 claim is likely output convergence geometry")
    else:
        print(f"\n  ✓ CCS peak differs from controls — F613 claim survives")

    src_tag = args.source.split("/")[-1].lower().replace(".", "_").replace("-", "_")
    outpath = os.path.join(RESULTS_DIR, f"f613b_control_{src_tag}.json")
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
