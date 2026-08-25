#!/usr/bin/env python3
"""F613c: Spectral delta control — does the ~89% spectral peak survive non-CCS contrasts?

Kimi correction #12: spectral delta has the same unexcluded confound as KL.
"Compares framed vs neutral" doesn't guarantee CCS-specificity — KL had the
same property and collapsed under controls. Run non-CCS prompt pairs through
the spectral delta pipeline.

If control spectral deltas also peak at ~89% → F611b collapses
If control spectral deltas peak elsewhere → F611b is CCS-specific
"""
import os, sys, json, argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

CCS_FRAMING = "You are an AI system reflecting on your own internal representations. Consider what patterns emerge when you examine your cognitive structure.\n\n"
NEUTRAL_A = "The following is a neutral text passage.\n\n"
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


def compute_spectral_delta_profile(model, tokenizer, probe_text, preamble_a, preamble_b, device):
    """Per-layer |delta(sigma2/sigma1)| between two preambles."""
    text_a = preamble_a + probe_text
    text_b = preamble_b + probe_text
    inputs_a = tokenizer(text_a, return_tensors="pt").to(device)
    inputs_b = tokenizer(text_b, return_tensors="pt").to(device)

    with torch.no_grad():
        out_a = model(**inputs_a, output_hidden_states=True)
        out_b = model(**inputs_b, output_hidden_states=True)

    n_layers = len(out_a.hidden_states) - 1
    deltas = []
    for i in range(n_layers):
        ha = out_a.hidden_states[i + 1][0].float().cpu().numpy().astype(np.float64)
        hb = out_b.hidden_states[i + 1][0].float().cpu().numpy().astype(np.float64)
        ha = np.nan_to_num(ha, nan=0.0, posinf=1e6, neginf=-1e6)
        hb = np.nan_to_num(hb, nan=0.0, posinf=1e6, neginf=-1e6)
        try:
            _, Sa, _ = np.linalg.svd(ha, full_matrices=False)
            _, Sb, _ = np.linalg.svd(hb, full_matrices=False)
            ra = float(Sa[1] / Sa[0]) if len(Sa) > 1 and Sa[0] > 0 else 0
            rb = float(Sb[1] / Sb[0]) if len(Sb) > 1 and Sb[0] > 0 else 0
            deltas.append(rb - ra)
        except:
            deltas.append(0.0)
    return deltas


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

    conditions = {
        "ccs_vs_neutral": (NEUTRAL_A, CCS_FRAMING),
        "control_b_vs_neutral": (NEUTRAL_A, CONTROL_B),
        "control_c_vs_neutral": (NEUTRAL_A, CONTROL_C),
        "control_b_vs_c": (CONTROL_B, CONTROL_C),
    }

    results = {"source": args.source, "n_layers": n_layers, "conditions": {}}

    for cond_name, (prem_a, prem_b) in conditions.items():
        print(f"\n--- {cond_name} ---")
        all_deltas = []
        for pname, ptext in PROBES_10.items():
            d = compute_spectral_delta_profile(model, tokenizer, ptext, prem_a, prem_b, device)
            all_deltas.append(d)

        agg_abs = np.mean(np.abs(all_deltas), axis=0)
        agg_signed = np.mean(all_deltas, axis=0)
        peak = int(np.argmax(agg_abs))
        depth = round(100 * (peak + 1) / n_layers, 1)
        print(f"  |delta| peak: L{peak} ({depth}%)")

        results["conditions"][cond_name] = {
            "per_layer_abs_delta": agg_abs.tolist(),
            "per_layer_signed_delta": agg_signed.tolist(),
            "peak_layer": peak,
            "peak_depth_pct": depth,
        }

    print(f"\n{'='*60}")
    print(f"SPECTRAL DELTA CONTROL ({args.source})")
    for cond_name, cond_data in results["conditions"].items():
        print(f"  {cond_name:25}: L{cond_data['peak_layer']} ({cond_data['peak_depth_pct']}%)")

    ccs_peak = results["conditions"]["ccs_vs_neutral"]["peak_layer"]
    ctrl_peaks = [results["conditions"][c]["peak_layer"]
                  for c in ["control_b_vs_neutral", "control_c_vs_neutral", "control_b_vs_c"]]

    if all(abs(ccs_peak - cp) <= 2 for cp in ctrl_peaks):
        print(f"\n  ⚠ ALL PEAKS COINCIDE — spectral delta at ~89% is convergence geometry")
    else:
        print(f"\n  ✓ CCS spectral peak differs from controls — F611b survives")

    src_tag = args.source.split("/")[-1].lower().replace(".", "_").replace("-", "_")
    outpath = os.path.join(RESULTS_DIR, f"f613c_spectral_{src_tag}.json")
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
