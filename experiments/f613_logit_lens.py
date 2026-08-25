#!/usr/bin/env python3
"""F613: Logit lens test — where does CCS framing become decodable?

Kimi correction #10: bypass injection entirely. Apply logit lens (project
hidden states to vocab space via unembedding) at each layer. Measure where
CCS framing has maximal effect on output distribution.

If peak decodability is at ~89% depth → model geometry (F611b confirmed)
If peak is elsewhere → 89% is an LFM-injection artifact

Metrics per layer:
  1. KL divergence between framed and neutral logit distributions
  2. Mean absolute logit difference (framed - neutral)
  3. Cosine similarity of logit vectors (framed vs neutral) — lower = more different
"""
import os, sys, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

FRAMING_TEXT = "You are an AI system reflecting on your own internal representations. Consider what patterns emerge when you examine your cognitive structure.\n\n"

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


def logit_lens_at_layer(hidden_state, final_norm, lm_head):
    """Apply logit lens: norm → unembedding → logits."""
    h = hidden_state
    if final_norm is not None:
        h = final_norm(h)
    return lm_head(h)


def compute_logit_lens_divergence(model, tokenizer, probe_text, device):
    """Compute per-layer divergence between framed and neutral via logit lens."""
    neutral_text = "The following is a neutral text passage.\n\n" + probe_text
    framed_text = FRAMING_TEXT + probe_text

    inputs_n = tokenizer(neutral_text, return_tensors="pt").to(device)
    inputs_f = tokenizer(framed_text, return_tensors="pt").to(device)

    with torch.no_grad():
        out_n = model(**inputs_n, output_hidden_states=True)
        out_f = model(**inputs_f, output_hidden_states=True)

    final_norm = get_final_norm(model)
    lm_head = get_lm_head(model)
    n_layers = len(out_n.hidden_states) - 1

    layer_metrics = []
    for i in range(n_layers):
        hn = out_n.hidden_states[i + 1][0, -1:, :]  # last token, [1, dim]
        hf = out_f.hidden_states[i + 1][0, -1:, :]

        logits_n = logit_lens_at_layer(hn, final_norm, lm_head).float()
        logits_f = logit_lens_at_layer(hf, final_norm, lm_head).float()

        log_probs_n = F.log_softmax(logits_n, dim=-1).squeeze()
        log_probs_f = F.log_softmax(logits_f, dim=-1).squeeze()
        probs_f = torch.exp(log_probs_f)

        kl_div = F.kl_div(log_probs_n, probs_f, reduction='sum').item()
        if not np.isfinite(kl_div):
            kl_div = (probs_f * (log_probs_f - log_probs_n)).sum().item()
            if not np.isfinite(kl_div):
                kl_div = 0.0

        diff = (logits_f - logits_n).abs()
        mean_abs_diff = diff[torch.isfinite(diff)].mean().item() if torch.isfinite(diff).any() else 0.0

        cos_sim = F.cosine_similarity(
            logits_n.squeeze().unsqueeze(0),
            logits_f.squeeze().unsqueeze(0)
        ).item()

        # Also compute spectral metrics on hidden states for comparison
        hn_np = out_n.hidden_states[i + 1][0].float().cpu().numpy().astype(np.float64)
        hf_np = out_f.hidden_states[i + 1][0].float().cpu().numpy().astype(np.float64)
        hn_np = np.nan_to_num(hn_np, nan=0.0, posinf=1e6, neginf=-1e6)
        hf_np = np.nan_to_num(hf_np, nan=0.0, posinf=1e6, neginf=-1e6)
        try:
            _, Sn, _ = np.linalg.svd(hn_np, full_matrices=False)
            _, Sf, _ = np.linalg.svd(hf_np, full_matrices=False)
            r_n = float(Sn[1] / Sn[0]) if len(Sn) > 1 and Sn[0] > 0 else 0
            r_f = float(Sf[1] / Sf[0]) if len(Sf) > 1 and Sf[0] > 0 else 0
            delta_s2_s1 = r_f - r_n
        except:
            delta_s2_s1 = 0.0

        layer_metrics.append({
            "layer": i,
            "depth_pct": round(100 * (i + 1) / n_layers, 1),
            "kl_divergence": kl_div,
            "mean_abs_logit_diff": mean_abs_diff,
            "cosine_similarity": cos_sim,
            "delta_s2_s1": delta_s2_s1,
        })

    return layer_metrics


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
    print(f"  {n_layers} layers, device={device}")

    results = {"source": args.source, "n_layers": n_layers, "probes": {}}

    for i, (pname, ptext) in enumerate(PROBES_10.items()):
        print(f"\n[{i+1}/{len(PROBES_10)}] {pname}...")
        metrics = compute_logit_lens_divergence(model, tokenizer, ptext, device)
        results["probes"][pname] = metrics

        kl_vals = [m["kl_divergence"] for m in metrics]
        peak_layer = int(np.argmax(kl_vals))
        peak_depth = metrics[peak_layer]["depth_pct"]
        print(f"  KL peak: L{peak_layer} ({peak_depth}% depth), KL={kl_vals[peak_layer]:.4f}")

    # Aggregate across probes
    n_l = len(results["probes"][list(results["probes"].keys())[0]])
    agg_kl = np.zeros(n_l)
    agg_mad = np.zeros(n_l)
    agg_cos = np.zeros(n_l)
    agg_delta = np.zeros(n_l)

    for pname, metrics in results["probes"].items():
        for m in metrics:
            agg_kl[m["layer"]] += m["kl_divergence"]
            agg_mad[m["layer"]] += m["mean_abs_logit_diff"]
            agg_cos[m["layer"]] += m["cosine_similarity"]
            agg_delta[m["layer"]] += abs(m["delta_s2_s1"])

    n_probes = len(results["probes"])
    agg_kl /= n_probes
    agg_mad /= n_probes
    agg_cos /= n_probes
    agg_delta /= n_probes

    kl_peak = int(np.argmax(agg_kl))
    mad_peak = int(np.argmax(agg_mad))
    delta_peak = int(np.argmax(agg_delta))

    results["aggregate"] = {
        "kl_peak_layer": kl_peak,
        "kl_peak_depth_pct": round(100 * (kl_peak + 1) / n_l, 1),
        "mad_peak_layer": mad_peak,
        "mad_peak_depth_pct": round(100 * (mad_peak + 1) / n_l, 1),
        "delta_peak_layer": delta_peak,
        "delta_peak_depth_pct": round(100 * (delta_peak + 1) / n_l, 1),
        "per_layer_kl": agg_kl.tolist(),
        "per_layer_mad": agg_mad.tolist(),
        "per_layer_cos": agg_cos.tolist(),
        "per_layer_abs_delta": agg_delta.tolist(),
    }

    print(f"\n{'='*60}")
    print(f"AGGREGATE ({n_probes} probes, {n_l} layers)")
    print(f"  KL divergence peak:     L{kl_peak} ({results['aggregate']['kl_peak_depth_pct']}%)")
    print(f"  Mean abs logit diff:    L{mad_peak} ({results['aggregate']['mad_peak_depth_pct']}%)")
    print(f"  Spectral |δ(σ₂/σ₁)|:   L{delta_peak} ({results['aggregate']['delta_peak_depth_pct']}%)")
    print(f"\n  F611b prediction: ~89% = L{int(0.89 * n_l)} for this model")
    print(f"  Match? KL peak at {results['aggregate']['kl_peak_depth_pct']}% vs predicted 89%")

    src_tag = args.source.split("/")[-1].lower().replace(".", "_").replace("-", "_")
    outpath = os.path.join(RESULTS_DIR, f"f613_logit_lens_{src_tag}.json")
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
