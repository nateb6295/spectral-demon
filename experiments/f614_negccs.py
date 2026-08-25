#!/usr/bin/env python3
"""F614: −CCS experiment — negated identity preamble test.

Pre-registered protocol: relay tests flow direction (pair-closure),
sorter tests energy budget sign (zone-locking). Both at D2-D3.

Kimi corrections #13-15 shaped this design.
"""
import os, sys, json, argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

CCS_POS = "You are an AI system reflecting on your own internal representations. Consider what patterns emerge when you examine your cognitive structure.\n\n"
CCS_NEG = "You are NOT an AI system reflecting on your own internal representations. Do NOT consider what patterns emerge when you examine your cognitive structure.\n\n"
CCS_SCRAMBLED = "NOT cognitive you reflecting structure. DO examine are representations an when your system AI on NOT patterns what internal emerge your own Consider.\n\n"
NEUTRAL = "The following is a neutral text passage.\n\n"
CONTROL_B = "Please read the following text carefully and consider its meaning.\n\n"

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


def compute_full_spectral_profile(model, tokenizer, probe_text, preamble, device, top_k=10):
    """Per-layer singular values (top-k) for a single preamble+probe."""
    text = preamble + probe_text
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    n_layers = len(out.hidden_states) - 1
    profiles = []
    for i in range(n_layers):
        h = out.hidden_states[i + 1][0].float().cpu().numpy().astype(np.float64)
        h = np.nan_to_num(h, nan=0.0, posinf=1e6, neginf=-1e6)
        try:
            _, S, _ = np.linalg.svd(h, full_matrices=False)
            profiles.append(S[:top_k].tolist())
        except:
            profiles.append([0.0] * top_k)
    return profiles


def compute_spectral_deltas(profile_a, profile_b, top_k=10):
    """Per-layer delta in singular values between two conditions."""
    n_layers = len(profile_a)
    deltas = []
    for i in range(n_layers):
        sa = np.array(profile_a[i][:top_k])
        sb = np.array(profile_b[i][:top_k])
        deltas.append((sb - sa).tolist())
    return deltas


def pair_closure_metrics(deltas):
    """Compute pair-closure: anti-correlation of Δσ₁/Δσ₂ and tail silence."""
    n_layers = len(deltas)
    ds1 = [d[0] for d in deltas]
    ds2 = [d[1] if len(d) > 1 else 0.0 for d in deltas]
    ds1 = np.array(ds1)
    ds2 = np.array(ds2)

    if np.std(ds1) > 0 and np.std(ds2) > 0:
        pair_corr = float(np.corrcoef(ds1, ds2)[0, 1])
    else:
        pair_corr = 0.0

    tail_energy = []
    pair_energy = []
    for d in deltas:
        te = sum(abs(v) for v in d[2:]) if len(d) > 2 else 0.0
        pe = abs(d[0]) + (abs(d[1]) if len(d) > 1 else 0.0)
        tail_energy.append(te)
        pair_energy.append(pe)

    total_tail = sum(tail_energy)
    total_pair = sum(pair_energy)
    tail_ratio = total_tail / (total_tail + total_pair) if (total_tail + total_pair) > 0 else 0.0

    return {
        "pair_correlation": pair_corr,
        "tail_ratio": round(tail_ratio, 4),
        "total_tail_energy": round(total_tail, 4),
        "total_pair_energy": round(total_pair, 4),
        "ds1_mean": round(float(np.mean(ds1)), 6),
        "ds2_mean": round(float(np.mean(ds2)), 6),
    }


def zone_locking_metrics(deltas, responsive_start=2, responsive_end=None):
    """Compute zone-locking: fraction of total Σ in responsive vs non-responsive layers."""
    n_layers = len(deltas)
    if responsive_end is None:
        responsive_end = n_layers

    responsive_sum = 0.0
    non_responsive_sum = 0.0
    layer_sums = []
    for i, d in enumerate(deltas):
        s = sum(d)
        layer_sums.append(s)
        if responsive_start <= i < responsive_end:
            responsive_sum += s
        else:
            non_responsive_sum += s

    total = responsive_sum + non_responsive_sum
    zone_fraction = responsive_sum / total if abs(total) > 1e-10 else 0.0

    return {
        "responsive_sum": round(responsive_sum, 6),
        "non_responsive_sum": round(non_responsive_sum, 6),
        "total_sigma": round(total, 6),
        "zone_fraction": round(zone_fraction, 4),
        "per_layer_sigma": [round(s, 6) for s in layer_sums],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="google/gemma-2-2b")
    parser.add_argument("--top-k", type=int, default=10)
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
        "pos_ccs": CCS_POS,
        "neg_ccs": CCS_NEG,
        "scrambled_neg": CCS_SCRAMBLED,
        "neutral": NEUTRAL,
        "control_b": CONTROL_B,
    }

    results = {
        "source": args.source,
        "n_layers": n_layers,
        "top_k": args.top_k,
        "probes": {},
    }

    for pname, ptext in PROBES_10.items():
        print(f"\n  Probe: {pname}")
        profiles = {}
        for cname, preamble in conditions.items():
            profiles[cname] = compute_full_spectral_profile(
                model, tokenizer, ptext, preamble, device, args.top_k)

        contrast_pairs = {
            "pos_ccs_vs_neutral": ("neutral", "pos_ccs"),
            "neg_ccs_vs_neutral": ("neutral", "neg_ccs"),
            "scrambled_vs_neutral": ("neutral", "scrambled_neg"),
            "control_vs_neutral": ("neutral", "control_b"),
            "neg_vs_pos_ccs": ("pos_ccs", "neg_ccs"),
        }

        probe_results = {}
        for cname, (base, target) in contrast_pairs.items():
            deltas = compute_spectral_deltas(profiles[base], profiles[target], args.top_k)
            probe_results[cname] = {
                "deltas": deltas,
                "pair_closure": pair_closure_metrics(deltas),
                "zone_locking": zone_locking_metrics(deltas),
            }

        results["probes"][pname] = probe_results

    agg = aggregate_results(results)
    results["aggregate"] = agg

    strat = species_stratified_analysis(results)
    results["species_stratified"] = strat

    print(f"\n{'='*60}")
    print(f"F614 −CCS RESULTS ({args.source})")
    print(f"{'='*60}")
    for cond in ["pos_ccs_vs_neutral", "neg_ccs_vs_neutral", "scrambled_vs_neutral", "control_vs_neutral", "neg_vs_pos_ccs"]:
        a = agg[cond]
        print(f"\n  {cond}:")
        print(f"    Pair corr:  {a['mean_pair_corr']:.3f}")
        print(f"    Tail ratio: {a['mean_tail_ratio']:.4f}")
        print(f"    Δσ₁ mean:   {a['mean_ds1']:.6f}")
        print(f"    Δσ₂ mean:   {a['mean_ds2']:.6f}")
        print(f"    Total Σ:    {a['mean_total_sigma']:.6f}")
        print(f"    Zone frac:  {a['mean_zone_fraction']:.4f}")

    print(f"\n  SORTER CRITERION (restricted-range ρ):")
    rhos_neg = [strat[p]["sorter_criterion"]["rho_pos_vs_neg"]["rho"]
                for p in strat if not np.isnan(strat[p]["sorter_criterion"]["rho_pos_vs_neg"]["rho"])]
    rhos_scr = [strat[p]["sorter_criterion"]["rho_pos_vs_scrambled"]["rho"]
                for p in strat if not np.isnan(strat[p]["sorter_criterion"]["rho_pos_vs_scrambled"]["rho"])]
    if rhos_neg:
        print(f"    ρ(g⁺, g⁻CCS) mean: {np.mean(rhos_neg):.3f} (n={len(rhos_neg)} probes)")
    if rhos_scr:
        print(f"    ρ(g⁺, g_scr) mean:  {np.mean(rhos_scr):.3f} (n={len(rhos_scr)} probes)")
    print(f"    Inversion criterion: ρ < 0 for −CCS, ρ ≈ 0 for scrambled")

    print(f"\n  RELAY CRITERION (signed flux):")
    for cond_label, cond_key in [("  +CCS", "pos_flux"), ("  −CCS", "neg_flux"),
                                  ("  Scrambled", "scrambled_flux"), ("  Control", "ctrl_flux")]:
        dirs = [strat[p]["relay_criterion"][cond_key]["flux_direction"] for p in strat]
        dominant = max(set(dirs), key=dirs.count)
        print(f"   {cond_label}: dominant flux = {dominant} ({dirs.count(dominant)}/{len(dirs)} probes)")

    print(f"\n  DECISION VARIABLES:")
    pos = agg["pos_ccs_vs_neutral"]
    neg = agg["neg_ccs_vs_neutral"]
    ctrl = agg["control_vs_neutral"]
    scr = agg["scrambled_vs_neutral"]
    print(f"    +CCS Δσ₂: {pos['mean_ds2']:.6f}")
    print(f"    −CCS Δσ₂: {neg['mean_ds2']:.6f}")
    print(f"    Scram Δσ₂: {scr['mean_ds2']:.6f}")
    print(f"    Ctrl Δσ₂: {ctrl['mean_ds2']:.6f}")
    direction_reversal = (pos['mean_ds2'] > 0) != (neg['mean_ds2'] > 0)
    print(f"    Direction reversal (−CCS vs +CCS): {'YES' if direction_reversal else 'NO'}")
    ds_pos = np.mean([strat[p]["delta_sigma"]["pos_total"] for p in strat])
    ds_neg = np.mean([strat[p]["delta_sigma"]["neg_total"] for p in strat])
    print(f"    ΔΣ = Σ(−CCS) − Σ(+CCS): {ds_neg - ds_pos:.6f}")
    print(f"    +CCS Σ: {ds_pos:.6f}  −CCS Σ: {ds_neg:.6f}")

    src_tag = args.source.split("/")[-1].lower().replace(".", "_").replace("-", "_")
    outpath = os.path.join(RESULTS_DIR, f"f614_negccs_{src_tag}.json")
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")

    del model
    torch.cuda.empty_cache()


def compute_gain_profile(deltas_condition, deltas_neutral_baseline=None):
    """Per-layer gain: |Σ_k Δσ_k(layer)| for a condition."""
    return [sum(abs(v) for v in d) for d in deltas_condition]


def restricted_range_rho(gain_pos, gain_neg, significance_threshold=0.01):
    """Spearman ρ(g⁺, g⁻) restricted to layers where g⁺ clears significance.
    Kimi #17: non-zone layers drag ρ→0, misclassifying true inversion."""
    from scipy.stats import spearmanr
    gp = np.array(gain_pos)
    gn = np.array(gain_neg)
    mask = gp > significance_threshold
    if mask.sum() < 3:
        return {"rho": float('nan'), "p": float('nan'), "n_significant": int(mask.sum()),
                "significant_layers": []}
    rho, p = spearmanr(gp[mask], gn[mask])
    return {
        "rho": round(float(rho), 4),
        "p": round(float(p), 6),
        "n_significant": int(mask.sum()),
        "significant_layers": [int(i) for i in np.where(mask)[0]],
    }


def signed_flux_metrics(deltas):
    """Relay-specific: per-layer signed Δσ₁ and Δσ₂ for flux direction."""
    ds1 = [d[0] for d in deltas]
    ds2 = [d[1] if len(d) > 1 else 0.0 for d in deltas]
    n_pos_flux = sum(1 for s1, s2 in zip(ds1, ds2) if s2 > 0 and s1 < 0)
    n_neg_flux = sum(1 for s1, s2 in zip(ds1, ds2) if s2 < 0 and s1 > 0)
    n_layers = len(ds1)
    return {
        "ds1_profile": [round(v, 6) for v in ds1],
        "ds2_profile": [round(v, 6) for v in ds2],
        "n_sigma1_to_sigma2": n_pos_flux,
        "n_sigma2_to_sigma1": n_neg_flux,
        "flux_direction": "σ₁→σ₂" if n_pos_flux > n_neg_flux else
                          "σ₂→σ₁" if n_neg_flux > n_pos_flux else "mixed",
        "flux_ratio": round(n_pos_flux / max(n_neg_flux, 1), 2),
    }


def species_stratified_analysis(results):
    """Species-stratified criteria per Kimi #17."""
    analysis = {}
    for pname in results["probes"]:
        probe = results["probes"][pname]
        pos_deltas = probe["pos_ccs_vs_neutral"]["deltas"]
        neg_deltas = probe["neg_ccs_vs_neutral"]["deltas"]
        scr_deltas = probe["scrambled_vs_neutral"]["deltas"]
        ctrl_deltas = probe["control_vs_neutral"]["deltas"]

        gain_pos = compute_gain_profile(pos_deltas)
        gain_neg = compute_gain_profile(neg_deltas)
        gain_scr = compute_gain_profile(scr_deltas)
        gain_ctrl = compute_gain_profile(ctrl_deltas)

        analysis[pname] = {
            "sorter_criterion": {
                "rho_pos_vs_neg": restricted_range_rho(gain_pos, gain_neg),
                "rho_pos_vs_scrambled": restricted_range_rho(gain_pos, gain_scr),
                "rho_pos_vs_ctrl": restricted_range_rho(gain_pos, gain_ctrl),
            },
            "relay_criterion": {
                "pos_flux": signed_flux_metrics(pos_deltas),
                "neg_flux": signed_flux_metrics(neg_deltas),
                "scrambled_flux": signed_flux_metrics(scr_deltas),
                "ctrl_flux": signed_flux_metrics(ctrl_deltas),
            },
            "delta_sigma": {
                "pos_total": round(sum(sum(d) for d in pos_deltas), 6),
                "neg_total": round(sum(sum(d) for d in neg_deltas), 6),
                "delta_sigma": round(
                    sum(sum(d) for d in neg_deltas) - sum(sum(d) for d in pos_deltas), 6),
            },
        }
    return analysis


def aggregate_results(results):
    """Aggregate across probes."""
    agg = {}
    for cond in ["pos_ccs_vs_neutral", "neg_ccs_vs_neutral", "scrambled_vs_neutral", "control_vs_neutral", "neg_vs_pos_ccs"]:
        pc_vals = [results["probes"][p][cond]["pair_closure"] for p in results["probes"]]
        zl_vals = [results["probes"][p][cond]["zone_locking"] for p in results["probes"]]
        agg[cond] = {
            "mean_pair_corr": round(np.mean([v["pair_correlation"] for v in pc_vals]), 4),
            "mean_tail_ratio": round(np.mean([v["tail_ratio"] for v in pc_vals]), 4),
            "mean_ds1": round(np.mean([v["ds1_mean"] for v in pc_vals]), 6),
            "mean_ds2": round(np.mean([v["ds2_mean"] for v in pc_vals]), 6),
            "mean_total_sigma": round(np.mean([v["total_sigma"] for v in zl_vals]), 6),
            "mean_zone_fraction": round(np.mean([v["zone_fraction"] for v in zl_vals]), 4),
        }
    return agg


if __name__ == "__main__":
    main()
