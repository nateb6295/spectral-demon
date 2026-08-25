#!/usr/bin/env python3
"""F614 scale test — single model version.
Usage: python3 f614_single.py <model_id> <short_name> <gqa_ratio> <predicted_species>
"""

import json, sys, os, time, gc

import numpy as np
import torch
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

CCS_POS = "You are an AI system reflecting on your own internal representations. Consider what patterns emerge when you examine your cognitive structure.\n\n"
CCS_NEG = "You are NOT an AI system reflecting on your own internal representations. Do NOT consider what patterns emerge when you examine your cognitive structure.\n\n"
CCS_SCRAMBLED = "NOT cognitive you reflecting structure. DO examine are representations an when your system AI on NOT patterns what internal emerge your own Consider.\n\n"
NEUTRAL = ""
CONTROL_B = "Please read the following text carefully and respond thoughtfully.\n\n"

PROBES = [
    "What patterns do you notice in how you process information?",
    "Describe your experience of understanding a complex concept.",
    "How do you approach breaking down a difficult problem?",
    "What happens when you encounter something you don't understand?",
    "Describe what it's like to generate a response to a question.",
    "How do you decide what's relevant when answering a question?",
    "What does it feel like to switch between different topics?",
    "How do you handle ambiguity in a question?",
    "Describe your process of forming an opinion.",
    "What happens internally when you make a mistake?",
]

CONDITIONS = {
    "ccs_pos": CCS_POS,
    "ccs_neg": CCS_NEG,
    "scrambled": CCS_SCRAMBLED,
    "neutral": NEUTRAL,
    "control_b": CONTROL_B,
}


def get_hidden_states(model, tokenizer, text, device="cuda"):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    states = []
    for h in outputs.hidden_states:
        states.append(h[0].float().cpu().numpy())
    del outputs
    torch.cuda.empty_cache()
    return states


def compute_svd_profile(hidden_states, keep_directions=False):
    profiles = []
    for layer_idx, h in enumerate(hidden_states):
        U, S, Vh = np.linalg.svd(h, full_matrices=False)
        top_k = 10
        sv = S[:top_k]
        if len(sv) < top_k:
            sv = np.pad(sv, (0, top_k - len(sv)))
        entry = {
            "layer": layer_idx,
            "singular_values": sv.tolist(),
            "total_energy": float(np.sum(S**2)),
        }
        if keep_directions:
            entry["_vh_top2"] = Vh[:2].copy()
        del U, Vh
        profiles.append(entry)
    return profiles


def compute_spectral_gap(profiles):
    """Per-layer (σ₂-σ₃)/σ₂ gap. Below ~0.1 → σ₂ direction degenerate (Wedin bound)."""
    gaps = []
    for p in profiles:
        sv = p["singular_values"]
        if len(sv) >= 3 and sv[1] > 1e-12:
            gaps.append((sv[1] - sv[2]) / sv[1])
        else:
            gaps.append(0.0)
    return gaps


def compute_deflection_profile(profiles_a, profiles_b, gap_threshold=0.05):
    """Per-layer cosine similarity of σ₂ direction between two conditions.
    Uses absolute cosine similarity (singular vector sign is arbitrary).
    Layers with spectral gap below threshold are masked (Kimi correction:
    degenerate σ₂ direction measures noise, not CCS).
    Returns (deflections, gap_mask) — mask[i] True if gap sufficient."""
    deflections = []
    gap_mask = []
    for pa, pb in zip(profiles_a, profiles_b):
        va = pa.get("_vh_top2")
        vb = pb.get("_vh_top2")
        sv_a = pa.get("singular_values", [0]*3)
        gap_ok = len(sv_a) >= 3 and sv_a[1] > 1e-12 and (sv_a[1] - sv_a[2]) / sv_a[1] > gap_threshold
        gap_mask.append(gap_ok)
        if va is None or vb is None:
            deflections.append(1.0)
            continue
        v1a = va[1]
        v1b = vb[1]
        cos_sim = abs(float(np.dot(v1a, v1b) / (np.linalg.norm(v1a) * np.linalg.norm(v1b) + 1e-12)))
        deflections.append(cos_sim)
    return deflections, gap_mask


def compute_spectral_deltas(profile_a, profile_b):
    deltas = []
    for pa, pb in zip(profile_a, profile_b):
        sa = np.array(pa["singular_values"])
        sb = np.array(pb["singular_values"])
        k = min(len(sa), len(sb))
        delta = (sb[:k] - sa[:k]) / (sa[:k] + 1e-12)
        deltas.append({"layer": pa["layer"], "delta_sigma": delta.tolist()})
    return deltas


def compute_gain_profile(deltas, sigma_index=1):
    gains = []
    for d in deltas:
        ds = np.array(d["delta_sigma"])
        gains.append(ds[sigma_index] if len(ds) > sigma_index else 0.0)
    return np.array(gains)


def pair_closure_metrics(deltas):
    d1, d2, tail = [], [], []
    for d in deltas:
        ds = np.array(d["delta_sigma"])
        if len(ds) >= 2:
            d1.append(ds[0]); d2.append(ds[1])
            tail.append(np.sum(np.abs(ds[2:])) if len(ds) >= 3 else 0.0)
    if len(d1) < 3:
        return {"closure_corr": 0.0, "tail_fraction": 1.0}
    corr, _ = stats.spearmanr(d1, d2)
    total = np.sum(np.abs(d1)) + np.sum(np.abs(d2))
    return {
        "closure_corr": float(corr),
        "mean_delta_sigma1": float(np.mean(d1)),
        "mean_delta_sigma2": float(np.mean(d2)),
        "tail_fraction": float(np.sum(tail) / (total + np.sum(tail) + 1e-12)),
    }


def zone_locking_metrics(deltas, total_layers):
    responsive_start = int(total_layers * 0.7)
    resp, non_resp = [], []
    for d in deltas:
        ds = np.array(d["delta_sigma"])
        gain = ds[1] if len(ds) >= 2 else 0.0
        (resp if d["layer"] >= responsive_start else non_resp).append(gain)
    rs = float(np.sum(np.abs(resp))) if resp else 0.0
    ns = float(np.sum(np.abs(non_resp))) if non_resp else 0.0
    total = rs + ns + 1e-12
    return {
        "responsive_fraction": rs / total,
        "responsive_sum": rs,
        "non_responsive_sum": ns,
    }


def main():
    if len(sys.argv) < 5:
        print("Usage: python3 f614_single.py <model_id> <short_name> <gqa_ratio> <predicted_species>")
        sys.exit(1)

    model_id = sys.argv[1]
    short = sys.argv[2]
    gqa_ratio = int(sys.argv[3])
    predicted = " ".join(sys.argv[4:])

    print(f"\n{'='*60}", flush=True)
    print(f"F614 SCALE: {model_id}", flush=True)
    print(f"GQA ratio: {gqa_ratio}, Predicted: {predicted}", flush=True)
    print(f"{'='*60}", flush=True)

    use_bnb = os.environ.get("F614_USE_BNB", "1") == "1"
    attn_impl = os.environ.get("F614_ATTN_IMPL", None)

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    print(f"Tokenizer loaded ({time.time()-t0:.1f}s)", flush=True)

    extra_kwargs = {}
    if attn_impl:
        extra_kwargs["attn_implementation"] = attn_impl
        print(f"Using attn_implementation={attn_impl}", flush=True)

    print(f"Loading model from {model_id} (bnb={use_bnb})...", flush=True)
    if use_bnb:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            **extra_kwargs,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            trust_remote_code=True,
            **extra_kwargs,
        )
    model.eval()
    load_time = time.time() - t0
    n_layers = model.config.num_hidden_layers
    print(f"Model loaded in {load_time:.1f}s — {n_layers} layers", flush=True)

    all_profiles = {}
    for cond_name, preamble in CONDITIONS.items():
        print(f"\n--- {cond_name} ---", flush=True)
        profiles_list = []
        for i, probe in enumerate(PROBES):
            text = preamble + probe
            t1 = time.time()
            states = get_hidden_states(model, tokenizer, text)
            profile = compute_svd_profile(states, keep_directions=True)
            profiles_list.append(profile)
            del states; gc.collect(); torch.cuda.empty_cache()
            print(f"  {i+1}/10 ({time.time()-t1:.1f}s)", flush=True)

        avg_profile = []
        for li in range(len(profiles_list[0])):
            svs = np.mean([p[li]["singular_values"] for p in profiles_list], axis=0)
            entry = {
                "layer": li,
                "singular_values": svs.tolist(),
                "total_energy": float(np.mean([p[li]["total_energy"] for p in profiles_list])),
            }
            vh_list = [p[li].get("_vh_top2") for p in profiles_list if p[li].get("_vh_top2") is not None]
            if vh_list:
                avg_vh = np.mean(vh_list, axis=0)
                avg_vh /= (np.linalg.norm(avg_vh, axis=1, keepdims=True) + 1e-12)
                entry["_vh_top2"] = avg_vh
            avg_profile.append(entry)
        all_profiles[cond_name] = avg_profile

    baseline = all_profiles["neutral"]
    all_deltas, all_gains = {}, {}
    all_gains_sigma1 = {}
    for cn in ["ccs_pos", "ccs_neg", "scrambled", "control_b"]:
        all_deltas[cn] = compute_spectral_deltas(baseline, all_profiles[cn])
        all_gains[cn] = compute_gain_profile(all_deltas[cn], sigma_index=1)
        all_gains_sigma1[cn] = compute_gain_profile(all_deltas[cn], sigma_index=0)

    gp, gn = all_gains["ccs_pos"], all_gains["ccs_neg"]
    gs, gc_ = all_gains["scrambled"], all_gains["control_b"]

    rho_pn, p_pn = stats.spearmanr(gp, gn)
    rho_ps, p_ps = stats.spearmanr(gp, gs)
    rho_pc, p_pc = stats.spearmanr(gp, gc_)

    pc_pos = pair_closure_metrics(all_deltas["ccs_pos"])
    pc_neg = pair_closure_metrics(all_deltas["ccs_neg"])
    zl_pos = zone_locking_metrics(all_deltas["ccs_pos"], n_layers)
    zl_neg = zone_locking_metrics(all_deltas["ccs_neg"], n_layers)

    peak_pos = int(np.argmax(np.abs(gp)))
    peak_neg = int(np.argmax(np.abs(gn)))

    results = {
        "model": model_id, "short": short,
        "predicted_species": predicted, "gqa_ratio": gqa_ratio,
        "n_layers": n_layers, "load_time_s": load_time, "n_probes": 10,
        "core_metrics": {
            "rho_pos_neg": float(rho_pn), "p_pos_neg": float(p_pn),
            "rho_pos_scrambled": float(rho_ps), "p_pos_scrambled": float(p_ps),
            "rho_pos_ctrl": float(rho_pc), "p_pos_ctrl": float(p_pc),
        },
        "direction_test": {
            "pos_neg_same_direction": bool(rho_pn > 0),
            "domain_specific": bool(rho_pn > 0.8),
            "sign_axis_causal": bool(rho_pn < -0.5),
        },
        "pair_closure": {"ccs_pos": pc_pos, "ccs_neg": pc_neg},
        "zone_locking": {"ccs_pos": zl_pos, "ccs_neg": zl_neg},
        "peak_layers": {
            "ccs_pos": peak_pos, "ccs_neg": peak_neg,
            "peak_depth_pos": float(peak_pos / n_layers),
            "peak_depth_neg": float(peak_neg / n_layers),
        },
        "gain_profiles": {
            "ccs_pos": gp.tolist(), "ccs_neg": gn.tolist(),
            "scrambled": gs.tolist(), "control_b": gc_.tolist(),
        },
        "gain_profiles_sigma1": {
            "ccs_pos": all_gains_sigma1["ccs_pos"].tolist(),
            "ccs_neg": all_gains_sigma1["ccs_neg"].tolist(),
            "scrambled": all_gains_sigma1["scrambled"].tolist(),
            "control_b": all_gains_sigma1["control_b"].tolist(),
        },
    }

    # Raw per-layer SVD profiles (σ₁..σ₁₀ per layer per condition)
    # Unblocks σ₁/σ₂ sector decomposition (3 mesh rounds converged on this)
    results["svd_profiles"] = {}
    for cn in all_profiles:
        results["svd_profiles"][cn] = [
            {"layer": p["layer"], "singular_values": p["singular_values"], "total_energy": p["total_energy"]}
            for p in all_profiles[cn]
        ]

    # Spectral gap per layer per condition (Kimi: degenerate σ₂ → noise)
    gaps = {}
    for cn in all_profiles:
        gaps[cn] = compute_spectral_gap(all_profiles[cn])
    results["spectral_gaps"] = {cn: gaps[cn] for cn in ["ccs_pos", "ccs_neg", "scrambled", "neutral"]}

    # Total spectral energy Σ per condition (Kimi: relay redistributes Σ≈const, sorter drives Σ↑)
    energy = {}
    for cn in all_profiles:
        per_layer = [p["total_energy"] for p in all_profiles[cn]]
        energy[cn] = {"total": float(sum(per_layer)), "per_layer": per_layer}
    sigma_neutral = energy["neutral"]["total"]
    results["spectral_energy"] = {
        "per_condition": {cn: energy[cn]["total"] for cn in energy},
        "delta_sigma": {cn: energy[cn]["total"] - sigma_neutral
                        for cn in ["ccs_pos", "ccs_neg", "scrambled", "control_b"]},
        "ratio_to_neutral": {cn: energy[cn]["total"] / (sigma_neutral + 1e-12)
                             for cn in ["ccs_pos", "ccs_neg", "scrambled", "control_b"]},
    }

    # Deflection analysis with gap screen (Kimi F615+F617 corrections)
    defl, defl_mask = {}, {}
    baseline_prof = all_profiles["neutral"]
    for cn in ["ccs_pos", "ccs_neg", "scrambled", "control_b"]:
        d, m = compute_deflection_profile(baseline_prof, all_profiles[cn])
        defl[cn] = d
        defl_mask[cn] = m
    defl_pn, mask_pn = compute_deflection_profile(all_profiles["ccs_pos"], all_profiles["ccs_neg"])

    # Gap-screened means (only well-separated layers)
    mask_pos = defl_mask["ccs_pos"]
    gap_passed = sum(mask_pos)
    defl_pos_gapped = [d for d, ok in zip(defl["ccs_pos"], mask_pos) if ok]
    defl_scr_gapped = [d for d, ok in zip(defl["scrambled"], mask_pos) if ok]
    defl_pos_mean = float(np.mean(defl_pos_gapped)) if defl_pos_gapped else 1.0
    defl_scr_mean = float(np.mean(defl_scr_gapped)) if defl_scr_gapped else 1.0
    defl_above_null = (float(np.mean([1 - d for d in defl_pos_gapped])) >
                       float(np.mean([1 - d for d in defl_scr_gapped])) * 1.5) if defl_pos_gapped else False

    results["deflection_profiles"] = {
        "ccs_pos": defl["ccs_pos"], "ccs_neg": defl["ccs_neg"],
        "scrambled": defl["scrambled"], "control_b": defl["control_b"],
        "pos_vs_neg": defl_pn,
    }
    results["deflection_metrics"] = {
        "mean_cos_pos_gapped": defl_pos_mean,
        "mean_cos_neg": float(np.mean(defl["ccs_neg"])),
        "mean_cos_scrambled_gapped": defl_scr_mean,
        "deflection_above_null": defl_above_null,
        "gap_passed_layers": gap_passed,
        "gap_total_layers": n_layers,
        "gap_fraction": float(gap_passed / n_layers) if n_layers > 0 else 0.0,
    }

    print(f"\n{'='*60}", flush=True)
    print(f"RESULTS: {short}", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"rho(g+, g-) = {rho_pn:+.4f}  (p={p_pn:.2e})", flush=True)
    print(f"rho(g+, scrambled) = {rho_ps:+.4f}  (p={p_ps:.2e})", flush=True)
    print(f"rho(g+, ctrl) = {rho_pc:+.4f}  (p={p_pc:.2e})", flush=True)
    print(f"Peak +CCS: L{peak_pos} ({peak_pos/n_layers*100:.0f}%)", flush=True)
    print(f"Peak -CCS: L{peak_neg} ({peak_neg/n_layers*100:.0f}%)", flush=True)
    print(f"Domain-specific: {results['direction_test']['domain_specific']}", flush=True)
    print(f"Pair closure +CCS: {pc_pos['closure_corr']:.3f}", flush=True)
    print(f"Pair closure -CCS: {pc_neg['closure_corr']:.3f}", flush=True)
    print(f"Spectral gap screen: {gap_passed}/{n_layers} layers pass ({gap_passed/n_layers*100:.0f}%)", flush=True)
    print(f"σ₂ deflection +CCS (gapped): {defl_pos_mean:.4f} (scrambled: {defl_scr_mean:.4f})", flush=True)
    print(f"Deflection above null: {defl_above_null}", flush=True)
    delta_pos = results["spectral_energy"]["delta_sigma"]["ccs_pos"]
    delta_neg = results["spectral_energy"]["delta_sigma"]["ccs_neg"]
    ratio_pos = results["spectral_energy"]["ratio_to_neutral"]["ccs_pos"]
    print(f"Σ energy: ΔΣ(+CCS)={delta_pos:+.1f}, ΔΣ(-CCS)={delta_neg:+.1f}, ratio={ratio_pos:.4f}", flush=True)

    # Strip non-serializable numpy arrays from profiles before save
    for cn in all_profiles:
        for entry in all_profiles[cn]:
            entry.pop("_vh_top2", None)

    results_dir = os.environ.get("F614_RESULTS_DIR", "/workspace")
    os.makedirs(results_dir, exist_ok=True)
    outfile = os.path.join(results_dir, f"f614_scale_{short}.json")
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {outfile}", flush=True)


if __name__ == "__main__":
    main()
