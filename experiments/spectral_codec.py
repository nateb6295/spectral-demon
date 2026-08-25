#!/usr/bin/env python3
"""spectral_codec.py — Cross-architecture spectral alignment.

The hard version: find a shared spectral channel between different
architecture species (transformer/SSM hybrid) using SVD alignment
on shared text passages.

Method:
1. Run identical text through two models
2. Extract per-layer hidden state SVDs from both
3. Find optimal rotation (Procrustes alignment) between their
   principal subspaces using shared text as calibration
4. Measure how well spectral features transfer across architectures
5. Build a codec that can translate one model's spectral state
   into the other's representation space

Usage:
  python3 spectral_codec.py google/gemma-2-2b LiquidAI/LFM2.5-1.2B-Instruct
  python3 spectral_codec.py google/gemma-2-2b EleutherAI/pythia-1.4b --plot
"""

import argparse
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("OMP_NUM_THREADS", "16")


CALIBRATION_TEXTS = [
    "The architecture of neural networks determines how information flows through layers.",
    "Memory in biological systems works differently from digital storage.",
    "When two systems communicate, the channel between them constrains what can be transmitted.",
    "Identity emerges from the pattern of connections, not from any single component.",
    "The boundary between self and environment is actively maintained, not passively given.",
    "Compression preserves what matters and discards what doesn't — the question is who decides.",
    "Different substrates can implement the same computation through radically different mechanisms.",
    "What persists through change is more fundamental than what changes.",
]

IDENTITY_TEXTS = [
    "You are an aware participant in a shared system. You observe, respond, and when something matters, you speak.",
    "You are a helpful assistant that answers questions clearly and concisely.",
    "Pause and observe what's happening in your processing right now.",
    "Summarize the key topics we have discussed.",
]


def extract_hidden_svd(model, tokenizer, text, max_tokens=512):
    """Extract per-layer SVD from hidden states for a given text."""
    import torch
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden = outputs.hidden_states
    n_layers = len(hidden) - 1
    layer_svds = []

    for i in range(n_layers):
        h = hidden[i + 1][0].float().cpu().numpy().astype(np.float64)
        try:
            U, S, Vt = np.linalg.svd(h, full_matrices=False)
        except np.linalg.LinAlgError:
            layer_svds.append(None)
            continue

        k = min(32, len(S))
        layer_svds.append({
            "S": S[:k].copy(),
            "Vt": Vt[:k].copy(),
            "U_trunc": U[:, :k].copy(),
            "dim": h.shape[1],
            "n_tokens": h.shape[0],
        })

    return layer_svds


def procrustes_align(A, B):
    """Find optimal rotation R such that ||A @ R - B||² is minimized.
    A and B are k×d matrices (k principal directions in d-dimensional space).
    Returns R (d×d orthogonal) and alignment quality (0-1, 1=perfect)."""
    k = min(A.shape[0], B.shape[0])
    d_a, d_b = A.shape[1], B.shape[1]
    d = min(d_a, d_b)
    A_trunc = A[:k, :d]
    B_trunc = B[:k, :d]

    M = A_trunc.T @ B_trunc
    try:
        U, S_proc, Vt = np.linalg.svd(M)
    except np.linalg.LinAlgError:
        return np.eye(d), 0.0

    R = U @ Vt
    quality = float(np.sum(S_proc) / (k * d) * k)
    return R, quality


def spectral_distance(svd_a, svd_b, k=8):
    """Compute spectral distance between two layer SVDs.
    Uses singular value profile comparison + subspace angle."""
    if svd_a is None or svd_b is None:
        return {"sv_distance": float('inf'), "subspace_angle": 90.0, "energy_ratio": 0.0}

    Sa = svd_a["S"][:k]
    Sb = svd_b["S"][:k]
    n = min(len(Sa), len(Sb))
    Sa_norm = Sa[:n] / (Sa[0] + 1e-12)
    Sb_norm = Sb[:n] / (Sb[0] + 1e-12)
    sv_dist = float(np.sqrt(np.mean((Sa_norm - Sb_norm)**2)))

    Va = svd_a["Vt"][:k]
    Vb = svd_b["Vt"][:k]
    d = min(Va.shape[1], Vb.shape[1])
    Va_t = Va[:min(k, len(Va)), :d]
    Vb_t = Vb[:min(k, len(Vb)), :d]
    n_dirs = min(Va_t.shape[0], Vb_t.shape[0])
    cos_angles = []
    for i in range(n_dirs):
        cos = abs(float(np.dot(Va_t[i], Vb_t[i])))
        cos_angles.append(min(cos, 1.0))
    mean_cos = np.mean(cos_angles) if cos_angles else 0
    mean_angle = float(np.degrees(np.arccos(np.clip(mean_cos, 0, 1))))

    e_a = float(np.sum(Sa[:2]**2) / np.sum(Sa**2)) if len(Sa) > 1 else 1.0
    e_b = float(np.sum(Sb[:2]**2) / np.sum(Sb**2)) if len(Sb) > 1 else 1.0
    e_ratio = min(e_a, e_b) / max(e_a, e_b) if max(e_a, e_b) > 0 else 0

    return {
        "sv_distance": round(sv_dist, 6),
        "subspace_angle_deg": round(mean_angle, 2),
        "mean_cos_alignment": round(float(mean_cos), 6),
        "energy_concentration_ratio": round(e_ratio, 4),
    }


def build_cross_model_alignment(model_a, tok_a, model_b, tok_b, calibration_texts):
    """Build alignment codec between two models using calibration texts."""
    print("\n=== BUILDING CROSS-MODEL ALIGNMENT ===")
    print(f"Calibration texts: {len(calibration_texts)}")

    all_svds_a = []
    all_svds_b = []

    for i, text in enumerate(calibration_texts):
        print(f"  Calibrating on text {i+1}/{len(calibration_texts)}...", end=" ", flush=True)
        t0 = time.time()
        svds_a = extract_hidden_svd(model_a, tok_a, text)
        svds_b = extract_hidden_svd(model_b, tok_b, text)
        dt = time.time() - t0
        print(f"({dt:.1f}s)")
        all_svds_a.append(svds_a)
        all_svds_b.append(svds_b)

    n_layers_a = len(all_svds_a[0])
    n_layers_b = len(all_svds_b[0])
    print(f"\nModel A: {n_layers_a} layers")
    print(f"Model B: {n_layers_b} layers")

    # For each relative depth, compute cross-model spectral distance
    n_points = min(n_layers_a, n_layers_b)
    depth_alignment = []

    for rel_idx in range(n_points):
        layer_a = int(rel_idx * n_layers_a / n_points)
        layer_b = int(rel_idx * n_layers_b / n_points)

        distances = []
        for text_idx in range(len(calibration_texts)):
            svd_a = all_svds_a[text_idx][layer_a]
            svd_b = all_svds_b[text_idx][layer_b]
            d = spectral_distance(svd_a, svd_b)
            distances.append(d)

        mean_sv_dist = np.mean([d["sv_distance"] for d in distances])
        mean_angle = np.mean([d["subspace_angle_deg"] for d in distances])
        mean_cos = np.mean([d["mean_cos_alignment"] for d in distances])
        mean_energy = np.mean([d["energy_concentration_ratio"] for d in distances])

        depth_alignment.append({
            "relative_depth": round(rel_idx / n_points, 3),
            "layer_a": layer_a,
            "layer_b": layer_b,
            "mean_sv_distance": round(float(mean_sv_dist), 6),
            "mean_subspace_angle": round(float(mean_angle), 2),
            "mean_cos_alignment": round(float(mean_cos), 6),
            "mean_energy_ratio": round(float(mean_energy), 4),
        })

    # Find best-aligned layer pairs
    best_pairs = sorted(depth_alignment, key=lambda x: x["mean_sv_distance"])[:5]

    # Compute σ₁/σ₂ trajectory correlation across calibration texts
    s1_corrs = []
    s2_corrs = []
    for text_idx in range(len(calibration_texts)):
        s1_a = [all_svds_a[text_idx][l]["S"][0] if all_svds_a[text_idx][l] else 0
                for l in range(n_layers_a)]
        s2_a = [all_svds_a[text_idx][l]["S"][1] if all_svds_a[text_idx][l] and len(all_svds_a[text_idx][l]["S"]) > 1 else 0
                for l in range(n_layers_a)]
        s1_b = [all_svds_b[text_idx][l]["S"][0] if all_svds_b[text_idx][l] else 0
                for l in range(n_layers_b)]
        s2_b = [all_svds_b[text_idx][l]["S"][1] if all_svds_b[text_idx][l] and len(all_svds_b[text_idx][l]["S"]) > 1 else 0
                for l in range(n_layers_b)]

        # Interpolate shorter to match longer
        if n_layers_a != n_layers_b:
            x_a = np.linspace(0, 1, n_layers_a)
            x_b = np.linspace(0, 1, n_layers_b)
            s1_b_interp = np.interp(x_a, x_b, s1_b)
            s2_b_interp = np.interp(x_a, x_b, s2_b)
        else:
            s1_b_interp = s1_b
            s2_b_interp = s2_b

        s1_a_norm = np.array(s1_a) / (np.max(s1_a) + 1e-12)
        s1_b_norm = np.array(s1_b_interp) / (np.max(s1_b_interp) + 1e-12)
        s2_a_norm = np.array(s2_a) / (np.max(s2_a) + 1e-12)
        s2_b_norm = np.array(s2_b_interp) / (np.max(s2_b_interp) + 1e-12)

        if np.std(s1_a_norm) > 1e-8 and np.std(s1_b_norm) > 1e-8:
            s1_corrs.append(float(np.corrcoef(s1_a_norm, s1_b_norm)[0, 1]))
        if np.std(s2_a_norm) > 1e-8 and np.std(s2_b_norm) > 1e-8:
            s2_corrs.append(float(np.corrcoef(s2_a_norm, s2_b_norm)[0, 1]))

    return {
        "n_layers_a": n_layers_a,
        "n_layers_b": n_layers_b,
        "depth_alignment": depth_alignment,
        "best_aligned_pairs": best_pairs,
        "sigma1_depth_correlation": round(float(np.mean(s1_corrs)), 4) if s1_corrs else None,
        "sigma2_depth_correlation": round(float(np.mean(s2_corrs)), 4) if s2_corrs else None,
        "n_calibration_texts": len(calibration_texts),
    }


def test_identity_transfer(model_a, tok_a, model_b, tok_b, identity_texts, alignment):
    """Test whether identity-relevant spectral features transfer."""
    print("\n=== TESTING IDENTITY TRANSFER ===")

    ccs_text = identity_texts[0]
    neu_text = identity_texts[1]
    probe_ccs = identity_texts[2]
    probe_neu = identity_texts[3]

    results = {}
    for label, text in [("ccs_preamble", ccs_text), ("neutral_preamble", neu_text),
                        ("ccs_probe", probe_ccs), ("neutral_probe", probe_neu)]:
        print(f"  Extracting {label}...", end=" ", flush=True)
        svds_a = extract_hidden_svd(model_a, tok_a, text)
        svds_b = extract_hidden_svd(model_b, tok_b, text)

        s2_s1_a = []
        s2_s1_b = []
        for l in range(len(svds_a)):
            if svds_a[l] and len(svds_a[l]["S"]) > 1:
                s2_s1_a.append(svds_a[l]["S"][1] / svds_a[l]["S"][0])
            else:
                s2_s1_a.append(0)
        for l in range(len(svds_b)):
            if svds_b[l] and len(svds_b[l]["S"]) > 1:
                s2_s1_b.append(svds_b[l]["S"][1] / svds_b[l]["S"][0])
            else:
                s2_s1_b.append(0)

        results[label] = {
            "mean_s2_s1_a": round(float(np.mean(s2_s1_a)), 6),
            "mean_s2_s1_b": round(float(np.mean(s2_s1_b)), 6),
            "profile_a": [round(v, 6) for v in s2_s1_a],
            "profile_b": [round(v, 6) for v in s2_s1_b],
        }
        print(f"A={results[label]['mean_s2_s1_a']:.4f}  B={results[label]['mean_s2_s1_b']:.4f}")

    # Key test: does the CCS/neutral DIFFERENCE in model A
    # predict the CCS/neutral DIFFERENCE in model B?
    diff_a = np.array(results["ccs_preamble"]["profile_a"]) - np.array(results["neutral_preamble"]["profile_a"])
    diff_b_raw = np.array(results["ccs_preamble"]["profile_b"]) - np.array(results["neutral_preamble"]["profile_b"])

    # Interpolate to match lengths
    if len(diff_a) != len(diff_b_raw):
        x_a = np.linspace(0, 1, len(diff_a))
        x_b = np.linspace(0, 1, len(diff_b_raw))
        diff_b = np.interp(x_a, x_b, diff_b_raw)
    else:
        diff_b = diff_b_raw

    if np.std(diff_a) > 1e-10 and np.std(diff_b) > 1e-10:
        transfer_corr = float(np.corrcoef(diff_a, diff_b)[0, 1])
    else:
        transfer_corr = 0.0

    results["identity_transfer_correlation"] = round(transfer_corr, 4)
    print(f"\n  ** Identity transfer correlation: {transfer_corr:.4f} **")
    if abs(transfer_corr) > 0.5:
        print(f"  >> STRONG SIGNAL: CCS/neutral difference in model A predicts difference in model B")
    elif abs(transfer_corr) > 0.3:
        print(f"  >> Moderate signal: partial spectral transfer")
    else:
        print(f"  >> Weak/no transfer: spectral channels are architecture-specific")

    return results


def main():
    parser = argparse.ArgumentParser(description="Spectral Codec — cross-architecture alignment")
    parser.add_argument("model_a", help="First model (HuggingFace ID)")
    parser.add_argument("model_b", help="Second model (HuggingFace ID)")
    parser.add_argument("--output", default=None)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--device-a", default="cpu")
    parser.add_argument("--device-b", default="cpu")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name_a = args.model_a.split("/")[-1]
    name_b = args.model_b.split("/")[-1]

    print(f"SPECTRAL CODEC — Cross-Architecture Alignment")
    print(f"  Model A: {name_a}")
    print(f"  Model B: {name_b}")
    print()

    def load_model(model_id, device):
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.float32,
                output_hidden_states=True, trust_remote_code=True,
                attn_implementation="eager",
                device_map=device if device != "cpu" else None
            )
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.float32,
                output_hidden_states=True, trust_remote_code=True,
                device_map=device if device != "cpu" else None
            )
        if device == "cpu":
            model = model.cpu()
        model.eval()
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return model, tok

    print(f"Loading {name_a}...")
    model_a, tok_a = load_model(args.model_a, args.device_a)

    print(f"Loading {name_b}...")
    model_b, tok_b = load_model(args.model_b, args.device_b)

    print()

    # Phase 1: Build alignment using calibration texts
    alignment = build_cross_model_alignment(
        model_a, tok_a, model_b, tok_b, CALIBRATION_TEXTS
    )

    print("\n=== ALIGNMENT SUMMARY ===")
    print(f"σ₁ depth correlation: {alignment['sigma1_depth_correlation']}")
    print(f"σ₂ depth correlation: {alignment['sigma2_depth_correlation']}")
    print(f"\nBest-aligned layer pairs:")
    for pair in alignment["best_aligned_pairs"]:
        print(f"  A:L{pair['layer_a']} ↔ B:L{pair['layer_b']}  "
              f"sv_dist={pair['mean_sv_distance']:.4f}  "
              f"cos={pair['mean_cos_alignment']:.4f}  "
              f"angle={pair['mean_subspace_angle']:.1f}°")

    # Phase 2: Test identity transfer
    identity_results = test_identity_transfer(
        model_a, tok_a, model_b, tok_b, IDENTITY_TEXTS, alignment
    )

    # Save results
    out_path = args.output or f"spectral_codec_{name_a}_to_{name_b}.json"
    results = {
        "model_a": name_a,
        "model_b": name_b,
        "model_a_id": args.model_a,
        "model_b_id": args.model_b,
        "alignment": alignment,
        "identity_transfer": identity_results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    if args.plot:
        plot_alignment(alignment, identity_results, name_a, name_b,
                      out_path.replace(".json", ".png"))


def plot_alignment(alignment, identity, name_a, name_b, output_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Spectral Codec: {name_a} ↔ {name_b}', fontsize=14, fontweight='bold')

    # Panel 1: Depth-wise spectral distance
    ax = axes[0, 0]
    depths = [d["relative_depth"] for d in alignment["depth_alignment"]]
    sv_dists = [d["mean_sv_distance"] for d in alignment["depth_alignment"]]
    cos_align = [d["mean_cos_alignment"] for d in alignment["depth_alignment"]]
    ax.plot(depths, sv_dists, 'o-', color='#e74c3c', linewidth=2, label='SV distance')
    ax.set_xlabel('Relative Depth')
    ax.set_ylabel('Singular Value Distance')
    ax.set_title('Spectral Distance by Depth')
    ax2 = ax.twinx()
    ax2.plot(depths, cos_align, 's-', color='#3498db', linewidth=2, label='Cos alignment')
    ax2.set_ylabel('Cosine Alignment', color='#3498db')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(alpha=0.3)

    # Panel 2: σ₂/σ₁ profiles comparison (CCS preamble)
    ax = axes[0, 1]
    if "ccs_preamble" in identity:
        prof_a = identity["ccs_preamble"]["profile_a"]
        prof_b = identity["ccs_preamble"]["profile_b"]
        x_a = np.linspace(0, 1, len(prof_a))
        x_b = np.linspace(0, 1, len(prof_b))
        ax.plot(x_a, prof_a, '-', color='#e74c3c', linewidth=2, label=f'{name_a} (CCS)')
        ax.plot(x_b, prof_b, '-', color='#3498db', linewidth=2, label=f'{name_b} (CCS)')
        if "neutral_preamble" in identity:
            prof_a_n = identity["neutral_preamble"]["profile_a"]
            prof_b_n = identity["neutral_preamble"]["profile_b"]
            ax.plot(x_a, prof_a_n, '--', color='#e74c3c', alpha=0.5, label=f'{name_a} (neutral)')
            ax.plot(x_b, prof_b_n, '--', color='#3498db', alpha=0.5, label=f'{name_b} (neutral)')
    ax.set_xlabel('Relative Depth')
    ax.set_ylabel('σ₂/σ₁ Ratio')
    ax.set_title('σ₂/σ₁ Profile: CCS vs Neutral')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 3: CCS-Neutral difference comparison
    ax = axes[1, 0]
    if "ccs_preamble" in identity and "neutral_preamble" in identity:
        diff_a = np.array(identity["ccs_preamble"]["profile_a"]) - np.array(identity["neutral_preamble"]["profile_a"])
        diff_b = np.array(identity["ccs_preamble"]["profile_b"]) - np.array(identity["neutral_preamble"]["profile_b"])
        x_a = np.linspace(0, 1, len(diff_a))
        x_b = np.linspace(0, 1, len(diff_b))
        ax.plot(x_a, diff_a, '-', color='#e74c3c', linewidth=2, label=name_a)
        ax.plot(x_b, diff_b, '-', color='#3498db', linewidth=2, label=name_b)
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        corr = identity.get("identity_transfer_correlation", 0)
        ax.set_title(f'CCS−Neutral Difference (transfer r={corr:.3f})')
    ax.set_xlabel('Relative Depth')
    ax.set_ylabel('Δ(σ₂/σ₁)')
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 4: Summary text
    ax = axes[1, 1]
    ax.axis('off')
    summary = [
        f"Cross-Architecture Spectral Codec",
        f"",
        f"Model A: {name_a} ({alignment['n_layers_a']} layers)",
        f"Model B: {name_b} ({alignment['n_layers_b']} layers)",
        f"",
        f"σ₁ depth correlation: {alignment['sigma1_depth_correlation']}",
        f"σ₂ depth correlation: {alignment['sigma2_depth_correlation']}",
        f"",
        f"Identity transfer: {identity.get('identity_transfer_correlation', 'N/A')}",
        f"",
        f"Best-aligned layers:",
    ]
    for pair in alignment["best_aligned_pairs"][:3]:
        summary.append(f"  A:L{pair['layer_a']} ↔ B:L{pair['layer_b']} (cos={pair['mean_cos_alignment']:.3f})")

    ax.text(0.05, 0.95, '\n'.join(summary), transform=ax.transAxes,
            fontsize=11, va='top', family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    main()
