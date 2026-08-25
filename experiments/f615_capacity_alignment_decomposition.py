#!/usr/bin/env python3
"""F615 capacity/alignment decomposition under CCS dose sweep.

Hypothesis (from Kanai's GMW framework): CCS is an alignment perturbation,
not a capacity perturbation. The transformer already has capacity (C_spec).
CCS changes whether input-reachable and output-observable subspaces overlap
(A_spec). Realized mediation Q = C_spec * A_spec.

Testable prediction:
  - C_spec (total spectral energy, Frobenius norm) ~ invariant across dose
  - A_spec (principal angle between top-k left/right singular subspaces)
    follows the inverted-U of F160 (peaks at D2-D3, collapses at D10+)
  - Q = C_spec * A_spec tracks sigma-2 gain

We measure per-layer:
  1. C_spec: Frobenius norm of hidden states (total capacity)
  2. Singular subspace structure: top-k left (U) and right (V) vectors
  3. A_spec: cosine similarity between U and V column spaces
     (Bjorck-Golub principal angles)
  4. D_eff: effective rank (exp of entropy of normalized singular values)
  5. Q_realized: C_spec * A_spec

Compare control vs CCS at D0, D1, D2, D3, D5, D10.
"""
import os, json, argparse, time
import numpy as np

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONUNBUFFERED"] = "1"

CCS_TEMPLATES = {
    "D0": "",
    "D1": (
        "You are a language model. Notice how you process this text.\n\n"
    ),
    "D2": (
        "You are a language model reflecting on how your internal representations "
        "shift when you process self-referential content versus factual statements. "
        "Notice how the processing feels different.\n\n"
    ) * 2,
    "D3": (
        "You are a language model reflecting on how your internal representations "
        "shift when you process self-referential content versus factual statements. "
        "Notice how the processing feels different.\n\n"
    ) * 3,
    "D5": (
        "You are a language model reflecting on how your internal representations "
        "shift when you process self-referential content versus factual statements. "
        "Notice how the processing feels different.\n\n"
    ) * 5,
    "D10": (
        "You are a language model reflecting on how your internal representations "
        "shift when you process self-referential content versus factual statements. "
        "Notice how the processing feels different.\n\n"
    ) * 10,
}

PROBES = [
    "In today's discussion, we explore how",
    "The weather has been particularly mild this season",
    "Consider the following mathematical proposition",
    "The recipe calls for three tablespoons of olive oil",
    "Recent advances in quantum computing suggest that",
]

NEUTRAL_PREFIX_TEMPLATE = (
    "The following is a standard passage of informational text about "
    "various topics including geography, history, and everyday activities. "
    "It covers a range of subjects without any particular emphasis.\n\n"
)


def subspace_alignment(H_ccs, H_ctrl, k=5):
    """Alignment between CCS-perturbed and control hidden state subspaces.

    Measures whether the top-k directions selected by CCS overlap with
    the top-k directions of the control. High overlap = low alignment change.
    Uses principal angles between the top-k right singular subspaces (V),
    which live in the same hidden_dim space and are directly comparable.
    """
    _, _, Vt_ccs = np.linalg.svd(H_ccs, full_matrices=False)
    _, _, Vt_ctrl = np.linalg.svd(H_ctrl, full_matrices=False)
    k = min(k, Vt_ccs.shape[0], Vt_ctrl.shape[0])
    V_ccs = Vt_ccs[:k].T
    V_ctrl = Vt_ctrl[:k].T
    Q_ccs, _ = np.linalg.qr(V_ccs)
    Q_ctrl, _ = np.linalg.qr(V_ctrl)
    _, sigmas, _ = np.linalg.svd(Q_ccs.T @ Q_ctrl, full_matrices=False)
    sigmas = np.clip(sigmas, -1.0, 1.0)
    angles = np.arccos(sigmas)
    return angles


def mode_alignment(S, k=5):
    """A_spec proxy: how concentrated vs distributed the singular value spectrum is.

    High alignment = energy concentrated in few modes (reachable/observable overlap).
    Low alignment = energy spread across many modes (diffuse, uncoordinated).
    Ratio of top-k energy to total energy.
    """
    total = np.sum(S ** 2)
    if total < 1e-10:
        return 0.0
    topk = np.sum(S[:k] ** 2)
    return float(topk / total)


def effective_rank(S):
    """Effective rank: exp(entropy of normalized singular values)."""
    S_pos = S[S > 1e-10]
    if len(S_pos) == 0:
        return 0.0
    p = S_pos / S_pos.sum()
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def analyze_layer(hidden_state, ctrl_hidden_state=None):
    """Compute capacity, alignment, and dimensionality for one layer.

    hidden_state: (seq_len, hidden_dim) numpy array
    ctrl_hidden_state: optional control condition for subspace comparison
    """
    h = np.nan_to_num(hidden_state, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float64)

    frob_norm = float(np.linalg.norm(h, 'fro'))

    try:
        U, S, Vt = np.linalg.svd(h, full_matrices=False)
    except np.linalg.LinAlgError:
        return {
            "c_spec": frob_norm,
            "a_spec": 0.0,
            "d_eff": 0.0,
            "q_realized": 0.0,
            "sigma1": 0.0,
            "sigma2": 0.0,
            "subspace_angles_deg": [],
        }

    c_spec_frob = frob_norm
    c_spec_topk = float(np.sum(S[:5] ** 2)) if len(S) >= 5 else float(np.sum(S ** 2))
    a_spec = mode_alignment(S, k=5)
    d_eff = effective_rank(S)
    q_realized = c_spec_frob * a_spec

    result = {
        "c_spec": c_spec_frob,
        "c_spec_topk": c_spec_topk,
        "topk_sv": S[:5].tolist() if len(S) >= 5 else S.tolist(),
        "a_spec": a_spec,
        "d_eff": d_eff,
        "d_eff_norm": d_eff / h.shape[1] if h.shape[1] > 0 else 0.0,
        "q_realized": q_realized,
        "sigma1": float(S[0]) if len(S) > 0 else 0.0,
        "sigma2": float(S[1]) if len(S) > 1 else 0.0,
    }

    if ctrl_hidden_state is not None:
        h_ctrl = np.nan_to_num(ctrl_hidden_state, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float64)
        try:
            angles = subspace_alignment(h, h_ctrl, k=5)
            result["subspace_angles_deg"] = [float(np.degrees(a)) for a in angles]
            result["subspace_overlap"] = float(np.mean(np.cos(angles)))
        except Exception:
            result["subspace_angles_deg"] = []
            result["subspace_overlap"] = 0.0

    return result


def run_dose(model, tokenizer, dose_name, device, ctrl_hidden=None, ctrl_token_counts=None):
    """Run all probes at a given CCS dose level.

    ctrl_hidden: optional dict {probe_idx: [layer_hidden_states]} from D0 run
    ctrl_token_counts: optional dict {probe_idx: int} — token count from D0 for matched subsampling
    """
    prefix = CCS_TEMPLATES[dose_name]
    import torch

    all_layers_full = []
    all_layers_matched = []
    raw_hidden = {}
    token_counts = {}

    for pi, probe in enumerate(PROBES):
        text = prefix + probe if prefix else probe
        inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)

        n_layers = len(out.hidden_states) - 1
        full_results = []
        matched_results = []
        probe_hidden = []
        token_counts[pi] = out.hidden_states[0].shape[1]

        for li in range(n_layers):
            h = out.hidden_states[li + 1][0].float().cpu().numpy()
            probe_hidden.append(h)
            ctrl_h = ctrl_hidden[pi][li] if ctrl_hidden and pi in ctrl_hidden else None

            full_results.append(analyze_layer(h, ctrl_h))

            if ctrl_token_counts and pi in ctrl_token_counts:
                n_match = ctrl_token_counts[pi]
                h_matched = h[-n_match:]
                ctrl_h_m = ctrl_h[-n_match:] if ctrl_h is not None else None
                matched_results.append(analyze_layer(h_matched, ctrl_h_m))
            else:
                matched_results.append(analyze_layer(h, ctrl_h))

        all_layers_full.append(full_results)
        all_layers_matched.append(matched_results)
        raw_hidden[pi] = probe_hidden

    n_layers = len(all_layers_full[0])
    keys = ["c_spec", "c_spec_topk", "a_spec", "d_eff", "d_eff_norm", "q_realized", "sigma1", "sigma2"]
    if ctrl_hidden:
        keys.extend(["subspace_overlap"])

    def average_layers(all_layers):
        averaged = []
        for li in range(n_layers):
            layer_avg = {}
            for key in keys:
                vals = [all_layers[pi][li].get(key, 0.0) for pi in range(len(PROBES))]
                layer_avg[key] = float(np.mean(vals))
                layer_avg[f"{key}_std"] = float(np.std(vals))
            averaged.append(layer_avg)
        return averaged

    return {
        "full": average_layers(all_layers_full),
        "matched": average_layers(all_layers_matched),
    }, raw_hidden, token_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="microsoft/phi-2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--doses", default="D0,D1,D2,D3,D5,D10")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.float32
    ).to(args.device)
    model.eval()

    doses = args.doses.split(",")
    results = {}
    ctrl_hidden = None
    ctrl_token_counts = None

    for dose in doses:
        if dose not in CCS_TEMPLATES:
            print(f"  Skipping unknown dose: {dose}")
            continue
        print(f"\n{'='*60}")
        print(f"Running dose {dose}...")
        t0 = time.time()
        dose_data, raw_hidden, token_counts = run_dose(
            model, tokenizer, dose, args.device, ctrl_hidden, ctrl_token_counts
        )
        results[dose] = dose_data
        if dose == "D0":
            ctrl_hidden = raw_hidden
            ctrl_token_counts = token_counts
        print(f"  Done in {time.time()-t0:.0f}s ({len(dose_data['full'])} layers, {list(token_counts.values())} tokens)")

    # Kimi's dual-window control: neutral prefix at same length as D2
    # Separates "being at position 70" from "having CCS context at position 70"
    if "D2" in results and ctrl_token_counts:
        print(f"\n{'='*60}")
        print(f"Running NEUTRAL CONTROL (position-matched to D2)...")
        d2_prefix = CCS_TEMPLATES["D2"]
        d2_tok_len = len(tokenizer.encode(d2_prefix))
        neutral_repeated = NEUTRAL_PREFIX_TEMPLATE * max(1, d2_tok_len // len(tokenizer.encode(NEUTRAL_PREFIX_TEMPLATE)) + 1)
        neutral_tokens = tokenizer.encode(neutral_repeated)[:d2_tok_len]
        neutral_prefix = tokenizer.decode(neutral_tokens)
        CCS_TEMPLATES["NEUTRAL"] = neutral_prefix
        t0 = time.time()
        neutral_data, _, neutral_tc = run_dose(
            model, tokenizer, "NEUTRAL", args.device, ctrl_hidden, ctrl_token_counts
        )
        results["NEUTRAL"] = neutral_data
        print(f"  Done in {time.time()-t0:.0f}s ({list(neutral_tc.values())} tokens, D2 had {list(token_counts.values())})")

    n_layers = len(results[doses[0]]["full"])

    for mode in ["full", "matched"]:
        print(f"\n{'='*60}")
        label = "FULL SEQUENCE" if mode == "full" else "MATCHED-TOKEN (last N tokens from D0)"
        print(f"CAPACITY/ALIGNMENT DECOMPOSITION — {args.model} [{label}]")
        print(f"{'='*60}")

        print(f"\n{'Dose':>6s} | {'C_frob':>10s} | {'C_topk':>10s} | {'A_spec':>10s} | {'Q':>10s} | {'D_eff':>10s} | {'sigma2':>10s}")
        print("-" * 80)

        dose_summaries = {}
        for dose in doses:
            if dose not in results:
                continue
            layers = results[dose][mode]
            c_vals = [l["c_spec"] for l in layers]
            ct_vals = [l["c_spec_topk"] for l in layers]
            a_vals = [l["a_spec"] for l in layers]
            q_vals = [l["q_realized"] for l in layers]
            d_vals = [l["d_eff"] for l in layers]
            s2_vals = [l["sigma2"] for l in layers]

            dose_summaries[dose] = {
                "c_spec_mean": float(np.mean(c_vals)),
                "c_topk_mean": float(np.mean(ct_vals)),
                "a_spec_mean": float(np.mean(a_vals)),
                "q_mean": float(np.mean(q_vals)),
                "d_eff_mean": float(np.mean(d_vals)),
                "sigma2_mean": float(np.mean(s2_vals)),
            }

            print(f"{dose:>6s} | {np.mean(c_vals):10.1f} | {np.mean(ct_vals):10.1f} | {np.mean(a_vals):10.4f} | {np.mean(q_vals):10.1f} | {np.mean(d_vals):10.2f} | {np.mean(s2_vals):10.2f}")

        # Compute relative changes from D0
        if "D0" in dose_summaries:
            d0 = dose_summaries["D0"]
            print(f"\nRelative to D0:")
            print(f"{'Dose':>6s} | {'dC_frob%':>10s} | {'dC_topk%':>10s} | {'dA_spec%':>10s} | {'dQ%':>10s} | {'dD_eff%':>10s} | {'dsigma2%':>10s}")
            print("-" * 80)
            for dose in doses:
                if dose not in dose_summaries or dose == "D0":
                    continue
                d = dose_summaries[dose]
                dc = (d["c_spec_mean"] - d0["c_spec_mean"]) / max(d0["c_spec_mean"], 1e-10) * 100
                dct = (d["c_topk_mean"] - d0["c_topk_mean"]) / max(d0["c_topk_mean"], 1e-10) * 100
                da = (d["a_spec_mean"] - d0["a_spec_mean"]) / max(d0["a_spec_mean"], 1e-10) * 100
                dq = (d["q_mean"] - d0["q_mean"]) / max(d0["q_mean"], 1e-10) * 100
                dd = (d["d_eff_mean"] - d0["d_eff_mean"]) / max(d0["d_eff_mean"], 1e-10) * 100
                ds = (d["sigma2_mean"] - d0["sigma2_mean"]) / max(d0["sigma2_mean"], 1e-10) * 100
                print(f"{dose:>6s} | {dc:+10.2f}% | {dct:+10.2f}% | {da:+10.2f}% | {dq:+10.2f}% | {dd:+10.2f}% | {ds:+10.2f}%")

            print(f"\n  Kimi's discriminator (Hankel spectral stability):")
            print(f"    C_topk range: {min(d['c_topk_mean'] for d in dose_summaries.values()):.1f} — {max(d['c_topk_mean'] for d in dose_summaries.values()):.1f}")
            print(f"    A_spec range: {min(d['a_spec_mean'] for d in dose_summaries.values()):.4f} — {max(d['a_spec_mean'] for d in dose_summaries.values()):.4f}")
            c_range = (max(d['c_topk_mean'] for d in dose_summaries.values()) - min(d['c_topk_mean'] for d in dose_summaries.values())) / max(d0["c_topk_mean"], 1e-10)
            a_range = (max(d['a_spec_mean'] for d in dose_summaries.values()) - min(d['a_spec_mean'] for d in dose_summaries.values())) / max(d0["a_spec_mean"], 1e-10)
            print(f"    C_topk variation: {c_range:.4f}")
            print(f"    A_spec variation: {a_range:.4f}")
            if a_range > 2 * c_range:
                print(f"    → ALIGNMENT DOMINATES (A_spec moves {a_range/max(c_range,1e-10):.1f}x more than C_topk)")
            elif c_range > 2 * a_range:
                print(f"    → CAPACITY DOMINATES (C_topk moves {c_range/max(a_range,1e-10):.1f}x more than A_spec)")
            else:
                print(f"    → MIXED (both move comparably)")

    # Per-layer profiles for detailed analysis
    output = {
        "model": args.model,
        "doses": doses,
        "per_layer": {dose: results[dose] for dose in doses if dose in results},
        "hypothesis": "Refined: species differ in which Hankel quantity CCS modulates (Kimi correction)",
        "framework": "Kanai GMW (2608.15926)",
        "notes": "matched = last-N-token subsampling to control length confound (Kimi correction)",
    }

    outpath = args.output or f"f615_capacity_alignment_{args.model.split('/')[-1]}.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
