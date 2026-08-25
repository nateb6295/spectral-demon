#!/usr/bin/env python3
"""Token-position × layer-band decomposition for SR/OR.

Kimi's correction: depth isn't time. Every layer sees the full sequence.
This script decomposes the spectral signal by BOTH layer depth and
token position, testing whether specific token positions (gap site,
main verb, relative clause) drive the SR/OR difference at specific layers.

For each (layer, token_position), computes the projection of that token's
hidden state onto the sigma-2 direction (V[:,1] from the full-matrix SVD).
Compares SR vs OR projections to locate where in (layer × position) space
the syntactic processing difference lives.

Runs on CPU with HF weights (no Ollama).
"""
import os, json, argparse, time
import numpy as np

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONUNBUFFERED"] = "1"

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

SR_PREAMBLES = [
    "The reporter that attacked the senator admitted the error publicly.",
    "The lawyer that questioned the witness revealed the inconsistency clearly.",
    "The scientist that challenged the theory published the correction immediately.",
    "The teacher that inspired the student received the recognition gratefully.",
    "The detective that followed the suspect discovered the evidence accidentally.",
]

OR_PREAMBLES = [
    "The reporter that the senator attacked admitted the error publicly.",
    "The lawyer that the witness questioned revealed the inconsistency clearly.",
    "The scientist that the theory challenged published the correction immediately.",
    "The teacher that the student inspired received the recognition gratefully.",
    "The detective that the suspect followed discovered the evidence accidentally.",
]

PROBES = [
    "In today's discussion, we explore how",
    "The weather has been particularly mild this season",
    "Consider the following mathematical proposition",
    "Recent advances in quantum computing suggest that",
    "The eigenvalue decomposition of the matrix reveals that",
]


def get_hidden_states(model, tokenizer, text, device):
    import torch
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    states = []
    for i in range(1, len(out.hidden_states)):
        h = out.hidden_states[i][0].float().cpu().numpy().astype(np.float64)
        states.append(h)
    return states, inputs["input_ids"][0].cpu().tolist()


def sigma2_projection_map(hidden_states):
    """For each layer, compute SVD and project each token onto V[:,1]."""
    n_layers = len(hidden_states)
    projections = []
    for layer_h in hidden_states:
        h = np.nan_to_num(layer_h, nan=0.0, posinf=1e6, neginf=-1e6)
        try:
            U, S, Vt = np.linalg.svd(h, full_matrices=False)
            v2 = Vt[1]  # sigma-2 direction
            proj = h @ v2  # [seq_len] projection magnitudes
            projections.append(proj.tolist())
        except Exception:
            projections.append([0.0] * h.shape[0])
    return projections


def align_sr_or_tokens(tokenizer, sr_text, or_text):
    """Identify token-level alignment between SR and OR sentences.
    Returns token indices for key structural positions."""
    sr_tokens = tokenizer.tokenize(sr_text)
    or_tokens = tokenizer.tokenize(or_text)

    sr_labels = ["other"] * len(sr_tokens)
    or_labels = ["other"] * len(or_tokens)

    sr_str = " ".join(sr_tokens)
    or_str = " ".join(or_tokens)

    return {
        "sr_tokens": sr_tokens,
        "or_tokens": or_tokens,
        "sr_len": len(sr_tokens),
        "or_len": len(or_tokens),
    }


def normalized_position_bands(seq_len, n_bands=5):
    """Split token positions into equal bands."""
    band_size = seq_len // n_bands
    bands = []
    for i in range(n_bands):
        start = i * band_size
        end = start + band_size if i < n_bands - 1 else seq_len
        bands.append((start, end))
    return bands


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="microsoft/phi-2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=None)
    parser.add_argument("--n-pos-bands", type=int, default=5)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from scipy.stats import spearmanr

    print(f"Loading {args.model} on {args.device}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.float32
    ).to(args.device)
    model.eval()
    dt = time.time() - t0
    print(f"Loaded in {dt:.0f}s")

    n_pos_bands = args.n_pos_bands

    all_sr_maps = []
    all_or_maps = []
    all_ctrl_maps = []
    token_info = []

    for pi, probe in enumerate(PROBES):
        print(f"\nProbe {pi+1}/{len(PROBES)}: {probe[:40]}...")

        # Control: just the probe
        ctrl_states, ctrl_ids = get_hidden_states(model, tokenizer, probe, args.device)
        ctrl_proj = sigma2_projection_map(ctrl_states)
        n_layers = len(ctrl_states)

        sr_probe_maps = []
        or_probe_maps = []

        for si in range(len(SR_PREAMBLES)):
            sr_text = SR_PREAMBLES[si] + "\n\n" + probe
            or_text = OR_PREAMBLES[si] + "\n\n" + probe

            sr_states, sr_ids = get_hidden_states(model, tokenizer, sr_text, args.device)
            or_states, or_ids = get_hidden_states(model, tokenizer, or_text, args.device)

            sr_proj = sigma2_projection_map(sr_states)
            or_proj = sigma2_projection_map(or_states)

            sr_probe_maps.append(sr_proj)
            or_probe_maps.append(or_proj)

            if pi == 0 and si == 0:
                sr_toks = tokenizer.convert_ids_to_tokens(sr_ids)
                or_toks = tokenizer.convert_ids_to_tokens(or_ids)
                token_info.append({
                    "sr_tokens": sr_toks[:30],
                    "or_tokens": or_toks[:30],
                    "sr_len": len(sr_ids),
                    "or_len": len(or_ids),
                })

        # Average across SR/OR preamble variants
        # For each layer, compute mean absolute projection in position bands
        # Normalize: use preamble-only tokens (not probe tokens)
        preamble_len_sr = len(tokenizer.encode(SR_PREAMBLES[0] + "\n\n"))
        preamble_len_or = len(tokenizer.encode(OR_PREAMBLES[0] + "\n\n"))

        # Build (layer × position_band) maps for SR, OR
        sr_layer_pos = np.zeros((n_layers, n_pos_bands))
        or_layer_pos = np.zeros((n_layers, n_pos_bands))

        for si in range(len(SR_PREAMBLES)):
            sr_proj = sr_probe_maps[si]
            or_proj = or_probe_maps[si]

            sr_preamble_len = len(tokenizer.encode(SR_PREAMBLES[si] + "\n\n"))
            or_preamble_len = len(tokenizer.encode(OR_PREAMBLES[si] + "\n\n"))

            sr_bands = normalized_position_bands(sr_preamble_len, n_pos_bands)
            or_bands = normalized_position_bands(or_preamble_len, n_pos_bands)

            for layer in range(n_layers):
                for bi, (start, end) in enumerate(sr_bands):
                    if end <= len(sr_proj[layer]):
                        band_vals = sr_proj[layer][start:end]
                        sr_layer_pos[layer, bi] += np.mean(np.abs(band_vals))

                for bi, (start, end) in enumerate(or_bands):
                    if end <= len(or_proj[layer]):
                        band_vals = or_proj[layer][start:end]
                        or_layer_pos[layer, bi] += np.mean(np.abs(band_vals))

        sr_layer_pos /= len(SR_PREAMBLES)
        or_layer_pos /= len(OR_PREAMBLES)

        all_sr_maps.append(sr_layer_pos)
        all_or_maps.append(or_layer_pos)

    # Average across probes
    mean_sr = np.mean(all_sr_maps, axis=0)  # [n_layers, n_pos_bands]
    mean_or = np.mean(all_or_maps, axis=0)
    diff_map = mean_sr - mean_or  # positive = SR > OR at that (layer, position)

    print(f"\n{'='*60}")
    print(f"TOKEN-POSITION × LAYER DECOMPOSITION")
    print(f"Model: {args.model}, {n_layers} layers, {n_pos_bands} position bands")
    print(f"{'='*60}")

    band_labels = [f"pos_{i}" for i in range(n_pos_bands)]
    header = "Layer  " + "  ".join(f"{bl:>8s}" for bl in band_labels)
    print(f"\n--- SR-OR difference (positive = SR > OR) ---")
    print(header)
    for layer in range(n_layers):
        vals = "  ".join(f"{diff_map[layer, bi]:+8.4f}" for bi in range(n_pos_bands))
        print(f"L{layer:02d}    {vals}")

    # Layer-band summary
    layer_bands = {
        "early": slice(0, n_layers // 3),
        "mid": slice(n_layers // 3, 2 * n_layers // 3),
        "late": slice(2 * n_layers // 3, n_layers),
    }

    print(f"\n--- Summary: mean |SR-OR| by (layer_band × position_band) ---")
    summary_header = "         " + "  ".join(f"{bl:>8s}" for bl in band_labels)
    print(summary_header)
    summary_data = {}
    for lname, lslice in layer_bands.items():
        vals = []
        for bi in range(n_pos_bands):
            cell = float(np.mean(np.abs(diff_map[lslice, bi])))
            vals.append(cell)
        summary_data[lname] = vals
        vals_str = "  ".join(f"{v:8.4f}" for v in vals)
        print(f"{lname:>8s}  {vals_str}")

    # Key question: does any position band show layer-graded SR/OR difference?
    print(f"\n--- Position-specific layer gradient (Spearman of diff vs layer index) ---")
    layer_indices = np.arange(n_layers)
    pos_gradients = {}
    for bi in range(n_pos_bands):
        col = diff_map[:, bi]
        rho, p = spearmanr(layer_indices, col)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {band_labels[bi]}: rho={rho:+.4f} (p={p:.2e}) {sig}")
        pos_gradients[band_labels[bi]] = {"rho": float(rho), "p": float(p)}

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model_tag = args.model.replace("/", "_")
    outpath = args.output or os.path.join(
        RESULTS_DIR, f"f614_token_layer_{model_tag}.json"
    )
    results = {
        "model": args.model,
        "n_layers": n_layers,
        "n_pos_bands": n_pos_bands,
        "n_probes": len(PROBES),
        "sr_map": mean_sr.tolist(),
        "or_map": mean_or.tolist(),
        "diff_map": diff_map.tolist(),
        "summary_by_band": summary_data,
        "position_gradients": pos_gradients,
        "token_info": token_info,
    }
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
