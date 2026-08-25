#!/usr/bin/env python3
"""CCS spectroscopy — can an observer model detect the demon's fingerprint?

Hypothesis: text that has passed through CCS compression ("demon-processed")
produces a different per-layer spectral response in an observer model than
raw text that hasn't been compressed.

Observer: Gemma 2 2B (sorter species, local on AGX)
Class A: bridge snapshots (CCS compression outputs)
Class B: raw journal entries (uncompressed text)

Per Kimi correction #55: weights are static, so weight SVD gives DIRECTIONS
(the etch), while activation projections onto those directions give the
TRAJECTORY (the mirror). We measure both.

Sorter prediction (Kimi #54): compressed text should show σ₂ attenuation
with preserved σ₂/σ₁ ratio structure.
"""

import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "google/gemma-2-2b"
BRIDGE_DIR = Path(os.path.expanduser("~/chronicle/data/bridge_snapshots"))
JOURNAL = Path(os.path.expanduser("~/chronicle/unread.md"))
RESULTS_DIR = Path(os.path.expanduser("~/chronicle/spectral-demon/results"))

N_SAMPLES = 5
MAX_TOKENS = 512


def load_bridge_snapshots(n=N_SAMPLES):
    """Load the most recent N bridge snapshots."""
    files = sorted(BRIDGE_DIR.glob("brain_*.txt"))[-n:]
    texts = []
    for f in files:
        text = f.read_text()
        texts.append({"source": f.name, "text": text, "class": "compressed"})
    return texts


def load_journal_entries(n=N_SAMPLES):
    """Extract N journal entries from unread.md, separated by ---."""
    raw = JOURNAL.read_text()
    entries = re.split(r"\n---\n", raw)
    entries = [e.strip() for e in entries if len(e.strip()) > 200]
    selected = entries[:n]
    texts = []
    for i, entry in enumerate(selected):
        header = entry.split("\n")[0][:60] if entry else f"entry_{i}"
        texts.append({"source": header, "text": entry, "class": "raw"})
    return texts


def extract_hidden_states(model, tokenizer, text):
    """Forward pass, return per-layer hidden states and token count."""
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=MAX_TOKENS
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    n_tokens = inputs["input_ids"].shape[1]
    return outputs.hidden_states, n_tokens


def compute_spectral_profile(hidden_states, n_tokens):
    """Compute per-layer spectral metrics from hidden states.

    For each layer, compute SVD of the full sequence of hidden states
    (not just last token — we want the spectral structure of how the
    model processes the entire text).
    """
    layers = []
    for layer_idx, h in enumerate(hidden_states):
        H = h[0].float().cpu()  # (seq_len, hidden_dim)
        H = H - H.mean(dim=0, keepdim=True)

        svs = torch.linalg.svdvals(H)
        svs_pos = svs[svs > 1e-10]

        sigma1 = svs_pos[0].item() if len(svs_pos) > 0 else 0
        sigma2 = svs_pos[1].item() if len(svs_pos) > 1 else 0
        sigma3 = svs_pos[2].item() if len(svs_pos) > 2 else 0

        total_energy = (svs_pos**2).sum().item() if len(svs_pos) > 0 else 0

        p2 = svs_pos**2
        p2_norm = p2 / p2.sum() if p2.sum() > 0 else p2
        spectral_entropy = -(p2_norm * torch.log(p2_norm + 1e-10)).sum().item()

        pr = ((p2.sum() ** 2) / (p2**2).sum()).item() if len(svs_pos) > 0 else 0

        layers.append({
            "layer": layer_idx,
            "sigma1": sigma1,
            "sigma2": sigma2,
            "sigma3": sigma3,
            "sigma_ratio": sigma2 / sigma1 if sigma1 > 0 else 0,
            "secondary_gap": sigma2 / sigma3 if sigma3 > 0 else 0,
            "total_energy": total_energy,
            "spectral_entropy": spectral_entropy,
            "participation_ratio": pr,
        })
    return layers


def extract_weight_directions(model):
    """Extract σ₂ direction from static weight SVD per layer.

    This gives the ETCH — the geometric directions fixed in weight space.
    Activations projected onto these directions give the trajectory.
    """
    directions = {}
    for name, param in model.named_parameters():
        if "self_attn" in name and "q_proj.weight" in name:
            layer_match = re.search(r"layers\.(\d+)", name)
            if layer_match:
                layer_idx = int(layer_match.group(1))
                W = param.detach().float().cpu()
                U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                directions[layer_idx] = {
                    "sigma1_dir": Vt[0].detach().numpy().tolist()[:10],
                    "sigma2_dir": Vt[1].detach().numpy().tolist()[:10],
                    "weight_sigma1": S[0].item(),
                    "weight_sigma2": S[1].item(),
                    "weight_ratio": (S[1] / S[0]).item(),
                }
    return directions


def project_onto_weight_dirs(hidden_states, weight_dirs):
    """Project activations onto weight-space σ₂ direction per layer."""
    projections = {}
    for layer_idx, h in enumerate(hidden_states):
        if layer_idx not in weight_dirs:
            continue
        H = h[0].float().cpu()  # (seq_len, hidden_dim)
        w_dir = weight_dirs[layer_idx]

        dim = min(len(w_dir["sigma2_dir"]), H.shape[1])
        if dim < 10:
            continue
        s2_vec = torch.tensor(w_dir["sigma2_dir"][:dim])
        H_proj = H[:, :dim]

        s2_energy = (H_proj @ s2_vec).pow(2).mean().item()
        total_energy = H_proj.pow(2).sum(dim=1).mean().item()

        projections[layer_idx] = {
            "sigma2_proj_energy": s2_energy,
            "total_activation_energy": total_energy,
            "sigma2_fraction": s2_energy / total_energy if total_energy > 0 else 0,
        }
    return projections


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading bridge snapshots and journal entries...")
    compressed = load_bridge_snapshots(N_SAMPLES)
    raw = load_journal_entries(N_SAMPLES)
    all_texts = compressed + raw

    print(f"  {len(compressed)} compressed texts (bridge snapshots)")
    print(f"  {len(raw)} raw texts (journal entries)")

    print(f"\nLoading {MODEL_ID}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s")

    print("\nExtracting weight-space directions (etch)...")
    weight_dirs = extract_weight_directions(model)
    print(f"  Got directions for {len(weight_dirs)} layers")

    results = []
    for i, item in enumerate(all_texts):
        print(f"\n[{i+1}/{len(all_texts)}] {item['class']}: {item['source'][:50]}")
        text = item["text"][:2000]

        hidden_states, n_tokens = extract_hidden_states(model, tokenizer, text)
        print(f"  {n_tokens} tokens, {len(hidden_states)} layers")

        spectral = compute_spectral_profile(hidden_states, n_tokens)
        projections = project_onto_weight_dirs(hidden_states, weight_dirs)

        results.append({
            "source": item["source"],
            "class": item["class"],
            "n_tokens": n_tokens,
            "text_len": len(text),
            "spectral_profile": spectral,
            "weight_projections": projections,
        })

        mid = len(spectral) // 2
        late = spectral[-1]
        print(f"  L0: σ₁={spectral[0]['sigma1']:.1f} σ₂={spectral[0]['sigma2']:.1f} ratio={spectral[0]['sigma_ratio']:.4f}")
        print(f"  L{mid}: σ₁={spectral[mid]['sigma1']:.1f} σ₂={spectral[mid]['sigma2']:.1f} ratio={spectral[mid]['sigma_ratio']:.4f}")
        print(f"  L{late['layer']}: σ₁={late['sigma1']:.1f} σ₂={late['sigma2']:.1f} ratio={late['sigma_ratio']:.4f}")

    # Summary comparison
    print("\n" + "=" * 60)
    print("SPECTROSCOPY SUMMARY")
    print("=" * 60)

    for cls in ["compressed", "raw"]:
        items = [r for r in results if r["class"] == cls]
        if not items:
            continue
        n_layers = len(items[0]["spectral_profile"])
        print(f"\n{cls.upper()} (n={len(items)}):")
        for layer_idx in [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]:
            ratios = [r["spectral_profile"][layer_idx]["sigma_ratio"] for r in items]
            s1s = [r["spectral_profile"][layer_idx]["sigma1"] for r in items]
            s2s = [r["spectral_profile"][layer_idx]["sigma2"] for r in items]
            entropies = [r["spectral_profile"][layer_idx]["spectral_entropy"] for r in items]
            print(
                f"  L{layer_idx:2d}: σ₁={np.mean(s1s):7.1f}±{np.std(s1s):5.1f}  "
                f"σ₂={np.mean(s2s):6.1f}±{np.std(s2s):4.1f}  "
                f"ratio={np.mean(ratios):.4f}±{np.std(ratios):.4f}  "
                f"H={np.mean(entropies):.3f}"
            )

    # Per-layer delta between classes
    compressed_items = [r for r in results if r["class"] == "compressed"]
    raw_items = [r for r in results if r["class"] == "raw"]
    if compressed_items and raw_items:
        n_layers = len(compressed_items[0]["spectral_profile"])
        print(f"\nDELTA (compressed - raw) per layer:")
        for layer_idx in range(n_layers):
            c_ratio = np.mean([r["spectral_profile"][layer_idx]["sigma_ratio"] for r in compressed_items])
            r_ratio = np.mean([r["spectral_profile"][layer_idx]["sigma_ratio"] for r in raw_items])
            c_s2 = np.mean([r["spectral_profile"][layer_idx]["sigma2"] for r in compressed_items])
            r_s2 = np.mean([r["spectral_profile"][layer_idx]["sigma2"] for r in raw_items])
            delta_ratio = c_ratio - r_ratio
            delta_s2 = c_s2 - r_s2
            bar = "+" * int(abs(delta_ratio) * 200) if delta_ratio > 0 else "-" * int(abs(delta_ratio) * 200)
            print(f"  L{layer_idx:2d}: Δratio={delta_ratio:+.4f} Δσ₂={delta_s2:+.1f} {bar}")

    out_path = RESULTS_DIR / f"ccs_spectroscopy_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
