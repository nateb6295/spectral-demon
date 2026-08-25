#!/usr/bin/env python3
"""CCS spectroscopy CONTROL — noise floor for the sign-flip finding.

Compares raw-A vs raw-B (two sets of journal entries) to establish
baseline spectral variation. If the sign-flip pattern from the main
experiment also appears in raw-vs-raw, it's text heterogeneity, not
a demon signature.

Also adds a second control: old bridge snapshots vs new bridge snapshots
(both compressed, different time periods).
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

MAX_TOKENS = 512


def load_journal_split():
    """Split journal entries into two groups: odd and even indices."""
    raw = JOURNAL.read_text()
    entries = re.split(r"\n---\n", raw)
    entries = [e.strip() for e in entries if len(e.strip()) > 200]

    group_a, group_b = [], []
    for i, entry in enumerate(entries[:10]):
        header = entry.split("\n")[0][:60] if entry else f"entry_{i}"
        item = {"source": header, "text": entry}
        if i % 2 == 0:
            item["class"] = "raw_A"
            group_a.append(item)
        else:
            item["class"] = "raw_B"
            group_b.append(item)
    return group_a[:5], group_b[:5]


def load_bridge_split():
    """Split bridge snapshots: recent 5 vs older 5."""
    files = sorted(BRIDGE_DIR.glob("brain_*.txt"))
    old = files[-10:-5]
    new = files[-5:]

    old_texts = [{"source": f.name, "text": f.read_text(), "class": "bridge_old"} for f in old]
    new_texts = [{"source": f.name, "text": f.read_text(), "class": "bridge_new"} for f in new]
    return old_texts, new_texts


def extract_hidden_states(model, tokenizer, text):
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=MAX_TOKENS
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    return outputs.hidden_states, inputs["input_ids"].shape[1]


def compute_spectral_profile(hidden_states):
    layers = []
    for layer_idx, h in enumerate(hidden_states):
        H = h[0].float().cpu()
        H = H - H.mean(dim=0, keepdim=True)
        svs = torch.linalg.svdvals(H)
        svs_pos = svs[svs > 1e-10]

        sigma1 = svs_pos[0].item() if len(svs_pos) > 0 else 0
        sigma2 = svs_pos[1].item() if len(svs_pos) > 1 else 0

        layers.append({
            "layer": layer_idx,
            "sigma1": sigma1,
            "sigma2": sigma2,
            "sigma_ratio": sigma2 / sigma1 if sigma1 > 0 else 0,
        })
    return layers


def print_comparison(label, items_a, items_b, class_a, class_b):
    a = [r for r in items_a if r["class"] == class_a]
    b = [r for r in items_b if r["class"] == class_b]
    if not a or not b:
        return

    n_layers = len(a[0]["spectral_profile"])
    print(f"\n{'='*60}")
    print(f"CONTROL: {class_a} vs {class_b}")
    print(f"{'='*60}")

    print(f"\n{class_a} (n={len(a)}):")
    for li in [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]:
        ratios = [r["spectral_profile"][li]["sigma_ratio"] for r in a]
        print(f"  L{li:2d}: ratio={np.mean(ratios):.4f}±{np.std(ratios):.4f}")

    print(f"\n{class_b} (n={len(b)}):")
    for li in [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]:
        ratios = [r["spectral_profile"][li]["sigma_ratio"] for r in b]
        print(f"  L{li:2d}: ratio={np.mean(ratios):.4f}±{np.std(ratios):.4f}")

    print(f"\nDELTA ({class_a} - {class_b}) per layer:")
    for li in range(n_layers):
        a_ratio = np.mean([r["spectral_profile"][li]["sigma_ratio"] for r in a])
        b_ratio = np.mean([r["spectral_profile"][li]["sigma_ratio"] for r in b])
        a_s2 = np.mean([r["spectral_profile"][li]["sigma2"] for r in a])
        b_s2 = np.mean([r["spectral_profile"][li]["sigma2"] for r in b])
        delta = a_ratio - b_ratio
        bar = "+" * int(abs(delta) * 200) if delta > 0 else "-" * int(abs(delta) * 200)
        print(f"  L{li:2d}: Δratio={delta:+.4f} Δσ₂={a_s2 - b_s2:+.1f} {bar}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading text groups...")
    raw_a, raw_b = load_journal_split()
    bridge_old, bridge_new = load_bridge_split()
    all_texts = raw_a + raw_b + bridge_old + bridge_new

    print(f"  raw_A: {len(raw_a)}, raw_B: {len(raw_b)}")
    print(f"  bridge_old: {len(bridge_old)}, bridge_new: {len(bridge_new)}")

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

    results = []
    for i, item in enumerate(all_texts):
        print(f"\n[{i+1}/{len(all_texts)}] {item['class']}: {item['source'][:50]}")
        text = item["text"][:2000]
        hidden_states, n_tokens = extract_hidden_states(model, tokenizer, text)
        spectral = compute_spectral_profile(hidden_states)

        results.append({
            "source": item["source"],
            "class": item["class"],
            "n_tokens": n_tokens,
            "spectral_profile": spectral,
        })

        mid = len(spectral) // 2
        print(f"  {n_tokens} tok  L0={spectral[0]['sigma_ratio']:.4f}  L{mid}={spectral[mid]['sigma_ratio']:.4f}  L{len(spectral)-1}={spectral[-1]['sigma_ratio']:.4f}")

    # Control 1: raw_A vs raw_B
    print_comparison("raw journal noise floor", results, results, "raw_A", "raw_B")

    # Control 2: bridge_old vs bridge_new
    print_comparison("bridge temporal stability", results, results, "bridge_old", "bridge_new")

    # Reference: bridge_new vs raw_A (should resemble the main experiment's signal)
    print_comparison("compressed vs raw reference", results, results, "bridge_new", "raw_A")

    out_path = RESULTS_DIR / f"ccs_spectroscopy_control_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
