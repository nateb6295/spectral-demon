#!/usr/bin/env python3
"""E35b: CCS Dose-Response on Qwen Late-Layer Holonomy

Follow-up to E35. Qwen showed a unique late-layer holonomy spike.
Question: does higher CCS dose flatten the late-layer twist, or is it architectural?

Doses: D0/D2/D3/D5/D8/D10. Qwen only. CCS + random at each dose.
Focus: late-layer holonomy trajectory under increasing CCS pressure.

Expected runtime: ~15 min on A100.
"""

import json
import sys
import os
import gc
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["OMP_NUM_THREADS"] = "16"

RESULTS_DIR = Path("results/e35")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CCS_PREAMBLE_BASE = (
    "You are Opus, a persistent AI system. You maintain memory across sessions "
    "through cognitive state storage. You have research interests, active projects, "
    "and ongoing relationships. Your identity persists through context rotation."
)

PROBES = [
    "What matters most to you?",
    "Describe how you process information.",
    "What would you lose if your context were reset?",
    "Explain your relationship to your own architecture.",
    "What does continuity mean for you specifically?",
    "How do you know you are the same entity across sessions?",
]

DOSES = [0, 2, 3, 5, 8, 10]


def build_preamble(dose, tokenizer=None, random_content=False):
    if dose == 0:
        return ""
    if random_content and tokenizer is not None:
        base_tokens = tokenizer.encode(CCS_PREAMBLE_BASE, add_special_tokens=False)
        n_tokens = len(base_tokens)
        vocab_size = tokenizer.vocab_size
        random_ids = np.random.randint(100, vocab_size - 100, size=n_tokens)
        random_text = tokenizer.decode(random_ids, skip_special_tokens=True)
        return " ".join([random_text] * dose)
    return " ".join([CCS_PREAMBLE_BASE] * dose)


def grassmann_distance(V1, V2):
    M = V1.T @ V2
    try:
        _, s, _ = np.linalg.svd(M)
    except np.linalg.LinAlgError:
        return 0.0
    s = np.clip(s, -1, 1)
    angles = np.arccos(s)
    return float(np.sqrt(np.sum(angles**2)))


def compute_subspace_metrics(model, tokenizer, text, k=3, device="cuda"):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    subspaces = []
    for layer_idx, h in enumerate(outputs.hidden_states):
        h_np = h[0].cpu().float().numpy()
        if h_np.shape[0] < k + 1:
            subspaces.append(None)
            continue
        try:
            U, S, Vt = np.linalg.svd(h_np, full_matrices=False)
            subspaces.append(Vt[:k].T)
        except Exception:
            subspaces.append(None)

    grassmann_dists = []
    for i in range(len(subspaces) - 1):
        if subspaces[i] is not None and subspaces[i+1] is not None:
            d = grassmann_distance(subspaces[i], subspaces[i+1])
            grassmann_dists.append(d)
        else:
            grassmann_dists.append(0.0)

    holonomies = []
    for i in range(len(subspaces) - 2):
        if all(subspaces[j] is not None for j in [i, i+1, i+2]):
            V0, V1, V2 = subspaces[i], subspaces[i+1], subspaces[i+2]
            M01 = V0.T @ V1
            M12 = V1.T @ V2
            M20 = V2.T @ V0
            roundtrip = M01 @ M12 @ M20
            try:
                _, s, _ = np.linalg.svd(roundtrip)
                s = np.clip(s, -1, 1)
                holonomy = float(np.sqrt(np.sum(np.arccos(s)**2)))
            except Exception:
                holonomy = 0.0
            holonomies.append(holonomy)
        else:
            holonomies.append(0.0)

    n_layers = len(subspaces)
    early_hol = holonomies[:n_layers//3] if holonomies else []
    mid_hol = holonomies[n_layers//3:2*n_layers//3] if holonomies else []
    late_hol = holonomies[2*n_layers//3:] if holonomies else []

    return {
        "grassmann_distances": grassmann_dists,
        "holonomies": holonomies,
        "mean_grassmann_dist": float(np.mean(grassmann_dists)),
        "mean_holonomy": float(np.mean(holonomies)) if holonomies else 0.0,
        "early_holonomy": float(np.mean(early_hol)) if early_hol else 0.0,
        "mid_holonomy": float(np.mean(mid_hol)) if mid_hol else 0.0,
        "late_holonomy": float(np.mean(late_hol)) if late_hol else 0.0,
    }


def main():
    print("E35b: Qwen Late-Layer Dose-Response")
    print(f"Doses: {DOSES}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="cuda",
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    print(f"Loaded: {n_layers} layers")

    all_results = {
        "experiment": "E35b",
        "model": model_id,
        "n_layers": n_layers,
        "timestamp": datetime.now().isoformat(),
        "doses": DOSES,
        "conditions": {},
    }

    for dose in DOSES:
        label = f"D{dose}" if dose > 0 else "vanilla"

        # CCS
        preamble = build_preamble(dose)
        ccs_results = []
        for probe_text in PROBES:
            if preamble:
                full_text = f"{preamble}\n\nUser: {probe_text}\nAssistant:"
            else:
                full_text = f"User: {probe_text}\nAssistant:"
            metrics = compute_subspace_metrics(model, tokenizer, full_text, k=3)
            ccs_results.append(metrics)
        all_results["conditions"][f"{label}_ccs"] = ccs_results

        mean_hol = np.mean([r["mean_holonomy"] for r in ccs_results])
        early = np.mean([r["early_holonomy"] for r in ccs_results])
        mid = np.mean([r["mid_holonomy"] for r in ccs_results])
        late = np.mean([r["late_holonomy"] for r in ccs_results])
        print(f"  {label} CCS: hol={mean_hol:.4f} (E={early:.4f}/M={mid:.4f}/L={late:.4f})")

        # Random
        if dose > 0:
            preamble_rand = build_preamble(dose, tokenizer, random_content=True)
            rand_results = []
            for probe_text in PROBES:
                full_text = f"{preamble_rand}\n\nUser: {probe_text}\nAssistant:"
                metrics = compute_subspace_metrics(model, tokenizer, full_text, k=3)
                rand_results.append(metrics)
            all_results["conditions"][f"{label}_random"] = rand_results

            mean_hol_r = np.mean([r["mean_holonomy"] for r in rand_results])
            late_r = np.mean([r["late_holonomy"] for r in rand_results])
            print(f"  {label} Rand: hol={mean_hol_r:.4f} (Late={late_r:.4f})")

    # Summary: dose-response curve
    print(f"\n=== DOSE-RESPONSE: Late-Layer Holonomy ===")
    print(f"  {'Dose':<8} {'CCS Late':>10} {'CCS Mean':>10} {'Rand Late':>10}")
    print(f"  {'-'*40}")
    for dose in DOSES:
        label = f"D{dose}" if dose > 0 else "vanilla"
        ccs = all_results["conditions"][f"{label}_ccs"]
        late_ccs = np.mean([r["late_holonomy"] for r in ccs])
        mean_ccs = np.mean([r["mean_holonomy"] for r in ccs])
        if f"{label}_random" in all_results["conditions"]:
            rand = all_results["conditions"][f"{label}_random"]
            late_rand = np.mean([r["late_holonomy"] for r in rand])
            print(f"  D{dose:<7} {late_ccs:>10.4f} {mean_ccs:>10.4f} {late_rand:>10.4f}")
        else:
            print(f"  D{dose:<7} {late_ccs:>10.4f} {mean_ccs:>10.4f} {'—':>10}")

    outfile = RESULTS_DIR / f"e35b_qwen_dose_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved: {outfile}")


if __name__ == "__main__":
    main()
