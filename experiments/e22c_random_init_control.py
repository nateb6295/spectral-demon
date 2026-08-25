#!/usr/bin/env python3
"""
E22c: Random-Init Control for MLP Orthogonality

Kimi's falsification test: if V₂ orthogonality to MLP is a geometric attractor,
it should appear even WITHOUT training. If it requires training, randomly
initialized models should show random V₂-MLP alignment (~0.03-0.05 for 4096-dim).

Result interpretation:
  - Random init shows orthogonality → geometric property of architecture
  - Random init shows random alignment → training produces orthogonality
  - Random init shows HIGH alignment → training actively separates V₂ from MLP
"""

import json, sys, os, time
import numpy as np
import torch
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
os.environ.setdefault("OMP_NUM_THREADS", "16")

RESULTS_DIR = Path(os.environ.get("E22_RESULTS_DIR",
    str(Path(__file__).parent.parent / "results" / "e22")))

sys.path.insert(0, str(Path(__file__).parent))
from e22_mlp_pathway_alignment import (
    CCS_PREAMBLE, PROMPTS, build_input, extract_hidden_states,
    get_component_svds, safe_svd, TOP_K
)


def run_random_init(model_key, model_name):
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

    print(f"\n{'='*60}")
    print(f"E22c: Random-Init {model_key}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load config but initialize with random weights
    print("Loading config (random init)...")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(
        config,
        torch_dtype=torch.float16,
        attn_implementation="eager",
    ).to("cuda:0")
    model.eval()

    n_layers = model.config.num_hidden_layers
    print(f"  {n_layers} layers (RANDOM WEIGHTS)")

    t0 = time.time()

    # Extract component SVDs from random weights
    print("  Extracting SVDs from random weights...")
    (lm_V, lm_S), mlp_svds, attn_svds = get_component_svds(model, TOP_K)

    # Run prompts with CCS preamble (activations will be random but structured by architecture)
    print("  Running prompts...")
    lm_profile, mlp_profile, attn_profile = [], [], []
    s2_profile = []

    all_results = []
    for pi, prompt in enumerate(PROMPTS[:5]):  # Only 5 prompts needed
        input_ids = build_input(tokenizer, CCS_PREAMBLE, prompt)
        input_ids = input_ids.to(model.device)
        states = extract_hidden_states(model, input_ids)

        for li, h in enumerate(states):
            h64 = h.astype(np.float64)
            try:
                U, S, Vt = np.linalg.svd(h64, full_matrices=False)
                if len(S) < 2:
                    continue
                v2 = Vt[1]

                mlp_idx = min(li, len(mlp_svds) - 1) if li > 0 else 0
                mlp_V, _ = mlp_svds[mlp_idx]
                if mlp_V is not None and v2.shape[0] == mlp_V.shape[0]:
                    cosines = [float(np.abs(np.dot(v2, mlp_V[:, j])))
                              for j in range(min(TOP_K, mlp_V.shape[1]))]
                    mlp_align = float(max(cosines))
                else:
                    mlp_align = 0.0

                if lm_V is not None:
                    cosines = [float(np.abs(np.dot(v2, lm_V[:, j])))
                              for j in range(min(TOP_K, lm_V.shape[1]))]
                    lm_align = float(max(cosines))
                else:
                    lm_align = 0.0

                attn_V, _ = attn_svds[mlp_idx]
                if attn_V is not None and v2.shape[0] == attn_V.shape[0]:
                    cosines = [float(np.abs(np.dot(v2, attn_V[:, j])))
                              for j in range(min(TOP_K, attn_V.shape[1]))]
                    attn_align = float(max(cosines))
                else:
                    attn_align = 0.0

                all_results.append({
                    "prompt": pi, "layer": li,
                    "mlp_align": mlp_align, "lm_align": lm_align,
                    "attn_align": attn_align, "sigma2": float(S[1]),
                })
            except:
                pass

        if (pi + 1) % 5 == 0:
            print(f"    {pi+1}/5 prompts done")

    elapsed = time.time() - t0

    # Compute summary statistics
    layers_data = {}
    for r in all_results:
        li = r["layer"]
        if li not in layers_data:
            layers_data[li] = {"mlp": [], "lm": [], "attn": []}
        layers_data[li]["mlp"].append(r["mlp_align"])
        layers_data[li]["lm"].append(r["lm_align"])
        layers_data[li]["attn"].append(r["attn_align"])

    summary = {
        "model": model_key,
        "n_layers": n_layers,
        "init": "random",
        "elapsed": elapsed,
        "mean_mlp_align": float(np.mean([r["mlp_align"] for r in all_results])),
        "mean_lm_align": float(np.mean([r["lm_align"] for r in all_results])),
        "mean_attn_align": float(np.mean([r["attn_align"] for r in all_results])),
        "per_layer": {
            str(li): {
                "mlp_mean": float(np.mean(d["mlp"])),
                "lm_mean": float(np.mean(d["lm"])),
                "attn_mean": float(np.mean(d["attn"])),
            }
            for li, d in sorted(layers_data.items())
        },
    }

    print(f"\n  RANDOM-INIT SUMMARY:")
    print(f"  Mean MLP alignment:  {summary['mean_mlp_align']:.4f}")
    print(f"  Mean lm_head align:  {summary['mean_lm_align']:.4f}")
    print(f"  Mean attn alignment: {summary['mean_attn_align']:.4f}")
    print(f"  (Compare to trained: MLP=0.000, lm_head~0.10, attn~0.03)")

    out_path = RESULTS_DIR / f"e22c_{model_key}_random.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {out_path}")
    print(f"  Time: {elapsed:.1f}s")

    del model
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


MODELS = {
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "yi9b": "01-ai/Yi-1.5-9B-Chat",
}

if __name__ == "__main__":
    targets = sys.argv[1:] if len(sys.argv) > 1 else ["mistral"]
    for key in targets:
        if key in MODELS:
            run_random_init(key, MODELS[key])
        elif key == "all":
            for k, v in MODELS.items():
                run_random_init(k, v)
