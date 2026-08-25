#!/usr/bin/env python3
"""
Species-typing sweep across 20+ models on A100.

Runs spectral_species.py on each model, collects results.
Models selected to cover: different GQA ratios, different sizes,
different families, base vs instruct variants.
"""

import subprocess
import json
import time
import sys
import os
import gc
import torch

MODELS = [
    # Potter candidates (GQA 7:1 or high ratio)
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-7B",  # base variant

    # Goldsmith candidates (GQA 4:1 or Llama family)
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-8B",  # base
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mistral-7B-v0.3",  # base

    # Equalizer candidates (GQA 2:1 or Gemma family)
    "google/gemma-2-9b-it",
    "google/gemma-2-2b-it",

    # Other architectures (species unknown — discovery)
    "microsoft/Phi-3.5-mini-instruct",
    "microsoft/Phi-3-medium-4k-instruct",
    "01-ai/Yi-1.5-9B-Chat",
    "internlm/internlm2_5-7b-chat",
    "deepseek-ai/DeepSeek-V2-Lite-Chat",
    "THUDM/glm-4-9b-chat",
    "Qwen/Qwen3-8B",

    # MHA models (GQA 1:1 — potential fourth species?)
    "tiiuae/falcon-7b-instruct",
    "EleutherAI/pythia-6.9b",
    "bigscience/bloom-7b1",
]


def run_species_typing(model_name, script_path, output_dir):
    """Run species typing on a single model, return results."""
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    t0 = time.time()

    try:
        safe_name = model_name.replace("/", "_").replace("-", "_")
        out_file = os.path.join(output_dir, f"{safe_name}.json")
        result = subprocess.run(
            ["python3", script_path, "--model", model_name, "--output", out_file],
            capture_output=True, text=True, timeout=600,
            env={**os.environ, "OMP_NUM_THREADS": "16", "PYTHONUNBUFFERED": "1"}
        )

        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.1f}s")

        if result.returncode != 0:
            print(f"  ERROR: {result.stderr[-500:]}")
            return {"model": model_name, "error": result.stderr[-500:], "time": elapsed}

        # Parse output for species classification
        stdout = result.stdout
        print(stdout[-500:])

        # Load the result file
        if os.path.exists(out_file):
            with open(out_file) as f:
                data = json.load(f)
            return {**data, "time": elapsed}

        return {"model": model_name, "stdout": stdout[-1000:], "time": elapsed}

    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after 600s")
        return {"model": model_name, "error": "timeout", "time": 600}
    except Exception as e:
        print(f"  EXCEPTION: {e}")
        return {"model": model_name, "error": str(e), "time": time.time() - t0}
    finally:
        gc.collect()
        torch.cuda.empty_cache()


def main():
    script_path = "/workspace/spectral_species.py"
    output_dir = "/workspace/species_results"
    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    total_start = time.time()

    for i, model_name in enumerate(MODELS):
        print(f"\n[{i+1}/{len(MODELS)}] Processing {model_name}")
        elapsed_total = time.time() - total_start
        print(f"  Total elapsed: {elapsed_total/60:.1f} min")

        result = run_species_typing(model_name, script_path, output_dir)
        all_results.append(result)

        # Save running results
        with open("/workspace/species_sweep_results.json", "w") as f:
            json.dump({
                "timestamp": time.strftime("%Y%m%d_%H%M%S"),
                "total_elapsed_min": (time.time() - total_start) / 60,
                "completed": i + 1,
                "total": len(MODELS),
                "results": all_results,
            }, f, indent=2)

    # Summary table
    print("\n" + "=" * 80)
    print("SPECIES CLASSIFICATION SUMMARY")
    print("=" * 80)
    print(f"{'Model':<45} {'Species':<15} {'M2 slope':>10} {'M3 relay':>10} {'Time':>8}")
    print("-" * 80)

    for r in all_results:
        model = r.get("model", "?")
        species = r.get("species", r.get("error", "?"))
        m2 = r.get("m2_depth_slope", "?")
        m3 = r.get("m3_relay_sign", "?")
        t = r.get("time", 0)
        if isinstance(m2, float):
            m2 = f"{m2:.3f}"
        if isinstance(m3, float):
            m3 = f"{m3:.3f}"
        print(f"{model:<45} {str(species):<15} {str(m2):>10} {str(m3):>10} {t:>7.0f}s")

    total = time.time() - total_start
    print(f"\nTotal time: {total/60:.1f} minutes")
    print(f"Results saved to /workspace/species_sweep_results.json")


if __name__ == "__main__":
    main()
