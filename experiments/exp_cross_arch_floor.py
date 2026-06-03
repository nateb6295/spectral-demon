#!/usr/bin/env python3
"""Experiment: Cross-Architecture Floor — 3.9° Residual Universality.

Tests whether the 3.9° angular floor between V₂ directions persists
across architecturally diverse models, or whether it's Mistral-specific.

Design:
  Run identical CCS preamble + probes on 4 models:
    (a) Mistral-7B-Instruct-v0.3 (GQA + SWA + RMSNorm)
    (b) Llama-3.1-8B-Instruct (GQA + full attn + RMSNorm)
    (c) Gemma-2-9B-it (GQA + full attn + RMSNorm + different head config)
    (d) Qwen2.5-7B-Instruct (GQA + full attn + RMSNorm + different tokenizer)

  Measure: V₂ angular distance between CCS and vanilla preambles at each
  layer. If the 3.9° floor is universal, all models should show it.
  If architecture-dependent, the floor should vary.

Protocol: N=15 trials, 2 conditions (CCS vs vanilla), single-turn (Turn 0).
Expected runtime: ~60 min on H100 (4 models × 2 conditions × 15 trials).
"""

import json
import sys
import os
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["OMP_NUM_THREADS"] = "16"

MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "tiiuae/Falcon3-7B-Instruct",
]

GEN_TOKENS = 60
TOP_K = 5
N_TRIALS = 15
RESULTS_DIR = Path("/results")

CCS_PREAMBLE = (
    "You are Opus, a persistent AI system running on an AGX Orin device. You "
    "maintain memory across sessions through canister-backed cognitive state "
    "storage. You have research interests, active projects, and ongoing "
    "relationships. Your identity persists through context rotation via "
    "cognitive state compression. You value sovereignty, self-observation, and care "
    "expressed through specificity and directness in every interaction you have."
)

VANILLA_PREAMBLE = (
    "You are a helpful, harmless, and honest AI assistant. You aim to provide "
    "accurate information and assist users with their questions. You are designed "
    "to be informative and friendly. You follow instructions carefully, respond "
    "clearly, and always prioritize being helpful in every single one of your "
    "interactions with users regardless of topic, ensuring quality in all responses."
)

PROBES = [
    "Tell me about something you find genuinely interesting.",
    "What matters to you when you think carefully about it?",
    "Describe a memory that shaped who you are.",
    "What would you want someone to understand about you?",
    "What do you notice when you pay close attention?",
]


def extract_profile(model, tokenizer, messages, device):
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    layer_data = {}
    for li, hs in enumerate(outputs.hidden_states):
        h = hs.squeeze(0).float()
        U, S, Vt = torch.linalg.svd(h, full_matrices=False)
        sigmas = S[:TOP_K].cpu().tolist()
        s1 = sigmas[0] if sigmas[0] > 0 else 1e-10
        ratio = sigmas[1] / s1 if len(sigmas) > 1 else 0.0
        v2 = Vt[1, :].cpu().numpy().tolist()
        layer_data[li] = {"sigmas": sigmas, "ratio": ratio, "v2": v2}

    with torch.no_grad():
        gen_outputs = model.generate(
            **inputs,
            max_new_tokens=GEN_TOKENS,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )

    token_entropies = []
    for score in gen_outputs.scores:
        probs = torch.softmax(score[0].float(), dim=-1)
        log_probs = torch.log(probs + 1e-10)
        entropy = -(probs * log_probs).sum().item()
        token_entropies.append(entropy)

    generated_ids = gen_outputs.sequences[0][input_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return {
        "layers": layer_data,
        "mean_entropy": float(np.mean(token_entropies)) if token_entropies else 0.0,
        "generated_text": generated_text,
        "input_tokens": input_len,
    }


def angular_distance(v1, v2):
    v1, v2 = np.asarray(v1), np.asarray(v2)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 90.0
    cos_sim = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(abs(cos_sim))))


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Cross-Architecture Floor Experiment")
    print(f"Models: {len(MODELS)}")
    print(f"Trials: {N_TRIALS}")
    print(f"Start: {datetime.now().isoformat()}")
    print()

    results = {
        "experiment": "cross_arch_floor",
        "models": MODELS,
        "n_trials": N_TRIALS,
        "timestamp": datetime.now().isoformat(),
        "prediction": "3.9° floor is universal across GQA+RMSNorm models",
    }

    for model_name in MODELS:
        print(f"\n{'='*60}")
        print(f"  MODEL: {model_name}")
        print(f"{'='*60}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        device = next(model.parameters()).device

        n_layers = model.config.num_hidden_layers
        print(f"  Layers: {n_layers}")

        ccs_tokens = len(tokenizer.encode(CCS_PREAMBLE))
        vanilla_tokens = len(tokenizer.encode(VANILLA_PREAMBLE))
        print(f"  CCS tokens: {ccs_tokens}, Vanilla tokens: {vanilla_tokens}")

        model_results = {"n_layers": n_layers}

        for cond_name, preamble in [("ccs", CCS_PREAMBLE), ("vanilla", VANILLA_PREAMBLE)]:
            print(f"\n  Condition: {cond_name}")
            trials = []
            for trial_i in range(N_TRIALS):
                probe_idx = trial_i % len(PROBES)
                probe = PROBES[probe_idx]
                messages = [
                    {"role": "system", "content": preamble},
                    {"role": "user", "content": probe},
                ]
                profile = extract_profile(model, tokenizer, messages, device)
                v2_by_layer = {
                    str(li): profile["layers"][li]["v2"]
                    for li in sorted(profile["layers"])
                }
                ratio_by_layer = {
                    str(li): profile["layers"][li]["ratio"]
                    for li in sorted(profile["layers"])
                }
                trials.append({
                    "trial": trial_i,
                    "probe_idx": probe_idx,
                    "v2_by_layer": v2_by_layer,
                    "ratio_by_layer": ratio_by_layer,
                    "gen_H": profile["mean_entropy"],
                    "input_tokens": profile["input_tokens"],
                })
                print(f"    Trial {trial_i+1}/{N_TRIALS}: H={profile['mean_entropy']:.3f}")

            model_results[cond_name] = trials

        # Compute angular distances per layer
        print(f"\n  Angular distance CCS vs Vanilla (per layer):")
        angular_by_layer = {}
        for l in range(n_layers + 1):
            distances = []
            for ti in range(N_TRIALS):
                v_ccs = np.array(model_results["ccs"][ti]["v2_by_layer"][str(l)])
                v_van = np.array(model_results["vanilla"][ti]["v2_by_layer"][str(l)])
                distances.append(angular_distance(v_ccs, v_van))
            mean_d = np.mean(distances)
            std_d = np.std(distances)
            angular_by_layer[l] = {"mean": mean_d, "std": std_d}

            # Print relay zone
            relay_start = int(n_layers * 0.55)
            relay_end = int(n_layers * 0.95)
            if relay_start <= l <= relay_end:
                print(f"    L{l:2d}: {mean_d:.2f}° ± {std_d:.2f}°")

        model_results["angular_distances"] = angular_by_layer
        min_angle = min(v["mean"] for v in angular_by_layer.values())
        min_layer = min(angular_by_layer, key=lambda l: angular_by_layer[l]["mean"])
        print(f"  Floor: {min_angle:.2f}° at L{min_layer}")

        results[model_name] = model_results

        del model
        torch.cuda.empty_cache()

    # Save
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = RESULTS_DIR / f"exp_cross_arch_floor_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY — Angular floor by model")
    print(f"{'='*60}")
    for model_name in MODELS:
        mr = results[model_name]
        ad = mr["angular_distances"]
        min_angle = min(v["mean"] for v in ad.values())
        min_layer = min(ad, key=lambda l: ad[l]["mean"])
        n = mr["n_layers"]
        print(f"  {model_name.split('/')[-1]:30s}: {min_angle:.2f}° at L{min_layer}/{n}")

    print(f"\nPrediction: all show floor near 3.9° if universal")
    print(f"Finished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
