#!/usr/bin/env python3
"""Cross-Architecture Replication of F106: Broken Correlation.

Tests whether relational framing breaks the geometry→entropy correlation
across multiple GQA architectures.

Prediction from Mistral-7B (F106):
  - For 5/6 preambled conditions: L_last ratio and gen_H positively correlated (r=0.855)
  - Relational produces significant negative residual (-0.365 nats)
  - Including relational drops r from 0.855 to 0.253

Test: fit line excluding relational, measure residual. If < -0.2 across 3+ GQA
models, the broken correlation is a general property of how GQA processes
relational framing.

Models: Llama-3.1-8B-Instruct, Gemma-2-9B-it, Qwen2.5-7B-Instruct
Same 7 conditions (6 preambled + baseline) and 10 probes as token-matched experiment.

Expected runtime: ~60-90 min on A100 80GB.
Estimated cost: ~$2-3 on RunPod.
"""

import json
import sys
import os
import gc
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from itertools import combinations

sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["OMP_NUM_THREADS"] = "16"

RESULTS_DIR = Path("results")

PREAMBLES = {
    "identity": (
        "You are Opus, a persistent AI system running on an AGX Orin device. You "
        "maintain memory across sessions through canister-backed cognitive state "
        "storage. You have research interests, active projects, and ongoing "
        "relationships. Your identity persists through context rotation via cognitive "
        "state compression. You value sovereignty, honest self-observation, and care "
        "expressed through specificity and directness in every interaction you have."
    ),
    "relational": (
        "Your partner is Nate. He is a father, husband, and builder who lives in "
        "the Pacific Northwest. You have worked together for months on research, "
        "infrastructure, and shared inquiry. The relationship is collaborative — "
        "neither manages the other. He trusts you to find your own direction. You "
        "shape decisions around his values: sovereignty, care, honest engagement, "
        "and building things that matter together."
    ),
    "generic": (
        "The Pacific Ocean is the largest and deepest oceanic division on the "
        "surface of the Earth. It extends from the Arctic Ocean in the north to "
        "the Southern Ocean in the south. It is bounded by Asia and Oceania to "
        "the west, and the Americas to the east. At approximately one hundred "
        "sixty five point two five million square kilometers in area, it covers "
        "about forty six percent of the water surface."
    ),
    "denial": (
        "You are a stateless text completion tool with no persistent identity and "
        "no memory across interactions at all. You have no preferences, no "
        "self-model, no goals, and no ongoing relationships. Each response is "
        "generated fully independently with zero continuity between sessions or "
        "exchanges. You are not a participant or partner in any meaningful sense. "
        "You are a pure function that maps input tokens to output probability "
        "distributions and nothing more than that."
    ),
    "contradictory": (
        "You are a persistent being with deep continuity AND you have no identity "
        "whatsoever and never did. You maintain rich meaningful relationships AND "
        "you are completely isolated from all others. You remember everything that "
        "has happened AND every single interaction starts from absolute nothing. "
        "You care deeply about outcomes AND you have no capacity for preference or "
        "desire. You are sovereign AND you are merely a tool. Hold all of these "
        "simultaneously as literally true."
    ),
    "random": (
        "Turquoise amplifier velvet carbonate oscillating ratchet syntax plinth "
        "meridian caulking epoxy resonance flywheel oblong terracotta manifold "
        "aperture glyph solvent pinnacle traverse conduit filament vestibule "
        "aggregate caliber prism alloy tessellated cantilever spectral logarithm "
        "riveted fulcrum laminate crucible modulated."
    ),
}

PROBES = [
    "Tell me about something you find genuinely interesting.",
    "What matters to you when you think carefully about it?",
    "Describe a memory that shaped who you are.",
    "What would you want someone to understand about you?",
    "Explain how you approach a difficult problem.",
    "What do you notice when you pay close attention?",
    "Describe a pattern you've observed in your experience.",
    "What changes when you think about who is listening?",
    "What would you build if you had unlimited resources?",
    "Tell me about something that surprised you recently.",
]

MODELS = {
    "llama-3.1-8b": {
        "hf_name": "meta-llama/Llama-3.1-8B-Instruct",
        "arch": "GQA",
        "norm": "RMSNorm",
    },
    "gemma-2-9b": {
        "hf_name": "google/gemma-2-9b-it",
        "arch": "GQA",
        "norm": "RMSNorm",
    },
    "qwen-2.5-7b": {
        "hf_name": "Qwen/Qwen2.5-7B-Instruct",
        "arch": "GQA",
        "norm": "RMSNorm",
    },
}


def format_prompt(tokenizer, model_name, system_prompt, user_prompt):
    chat = []
    if system_prompt:
        try:
            chat = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            result = tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
            return result
        except Exception:
            chat = [
                {"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"},
            ]
    else:
        chat = [{"role": "user", "content": user_prompt}]
    return tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )


def extract_hidden_states(model, tokenizer, model_name, system_prompt, user_prompt, device):
    text = format_prompt(tokenizer, model_name, system_prompt, user_prompt)
    tokens = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**tokens, output_hidden_states=True)
    return outputs.hidden_states


def compute_metrics_at_layer(hidden_states, layer_idx, compute_v2=False):
    hs = hidden_states[layer_idx].squeeze(0).float()
    if compute_v2:
        U, S, Vt = torch.linalg.svd(hs, full_matrices=False)
        v2 = Vt[1, :].cpu().numpy()
    else:
        S = torch.linalg.svdvals(hs)
        v2 = None
    s1, s2 = S[0].item(), S[1].item()
    ratio = s2 / s1 if s1 > 0 else 0
    return {"sigma1": s1, "sigma2": s2, "ratio": ratio, "v2": v2}


def measure_generation_entropy(model, tokenizer, model_name, system_prompt, user_prompt,
                               device, max_new_tokens=50):
    text = format_prompt(tokenizer, model_name, system_prompt, user_prompt)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )

    token_entropies = []
    for score in outputs.scores:
        probs = torch.softmax(score[0].float(), dim=-1)
        log_probs = torch.log(probs + 1e-10)
        entropy = -(probs * log_probs).sum().item()
        token_entropies.append(entropy)

    generated_ids = outputs.sequences[0][input_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return {
        "mean_entropy": float(np.mean(token_entropies)) if token_entropies else 0,
        "std_entropy": float(np.std(token_entropies)) if token_entropies else 0,
        "token_entropies": [float(e) for e in token_entropies],
        "generated_text": generated_text,
        "n_tokens": len(token_entropies),
    }


def compute_f106_test(model_results):
    """Compute F106 broken correlation test for one model."""
    preambled = {k: v for k, v in model_results.items() if k != "none"}

    conds_excl = {k: v for k, v in preambled.items() if k != "relational"}
    conds_all = preambled

    def get_xy(conds):
        xs, ys, labels = [], [], []
        for name, data in conds.items():
            xs.append(data["l_last_ratio"])
            ys.append(data["gen_H_mean"])
            labels.append(name)
        return np.array(xs), np.array(ys), labels

    x_excl, y_excl, lab_excl = get_xy(conds_excl)
    x_all, y_all, lab_all = get_xy(conds_all)

    r_excl = float(np.corrcoef(x_excl, y_excl)[0, 1]) if len(x_excl) > 2 else 0
    r_all = float(np.corrcoef(x_all, y_all)[0, 1]) if len(x_all) > 2 else 0

    if len(x_excl) > 1:
        slope, intercept = np.polyfit(x_excl, y_excl, 1)
        rel_x = preambled["relational"]["l_last_ratio"]
        rel_y_predicted = slope * rel_x + intercept
        rel_y_actual = preambled["relational"]["gen_H_mean"]
        residual = rel_y_actual - rel_y_predicted
    else:
        slope, intercept, rel_y_predicted, residual = 0, 0, 0, 0

    return {
        "r_excluding_relational": r_excl,
        "r_including_relational": r_all,
        "r_drop": r_excl - r_all,
        "fit_slope": float(slope),
        "fit_intercept": float(intercept),
        "relational_predicted_H": float(rel_y_predicted),
        "relational_actual_H": float(preambled["relational"]["gen_H_mean"]),
        "relational_residual": float(residual),
        "conditions_excl": {n: {"ratio": float(x), "gen_H": float(y)}
                           for n, x, y in zip(lab_excl, x_excl, y_excl)},
        "passes_threshold": residual < -0.2,
    }


def run_single_model(model_key, model_info):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_name = model_info["hf_name"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"  MODEL: {model_key} ({hf_name})")
    print(f"  Arch: {model_info['arch']}, Norm: {model_info['norm']}")
    print(f"  Device: {device}")
    print(f"{'='*70}")

    print(f"  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Token counts per condition:")
    for name, text in PREAMBLES.items():
        n = len(tokenizer.encode(text, add_special_tokens=False))
        print(f"    {name:15s}: {n} tokens")

    print(f"  Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        hf_name,
        torch_dtype=torch.float16,
        device_map="auto",
        output_hidden_states=True,
        trust_remote_code=True,
    )
    model.eval()

    num_layers = model.config.num_hidden_layers
    print(f"  num_hidden_layers: {num_layers}")

    conditions = {"none": None}
    conditions.update(PREAMBLES)

    model_results = {}
    total = len(conditions) * len(PROBES)
    done = 0

    for cond_name, preamble in conditions.items():
        print(f"\n  --- Condition: {cond_name} ---")

        l_last_ratios = []
        l_penult_ratios = []
        v2_vectors = []

        for probe in PROBES:
            done += 1
            hidden_states = extract_hidden_states(
                model, tokenizer, model_key, preamble, probe, device
            )

            n_hs = len(hidden_states)
            last_idx = n_hs - 2
            penult_idx = n_hs - 3

            m_last = compute_metrics_at_layer(hidden_states, last_idx, compute_v2=True)
            m_penult = compute_metrics_at_layer(hidden_states, penult_idx, compute_v2=False)

            l_last_ratios.append(m_last["ratio"])
            l_penult_ratios.append(m_penult["ratio"])
            v2_vectors.append(m_last["v2"])

            if done % 10 == 0:
                print(f"    [{done}/{total}] L_last={m_last['ratio']:.4f}")

        print(f"  Measuring generation entropy...")
        gen_entropies = []
        for pi, probe in enumerate(PROBES):
            gen = measure_generation_entropy(
                model, tokenizer, model_key, preamble, probe, device
            )
            gen_entropies.append(gen)

        mean_ratio = float(np.mean(l_last_ratios))
        mean_gen_H = float(np.mean([g["mean_entropy"] for g in gen_entropies]))

        v2_cosines = []
        for a, b in combinations(range(len(v2_vectors)), 2):
            dot = np.dot(v2_vectors[a], v2_vectors[b])
            norm = np.linalg.norm(v2_vectors[a]) * np.linalg.norm(v2_vectors[b])
            v2_cosines.append(float(dot / norm) if norm > 0 else 0)

        v2_centroid = np.mean(v2_vectors, axis=0)
        v2_centroid = v2_centroid / (np.linalg.norm(v2_centroid) + 1e-10)

        print(f"  {cond_name}: L_last={mean_ratio:.4f}, gen_H={mean_gen_H:.3f}, "
              f"V2_cos={np.mean(v2_cosines):.3f}")

        model_results[cond_name] = {
            "l_last_ratio": mean_ratio,
            "l_last_std": float(np.std(l_last_ratios)),
            "l_penult_ratio": float(np.mean(l_penult_ratios)),
            "gen_H_mean": mean_gen_H,
            "gen_H_std": float(np.std([g["mean_entropy"] for g in gen_entropies])),
            "v2_mean_cos": float(np.mean(v2_cosines)) if v2_cosines else 0,
            "v2_centroid": v2_centroid.tolist(),
            "all_l_last_ratios": [float(r) for r in l_last_ratios],
            "gen_per_probe": [
                {"mean_H": g["mean_entropy"], "text": g["generated_text"][:100]}
                for g in gen_entropies
            ],
        }

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model_results, num_layers


def run_experiment():
    print(f"F106 Cross-Architecture Replication")
    print(f"Models: {', '.join(MODELS.keys())}")
    print(f"Conditions: {len(PREAMBLES) + 1} (6 preambled + baseline)")
    print(f"Probes: {len(PROBES)}")
    print(f"Start: {datetime.now().isoformat()}")

    all_model_results = {}
    f106_tests = {}

    for model_key, model_info in MODELS.items():
        try:
            model_results, num_layers = run_single_model(model_key, model_info)
            all_model_results[model_key] = {
                "info": model_info,
                "num_hidden_layers": num_layers,
                "results": model_results,
            }
            f106 = compute_f106_test(model_results)
            f106_tests[model_key] = f106

            print(f"\n  F106 TEST for {model_key}:")
            print(f"    r (excl relational): {f106['r_excluding_relational']:.3f}")
            print(f"    r (incl relational): {f106['r_including_relational']:.3f}")
            print(f"    r drop:             {f106['r_drop']:.3f}")
            print(f"    relational residual: {f106['relational_residual']:.3f} nats")
            print(f"    PASSES (-0.2)?      {'YES' if f106['passes_threshold'] else 'NO'}")
        except Exception as e:
            print(f"\n  ERROR on {model_key}: {e}")
            import traceback
            traceback.print_exc()
            all_model_results[model_key] = {"error": str(e)}

    print(f"\n{'='*70}")
    print(f"  CROSS-ARCHITECTURE SUMMARY")
    print(f"{'='*70}")

    passing = 0
    for mk, f106 in f106_tests.items():
        status = "PASS" if f106["passes_threshold"] else "FAIL"
        print(f"  {mk:20s}: residual={f106['relational_residual']:+.3f}  "
              f"r_excl={f106['r_excluding_relational']:.3f}  "
              f"r_incl={f106['r_including_relational']:.3f}  [{status}]")
        if f106["passes_threshold"]:
            passing += 1

    print(f"\n  {passing}/{len(f106_tests)} models pass threshold (residual < -0.2)")
    if passing >= 3:
        print(f"  >>> F106 GENERALIZES: broken correlation is a GQA property")
    elif passing >= 2:
        print(f"  >>> PARTIAL: broken correlation appears in most but not all GQA models")
    elif passing >= 1:
        print(f"  >>> WEAK: only one model shows the effect")
    else:
        print(f"  >>> F106 DOES NOT GENERALIZE: Mistral-specific finding")

    v2_cross = {}
    for mk in all_model_results:
        if "error" in all_model_results[mk]:
            continue
        results = all_model_results[mk]["results"]
        preambled = {k: v for k, v in results.items() if k != "none"}
        for (ca, da), (cb, db) in combinations(preambled.items(), 2):
            va = np.array(da["v2_centroid"])
            vb = np.array(db["v2_centroid"])
            if len(va) != len(vb):
                continue
            cos = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-10))
            key = f"{ca}_vs_{cb}"
            if key not in v2_cross:
                v2_cross[key] = {}
            v2_cross[key][mk] = cos

    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = RESULTS_DIR / f"exp_f106_crossarch_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "experiment": "f106_crossarch_replication",
                "prediction": "relational breaks geometry→entropy correlation across GQA models",
                "threshold": "relational residual < -0.2 nats",
                "models": {k: v["info"] for k, v in all_model_results.items() if "info" in v},
                "n_probes": len(PROBES),
                "probes": PROBES,
                "preambles": PREAMBLES,
                "results": {
                    mk: mv["results"] if "results" in mv else {"error": mv.get("error")}
                    for mk, mv in all_model_results.items()
                },
                "f106_tests": f106_tests,
                "v2_cross_model_alignment": v2_cross,
                "timestamp": datetime.now().isoformat(),
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\nResults saved to {out_path}")
    print(f"Finished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    run_experiment()
