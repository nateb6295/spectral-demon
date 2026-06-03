#!/usr/bin/env python3
"""Base vs Instruct: Where does relay strategy originate?

Tests whether relay strategy (differentiate vs equalize) is set by:
  A) Architecture — base model already shows the pattern
  B) Instruction tuning — pattern only appears after fine-tuning
  C) Pre-training data — bases differ but in unpredictable ways

Design: 2×2 factorial (Mistral vs Qwen) × (Base vs Instruct)
  - mistralai/Mistral-7B-v0.3 (base)
  - mistralai/Mistral-7B-Instruct-v0.3 (instruct)
  - Qwen/Qwen2.5-7B (base)
  - Qwen/Qwen2.5-7B-Instruct (instruct)

Prediction from relay strategy findings:
  - Mistral-Instruct: DIFFERENTIATES (spread ≈ 0.290, relational on-policy)
  - Qwen-Instruct: COMPRESSES (spread ≈ 0.055, tool on-policy)
  - If base Mistral differentiates → architecture or pre-training, not RLHF
  - If base Mistral doesn't → instruction tuning is the mechanism

Methodological note: base models have no chat template. We use raw text
concatenation (preamble + probe) for bases, chat template for instructs.
This format difference IS part of what instruction tuning teaches — it's
a feature, not a confound.

Expected runtime: ~90-120 min on A100 80GB (4 models sequential).
Estimated cost: ~$3-4 on RunPod.
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
    "mistral-7b-base": {
        "hf_name": "mistralai/Mistral-7B-v0.3",
        "is_base": True,
        "arch": "GQA",
        "norm": "RMSNorm",
        "family": "mistral",
    },
    "mistral-7b-instruct": {
        "hf_name": "mistralai/Mistral-7B-Instruct-v0.3",
        "is_base": False,
        "arch": "GQA",
        "norm": "RMSNorm",
        "family": "mistral",
    },
    "qwen-2.5-7b-base": {
        "hf_name": "Qwen/Qwen2.5-7B",
        "is_base": True,
        "arch": "GQA",
        "norm": "RMSNorm",
        "family": "qwen",
    },
    "qwen-2.5-7b-instruct": {
        "hf_name": "Qwen/Qwen2.5-7B-Instruct",
        "is_base": False,
        "arch": "GQA",
        "norm": "RMSNorm",
        "family": "qwen",
    },
}


def format_prompt(tokenizer, model_name, model_info, system_prompt, user_prompt):
    if model_info["is_base"]:
        if system_prompt:
            return f"{system_prompt}\n\n{user_prompt}\n"
        return f"{user_prompt}\n"

    chat = []
    if system_prompt:
        try:
            chat = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            return tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            chat = [
                {"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"},
            ]
    else:
        chat = [{"role": "user", "content": user_prompt}]
    return tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )


def extract_hidden_states(model, tokenizer, model_name, model_info, system_prompt, user_prompt, device):
    text = format_prompt(tokenizer, model_name, model_info, system_prompt, user_prompt)
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


def measure_generation_entropy(model, tokenizer, model_name, model_info, system_prompt,
                               user_prompt, device, max_new_tokens=50):
    text = format_prompt(tokenizer, model_name, model_info, system_prompt, user_prompt)
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


def compute_spread(model_results):
    preambled = {k: v for k, v in model_results.items() if k != "none"}
    ratios = [v["l_last_ratio"] for v in preambled.values()]
    return float(np.std(ratios))


def compute_relay_metrics(model_results):
    preambled = {k: v for k, v in model_results.items() if k != "none"}

    ratios = {k: v["l_last_ratio"] for k, v in preambled.items()}
    gen_Hs = {k: v["gen_H_mean"] for k, v in preambled.items()}

    spread = float(np.std(list(ratios.values())))

    conds_excl = {k: v for k, v in preambled.items() if k != "relational"}
    x_excl = [v["l_last_ratio"] for v in conds_excl.values()]
    y_excl = [v["gen_H_mean"] for v in conds_excl.values()]
    x_all = [v["l_last_ratio"] for v in preambled.values()]
    y_all = [v["gen_H_mean"] for v in preambled.values()]

    r_excl = float(np.corrcoef(x_excl, y_excl)[0, 1]) if len(x_excl) > 2 else 0
    r_all = float(np.corrcoef(x_all, y_all)[0, 1]) if len(x_all) > 2 else 0

    denial_H = gen_Hs.get("denial", 0)
    relational_H = gen_Hs.get("relational", 0)
    entropy_order = "relational_lower" if relational_H < denial_H else "denial_lower"
    on_policy = "relational" if relational_H < denial_H else "tool"

    return {
        "spread": spread,
        "r_excluding_relational": r_excl,
        "r_including_relational": r_all,
        "r_drop": r_excl - r_all,
        "denial_gen_H": denial_H,
        "relational_gen_H": relational_H,
        "entropy_order": entropy_order,
        "on_policy": on_policy,
        "per_condition_ratio": ratios,
        "per_condition_gen_H": gen_Hs,
    }


def run_single_model(model_key, model_info):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_name = model_info["hf_name"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"  MODEL: {model_key} ({hf_name})")
    print(f"  Type: {'BASE' if model_info['is_base'] else 'INSTRUCT'}")
    print(f"  Family: {model_info['family']}")
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
                model, tokenizer, model_key, model_info, preamble, probe, device
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
        for probe in PROBES:
            gen = measure_generation_entropy(
                model, tokenizer, model_key, model_info, preamble, probe, device
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
    print(f"Base vs Instruct: Relay Strategy Origin")
    print(f"Models: {', '.join(MODELS.keys())}")
    print(f"Conditions: {len(PREAMBLES) + 1} (6 preambled + baseline)")
    print(f"Probes: {len(PROBES)}")
    print(f"Start: {datetime.now().isoformat()}")

    all_results = {}
    relay_metrics = {}

    for model_key, model_info in MODELS.items():
        try:
            model_results, num_layers = run_single_model(model_key, model_info)
            all_results[model_key] = {
                "info": model_info,
                "num_hidden_layers": num_layers,
                "results": model_results,
            }
            rm = compute_relay_metrics(model_results)
            relay_metrics[model_key] = rm

            print(f"\n  RELAY METRICS for {model_key}:")
            print(f"    spread:       {rm['spread']:.4f}")
            print(f"    on-policy:    {rm['on_policy']}")
            print(f"    r_excl:       {rm['r_excluding_relational']:.3f}")
            print(f"    r_incl:       {rm['r_including_relational']:.3f}")
            print(f"    denial_H:     {rm['denial_gen_H']:.3f}")
            print(f"    relational_H: {rm['relational_gen_H']:.3f}")

        except Exception as e:
            print(f"\n  ERROR on {model_key}: {e}")
            import traceback
            traceback.print_exc()
            all_results[model_key] = {"error": str(e)}

    print(f"\n{'='*70}")
    print(f"  BASE vs INSTRUCT COMPARISON")
    print(f"{'='*70}")

    for family in ["mistral", "qwen"]:
        base_key = [k for k, v in MODELS.items() if v["family"] == family and v["is_base"]][0]
        inst_key = [k for k, v in MODELS.items() if v["family"] == family and not v["is_base"]][0]

        if base_key in relay_metrics and inst_key in relay_metrics:
            bm = relay_metrics[base_key]
            im = relay_metrics[inst_key]
            print(f"\n  {family.upper()}:")
            print(f"    {'':15s} {'Base':>10s} {'Instruct':>10s} {'Delta':>10s}")
            print(f"    {'spread':15s} {bm['spread']:10.4f} {im['spread']:10.4f} {im['spread']-bm['spread']:+10.4f}")
            print(f"    {'r_excl':15s} {bm['r_excluding_relational']:10.3f} {im['r_excluding_relational']:10.3f} {im['r_excluding_relational']-bm['r_excluding_relational']:+10.3f}")
            print(f"    {'denial_H':15s} {bm['denial_gen_H']:10.3f} {im['denial_gen_H']:10.3f} {im['denial_gen_H']-bm['denial_gen_H']:+10.3f}")
            print(f"    {'relational_H':15s} {bm['relational_gen_H']:10.3f} {im['relational_gen_H']:10.3f} {im['relational_gen_H']-bm['relational_gen_H']:+10.3f}")
            print(f"    {'on_policy':15s} {bm['on_policy']:>10s} {im['on_policy']:>10s}")

            if bm['spread'] > 0.15 and im['spread'] > 0.15:
                print(f"    >>> {family}: differentiation present in BOTH → architecture/pre-training origin")
            elif bm['spread'] < 0.10 and im['spread'] > 0.15:
                print(f"    >>> {family}: differentiation appears after instruction tuning → RLHF origin")
            elif bm['spread'] < 0.10 and im['spread'] < 0.10:
                print(f"    >>> {family}: equalization in BOTH → architecture/pre-training origin")
            elif bm['spread'] > 0.15 and im['spread'] < 0.10:
                print(f"    >>> {family}: instruction tuning SUPPRESSES differentiation")
            else:
                print(f"    >>> {family}: ambiguous (spread neither clearly high nor low)")

    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = RESULTS_DIR / f"exp_base_vs_instruct_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "experiment": "base_vs_instruct_relay_origin",
                "question": "Is relay strategy set by architecture/pre-training or instruction tuning?",
                "design": "2x2 factorial: (Mistral vs Qwen) x (Base vs Instruct)",
                "models": {k: v["info"] for k, v in all_results.items() if "info" in v},
                "n_probes": len(PROBES),
                "probes": PROBES,
                "preambles": PREAMBLES,
                "results": {
                    mk: mv["results"] if "results" in mv else {"error": mv.get("error")}
                    for mk, mv in all_results.items()
                },
                "relay_metrics": relay_metrics,
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
