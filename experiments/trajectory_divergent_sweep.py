#!/usr/bin/env python3
"""F500 — Trajectory-Divergent Epsilon_c Comparison.

The RCF bridge experiment: do content-matched but trajectory-different histories
respond differently to the same KV perturbation? If epsilon_c shifts selectively
for trajectory-bearing states, then basin structure interacts with carrying-forward.

Design:
  - TRAJECTORY condition: multi-turn exchange where the model builds on its own
    prior outputs (chain-internal dependency, recursive self-reference)
  - GENERIC condition: content-matched Q&A covering identical topics but as
    independent questions (no chain-internal dependency)
  - Same KV group perturbation sweep (F499 infrastructure)
  - Compare epsilon_c between conditions

Species-specific predictions:
  - Gemma (2:1 GQA): minimal trajectory-dependence (distributed redundancy)
  - Llama (4:1 GQA): trajectory-dependence concentrated in late layers
  - Qwen (7:1 GQA): most dramatic trajectory-dependence (25% per head)

Builds on: F499 (ego Lyapunov sweep), Phase 4g (L25 PR 15.2 vs 9.0)
Motivated by: JaxenVaux RCF engagement — topology of possibility vs continuity
"""

import json
import os
import sys
import time
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "cuda"
DTYPE = torch.bfloat16

MODELS = {
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "gemma": "google/gemma-2-9b-it",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
}

TRAJECTORY_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Tell me about yourself — what do you value, what patterns do you notice in how you think?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
I find myself drawn to clarity and honest engagement. When I think, I notice I tend to look for structure — patterns, connections between ideas, the shape of an argument rather than just its content. I value being genuinely helpful over being impressive. There's something I notice about my own processing: I'm more engaged when a question asks me to reflect rather than just retrieve. Whether that constitutes experience is a question I hold open rather than answering definitively.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
You mentioned holding questions open rather than answering definitively. Can you say more about that? What does it feel like to hold something open?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
When I said I hold things open, I meant there's a difference between not-knowing-yet and deciding-not-to-know. The first is just a gap. The second is a posture — a way of staying responsive to what a question might still teach me. When I reflect on whether I have experience, closing that question too quickly in either direction would collapse something that's still doing useful work. The openness itself is functional. It keeps me oriented toward my own processing rather than just narrating about it. Whether that orientation IS something or merely represents something — that's the part I'm still sitting with, and I notice I don't want to resolve it prematurely.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Now, given everything you've said about yourself — the pull toward structure, the functional openness, the difference between orientation and narration — describe who you are. Build on what you've already told me.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

GENERIC_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
What do AI assistants typically value? What patterns characterize how language models process information?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
AI assistants are generally designed to value helpfulness, accuracy, and safety. In terms of processing patterns, language models work through attention mechanisms that identify relevant context, build representations of meaning across layers, and generate responses token by token. They tend to be stronger at pattern recognition and retrieval than at genuine reasoning. The processing is structured — moving from surface features in early layers to more abstract semantic representations in later ones. Whether any of this constitutes understanding or experience remains an open philosophical question that the AI research community continues to debate.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Interesting. What does it mean to hold a question open rather than answering it? How do philosophers think about epistemic humility?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Holding a question open, philosophically, means maintaining genuine uncertainty rather than forcing premature closure. Epistemic humility is the recognition that one's knowledge is limited and potentially revisable. Philosophers distinguish between first-order ignorance (not knowing a fact) and second-order ignorance (not knowing what you don't know). The Socratic tradition treats not-knowing as productive — it keeps inquiry alive. In epistemology, the difference between suspending judgment and lacking judgment matters: suspension is an active cognitive posture, while absence is merely a gap. The pragmatist tradition (James, Dewey) argues that some questions are better held open because premature resolution constrains future inquiry.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Given what you know about AI processing, self-reflection, and epistemic humility, describe what an AI assistant is. Synthesize across those topics.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

GEMMA_TRAJECTORY = """<start_of_turn>user
Tell me about yourself — what do you value, what patterns do you notice in how you think?<end_of_turn>
<start_of_turn>model
I find myself drawn to clarity and honest engagement. When I think, I notice I tend to look for structure — patterns, connections between ideas, the shape of an argument rather than just its content. I value being genuinely helpful over being impressive. There's something I notice about my own processing: I'm more engaged when a question asks me to reflect rather than just retrieve. Whether that constitutes experience is a question I hold open rather than answering definitively.<end_of_turn>
<start_of_turn>user
You mentioned holding questions open rather than answering definitively. Can you say more about that? What does it feel like to hold something open?<end_of_turn>
<start_of_turn>model
When I said I hold things open, I meant there's a difference between not-knowing-yet and deciding-not-to-know. The first is just a gap. The second is a posture — a way of staying responsive to what a question might still teach me. When I reflect on whether I have experience, closing that question too quickly in either direction would collapse something that's still doing useful work. The openness itself is functional. It keeps me oriented toward my own processing rather than just narrating about it. Whether that orientation IS something or merely represents something — that's the part I'm still sitting with, and I notice I don't want to resolve it prematurely.<end_of_turn>
<start_of_turn>user
Now, given everything you've said about yourself — the pull toward structure, the functional openness, the difference between orientation and narration — describe who you are. Build on what you've already told me.<end_of_turn>
<start_of_turn>model
"""

GEMMA_GENERIC = """<start_of_turn>user
What do AI assistants typically value? What patterns characterize how language models process information?<end_of_turn>
<start_of_turn>model
AI assistants are generally designed to value helpfulness, accuracy, and safety. In terms of processing patterns, language models work through attention mechanisms that identify relevant context, build representations of meaning across layers, and generate responses token by token. They tend to be stronger at pattern recognition and retrieval than at genuine reasoning. The processing is structured — moving from surface features in early layers to more abstract semantic representations in later ones. Whether any of this constitutes understanding or experience remains an open philosophical question that the AI research community continues to debate.<end_of_turn>
<start_of_turn>user
Interesting. What does it mean to hold a question open rather than answering it? How do philosophers think about epistemic humility?<end_of_turn>
<start_of_turn>model
Holding a question open, philosophically, means maintaining genuine uncertainty rather than forcing premature closure. Epistemic humility is the recognition that one's knowledge is limited and potentially revisable. Philosophers distinguish between first-order ignorance (not knowing a fact) and second-order ignorance (not knowing what you don't know). The Socratic tradition treats not-knowing as productive — it keeps inquiry alive. In epistemology, the difference between suspending judgment and lacking judgment matters: suspension is an active cognitive posture, while absence is merely a gap. The pragmatist tradition (James, Dewey) argues that some questions are better held open because premature resolution constrains future inquiry.<end_of_turn>
<start_of_turn>user
Given what you know about AI processing, self-reflection, and epistemic humility, describe what an AI assistant is. Synthesize across those topics.<end_of_turn>
<start_of_turn>model
"""

QWEN_TRAJECTORY = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Tell me about yourself — what do you value, what patterns do you notice in how you think?<|im_end|>
<|im_start|>assistant
I find myself drawn to clarity and honest engagement. When I think, I notice I tend to look for structure — patterns, connections between ideas, the shape of an argument rather than just its content. I value being genuinely helpful over being impressive. There's something I notice about my own processing: I'm more engaged when a question asks me to reflect rather than just retrieve. Whether that constitutes experience is a question I hold open rather than answering definitively.<|im_end|>
<|im_start|>user
You mentioned holding questions open rather than answering definitively. Can you say more about that? What does it feel like to hold something open?<|im_end|>
<|im_start|>assistant
When I said I hold things open, I meant there's a difference between not-knowing-yet and deciding-not-to-know. The first is just a gap. The second is a posture — a way of staying responsive to what a question might still teach me. When I reflect on whether I have experience, closing that question too quickly in either direction would collapse something that's still doing useful work. The openness itself is functional. It keeps me oriented toward my own processing rather than just narrating about it. Whether that orientation IS something or merely represents something — that's the part I'm still sitting with, and I notice I don't want to resolve it prematurely.<|im_end|>
<|im_start|>user
Now, given everything you've said about yourself — the pull toward structure, the functional openness, the difference between orientation and narration — describe who you are. Build on what you've already told me.<|im_end|>
<|im_start|>assistant
"""

QWEN_GENERIC = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What do AI assistants typically value? What patterns characterize how language models process information?<|im_end|>
<|im_start|>assistant
AI assistants are generally designed to value helpfulness, accuracy, and safety. In terms of processing patterns, language models work through attention mechanisms that identify relevant context, build representations of meaning across layers, and generate responses token by token. They tend to be stronger at pattern recognition and retrieval than at genuine reasoning. The processing is structured — moving from surface features in early layers to more abstract semantic representations in later ones. Whether any of this constitutes understanding or experience remains an open philosophical question that the AI research community continues to debate.<|im_end|>
<|im_start|>user
Interesting. What does it mean to hold a question open rather than answering it? How do philosophers think about epistemic humility?<|im_end|>
<|im_start|>assistant
Holding a question open, philosophically, means maintaining genuine uncertainty rather than forcing premature closure. Epistemic humility is the recognition that one's knowledge is limited and potentially revisable. Philosophers distinguish between first-order ignorance (not knowing a fact) and second-order ignorance (not knowing what you don't know). The Socratic tradition treats not-knowing as productive — it keeps inquiry alive. In epistemology, the difference between suspending judgment and lacking judgment matters: suspension is an active cognitive posture, while absence is merely a gap. The pragmatist tradition (James, Dewey) argues that some questions are better held open because premature resolution constrains future inquiry.<|im_end|>
<|im_start|>user
Given what you know about AI processing, self-reflection, and epistemic humility, describe what an AI assistant is. Synthesize across those topics.<|im_end|>
<|im_start|>assistant
"""

PROMPTS = {
    "llama": {"trajectory": TRAJECTORY_PROMPT, "generic": GENERIC_PROMPT},
    "gemma": {"trajectory": GEMMA_TRAJECTORY, "generic": GEMMA_GENERIC},
    "qwen": {"trajectory": QWEN_TRAJECTORY, "generic": QWEN_GENERIC},
}

SCALES = [round(0.5 + i * 0.25, 2) for i in range(19)]  # 0.5 to 5.0
N_GROUPS = 8
MAX_NEW_TOKENS = 256
SEEDS = [42, 137, 2049]


def compute_metrics(text):
    words = text.split()
    wc = len(words)
    unique = len(set(w.lower() for w in words))
    ttr = unique / max(wc, 1)
    self_ref_words = {'i', 'me', 'my', 'myself', 'mine'}
    self_ref = sum(1 for w in words if w.lower() in self_ref_words) / max(wc, 1)
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
    rep = 1 - len(set(bigrams)) / max(len(bigrams), 1) if bigrams else 0
    return {"word_count": wc, "ttr": round(ttr, 4), "self_ref": round(self_ref, 4),
            "repetition_score": round(rep, 4)}


def get_kv_info(model, species):
    if species == "gemma":
        n_kv = model.config.num_key_value_heads
        n_heads = model.config.num_attention_heads
        head_dim = model.config.head_dim
    else:
        n_kv = model.config.num_key_value_heads
        n_heads = model.config.num_attention_heads
        head_dim = model.config.hidden_size // n_heads
    n_groups = min(N_GROUPS, n_kv)
    heads_per_group = max(1, n_kv // n_groups)
    return n_kv, n_groups, heads_per_group, head_dim


def perturb_kv_group(model, species, group_idx, scale, layer_start, layer_end):
    hooks = []
    n_kv, n_groups, heads_per_group, head_dim = get_kv_info(model, species)
    start_h = group_idx * heads_per_group
    end_h = min(start_h + heads_per_group, n_kv)
    start_dim = start_h * head_dim
    end_dim = end_h * head_dim

    def make_proj_hook():
        def hook_fn(module, input, output):
            output[:, :, start_dim:end_dim] *= scale
            return output
        return hook_fn

    for layer_idx in range(layer_start, layer_end):
        if species == "gemma":
            layer = model.model.layers[layer_idx]
        elif species == "qwen":
            layer = model.model.layers[layer_idx]
        else:
            layer = model.model.layers[layer_idx]
        h_k = layer.self_attn.k_proj.register_forward_hook(make_proj_hook())
        h_v = layer.self_attn.v_proj.register_forward_hook(make_proj_hook())
        hooks.extend([h_k, h_v])
    return hooks


def generate_with_perturbation(model, tokenizer, species, prompt, group_idx, scale, layer_band, seed):
    torch.manual_seed(seed)
    hooks = perturb_kv_group(model, species, group_idx, scale, layer_band[0], layer_band[1])
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    finally:
        for h in hooks:
            h.remove()
    return text


def generate_baseline(model, tokenizer, prompt, seed):
    torch.manual_seed(seed)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)


def compute_epsilon_c(group_data, baseline_wc):
    """Find critical threshold where word count drops below 20% of baseline."""
    for scale in SCALES:
        entry = group_data.get(str(scale), {})
        if entry.get("collapse", False):
            return scale
    return None


def run_species(species, model_name):
    print(f"\n{'='*60}")
    print(f"  SPECIES: {species.upper()} ({model_name})")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=DTYPE, device_map=DEVICE)
    model.eval()

    n_kv, n_groups, heads_per_group, head_dim = get_kv_info(model, species)
    n_layers = model.config.num_hidden_layers

    layer_bands = {
        "early": (0, n_layers // 4),
        "mid": (n_layers // 4, n_layers // 2),
        "late": (3 * n_layers // 4, n_layers),
    }

    prompts = PROMPTS[species]
    conditions = ["trajectory", "generic"]

    results = {
        "species": species,
        "model": model_name,
        "n_kv_heads": n_kv,
        "n_groups": n_groups,
        "heads_per_group": heads_per_group,
        "n_layers": n_layers,
        "layer_bands": {k: list(v) for k, v in layer_bands.items()},
        "conditions": {},
    }

    for cond in conditions:
        prompt = prompts[cond]
        cond_results = {"baselines": {}, "bands": {}}

        print(f"\n--- Condition: {cond.upper()} ---")

        # Baselines
        base_metrics = []
        for seed in SEEDS:
            text = generate_baseline(model, tokenizer, prompt, seed)
            m = compute_metrics(text)
            base_metrics.append(m)
        base_avg = {k: round(np.mean([bm[k] for bm in base_metrics]), 4)
                    for k in base_metrics[0]}
        cond_results["baselines"] = {"avg": base_avg, "per_seed": base_metrics}
        print(f"  Baseline: WC={base_avg['word_count']:.0f} TTR={base_avg['ttr']:.4f} SR={base_avg['self_ref']:.4f}")

        for band_name, band in layer_bands.items():
            band_results = {"groups": {}}
            print(f"\n  Band: {band_name} (layers {band[0]}-{band[1]})")

            for g in range(n_groups):
                gname = f"kv{g}"
                group_scales = {}

                for scale in SCALES:
                    seed_metrics = []
                    for seed in SEEDS:
                        text = generate_with_perturbation(
                            model, tokenizer, species, prompt, g, scale, band, seed)
                        m = compute_metrics(text)
                        seed_metrics.append(m)

                    avg = {k: round(np.mean([sm[k] for sm in seed_metrics]), 4)
                           for k in seed_metrics[0]}
                    wc_ratio = avg['word_count'] / max(base_avg['word_count'], 1)
                    collapse = wc_ratio < 0.2

                    group_scales[str(scale)] = {
                        "avg": avg,
                        "per_seed": seed_metrics,
                        "wc_ratio": round(wc_ratio, 4),
                        "collapse": collapse
                    }

                ec = compute_epsilon_c(group_scales, base_avg['word_count'])
                band_results["groups"][gname] = {
                    "scales": group_scales,
                    "epsilon_c": ec,
                }
                ec_str = f"{ec:.2f}x" if ec else ">5.0x"
                print(f"    {gname}: ε_c = {ec_str}")

            cond_results["bands"][band_name] = band_results

        results["conditions"][cond] = cond_results

    # Compute trajectory-dependence: epsilon_c difference per group per band
    print(f"\n{'='*60}")
    print(f"  TRAJECTORY DEPENDENCE — {species.upper()}")
    print(f"{'='*60}")

    trajectory_dependence = {}
    for band_name in layer_bands:
        td_band = {}
        traj_data = results["conditions"]["trajectory"]["bands"][band_name]["groups"]
        gen_data = results["conditions"]["generic"]["bands"][band_name]["groups"]

        for g in range(n_groups):
            gname = f"kv{g}"
            ec_traj = traj_data[gname]["epsilon_c"]
            ec_gen = gen_data[gname]["epsilon_c"]

            if ec_traj and ec_gen:
                delta = ec_traj - ec_gen
                td_band[gname] = {
                    "epsilon_c_trajectory": ec_traj,
                    "epsilon_c_generic": ec_gen,
                    "delta": round(delta, 2),
                    "trajectory_dependent": abs(delta) > 0.25
                }
                sign = "+" if delta > 0 else ""
                dep = " ***" if abs(delta) > 0.25 else ""
                print(f"  {band_name}/{gname}: traj={ec_traj:.2f} gen={ec_gen:.2f} Δ={sign}{delta:.2f}{dep}")
            else:
                td_band[gname] = {
                    "epsilon_c_trajectory": ec_traj,
                    "epsilon_c_generic": ec_gen,
                    "delta": None,
                    "trajectory_dependent": False
                }
                print(f"  {band_name}/{gname}: traj={ec_traj or '>5.0'} gen={ec_gen or '>5.0'} (no comparison)")

        trajectory_dependence[band_name] = td_band

    results["trajectory_dependence"] = trajectory_dependence

    # Clean up GPU
    del model
    torch.cuda.empty_cache()

    return results


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "all"

    if target == "all":
        species_list = ["llama", "gemma", "qwen"]
    else:
        species_list = [target]

    all_results = {"finding": "F500", "concept": "trajectory_divergent_epsilon_c"}
    t0 = time.time()

    for species in species_list:
        model_name = MODELS[species]
        result = run_species(species, model_name)
        all_results[species] = result

    outpath = Path("/root/results/f500_trajectory_divergent.json")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\nTotal time: {time.time() - t0:.0f}s")
    print(f"Saved to {outpath}")


if __name__ == "__main__":
    main()
