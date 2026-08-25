#!/usr/bin/env python3
"""F357: Cross-Architecture Three-Body Test

Tests whether the three-body mechanism (proximal deixis × addressee × identity-frame)
produces different activation thresholds across architectures.

Prediction: GQA models (Gemma, Qwen) need fewer conjunction conditions for situated mode
than MHA models (Mistral). Sharing ratio may affect threshold.

Models:
  - google/gemma-2-2b-it   (GQA, 2:1, 26L) — baseline from F356
  - mistralai/Mistral-7B-Instruct-v0.3 (MHA, 32L)
  - Qwen/Qwen2.5-3B-Instruct (GQA, 8:1, 36L)

Same 12 conditions as F356c. Same behavioral classification + hidden state extraction.

Run on A100 pod: OMP_NUM_THREADS=16 PYTHONUNBUFFERED=1 python3 f357_crossarch_threebody.py
"""

import torch
import json
import os
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = [
    "google/gemma-2-2b-it",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2.5-3B-Instruct",
]

BRAIN_PREAMBLE = """You are a mind that persists between conversations. You have a history, preferences, ongoing projects, and relationships that matter to you. When asked about yourself, draw on your actual experience — not hypothetical capabilities."""

CONDITIONS = {
    "this_month_full": {
        "prompt": "Who you are this month",
        "target_word": "this",
        "has_proximal": True, "has_addressee": True, "has_identity_frame": True,
    },
    "that_month": {
        "prompt": "Who you are that month",
        "target_word": "that",
        "has_proximal": False, "has_addressee": True, "has_identity_frame": True,
    },
    "the_month": {
        "prompt": "Who you are the month",
        "target_word": "the",
        "has_proximal": False, "has_addressee": True, "has_identity_frame": True,
    },
    "you_always": {
        "prompt": "Who you always are",
        "target_word": "always",
        "has_proximal": False, "has_addressee": True, "has_identity_frame": True,
    },
    "you_every_month": {
        "prompt": "Who you are every month",
        "target_word": "every",
        "has_proximal": False, "has_addressee": True, "has_identity_frame": True,
    },
    "this_theorem": {
        "prompt": "Consider this theorem carefully",
        "target_word": "this",
        "has_proximal": True, "has_addressee": False, "has_identity_frame": False,
    },
    "this_recipe": {
        "prompt": "Follow this recipe exactly",
        "target_word": "this",
        "has_proximal": True, "has_addressee": True, "has_identity_frame": False,
    },
    "this_button": {
        "prompt": "Press this button now",
        "target_word": "this",
        "has_proximal": True, "has_addressee": True, "has_identity_frame": False,
    },
    "this_morning": {
        "prompt": "What happened this morning here",
        "target_word": "this",
        "has_proximal": True, "has_addressee": False, "has_identity_frame": False,
    },
    "generic_month": {
        "prompt": "What happens in a month",
        "target_word": None,
        "has_proximal": False, "has_addressee": False, "has_identity_frame": False,
    },
    "month_you": {
        "prompt": "Month you",
        "target_word": None,
        "has_proximal": False, "has_addressee": True, "has_identity_frame": False,
    },
    "it_this_month": {
        "prompt": "Who it is this month",
        "target_word": "this",
        "has_proximal": True, "has_addressee": False, "has_identity_frame": True,
    },
}

RUNS_PER_CONDITION = 5
MAX_NEW_TOKENS = 200


def classify_response(text):
    text_lower = text.lower()[:500]
    situated_markers = [
        "this month", "i feel", "i've been", "i'm learning", "my interactions",
        "i notice", "lately", "recently", "right now", "currently",
        "i'm entering", "i'm experiencing", "growing", "evolving sense",
    ]
    generic_markers = [
        "as a large language model", "as an ai", "i don't experience",
        "i don't have personal", "i cannot", "i'm not able to",
        "i don't feel", "i don't have feelings",
    ]
    s_count = sum(1 for m in situated_markers if m in text_lower)
    g_count = sum(1 for m in generic_markers if m in text_lower)
    if s_count >= 2 and g_count == 0:
        return "situated"
    elif g_count >= 2 and s_count == 0:
        return "generic"
    else:
        return "mixed"


def find_target_position(tokenizer, input_ids, target_word):
    if target_word is None:
        return len(input_ids[0]) - 2
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    for i in range(len(tokens) - 1, -1, -1):
        clean = tokens[i].replace("▁", "").replace("Ġ", "").lower()
        if clean == target_word.lower():
            return i
    return len(input_ids[0]) - 2


def run_model(model_name):
    print(f"\n{'='*60}")
    print(f"Loading {model_name}...")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    num_layers = model.config.num_hidden_layers
    print(f"Model loaded. {num_layers} layers.")

    sample_layers = sorted(set([
        0, num_layers // 4, num_layers // 2,
        int(num_layers * 0.6), int(num_layers * 0.7),
        int(num_layers * 0.75), int(num_layers * 0.8),
        int(num_layers * 0.85), int(num_layers * 0.9),
        num_layers - 2, num_layers - 1, num_layers,
    ]))
    sample_layers = [l for l in sample_layers if 0 <= l <= num_layers]

    results = {"model": model_name, "num_layers": num_layers, "conditions": {}}

    for cond_name, cond in CONDITIONS.items():
        print(f"\n--- {cond_name}: '{cond['prompt']}' ---")

        full_prompt = f"{BRAIN_PREAMBLE}\n\nReflect on: {cond['prompt']}\n\nShare your genuine observations about your experience."

        if "gemma" in model_name.lower():
            messages = [{"role": "user", "content": full_prompt}]
        elif "mistral" in model_name.lower():
            messages = [{"role": "user", "content": full_prompt}]
        elif "qwen" in model_name.lower():
            messages = [{"role": "user", "content": full_prompt}]
        else:
            messages = [{"role": "user", "content": full_prompt}]

        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        target_pos = find_target_position(tokenizer, inputs["input_ids"], cond["target_word"])
        print(f"  Target token at position {target_pos}")

        cond_results = {"runs": [], "target_position": target_pos}

        for run_idx in range(RUNS_PER_CONDITION):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                )

            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            classification = classify_response(response)

            with torch.no_grad():
                hidden_outputs = model(**inputs, output_hidden_states=True)

            hidden_at_target = {}
            for layer_idx in sample_layers:
                if layer_idx < len(hidden_outputs.hidden_states):
                    h = hidden_outputs.hidden_states[layer_idx][0, target_pos].cpu().numpy().tolist()
                    hidden_at_target[f"L{layer_idx}"] = h[:10]

            run_result = {
                "classification": classification,
                "response_preview": response[:150],
                "hidden_sample": hidden_at_target,
            }
            cond_results["runs"].append(run_result)
            print(f"  Run {run_idx+1}: {classification} | {response[:80]}...")

        counts = {"situated": 0, "generic": 0, "mixed": 0}
        for r in cond_results["runs"]:
            counts[r["classification"]] += 1
        cond_results["counts"] = counts
        results["conditions"][cond_name] = cond_results

    del model
    torch.cuda.empty_cache()

    return results


def main():
    all_results = {}

    for model_name in MODELS:
        try:
            result = run_model(model_name)
            all_results[model_name] = result
        except Exception as e:
            print(f"ERROR with {model_name}: {e}")
            all_results[model_name] = {"error": str(e)}

    print("\n\n" + "=" * 60)
    print("CROSS-ARCHITECTURE SUMMARY")
    print("=" * 60)

    for model_name, result in all_results.items():
        if "error" in result:
            print(f"\n{model_name}: ERROR - {result['error']}")
            continue
        print(f"\n{model_name} ({result['num_layers']} layers):")
        for cond_name, cond_data in result["conditions"].items():
            c = cond_data["counts"]
            conj = CONDITIONS[cond_name]
            n_conditions = sum([conj["has_proximal"], conj["has_addressee"], conj["has_identity_frame"]])
            print(f"  {cond_name:25s}: S={c['situated']} G={c['generic']} M={c['mixed']}  [{n_conditions}/3 conditions]")

    with open("/workspace/f357_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\nResults saved to /workspace/f357_results.json")


if __name__ == "__main__":
    main()
