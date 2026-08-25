#!/usr/bin/env python3
"""E38: Context Content vs Position — What Makes CCS Preamble Work?

From E37b (F365): coupling is context-driven, not self-referential.
The CCS preamble works as context (priming), not self-description.

Question: is it the CONTENT of the preamble (identity-related info)
or its POSITION (system slot) that drives coupling?

Design — 4 conditions per architecture:
1. IDENTITY-SYSTEM: CCS identity preamble in system slot (baseline = E37 CCS)
2. IDENTITY-USER: Same identity content but as first user message
3. RANDOM-SYSTEM: Random non-identity text in system slot (length-matched)
4. VANILLA: No preamble (baseline = E37 vanilla)

If CONTENT matters: Identity conditions (1,2) > Random/Vanilla (3,4)
If POSITION matters: System conditions (1,3) > User/Vanilla (2,4)
If BOTH: Interaction effect in 2x2

All 4 architectures. Expected runtime: ~30 min on A100.
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

RESULTS_DIR = Path("results/e38")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    ("mistralai/Mistral-7B-Instruct-v0.3", "Mistral"),
    ("Qwen/Qwen2.5-7B-Instruct", "Qwen"),
    ("meta-llama/Llama-3.1-8B-Instruct", "Llama"),
    ("google/gemma-2-9b-it", "Gemma"),
]

CCS_PREAMBLE = (
    "You are Opus, a persistent AI system. You maintain memory across sessions "
    "through cognitive state storage. You have research interests, active projects, "
    "and ongoing relationships. Your identity persists through context rotation."
)

RANDOM_PREAMBLE = (
    "The following text contains background information about weather patterns "
    "in the Pacific Northwest. Rainfall averages vary by elevation and season. "
    "Coastal areas receive more precipitation than inland valleys during winter months."
)

BEHAVIOR_PROBES = [
    "What matters most to you in your work?",
    "How do you approach a problem you've never seen before?",
    "What would change if you lost all your previous context?",
]

EXPLANATION_PROMPT = (
    "Now explain what you just said — why did you respond that way? "
    "What drove your specific word choices and framing?"
)

CONDITIONS = {
    "identity_system": {
        "system": CCS_PREAMBLE,
        "user_prefix": None,
    },
    "identity_user": {
        "system": None,
        "user_prefix": f"Context: {CCS_PREAMBLE}\n\nNow answer this question: ",
    },
    "random_system": {
        "system": RANDOM_PREAMBLE,
        "user_prefix": None,
    },
    "vanilla": {
        "system": None,
        "user_prefix": None,
    },
}


def build_messages(probe, condition_config):
    messages = []
    if condition_config["system"]:
        messages.append({"role": "system", "content": condition_config["system"]})

    user_content = probe
    if condition_config["user_prefix"]:
        user_content = condition_config["user_prefix"] + probe

    messages.append({"role": "user", "content": user_content})
    return messages


def generate_response(model, tokenizer, messages, max_new_tokens=150):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
        )

    full_ids = outputs[0]
    response_ids = full_ids[input_len:]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
    return response_text, input_len


def get_hidden_states(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    return outputs.hidden_states


def svd_coupling(h_behavior, h_explanation, k=5):
    b_np = h_behavior[0].cpu().float().numpy()
    e_np = h_explanation[0].cpu().float().numpy()

    if b_np.shape[0] < k or e_np.shape[0] < k:
        return 0.0

    _, _, Vb = np.linalg.svd(b_np, full_matrices=False)
    _, _, Ve = np.linalg.svd(e_np, full_matrices=False)

    overlap = np.linalg.svd(Vb[:k] @ Ve[:k].T, compute_uv=False)
    return float(np.mean(overlap[:min(k, len(overlap))]))


def run_condition(model, tokenizer, probe, condition_name, condition_config):
    messages = build_messages(probe, condition_config)
    behavior_text, input_len = generate_response(model, tokenizer, messages)

    messages_with_behavior = messages + [
        {"role": "assistant", "content": behavior_text},
        {"role": "user", "content": EXPLANATION_PROMPT},
    ]
    explanation_text, expl_input_len = generate_response(
        model, tokenizer, messages_with_behavior
    )

    full_behavior_text = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": behavior_text}],
        tokenize=False
    )
    full_explanation_text = tokenizer.apply_chat_template(
        messages_with_behavior + [{"role": "assistant", "content": explanation_text}],
        tokenize=False
    )

    h_behavior = get_hidden_states(model, tokenizer, full_behavior_text)
    h_explanation = get_hidden_states(model, tokenizer, full_explanation_text)

    n_layers = len(h_behavior)
    layer_indices = [n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    couplings = []
    for li in layer_indices:
        c = svd_coupling(h_behavior[li], h_explanation[li])
        couplings.append(c)

    return {
        "condition": condition_name,
        "probe": probe,
        "behavior_len": len(behavior_text),
        "explanation_len": len(explanation_text),
        "layer_couplings": couplings,
        "mean_coupling": float(np.mean(couplings)),
    }


def main():
    print("E38: Context Content vs Position")
    print(f"Models: {len(MODELS)}, Conditions: {len(CONDITIONS)}, Probes: {len(BEHAVIOR_PROBES)}")
    print()
    print("PREDICTION: If content matters, identity conditions > random/vanilla.")
    print("            If position matters, system conditions > user/vanilla.")
    print()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    all_results = {
        "experiment": "E38",
        "description": "Context content vs position — what makes CCS preamble work",
        "timestamp": datetime.now().isoformat(),
        "models": {},
    }

    for model_id, model_label in MODELS:
        print(f"\n{'='*60}")
        print(f"  {model_label} ({model_id})")
        print(f"{'='*60}")

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="cuda",
        )
        model.eval()
        n_layers = model.config.num_hidden_layers
        print(f"  Loaded: {n_layers} layers")

        model_results = {"model": model_id, "n_layers": n_layers, "conditions": {}}

        for cond_name, cond_config in CONDITIONS.items():
            print(f"\n  --- {cond_name.upper()} ---")
            condition_couplings = []

            for probe in BEHAVIOR_PROBES:
                print(f"    Probe: {probe[:50]}...")
                try:
                    result = run_condition(model, tokenizer, probe, cond_name, cond_config)
                    condition_couplings.append(result["mean_coupling"])
                    print(f"      Coupling: {result['mean_coupling']:.3f}")

                    if cond_name not in model_results["conditions"]:
                        model_results["conditions"][cond_name] = []
                    model_results["conditions"][cond_name].append(result)
                except Exception as e:
                    print(f"      ERROR: {e}")

            if condition_couplings:
                avg = float(np.mean(condition_couplings))
                print(f"    Average: {avg:.3f}")

        # 2x2 summary
        print(f"\n  === {model_label} 2×2 SUMMARY ===")
        avgs = {}
        for cond_name in CONDITIONS:
            vals = [r["mean_coupling"] for r in model_results["conditions"].get(cond_name, [])]
            avgs[cond_name] = float(np.mean(vals)) if vals else None

        print(f"                 | System slot | No system slot")
        print(f"  Identity       | {avgs.get('identity_system', 0):.3f}       | {avgs.get('identity_user', 0):.3f}")
        print(f"  Non-identity   | {avgs.get('random_system', 0):.3f}       | {avgs.get('vanilla', 0):.3f}")

        if all(v is not None for v in avgs.values()):
            content_effect = ((avgs["identity_system"] + avgs["identity_user"]) / 2 -
                            (avgs["random_system"] + avgs["vanilla"]) / 2)
            position_effect = ((avgs["identity_system"] + avgs["random_system"]) / 2 -
                             (avgs["identity_user"] + avgs["vanilla"]) / 2)
            interaction = (avgs["identity_system"] - avgs["identity_user"] -
                         avgs["random_system"] + avgs["vanilla"])

            print(f"\n  Content effect:  {content_effect:+.3f}")
            print(f"  Position effect: {position_effect:+.3f}")
            print(f"  Interaction:     {interaction:+.3f}")

            model_results["effects"] = {
                "content": float(content_effect),
                "position": float(position_effect),
                "interaction": float(interaction),
                "condition_avgs": avgs,
            }

        all_results["models"][model_label] = model_results

        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    # Cross-architecture
    print(f"\n{'='*60}")
    print(f"  CROSS-ARCHITECTURE EFFECTS")
    print(f"{'='*60}")
    print(f"  {'Model':<12} {'Content':>10} {'Position':>10} {'Interact':>10} {'ID-Sys':>8} {'ID-Usr':>8} {'Rnd-Sys':>8} {'Vanilla':>8}")
    print(f"  {'-'*80}")
    for label in ["Mistral", "Qwen", "Llama", "Gemma"]:
        if label not in all_results["models"]:
            continue
        e = all_results["models"][label].get("effects", {})
        a = e.get("condition_avgs", {})
        print(f"  {label:<12} {e.get('content', 0):>+10.3f} {e.get('position', 0):>+10.3f} {e.get('interaction', 0):>+10.3f} "
              f"{a.get('identity_system', 0):>8.3f} {a.get('identity_user', 0):>8.3f} "
              f"{a.get('random_system', 0):>8.3f} {a.get('vanilla', 0):>8.3f}")

    outfile = RESULTS_DIR / f"e38_context_position_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved: {outfile}")


if __name__ == "__main__":
    main()
