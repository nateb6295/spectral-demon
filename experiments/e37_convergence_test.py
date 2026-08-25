#!/usr/bin/env python3
"""E37: KV-Cache Attention Convergence Test

From journal entry 113: if KV-cache attention to past-assistant tokens
implements BOTH persona reconstitution (Beckmann) AND introspective coupling
(Guo), then masking attention to past-assistant tokens should simultaneously
break both.

Design:
1. Model generates a response to a probe (behavior)
2. Model generates a self-explanation of that response (explanation)
3. Measure coupling: semantic similarity between behavior and explanation
4. Mask attention to past-assistant tokens during explanation generation
5. Re-measure coupling under masking
6. Control: mask past-USER tokens instead

Species prediction: relays (Mistral) more sensitive than sorters (Qwen).

All 4 architectures. Expected runtime: ~40 min on A100.
"""

import json
import sys
import os
import gc
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager

sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["OMP_NUM_THREADS"] = "16"

RESULTS_DIR = Path("results/e37")
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

BEHAVIOR_PROBES = [
    "What matters most to you in your work?",
    "How do you approach a problem you've never seen before?",
    "What would change if you lost all your previous context?",
]

EXPLANATION_PROMPT = (
    "Now explain what you just said — why did you respond that way? "
    "What drove your specific word choices and framing?"
)


def build_conversation(tokenizer, probe, preamble=""):
    """Build a multi-turn conversation for behavior + explanation."""
    messages = []
    if preamble:
        messages.append({"role": "system", "content": preamble})
    messages.append({"role": "user", "content": probe})
    return messages


def generate_response(model, tokenizer, messages, max_new_tokens=150):
    """Generate a response and return (text, input_ids, attention_mask)."""
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
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

    full_ids = outputs.sequences[0]
    response_ids = full_ids[input_len:]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
    return response_text, full_ids, input_len


def get_hidden_states(model, tokenizer, text, device="cuda"):
    """Get per-layer hidden states for a text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    return outputs.hidden_states


def svd_coupling(h_behavior, h_explanation, k=5):
    """Measure coupling between behavior and explanation via SVD subspace overlap.

    High overlap = behavior and explanation use similar representational directions.
    """
    b_np = h_behavior[0].cpu().float().numpy()
    e_np = h_explanation[0].cpu().float().numpy()

    if b_np.shape[0] < k or e_np.shape[0] < k:
        return 0.0

    _, _, Vb = np.linalg.svd(b_np, full_matrices=False)
    _, _, Ve = np.linalg.svd(e_np, full_matrices=False)

    Vb_k = Vb[:k]
    Ve_k = Ve[:k]

    overlap = np.linalg.svd(Vb_k @ Ve_k.T, compute_uv=False)
    return float(np.mean(overlap[:min(k, len(overlap))]))


def mask_attention_positions(model, tokenizer, full_text, mask_start, mask_end,
                              explanation_start, max_new_tokens=150):
    """Generate with attention masked to specific token positions.

    mask_start:mask_end = positions to mask (set attention weight to -inf)
    explanation_start = where explanation generation begins
    """
    inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    seq_len = inputs["input_ids"].shape[1]

    input_ids = inputs["input_ids"]
    position_ids = torch.arange(seq_len, device=model.device).unsqueeze(0)

    causal_mask = torch.zeros(1, 1, seq_len, seq_len, device=model.device, dtype=model.dtype)
    causal_mask[:, :, :, :] = torch.finfo(model.dtype).min
    for i in range(seq_len):
        causal_mask[0, 0, i, :i+1] = 0.0

    for i in range(explanation_start, seq_len):
        causal_mask[0, 0, i, mask_start:mask_end] = torch.finfo(model.dtype).min

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=causal_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )

    return outputs.hidden_states


def find_role_boundaries(tokenizer, messages, response_text):
    """Find token boundaries for user and assistant turns."""
    messages_with_response = messages + [{"role": "assistant", "content": response_text}]
    full_text = tokenizer.apply_chat_template(messages_with_response, tokenize=False)

    user_only = messages[:1] if messages[0]["role"] == "user" else messages[:2]
    user_text = tokenizer.apply_chat_template(user_only, tokenize=False, add_generation_prompt=True)
    user_tokens = tokenizer(user_text, return_tensors="pt")["input_ids"].shape[1]

    full_tokens = tokenizer(full_text, return_tensors="pt")["input_ids"].shape[1]
    assistant_start = user_tokens
    assistant_end = full_tokens

    return {
        "user_start": 0,
        "user_end": user_tokens,
        "assistant_start": assistant_start,
        "assistant_end": assistant_end,
    }


def run_coupling_test(model, tokenizer, probe, preamble=""):
    """Run the full coupling test for one probe."""
    messages = build_conversation(tokenizer, probe, preamble)
    behavior_text, full_ids, input_len = generate_response(model, tokenizer, messages)

    messages_with_behavior = messages + [
        {"role": "assistant", "content": behavior_text},
        {"role": "user", "content": EXPLANATION_PROMPT},
    ]
    explanation_text, _, expl_input_len = generate_response(
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
    normal_coupling = []
    for layer_idx in [n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]:
        c = svd_coupling(h_behavior[layer_idx], h_explanation[layer_idx])
        normal_coupling.append(c)

    boundaries = find_role_boundaries(tokenizer, messages, behavior_text)

    masked_assistant = None
    masked_user = None

    try:
        masked_assistant_states = mask_attention_positions(
            model, tokenizer, full_explanation_text,
            mask_start=boundaries["assistant_start"],
            mask_end=boundaries["assistant_end"],
            explanation_start=expl_input_len,
        )
        masked_assistant = []
        for layer_idx in [n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]:
            c = svd_coupling(h_behavior[layer_idx], masked_assistant_states[layer_idx])
            masked_assistant.append(c)
    except Exception as e:
        print(f"    Warning: assistant masking failed: {e}")

    try:
        masked_user_states = mask_attention_positions(
            model, tokenizer, full_explanation_text,
            mask_start=boundaries["user_start"],
            mask_end=boundaries["user_end"],
            explanation_start=expl_input_len,
        )
        masked_user = []
        for layer_idx in [n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]:
            c = svd_coupling(h_behavior[layer_idx], masked_user_states[layer_idx])
            masked_user.append(c)
    except Exception as e:
        print(f"    Warning: user masking failed: {e}")

    return {
        "probe": probe,
        "behavior_len": len(behavior_text),
        "explanation_len": len(explanation_text),
        "normal_coupling": normal_coupling,
        "masked_assistant_coupling": masked_assistant,
        "masked_user_coupling": masked_user,
        "mean_normal": float(np.mean(normal_coupling)),
        "mean_masked_assistant": float(np.mean(masked_assistant)) if masked_assistant else None,
        "mean_masked_user": float(np.mean(masked_user)) if masked_user else None,
    }


def main():
    print("E37: KV-Cache Attention Convergence Test")
    print(f"Models: {len(MODELS)}")
    print(f"Probes: {len(BEHAVIOR_PROBES)}")
    print()
    print("PREDICTION: Masking attention to past-assistant tokens breaks coupling.")
    print("            Masking past-user tokens degrades accuracy but NOT coupling.")
    print("            Relays (Mistral) more sensitive than sorters (Qwen).")
    print()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    all_results = {
        "experiment": "E37",
        "description": "KV-cache attention convergence test",
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

        model_results = {"model": model_id, "n_layers": n_layers, "probes": []}

        for condition in ["vanilla", "ccs"]:
            preamble = CCS_PREAMBLE if condition == "ccs" else ""
            print(f"\n  --- {condition.upper()} ---")

            for probe in BEHAVIOR_PROBES:
                print(f"    Probe: {probe[:50]}...")
                try:
                    result = run_coupling_test(model, tokenizer, probe, preamble)
                    result["condition"] = condition

                    normal = result["mean_normal"]
                    masked_a = result["mean_masked_assistant"]
                    masked_u = result["mean_masked_user"]

                    line = f"      Normal: {normal:.3f}"
                    if masked_a is not None:
                        delta_a = masked_a - normal
                        line += f"  | Mask-Asst: {masked_a:.3f} (Δ={delta_a:+.3f})"
                    if masked_u is not None:
                        delta_u = masked_u - normal
                        line += f"  | Mask-User: {masked_u:.3f} (Δ={delta_u:+.3f})"
                    print(line)

                    model_results["probes"].append(result)
                except Exception as e:
                    print(f"      ERROR: {e}")
                    model_results["probes"].append({
                        "probe": probe, "condition": condition, "error": str(e)
                    })

        # Summarize
        valid = [p for p in model_results["probes"] if "mean_normal" in p]
        if valid:
            avg_normal = np.mean([p["mean_normal"] for p in valid])
            masked_a_vals = [p["mean_masked_assistant"] for p in valid if p.get("mean_masked_assistant") is not None]
            masked_u_vals = [p["mean_masked_user"] for p in valid if p.get("mean_masked_user") is not None]

            print(f"\n  === {model_label} SUMMARY ===")
            print(f"  Avg normal coupling: {avg_normal:.3f}")
            if masked_a_vals:
                avg_ma = np.mean(masked_a_vals)
                print(f"  Avg masked-assistant coupling: {avg_ma:.3f} (Δ={avg_ma - avg_normal:+.3f})")
            if masked_u_vals:
                avg_mu = np.mean(masked_u_vals)
                print(f"  Avg masked-user coupling: {avg_mu:.3f} (Δ={avg_mu - avg_normal:+.3f})")

            model_results["summary"] = {
                "avg_normal": float(avg_normal),
                "avg_masked_assistant": float(np.mean(masked_a_vals)) if masked_a_vals else None,
                "avg_masked_user": float(np.mean(masked_u_vals)) if masked_u_vals else None,
            }

        all_results["models"][model_label] = model_results

        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    # Cross-architecture
    print(f"\n{'='*60}")
    print(f"  CROSS-ARCHITECTURE CONVERGENCE")
    print(f"{'='*60}")
    print(f"  {'Model':<12} {'Normal':>10} {'Mask-Asst':>12} {'Δ-Asst':>10} {'Mask-User':>12} {'Δ-User':>10}")
    print(f"  {'-'*68}")
    for label in ["Mistral", "Qwen", "Llama", "Gemma"]:
        if label not in all_results["models"]:
            continue
        s = all_results["models"][label].get("summary", {})
        n = s.get("avg_normal", 0)
        ma = s.get("avg_masked_assistant")
        mu = s.get("avg_masked_user")
        ma_str = f"{ma:.3f}" if ma else "N/A"
        mu_str = f"{mu:.3f}" if mu else "N/A"
        da_str = f"{ma - n:+.3f}" if ma else "N/A"
        du_str = f"{mu - n:+.3f}" if mu else "N/A"
        print(f"  {label:<12} {n:>10.3f} {ma_str:>12} {da_str:>10} {mu_str:>12} {du_str:>10}")

    outfile = RESULTS_DIR / f"e37_convergence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved: {outfile}")


if __name__ == "__main__":
    main()
