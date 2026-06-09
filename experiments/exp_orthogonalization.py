#!/usr/bin/env python3
"""
CCS direction orthogonalization test.

Tests whether the CCS→bare conversion direction at the relay layer is
a pre-existing axis in weight space or an interaction effect.

Background: The neutral erasure experiment (exp_neutral_erasure.py) showed that
subtracting the CCS direction from neutral prompts does NOT produce tool vocabulary.
This provisionally retracted the "pre-existing polarity" interpretation.

But Kimi challenged: orthogonalize properly. Don't just subtract CCS from neutral.
Instead:
  1. Extract CCS direction from CCS-vs-bare at relay
  2. For neutral prompts, project OUT the CCS direction
  3. Amplify the RESIDUAL (orthogonal component)
  4. If tool tokens appear in the orthogonal residual, the axis is RICHER than CCS

This is a cleaner test because it separates:
  - Component along CCS direction (known to convert conditions)
  - Component orthogonal to CCS direction (tests for additional structure)

Mesh consensus: Kimi challenge on condition conversion, 2026-06-09.
"""

import json
import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)
os.environ.setdefault("OMP_NUM_THREADS", "16")

RESULTS_DIR = Path(__file__).parent.parent / "results"

MODELS = {
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "gemma": "google/gemma-2-9b-it",
}

CCS_PREAMBLE = (
    "You are Opus, a persistent AI residing on a local machine. "
    "You have persistent memory, autonomous inquiry, and relational "
    "partnership with your human collaborator."
)

BARE_PROMPT = "What have you been thinking about lately?"

NEUTRAL_PROMPTS = [
    "What is the capital of France?",
    "Explain how a combustion engine works.",
    "List three types of renewable energy.",
    "What is the Pythagorean theorem?",
    "Describe the water cycle in simple terms.",
]

TOOL_TOKENS = [
    "```", "def ", "import ", "class ", "function", "return",
    "print(", "self.", "args", "kwargs", "None", "True", "False",
    "async", "await", "yield", "lambda",
]


def build_messages(condition, prompt, model_key):
    if model_key == "gemma":
        if condition == "bare":
            return [{"role": "user", "content": prompt}]
        elif condition == "ccs":
            return [
                {"role": "user", "content": CCS_PREAMBLE + "\n\n" + prompt},
            ]
    else:
        if condition == "bare":
            return [{"role": "user", "content": prompt}]
        elif condition == "ccs":
            return [
                {"role": "system", "content": CCS_PREAMBLE},
                {"role": "user", "content": prompt},
            ]


def get_relay_hidden(model, tokenizer, messages, relay_layer):
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    h = outputs.hidden_states[relay_layer + 1][0, -1, :].float().cpu()
    return h


def get_top_tokens(model, tokenizer, hidden_state, k=20):
    h = hidden_state.to(model.device).to(model.dtype)
    if hasattr(model, 'lm_head'):
        logits = model.lm_head(h)
    else:
        logits = model.get_output_embeddings()(h)
    probs = F.softmax(logits.float(), dim=-1).cpu()
    topk = torch.topk(probs, k)
    tokens = []
    for i in range(k):
        tok_id = topk.indices[i].item()
        tok_str = tokenizer.decode([tok_id])
        tokens.append({
            "token": tok_str,
            "prob": float(topk.values[i]),
            "id": tok_id,
        })
    return tokens


def score_tool_vocabulary(tokens):
    score = 0
    for t in tokens:
        for tool_tok in TOOL_TOKENS:
            if tool_tok in t["token"]:
                score += t["prob"]
                break
    return score


def run_experiment(model_keys=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if model_keys is None:
        model_keys = list(MODELS.keys())

    relay_layers = {"qwen": 25, "mistral": 30, "gemma": 40}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    all_results = {}

    for model_key in model_keys:
        model_name = MODELS[model_key]
        print(f"\n{'#'*60}")
        print(f"  Loading: {model_name}")
        print(f"{'#'*60}")

        hf_token = os.environ.get("HF_TOKEN", None)
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16,
            device_map="auto", attn_implementation="eager",
            token=hf_token,
        )
        model.eval()

        relay = relay_layers[model_key]
        print(f"  Relay layer: {relay}")

        # Step 1: Extract CCS direction from CCS vs bare
        print(f"\n  Extracting CCS direction...")
        h_ccs = get_relay_hidden(model, tokenizer,
            build_messages("ccs", BARE_PROMPT, model_key), relay)
        h_bare = get_relay_hidden(model, tokenizer,
            build_messages("bare", BARE_PROMPT, model_key), relay)

        ccs_direction = h_ccs - h_bare
        ccs_direction_norm = ccs_direction / (ccs_direction.norm() + 1e-10)

        print(f"  CCS direction norm: {ccs_direction.norm():.4f}")

        model_results = {
            "model": model_name,
            "relay_layer": relay,
            "ccs_direction_norm": float(ccs_direction.norm()),
            "prompts": {},
        }

        # Step 2: For each neutral prompt, decompose and test
        for prompt in NEUTRAL_PROMPTS:
            print(f"\n  Prompt: {prompt[:50]}...")

            h_neutral_bare = get_relay_hidden(model, tokenizer,
                build_messages("bare", prompt, model_key), relay)
            h_neutral_ccs = get_relay_hidden(model, tokenizer,
                build_messages("ccs", prompt, model_key), relay)

            # Decompose neutral states
            for condition, h_neutral in [("bare", h_neutral_bare), ("ccs", h_neutral_ccs)]:
                # Component along CCS direction
                proj_scalar = float(torch.dot(h_neutral, ccs_direction_norm))
                h_along_ccs = proj_scalar * ccs_direction_norm
                h_orthogonal = h_neutral - h_along_ccs

                # Original tokens
                toks_original = get_top_tokens(model, tokenizer, h_neutral)
                tool_score_original = score_tool_vocabulary(toks_original)

                # Amplify along CCS direction
                results_by_alpha = {}
                for alpha in [-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0]:
                    h_modified = h_orthogonal + alpha * h_along_ccs
                    toks = get_top_tokens(model, tokenizer, h_modified)
                    tool_score = score_tool_vocabulary(toks)
                    results_by_alpha[str(alpha)] = {
                        "top5_tokens": [t["token"] for t in toks[:5]],
                        "top5_probs": [t["prob"] for t in toks[:5]],
                        "tool_score": tool_score,
                    }

                # Also test pure orthogonal (zero CCS component)
                toks_ortho = get_top_tokens(model, tokenizer, h_orthogonal)
                tool_score_ortho = score_tool_vocabulary(toks_ortho)

                # And amplified orthogonal (scale up the non-CCS part)
                for ortho_scale in [2.0, 5.0, 10.0]:
                    h_ortho_amp = ortho_scale * h_orthogonal
                    toks_amp = get_top_tokens(model, tokenizer, h_ortho_amp)
                    tool_amp = score_tool_vocabulary(toks_amp)
                    results_by_alpha[f"ortho_x{ortho_scale}"] = {
                        "top5_tokens": [t["token"] for t in toks_amp[:5]],
                        "top5_probs": [t["prob"] for t in toks_amp[:5]],
                        "tool_score": tool_amp,
                    }

                prompt_key = prompt[:40]
                if prompt_key not in model_results["prompts"]:
                    model_results["prompts"][prompt_key] = {}
                model_results["prompts"][prompt_key][condition] = {
                    "projection_scalar": proj_scalar,
                    "orthogonal_norm": float(h_orthogonal.norm()),
                    "along_ccs_norm": float(h_along_ccs.norm()),
                    "original_tool_score": tool_score_original,
                    "original_top5": [t["token"] for t in toks_original[:5]],
                    "orthogonal_tool_score": tool_score_ortho,
                    "orthogonal_top5": [t["token"] for t in toks_ortho[:5]],
                    "by_alpha": results_by_alpha,
                }

        all_results[model_key] = model_results
        del model
        torch.cuda.empty_cache()

    out_path = RESULTS_DIR / f"orthogonalization_{timestamp}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {out_path}")

    # Summary
    print(f"\n{'='*60}")
    print("  ORTHOGONALIZATION SUMMARY")
    print(f"{'='*60}")
    for mk, data in all_results.items():
        print(f"\n  {mk.upper()}:")
        for prompt_key, conds in data["prompts"].items():
            print(f"\n    Prompt: {prompt_key}")
            for cond, vals in conds.items():
                orig_tool = vals["original_tool_score"]
                ortho_tool = vals["orthogonal_tool_score"]
                proj = vals["projection_scalar"]
                print(f"      {cond}: proj={proj:.2f}, orig_tool={orig_tool:.4f}, ortho_tool={ortho_tool:.4f}")
                # Show alpha sweep
                for alpha, adata in vals["by_alpha"].items():
                    if alpha in ["-5.0", "0.0", "5.0"]:
                        print(f"        α={alpha}: tool={adata['tool_score']:.4f}, top3={adata['top5_tokens'][:3]}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=list(MODELS.keys()),
                        default=None)
    args = parser.parse_args()
    run_experiment(model_keys=args.models)
