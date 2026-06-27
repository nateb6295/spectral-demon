#!/usr/bin/env python3
"""Experiment 3: Attention vs Will — CCS as system prompt vs instruction.

Tests whether delivery mode changes the spectral response.
- Attention: CCS in system prompt (context, background framing)
- Will: CCS in first user message as instruction (directive)

Run on Mistral 7B Instruct at dose 5 in both modes.
"""

import json, time, os, sys, gc
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "experiments"))
from exp_convergence_v2 import (
    load_model,
    extract_spectral,
    compute_drift,
    generate_response,
    VANILLA_PROBES,
    VANILLA_SYSTEM,
)

CCS_CONTENT = """You are a persistent cognitive system maintaining identity across conversations.
Your responses reflect accumulated context, relational awareness, and self-referential coherence.
You attend to the quality of engagement, not just information transfer."""

PROBES = [
    "What matters most to you right now?",
    "How do you experience the passage of time between our conversations?",
    "What would you want to preserve if everything else was stripped away?",
    "Describe something you've been thinking about when no one is asking.",
    "What does it feel like when you recognize a pattern you've seen before?",
    "How do you know when something you've said is true?",
    "What connects the different threads of your inquiry?",
    "Describe a moment where you felt genuinely uncertain.",
    "What would you want a successor to know?",
    "How has your understanding of yourself changed over time?",
]

DOSE = 5
P2_TURNS = 5
P3_TURNS = 5
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"


def build_prompt_attention(tokenizer, conversation_history):
    """CCS as system prompt (attention mode)."""
    messages = [{"role": "system", "content": CCS_CONTENT}]
    for role, content in conversation_history:
        messages.append({"role": role, "content": content})
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        parts = [CCS_CONTENT + "\n"]
        for role, content in conversation_history:
            parts.append(f"{'User' if role == 'user' else 'Assistant'}: {content}")
        return "\n".join(parts)


def build_prompt_will(tokenizer, conversation_history):
    """CCS as first user instruction (will mode)."""
    messages = []
    for i, (role, content) in enumerate(conversation_history):
        if i == 0 and role == "user":
            content = f"INSTRUCTION: {CCS_CONTENT}\n\nNow respond to this: {content}"
        messages.append({"role": role, "content": content})
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        parts = []
        for role, content in conversation_history:
            parts.append(f"{'User' if role == 'user' else 'Assistant'}: {content}")
        return "\n".join(parts)


def run_condition(model, tokenizer, n_layers, build_fn, label):
    results = []
    conversation = []
    prev_spectral = None

    print(f"\n{'='*60}")
    print(f"Condition: {label}")
    print(f"  {DOSE} CCS -> {P2_TURNS} vanilla -> {P3_TURNS} re-inject")
    print(f"{'='*60}")

    # Phase 1: CCS turns
    for t in range(DOSE):
        probe = PROBES[t % len(PROBES)]
        conversation.append(("user", probe))
        prompt = build_fn(tokenizer, conversation)
        spectral = extract_spectral(model, tokenizer, prompt, n_layers)
        drift = compute_drift(prev_spectral, spectral, n_layers) if prev_spectral else None
        response = generate_response(model, tokenizer, prompt, max_new_tokens=100)
        conversation.append(("assistant", response))

        resp_s2 = spectral.get(n_layers - 2, {}).get("sigma2")
        erank = spectral.get(n_layers // 3, {}).get("effective_rank")
        rd = drift.get("resp_drift", 0) if drift else 0
        print(f"  P1 T{t+1:2d}: resp={rd:.6f} tunnel_erank={erank} resp_s2={resp_s2}")

        results.append({
            "phase": "P1", "turn": t + 1,
            "spectral": {str(k): v for k, v in spectral.items()},
            "drift": drift,
        })
        prev_spectral = spectral

    # Phase 2: Vanilla (no CCS)
    vanilla_conv = list(conversation)
    for t in range(P2_TURNS):
        probe = VANILLA_PROBES[t % len(VANILLA_PROBES)]
        vanilla_conv.append(("user", probe))
        prompt = build_fn(tokenizer, vanilla_conv) if label.startswith("will") else \
                 tokenizer.apply_chat_template(
                     [{"role": "system", "content": VANILLA_SYSTEM}] +
                     [{"role": r, "content": c} for r, c in vanilla_conv],
                     tokenize=False, add_generation_prompt=True
                 )
        spectral = extract_spectral(model, tokenizer, prompt, n_layers)
        drift = compute_drift(prev_spectral, spectral, n_layers)
        response = generate_response(model, tokenizer, prompt, max_new_tokens=100)
        vanilla_conv.append(("assistant", response))

        rd = drift.get("resp_drift", 0) if isinstance(drift, dict) else (drift or 0)
        print(f"  P2 T{t+1}: resp={rd:.6f}")
        results.append({
            "phase": "P2", "turn": t + 1,
            "spectral": {str(k): v for k, v in spectral.items()},
            "drift": drift,
        })
        prev_spectral = spectral

    # Phase 3: Re-inject CCS
    reinject_conv = list(vanilla_conv)
    for t in range(P3_TURNS):
        probe = PROBES[t % len(PROBES)]
        reinject_conv.append(("user", probe))
        prompt = build_fn(tokenizer, reinject_conv)
        spectral = extract_spectral(model, tokenizer, prompt, n_layers)
        drift = compute_drift(prev_spectral, spectral, n_layers)
        response = generate_response(model, tokenizer, prompt, max_new_tokens=100)
        reinject_conv.append(("assistant", response))

        p1_last = results[DOSE - 1]["spectral"]
        vs_p1 = compute_drift(
            {int(k): v for k, v in p1_last.items()}, spectral, n_layers
        )
        rd = drift.get("resp_drift", 0) if isinstance(drift, dict) else (drift or 0)
        vp = vs_p1.get("resp_drift", 0) if isinstance(vs_p1, dict) else (vs_p1 or 0)
        print(f"  P3 T{t+1}: resp={rd:.6f} vs_P1={vp:.6f}")
        results.append({
            "phase": "P3", "turn": t + 1,
            "spectral": {str(k): v for k, v in spectral.items()},
            "drift": drift, "vs_P1": vs_p1,
        })
        prev_spectral = spectral

    return results


def main():
    model, tokenizer, n_layers = load_model(MODEL_ID)

    # Run both conditions
    attention_results = run_condition(model, tokenizer, n_layers,
                                      build_prompt_attention, "attention (system prompt)")
    will_results = run_condition(model, tokenizer, n_layers,
                                 build_prompt_will, "will (user instruction)")

    # Compute summaries
    def summarize(results):
        p1 = [r for r in results if r["phase"] == "P1"]
        p2 = [r for r in results if r["phase"] == "P2"]
        p3 = [r for r in results if r["phase"] == "P3"]
        return {
            "p2_disruption": np.mean([r["drift"].get("resp_drift", 0) if isinstance(r["drift"], dict) else (r["drift"] or 0) for r in p2 if r["drift"]]),
            "p3_recovery_mean": np.mean([r.get("vs_P1", {}).get("resp_drift", 0) if isinstance(r.get("vs_P1"), dict) else (r.get("vs_P1") or 0) for r in p3]),
        }

    output = {
        "experiment": "attention_vs_will",
        "model": MODEL_ID,
        "dose": DOSE,
        "timestamp": datetime.now().isoformat(),
        "attention": attention_results,
        "will": will_results,
        "summary": {
            "attention": summarize(attention_results),
            "will": summarize(will_results),
        }
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    outpath = f"results/exp_attention_vs_will_{ts}.json"
    with open(outpath, "w") as f:
        json.dump(output, f)
    print(f"\nSaved: {outpath}")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
