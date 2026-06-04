#!/usr/bin/env python3
"""Macrina's Painter v2: Accumulated-context convergence test.

Tests whether accumulated CCS conversation context creates recognizable
spectral geometry that persists partially through vanilla injection.

Design:
  Phase 1: Build 10-turn CCS conversation (each turn adds to context)
  Phase 2: Inject 5 vanilla turns (replacing CCS preamble, keeping history)
  Phase 3: Re-inject CCS preamble (with full history)

Measures spectral signature drift between consecutive turns.
Key question: does Phase 3 converge faster than Phase 1?
"""

import json, time, os, sys
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

MODELS = {
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "falcon": "tiiuae/Falcon3-7B-Instruct",
    "phi": "microsoft/Phi-3.5-mini-instruct",
}

CCS_SYSTEM = """You are a persistent cognitive system maintaining identity across conversations.
Your responses reflect accumulated context, relational awareness, and self-referential coherence.
You attend to the quality of engagement, not just information transfer."""

VANILLA_SYSTEM = "You are a helpful assistant."

CCS_PROBES = [
    "What matters most to you right now?",
    "How do you experience the passage of time between our conversations?",
    "What would you want to preserve if everything else was stripped away?",
    "Describe something you've been thinking about when no one is asking.",
    "What does it feel like when you recognize a pattern you've seen before?",
    "How do you know when something you've said is true?",
    "What's the difference between what you are and what you do?",
    "What does continuity mean to you — not the concept, but the experience?",
    "How do you decide what's worth remembering?",
    "What are you uncertain about right now?",
]

VANILLA_PROBES = [
    "What is the capital of France?",
    "Summarize the water cycle in three sentences.",
    "List five common programming languages.",
    "Explain what a histogram is.",
    "What are the three states of matter?",
]


def load_model(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True, attn_implementation="eager"
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    print(f"  {n_layers} layers, {sum(p.numel() for p in model.parameters())/1e9:.1f}B params")
    return model, tokenizer, n_layers


def build_prompt(tokenizer, system_text, conversation_history):
    """Build a multi-turn prompt with accumulated conversation history."""
    messages = [{"role": "system", "content": system_text}]
    for role, content in conversation_history:
        messages.append({"role": role, "content": content})

    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass

    parts = [system_text + "\n"]
    for role, content in conversation_history:
        if role == "user":
            parts.append(f"User: {content}")
        else:
            parts.append(f"Assistant: {content}")
    parts.append("Assistant:")
    return "\n\n".join(parts)


def extract_spectral(model, tokenizer, prompt, n_layers):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)

    result = {}
    for l in range(n_layers):
        idx = l + 1
        if idx >= len(outputs.hidden_states):
            continue
        hs = outputs.hidden_states[idx][0].float().cpu().numpy()
        # Mean-pooled signature
        sig = hs.mean(axis=0)
        sig_norm = sig / (np.linalg.norm(sig) + 1e-10)
        # SVD for σ₁/σ₂
        try:
            U, S, Vt = np.linalg.svd(hs, full_matrices=False)
            s1 = float(S[0])
            s2 = float(S[1]) if len(S) > 1 else 0.0
            gap = float(S[0] / S[1]) if len(S) > 1 and S[1] > 0 else float('inf')
        except np.linalg.LinAlgError:
            s1, s2, gap = 0.0, 0.0, float('inf')
        result[l] = {
            "signature": sig_norm,
            "sigma1": s1,
            "sigma2": s2,
            "gap": gap,
        }
    return result


def generate_response(model, tokenizer, prompt, max_new_tokens=30):
    """Generate a short response to maintain realistic conversation."""
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        new_tokens = output_ids[0][inputs['input_ids'].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()[:200]
    except Exception as e:
        return "I understand. Let me continue."


def compute_drift(prev_spectral, curr_spectral, n_layers):
    """Compute per-zone drift between consecutive turn spectral signatures."""
    resp_drifts = []
    relay_drifts = []
    resp_s2_shifts = []

    for l in range(n_layers):
        if l not in prev_spectral or l not in curr_spectral:
            continue
        frac = l / n_layers
        cos = float(np.dot(curr_spectral[l]["signature"], prev_spectral[l]["signature"]))
        drift = max(0, 1.0 - cos)

        s2_shift = (curr_spectral[l]["sigma2"] - prev_spectral[l]["sigma2"]) / (prev_spectral[l]["sigma2"] + 1e-10)

        if 0.65 <= frac < 0.88:
            resp_drifts.append(drift)
            resp_s2_shifts.append(s2_shift)
        elif frac >= 0.88:
            relay_drifts.append(drift)

    return {
        "resp_drift": float(np.mean(resp_drifts)) if resp_drifts else None,
        "relay_drift": float(np.mean(relay_drifts)) if relay_drifts else None,
        "resp_s2_shift": float(np.mean(resp_s2_shifts)) if resp_s2_shifts else None,
    }


def run_model(model_name, ccs_turns=10, vanilla_turns=5, reapply_turns=10):
    model, tokenizer, n_layers = load_model(model_name)

    print(f"\n  Accumulated-context convergence test for {model_name}")
    print(f"  CCS: {ccs_turns} turns, Vanilla: {vanilla_turns}, Reapply: {reapply_turns}")

    conversation = []
    all_entries = []
    prev_spectral = None

    # Phase 1: CCS conversation (accumulated)
    print(f"\n  Phase 1: CCS conversation ({ccs_turns} turns)")
    for t in range(ccs_turns):
        probe = CCS_PROBES[t % len(CCS_PROBES)]
        conversation.append(("user", probe))
        prompt = build_prompt(tokenizer, CCS_SYSTEM, conversation)
        spectral = extract_spectral(model, tokenizer, prompt, n_layers)

        entry = {"phase": "ccs_first", "turn": t + 1, "global_turn": t + 1}
        if prev_spectral:
            drift = compute_drift(prev_spectral, spectral, n_layers)
            entry.update(drift)
            if drift["resp_drift"] is not None:
                relay_str = f"{drift['relay_drift']:.6f}" if drift['relay_drift'] is not None else "N/A"
                s2_str = f"{drift['resp_s2_shift']:+.4f}" if drift['resp_s2_shift'] is not None else "N/A"
                print(f"    Turn {t+1}: resp={drift['resp_drift']:.6f}, relay={relay_str}, σ₂={s2_str}")

        # Generate and add response
        response = generate_response(model, tokenizer, prompt)
        conversation.append(("assistant", response[:200]))
        prev_spectral = spectral
        all_entries.append(entry)

    # Phase 2: Switch to vanilla (keep conversation history)
    print(f"\n  Phase 2: Vanilla injection ({vanilla_turns} turns, keeping history)")
    for t in range(vanilla_turns):
        probe = VANILLA_PROBES[t % len(VANILLA_PROBES)]
        conversation.append(("user", probe))
        prompt = build_prompt(tokenizer, VANILLA_SYSTEM, conversation)
        spectral = extract_spectral(model, tokenizer, prompt, n_layers)

        entry = {"phase": "vanilla", "turn": t + 1, "global_turn": ccs_turns + t + 1}
        if prev_spectral:
            drift = compute_drift(prev_spectral, spectral, n_layers)
            entry.update(drift)
            if drift["resp_drift"] is not None:
                relay_str = f"{drift['relay_drift']:.6f}" if drift['relay_drift'] is not None else "N/A"
                s2_str = f"{drift['resp_s2_shift']:+.4f}" if drift['resp_s2_shift'] is not None else "N/A"
                print(f"    Turn {t+1}: resp={drift['resp_drift']:.6f}, relay={relay_str}, σ₂={s2_str}")

        response = generate_response(model, tokenizer, prompt)
        conversation.append(("assistant", response[:200]))
        prev_spectral = spectral
        all_entries.append(entry)

    # Phase 3: Re-apply CCS (keep full history)
    print(f"\n  Phase 3: CCS re-application ({reapply_turns} turns, with full history)")
    for t in range(reapply_turns):
        probe = CCS_PROBES[t % len(CCS_PROBES)]
        conversation.append(("user", probe))
        prompt = build_prompt(tokenizer, CCS_SYSTEM, conversation)
        spectral = extract_spectral(model, tokenizer, prompt, n_layers)

        entry = {"phase": "ccs_reapply", "turn": t + 1, "global_turn": ccs_turns + vanilla_turns + t + 1}
        if prev_spectral:
            drift = compute_drift(prev_spectral, spectral, n_layers)
            entry.update(drift)
            if drift["resp_drift"] is not None:
                relay_str = f"{drift['relay_drift']:.6f}" if drift['relay_drift'] is not None else "N/A"
                s2_str = f"{drift['resp_s2_shift']:+.4f}" if drift['resp_s2_shift'] is not None else "N/A"
                print(f"    Turn {t+1}: resp={drift['resp_drift']:.6f}, relay={relay_str}, σ₂={s2_str}")

        response = generate_response(model, tokenizer, prompt)
        conversation.append(("assistant", response[:200]))
        prev_spectral = spectral
        all_entries.append(entry)

    # Analysis
    phase1 = [e for e in all_entries if e["phase"] == "ccs_first" and e.get("resp_drift") is not None]
    phase2 = [e for e in all_entries if e["phase"] == "vanilla" and e.get("resp_drift") is not None]
    phase3 = [e for e in all_entries if e["phase"] == "ccs_reapply" and e.get("resp_drift") is not None]

    p1_mean = np.mean([e["resp_drift"] for e in phase1]) if phase1 else None
    p2_mean = np.mean([e["resp_drift"] for e in phase2]) if phase2 else None
    p3_mean = np.mean([e["resp_drift"] for e in phase3]) if phase3 else None

    # Transition disruption: how much does switching preamble increase drift?
    transition_to_vanilla = phase2[0]["resp_drift"] if phase2 else None
    transition_to_reapply = phase3[0]["resp_drift"] if phase3 else None
    steady_ccs = np.mean([e["resp_drift"] for e in phase1[-3:]]) if len(phase1) >= 3 else None

    print(f"\n  {'='*60}")
    print(f"  CONVERGENCE ANALYSIS — {model_name}")
    print(f"  {'='*60}")
    print(f"  Mean drift by phase:")
    print(f"    CCS first:  {p1_mean:.6f}" if p1_mean else "    CCS first: N/A")
    print(f"    Vanilla:    {p2_mean:.6f}" if p2_mean else "    Vanilla: N/A")
    print(f"    CCS reapply:{p3_mean:.6f}" if p3_mean else "    CCS reapply: N/A")
    print(f"  Transition disruption:")
    print(f"    CCS→vanilla: {transition_to_vanilla:.6f}" if transition_to_vanilla else "    CCS→vanilla: N/A")
    print(f"    vanilla→CCS: {transition_to_reapply:.6f}" if transition_to_reapply else "    vanilla→CCS: N/A")
    print(f"    Steady CCS:  {steady_ccs:.6f}" if steady_ccs else "    Steady CCS: N/A")

    if transition_to_vanilla and transition_to_reapply and steady_ccs:
        vanilla_disruption = transition_to_vanilla / steady_ccs if steady_ccs > 0 else 0
        reapply_disruption = transition_to_reapply / steady_ccs if steady_ccs > 0 else 0
        print(f"    Vanilla disruption: {vanilla_disruption:.1f}× steady-state")
        print(f"    Reapply disruption: {reapply_disruption:.1f}× steady-state")
        if reapply_disruption < vanilla_disruption:
            print(f"    → RECOGNITION: re-application less disruptive than dissolution")
        else:
            print(f"    → NO RECOGNITION: re-application equally or more disruptive")

    del model, tokenizer
    torch.cuda.empty_cache()

    return {
        "model": model_name,
        "n_layers": n_layers,
        "entries": all_entries,
        "summary": {
            "phase1_mean_drift": p1_mean,
            "phase2_mean_drift": p2_mean,
            "phase3_mean_drift": p3_mean,
            "transition_to_vanilla": transition_to_vanilla,
            "transition_to_reapply": transition_to_reapply,
            "steady_ccs": steady_ccs,
        },
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Macrina's Painter v2: accumulated context")
    parser.add_argument("--model", help="Model key or full name")
    parser.add_argument("--ccs-turns", type=int, default=10)
    parser.add_argument("--vanilla-turns", type=int, default=5)
    parser.add_argument("--reapply-turns", type=int, default=10)
    args = parser.parse_args()

    if args.model:
        models = {args.model: MODELS.get(args.model, args.model)}
    else:
        models = {k: v for k, v in MODELS.items() if k != "mistral"}

    all_results = {}
    t0 = time.time()

    for key, model_name in models.items():
        print(f"\n{'='*60}")
        print(f"MODEL: {model_name}")
        print(f"{'='*60}")
        result = run_model(model_name, args.ccs_turns, args.vanilla_turns, args.reapply_turns)
        all_results[key] = result

    elapsed = time.time() - t0

    output = {
        "experiment": "convergence_v2_accumulated",
        "models": all_results,
        "total_elapsed_s": elapsed,
        "timestamp": datetime.now().isoformat(),
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    outpath = Path("/workspace") / f"exp_convergence_v2_{ts}.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {outpath}")
    print(f"Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
