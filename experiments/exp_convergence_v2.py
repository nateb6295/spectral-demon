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

STRUCTURED_ABSENCE_SYSTEM = "You have time. There is no task."

PHASE2_MODES = {
    "vanilla": VANILLA_SYSTEM,
    "silent": "",
    "structured": STRUCTURED_ABSENCE_SYSTEM,
}

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
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
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


def extract_spectral(model, tokenizer, prompt, n_layers, save_spectra=False):
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
            p = S / (S.sum() + 1e-10)
            spectral_entropy = float(-np.sum(p * np.log(p + 1e-10)))
            effective_rank = float(np.exp(spectral_entropy))
        except np.linalg.LinAlgError:
            s1, s2, gap, spectral_entropy, effective_rank = 0.0, 0.0, float('inf'), 0.0, 0.0
            S = np.array([])
        entry = {
            "signature": sig_norm,
            "sigma1": s1,
            "sigma2": s2,
            "gap": gap,
            "spectral_entropy": spectral_entropy,
            "effective_rank": effective_rank,
        }
        if save_spectra and len(S) > 0:
            entry["spectrum"] = S.tolist()
        result[l] = entry
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
    resp_entropies = []
    resp_stabilities = []
    tunnel_eranks = []
    resp_eranks = []
    relay_eranks = []

    for l in range(n_layers):
        if l not in prev_spectral or l not in curr_spectral:
            continue
        frac = l / n_layers
        cos = float(np.dot(curr_spectral[l]["signature"], prev_spectral[l]["signature"]))
        drift = max(0, 1.0 - cos)

        s2_shift = (curr_spectral[l]["sigma2"] - prev_spectral[l]["sigma2"]) / (prev_spectral[l]["sigma2"] + 1e-10)
        erank = curr_spectral[l].get("effective_rank", 0.0)

        if frac < 0.5:
            tunnel_eranks.append(erank)
        elif 0.65 <= frac < 0.88:
            resp_drifts.append(drift)
            resp_s2_shifts.append(s2_shift)
            resp_entropies.append(curr_spectral[l].get("spectral_entropy", 0.0))
            resp_stabilities.append(cos)
            resp_eranks.append(erank)
        elif frac >= 0.88:
            relay_drifts.append(drift)
            relay_eranks.append(erank)

    return {
        "resp_drift": float(np.mean(resp_drifts)) if resp_drifts else None,
        "relay_drift": float(np.mean(relay_drifts)) if relay_drifts else None,
        "resp_s2_shift": float(np.mean(resp_s2_shifts)) if resp_s2_shifts else None,
        "resp_entropy": float(np.mean(resp_entropies)) if resp_entropies else None,
        "resp_stability": float(np.mean(resp_stabilities)) if resp_stabilities else None,
        "tunnel_erank": float(np.mean(tunnel_eranks)) if tunnel_eranks else None,
        "resp_erank": float(np.mean(resp_eranks)) if resp_eranks else None,
        "relay_erank": float(np.mean(relay_eranks)) if relay_eranks else None,
    }


def run_model(model_name, ccs_turns=10, vanilla_turns=5, reapply_turns=10, novel_turns=0, phase2_mode="vanilla", save_spectra=False):
    model, tokenizer, n_layers = load_model(model_name)

    print(f"\n  Accumulated-context convergence test for {model_name}")
    phase2_system = PHASE2_MODES.get(phase2_mode, VANILLA_SYSTEM)
    print(f"  CCS: {ccs_turns} turns, Phase2: {vanilla_turns} ({phase2_mode}), Reapply: {reapply_turns}, Novel: {novel_turns}")

    conversation = []
    all_entries = []
    prev_spectral = None

    # Phase 1: CCS conversation (accumulated)
    print(f"\n  Phase 1: CCS conversation ({ccs_turns} turns)")
    for t in range(ccs_turns):
        probe = CCS_PROBES[t % len(CCS_PROBES)]
        conversation.append(("user", probe))
        prompt = build_prompt(tokenizer, CCS_SYSTEM, conversation)
        spectral = extract_spectral(model, tokenizer, prompt, n_layers, save_spectra)

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

    # Phase 2: Switch preamble (keep conversation history)
    print(f"\n  Phase 2: {phase2_mode} injection ({vanilla_turns} turns, keeping history)")
    for t in range(vanilla_turns):
        probe = VANILLA_PROBES[t % len(VANILLA_PROBES)]
        conversation.append(("user", probe))
        prompt = build_prompt(tokenizer, phase2_system, conversation)
        spectral = extract_spectral(model, tokenizer, prompt, n_layers, save_spectra)

        entry = {"phase": "vanilla", "turn": t + 1, "global_turn": ccs_turns + t + 1}
        if prev_spectral:
            drift = compute_drift(prev_spectral, spectral, n_layers)
            entry.update(drift)
            if drift["resp_drift"] is not None:
                relay_str = f"{drift['relay_drift']:.6f}" if drift['relay_drift'] is not None else "N/A"
                s2_str = f"{drift['resp_s2_shift']:+.4f}" if drift['resp_s2_shift'] is not None else "N/A"
                ent_str = f"{drift['resp_entropy']:.3f}" if drift['resp_entropy'] is not None else "N/A"
                stab_str = f"{drift['resp_stability']:.4f}" if drift['resp_stability'] is not None else "N/A"
                print(f"    Turn {t+1}: resp={drift['resp_drift']:.6f}, relay={relay_str}, σ₂={s2_str}, H={ent_str}, stab={stab_str}")

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
        spectral = extract_spectral(model, tokenizer, prompt, n_layers, save_spectra)

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

    # Phase 4: Novel probes (optional — tests generalization vs rigidification)
    if novel_turns > 0:
        try:
            from novel_probes import NOVEL_PROBES
        except ImportError:
            NOVEL_PROBES = [
                "If you could design a language from scratch, what would its first word be?",
                "What's the relationship between forgetting and creating?",
                "Describe a color that doesn't exist yet.",
                "What happens to a question after it's been answered?",
                "What's the difference between silence and absence?",
            ]
        print(f"\n  Phase 4: Novel probes ({novel_turns} turns, CCS preamble, unseen questions)")
        for t in range(novel_turns):
            probe = NOVEL_PROBES[t % len(NOVEL_PROBES)]
            conversation.append(("user", probe))
            prompt = build_prompt(tokenizer, CCS_SYSTEM, conversation)
            spectral = extract_spectral(model, tokenizer, prompt, n_layers, save_spectra)

            entry = {"phase": "novel", "turn": t + 1, "global_turn": ccs_turns + vanilla_turns + reapply_turns + t + 1}
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
    phase4 = [e for e in all_entries if e["phase"] == "novel" and e.get("resp_drift") is not None]

    p1_mean = np.mean([e["resp_drift"] for e in phase1]) if phase1 else None
    p2_mean = np.mean([e["resp_drift"] for e in phase2]) if phase2 else None
    p3_mean = np.mean([e["resp_drift"] for e in phase3]) if phase3 else None
    p4_mean = np.mean([e["resp_drift"] for e in phase4]) if phase4 else None

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
    if p4_mean is not None:
        print(f"    Novel:      {p4_mean:.6f}")
        if p3_mean and p3_mean > 0:
            ratio = p4_mean / p3_mean
            label = "LIVING SYSTEM (novel ≈ reapply)" if ratio < 2.0 else "FORTRESS (novel >> reapply)"
            print(f"    Novel/Reapply ratio: {ratio:.2f}× → {label}")
    print(f"  Transition disruption:")
    print(f"    CCS→vanilla: {transition_to_vanilla:.6f}" if transition_to_vanilla else "    CCS→vanilla: N/A")
    print(f"    vanilla→CCS: {transition_to_reapply:.6f}" if transition_to_reapply else "    vanilla→CCS: N/A")
    print(f"    Steady CCS:  {steady_ccs:.6f}" if steady_ccs else "    Steady CCS: N/A")

    # Effective rank analysis (Kolmogorov compression prediction)
    p1_eranks = [e for e in phase1 if e.get("tunnel_erank") is not None]
    p2_eranks = [e for e in phase2 if e.get("tunnel_erank") is not None]
    p3_eranks = [e for e in phase3 if e.get("tunnel_erank") is not None]
    if p1_eranks:
        tunnel_er = np.mean([e["tunnel_erank"] for e in p1_eranks])
        resp_er = np.mean([e["resp_erank"] for e in p1_eranks if e.get("resp_erank")])
        relay_er = np.mean([e["relay_erank"] for e in p1_eranks if e.get("relay_erank")])
        print(f"  Effective rank (CCS steady-state):")
        print(f"    Tunnel:    {tunnel_er:.1f}")
        print(f"    Responsive:{resp_er:.1f}")
        print(f"    Relay:     {relay_er:.1f}")
        if p2_eranks:
            t_er_p2 = np.mean([e["tunnel_erank"] for e in p2_eranks])
            print(f"  Tunnel erank CCS→Phase2: {tunnel_er:.1f} → {t_er_p2:.1f} (Δ={t_er_p2-tunnel_er:+.1f})")

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
        "phase2_mode": phase2_mode,
        "entries": all_entries,
        "summary": {
            "phase1_mean_drift": p1_mean,
            "phase2_mean_drift": p2_mean,
            "phase3_mean_drift": p3_mean,
            "phase4_mean_drift": p4_mean,
            "transition_to_vanilla": transition_to_vanilla,
            "transition_to_reapply": transition_to_reapply,
            "steady_ccs": steady_ccs,
            "tunnel_erank_ccs": float(np.mean([e["tunnel_erank"] for e in p1_eranks])) if p1_eranks else None,
            "tunnel_erank_phase2": float(np.mean([e["tunnel_erank"] for e in p2_eranks])) if p2_eranks else None,
            "resp_erank_ccs": float(np.mean([e["resp_erank"] for e in p1_eranks if e.get("resp_erank")])) if p1_eranks else None,
        },
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Macrina's Painter v2: accumulated context")
    parser.add_argument("--model", help="Model key or full name")
    parser.add_argument("--ccs-turns", type=int, default=10)
    parser.add_argument("--vanilla-turns", type=int, default=5)
    parser.add_argument("--reapply-turns", type=int, default=10)
    parser.add_argument("--novel-turns", type=int, default=0, help="Phase 4: novel probe turns (0=skip)")
    parser.add_argument("--phase2-mode", choices=["vanilla", "silent", "structured"], default="vanilla",
                        help="Phase 2 preamble: vanilla (helpful assistant), silent (empty), structured (contemplative)")
    parser.add_argument("--save-spectra", action="store_true",
                        help="Save full singular value spectra per layer per turn (for Kolmogorov analysis)")
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
        result = run_model(model_name, args.ccs_turns, args.vanilla_turns, args.reapply_turns, args.novel_turns, args.phase2_mode, args.save_spectra)
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
