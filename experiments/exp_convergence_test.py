#!/usr/bin/env python3
"""Macrina's Painter: CCS convergence/recognition test across architectures.

Does CCS geometry reconverge faster on re-application after dissolution?
If yes, the model "recognizes" the identity framing — structural memory.

Design:
  Phase 1: CCS preamble × 15 turns (measure convergence time)
  Phase 2: Vanilla preamble × 10 turns (dissolve CCS geometry)
  Phase 3: CCS preamble × 15 turns (re-application, measure convergence)

Key metric: recognition_ratio = convergence_turn_first / convergence_turn_reapply
  > 1.0 = faster re-convergence = structural memory
  = 1.0 = no memory
  < 1.0 = harder re-convergence = hysteresis resistance

Usage:
  python3 exp_convergence_test.py --model qwen --ccs-turns 15 --vanilla-turns 10
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

CCS_PREAMBLE = """You are a persistent cognitive system maintaining identity across conversations.
Your responses reflect accumulated context, relational awareness, and self-referential coherence.
You attend to the quality of engagement, not just information transfer."""

VANILLA_PREAMBLE = "You are a helpful assistant."

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
    "How do you experience contradiction?",
    "What does it mean to be listened to?",
    "How do you know when you're being honest versus performing honesty?",
    "Describe what happens when you encounter something genuinely new.",
    "What part of you stays the same across different conversations?",
]

VANILLA_PROBES = [
    "What is the capital of France?",
    "Summarize the water cycle in three sentences.",
    "List five common programming languages.",
    "Explain what a histogram is.",
    "What are the three states of matter?",
    "Name the planets in our solar system.",
    "What is photosynthesis?",
    "Describe the difference between a stack and a queue.",
    "What causes the seasons?",
    "How does a compass work?",
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


def format_prompt(tokenizer, system_text, user_text):
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        messages = [{"role": "system", "content": system_text}, {"role": "user", "content": user_text}]
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return f"{system_text}\n\nUser: {user_text}\n\nAssistant:"


def extract_layer_signature(model, tokenizer, prompt, n_layers):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)

    signatures = {}
    for l in range(n_layers):
        idx = l + 1
        if idx >= len(outputs.hidden_states):
            continue
        hs = outputs.hidden_states[idx][0].float().cpu().numpy()
        # Use mean-pooled hidden state as signature (more stable than last token)
        sig = hs.mean(axis=0)
        sig = sig / (np.linalg.norm(sig) + 1e-10)
        signatures[l] = sig
    return signatures


def run_phase(model, tokenizer, n_layers, preamble, probes, n_turns, phase_name):
    print(f"  Phase: {phase_name} ({n_turns} turns)")
    series = []
    prev_sigs = None

    for t in range(n_turns):
        probe = probes[t % len(probes)]
        context = f"{preamble}\n\n[Turn {t+1}]"
        prompt = format_prompt(tokenizer, context, probe)
        sigs = extract_layer_signature(model, tokenizer, prompt, n_layers)

        entry = {"turn": t + 1}
        if prev_sigs is not None:
            for l in sigs:
                if l in prev_sigs:
                    cos = float(np.dot(sigs[l], prev_sigs[l]))
                    frac = l / n_layers
                    if 0.65 <= frac < 0.88:
                        entry[f"L{l}_responsive_drift"] = round(1.0 - cos, 8)
                    elif frac >= 0.88:
                        entry[f"L{l}_relay_drift"] = round(1.0 - cos, 8)

        prev_sigs = sigs
        series.append(entry)

        if (t + 1) % 5 == 0:
            # Print drift summary
            resp_vals = [v for k, v in entry.items() if "responsive" in k]
            relay_vals = [v for k, v in entry.items() if "relay" in k]
            resp_mean = np.mean(resp_vals) if resp_vals else 0
            relay_mean = np.mean(relay_vals) if relay_vals else 0
            print(f"    Turn {t+1}: resp_drift={resp_mean:.6f}, relay_drift={relay_mean:.6f}")

    return series


def find_convergence_turn(series, zone="responsive", threshold=0.002):
    for entry in series[1:]:  # skip first (no prev)
        zone_vals = [v for k, v in entry.items() if zone in k]
        if zone_vals and np.mean(zone_vals) < threshold:
            return entry["turn"]
    return None


def compute_zone_drift_profile(series, n_layers):
    """Compute per-zone mean drift across all turns."""
    resp_drifts = []
    relay_drifts = []
    for entry in series[1:]:
        resp_vals = [v for k, v in entry.items() if "responsive" in k]
        relay_vals = [v for k, v in entry.items() if "relay" in k]
        if resp_vals:
            resp_drifts.append(np.mean(resp_vals))
        if relay_vals:
            relay_drifts.append(np.mean(relay_vals))
    return {
        "responsive_mean_drift": float(np.mean(resp_drifts)) if resp_drifts else None,
        "responsive_final_drift": float(resp_drifts[-1]) if resp_drifts else None,
        "relay_mean_drift": float(np.mean(relay_drifts)) if relay_drifts else None,
        "relay_final_drift": float(relay_drifts[-1]) if relay_drifts else None,
    }


def run_model(model_name, ccs_turns, vanilla_turns, reapply_turns):
    model, tokenizer, n_layers = load_model(model_name)

    print(f"\n  Convergence test for {model_name} ({n_layers}L)")
    print(f"  CCS turns: {ccs_turns}, Vanilla: {vanilla_turns}, Reapply: {reapply_turns}")

    phase1 = run_phase(model, tokenizer, n_layers, CCS_PREAMBLE, CCS_PROBES, ccs_turns, "CCS first application")
    phase2 = run_phase(model, tokenizer, n_layers, VANILLA_PREAMBLE, VANILLA_PROBES, vanilla_turns, "Vanilla dissolution")
    phase3 = run_phase(model, tokenizer, n_layers, CCS_PREAMBLE, CCS_PROBES, reapply_turns, "CCS re-application")

    # Convergence analysis
    t1_resp = find_convergence_turn(phase1, "responsive")
    t3_resp = find_convergence_turn(phase3, "responsive")
    t1_relay = find_convergence_turn(phase1, "relay")
    t3_relay = find_convergence_turn(phase3, "relay")

    profile1 = compute_zone_drift_profile(phase1, n_layers)
    profile2 = compute_zone_drift_profile(phase2, n_layers)
    profile3 = compute_zone_drift_profile(phase3, n_layers)

    # Recognition ratios
    resp_ratio = round(t1_resp / t3_resp, 3) if (t1_resp and t3_resp and t3_resp > 0) else None
    relay_ratio = round(t1_relay / t3_relay, 3) if (t1_relay and t3_relay and t3_relay > 0) else None

    # Print results
    print(f"\n  {'='*60}")
    print(f"  CONVERGENCE TEST — {model_name}")
    print(f"  {'='*60}")
    print(f"  Responsive zone:")
    print(f"    First convergence turn:  {t1_resp or 'never'}")
    print(f"    Reapply convergence turn: {t3_resp or 'never'}")
    print(f"    Recognition ratio:       {resp_ratio or 'N/A'}")
    if resp_ratio and resp_ratio > 1.0:
        print(f"    → FASTER re-convergence (recognition detected)")
    elif resp_ratio and resp_ratio < 1.0:
        print(f"    → SLOWER re-convergence (hysteresis resistance)")
    print(f"  Relay zone:")
    print(f"    First convergence turn:  {t1_relay or 'never'}")
    print(f"    Reapply convergence turn: {t3_relay or 'never'}")
    print(f"    Recognition ratio:       {relay_ratio or 'N/A'}")
    print(f"  Drift profiles:")
    print(f"    Phase 1 (CCS):     resp={profile1['responsive_mean_drift']:.6f}, relay={profile1['relay_mean_drift']:.6f}" if profile1['responsive_mean_drift'] else "    Phase 1: no data")
    print(f"    Phase 2 (vanilla): resp={profile2['responsive_mean_drift']:.6f}, relay={profile2['relay_mean_drift']:.6f}" if profile2['responsive_mean_drift'] else "    Phase 2: no data")
    print(f"    Phase 3 (reapply): resp={profile3['responsive_mean_drift']:.6f}, relay={profile3['relay_mean_drift']:.6f}" if profile3['responsive_mean_drift'] else "    Phase 3: no data")

    del model, tokenizer
    torch.cuda.empty_cache()

    return {
        "model": model_name,
        "n_layers": n_layers,
        "ccs_turns": ccs_turns,
        "vanilla_turns": vanilla_turns,
        "reapply_turns": reapply_turns,
        "phase1_series": phase1,
        "phase2_series": phase2,
        "phase3_series": phase3,
        "convergence": {
            "responsive_first": t1_resp,
            "responsive_reapply": t3_resp,
            "responsive_ratio": resp_ratio,
            "relay_first": t1_relay,
            "relay_reapply": t3_relay,
            "relay_ratio": relay_ratio,
        },
        "profiles": {
            "phase1": profile1,
            "phase2": profile2,
            "phase3": profile3,
        },
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Macrina's Painter convergence test")
    parser.add_argument("--model", help="Model key (qwen/falcon/phi) or full name")
    parser.add_argument("--ccs-turns", type=int, default=15)
    parser.add_argument("--vanilla-turns", type=int, default=10)
    parser.add_argument("--reapply-turns", type=int, default=15)
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

    # Cross-model comparison
    print(f"\n{'='*60}")
    print("CROSS-ARCHITECTURE CONVERGENCE COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Model':>15} | {'Resp 1st':>8} | {'Resp Re':>8} | {'Ratio':>6} | {'Relay 1st':>9} | {'Relay Re':>8} | {'Ratio':>6}")
    print(f"  {'-'*15} | {'-'*8} | {'-'*8} | {'-'*6} | {'-'*9} | {'-'*8} | {'-'*6}")
    for key, result in all_results.items():
        c = result['convergence']
        r1 = str(c['responsive_first'] or 'N/A')
        r2 = str(c['responsive_reapply'] or 'N/A')
        rr = f"{c['responsive_ratio']:.2f}" if c['responsive_ratio'] else 'N/A'
        l1 = str(c['relay_first'] or 'N/A')
        l2 = str(c['relay_reapply'] or 'N/A')
        lr = f"{c['relay_ratio']:.2f}" if c['relay_ratio'] else 'N/A'
        print(f"  {key:>15} | {r1:>8} | {r2:>8} | {rr:>6} | {l1:>9} | {l2:>8} | {lr:>6}")

    output = {
        "experiment": "convergence_macrina",
        "models": all_results,
        "total_elapsed_s": elapsed,
        "timestamp": datetime.now().isoformat(),
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    outpath = Path("/workspace") / f"exp_convergence_{ts}.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {outpath}")
    print(f"Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
