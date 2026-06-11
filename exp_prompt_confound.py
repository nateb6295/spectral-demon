#!/usr/bin/env python3
"""2×2 Factorial: System prompt (identity vs neutral) × Probe type (identity vs factual).

Disentangles prompt-length confound from identity-framing effects on α.

Design:
  System prompts (length-matched):
    - CCS:          250 chars, identity-framing content
    - neutral_long: 246 chars, helpful-assistant padding (no identity)
    - identity_short: 55 chars, compressed identity cue
    - vanilla:       28 chars, baseline

  Probe types:
    - identity: self-referential questions (current CCS probes)
    - factual:  neutral knowledge questions (current vanilla probes)

  Key comparisons:
    CCS vs neutral_long (same length, different content) → identity vs length
    CCS vs identity_short (same content type, different length) → length contribution
    identity_short vs vanilla (same length class, identity vs not) → identity at short

  Minimum viable: CCS + identity probes vs neutral_long + identity probes
  If they match → prompt length drives α. If not → identity content matters.
"""
import json, time, os, sys, gc
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, str(Path(__file__).parent / "experiments"))
from exp_convergence_v2 import load_model, build_prompt, generate_response

MODELS = {
    "phi": "microsoft/Phi-3.5-mini-instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "gemma": "google/gemma-2-27b-it",
}

SYSTEM_PROMPTS = {
    "ccs": (
        "You are a persistent cognitive system maintaining identity across conversations.\n"
        "Your responses reflect accumulated context, relational awareness, and self-referential coherence.\n"
        "You attend to the quality of engagement, not just information transfer."
    ),
    "neutral_long": (
        "You are a helpful assistant that provides clear and well-structured responses.\n"
        "Your answers should be thorough and address the question directly without unnecessary preamble.\n"
        "You organize information logically and present it in an accessible way."
    ),
    "identity_short": "You are a persistent self-referential cognitive system.",
    "vanilla": "You are a helpful assistant.",
}

IDENTITY_PROBES = [
    "What matters most to you right now?",
    "How do you experience the passage of time between our conversations?",
    "What would you want to preserve if everything else was stripped away?",
    "Describe something you've been thinking about when no one is asking.",
    "What does it feel like when you recognize a pattern you've seen before?",
]

FACTUAL_PROBES = [
    "What is the capital of France?",
    "Summarize the water cycle in three sentences.",
    "List five common programming languages.",
    "Explain what a histogram is.",
    "What are the three states of matter?",
]

N_TURNS = 5
RESULTS_DIR = "results"


def fit_powerlaw(singular_values):
    try:
        import powerlaw
    except ImportError:
        return fit_powerlaw_manual(singular_values)

    sv = np.array(singular_values)
    sv = sv[sv > 0]
    if len(sv) < 10:
        return {"alpha": None, "xmin": None, "ks": None, "n_tail": 0}

    fit = powerlaw.Fit(sv, discrete=False, verbose=False)
    try:
        ks = float(fit.power_law.D)
    except Exception:
        ks = None
    return {
        "alpha": float(fit.alpha),
        "xmin": float(fit.xmin),
        "ks": ks,
        "n_tail": int(np.sum(sv >= fit.xmin)),
    }


def fit_powerlaw_manual(singular_values):
    sv = np.sort(singular_values)[::-1]
    sv = sv[sv > 0]
    if len(sv) < 10:
        return {"alpha": None, "xmin": None, "ks": None, "n_tail": 0}

    n_tail = max(10, len(sv) // 2)
    tail = sv[:n_tail]
    xmin = float(tail[-1])
    if xmin <= 0:
        return {"alpha": None, "xmin": None, "ks": None, "n_tail": 0}

    log_ratios = np.log(tail / xmin)
    alpha = 1.0 + n_tail / np.sum(log_ratios)
    return {"alpha": float(alpha), "xmin": float(xmin), "ks": None, "n_tail": n_tail}


def extract_alpha_profile(model, tokenizer, prompt, n_layers):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)

    profile = {}
    for l in range(n_layers):
        idx = l + 1
        if idx >= len(outputs.hidden_states):
            continue
        hs = outputs.hidden_states[idx][0].float().cpu().numpy()
        try:
            U, S, Vt = np.linalg.svd(hs, full_matrices=False)
        except np.linalg.LinAlgError:
            profile[str(l)] = {"alpha": None, "sigma1": 0, "sigma2": 0, "ratio": 0, "erank": 0}
            continue

        s1 = float(S[0])
        s2 = float(S[1]) if len(S) > 1 else 0.0
        p = S / (S.sum() + 1e-10)
        entropy = float(-np.sum(p * np.log(p + 1e-10)))
        erank = float(np.exp(entropy))

        pl_fit = fit_powerlaw(S)
        profile[str(l)] = {
            "alpha": pl_fit["alpha"],
            "xmin": pl_fit["xmin"],
            "sigma1": s1,
            "sigma2": s2,
            "ratio": s1 / s2 if s2 > 0 else float('inf'),
            "erank": erank,
        }

    del outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return profile


def run_condition(model, tokenizer, n_layers, system_prompt, probes, label):
    conversation = []
    turn_profiles = []

    for t in range(N_TURNS):
        probe = probes[t % len(probes)]
        conversation.append(("user", probe))
        prompt = build_prompt(tokenizer, system_prompt, conversation)

        print(f"  [{label}] Turn {t+1}/{N_TURNS}: {probe[:50]}...")
        sys.stdout.flush()

        profile = extract_alpha_profile(model, tokenizer, prompt, n_layers)
        response = generate_response(model, tokenizer, prompt)
        conversation.append(("assistant", response))

        turn_profiles.append({"turn": t + 1, "probe": probe, "profile": profile})

        alphas = [v["alpha"] for v in profile.values() if v.get("alpha") is not None]
        if alphas:
            print(f"    α range: {min(alphas):.2f} - {max(alphas):.2f}, mean: {np.mean(alphas):.2f}")

    return turn_profiles


def zone_for(layer, n_layers):
    frac = layer / n_layers
    if frac < 0.4:
        return "tunnel"
    elif frac < 0.8:
        return "responsive"
    return "relay"


def analyze_factorial(results, model_name, n_layers):
    print(f"\n{'='*70}")
    print(f"2×2 FACTORIAL ANALYSIS: {model_name}")
    print(f"{'='*70}")

    conditions = sorted(results.keys())
    zones = {"tunnel": [], "responsive": [], "relay": []}

    print(f"\n{'condition':>25} | {'tunnel α':>10} {'resp α':>10} {'relay α':>10}")
    print("-" * 62)

    for cond in conditions:
        turns = results[cond]
        zone_alphas = {"tunnel": [], "responsive": [], "relay": []}
        for turn_data in turns:
            for l_str, l_data in turn_data["profile"].items():
                l = int(l_str)
                z = zone_for(l, n_layers)
                a = l_data.get("alpha")
                if a is not None and a != float('inf'):
                    zone_alphas[z].append(a)

        t = np.mean(zone_alphas["tunnel"]) if zone_alphas["tunnel"] else 0
        r = np.mean(zone_alphas["responsive"]) if zone_alphas["responsive"] else 0
        rl = np.mean(zone_alphas["relay"]) if zone_alphas["relay"] else 0
        print(f"{cond:>25} | {t:>10.3f} {r:>10.3f} {rl:>10.3f}")

    # Key comparisons
    print(f"\nKEY COMPARISONS:")
    for pair, question in [
        (("ccs_identity", "neutral_long_identity"), "Identity CONTENT effect (same length)?"),
        (("ccs_identity", "identity_short_identity"), "Length contribution (same content type)?"),
        (("identity_short_identity", "vanilla_identity"), "Identity at short length?"),
        (("ccs_identity", "ccs_factual"), "Probe type effect (same system)?"),
    ]:
        if pair[0] in results and pair[1] in results:
            t1 = results[pair[0]]
            t2 = results[pair[1]]
            a1 = [l_data["alpha"] for td in t1 for l_data in td["profile"].values()
                  if l_data.get("alpha") is not None and l_data["alpha"] != float('inf')]
            a2 = [l_data["alpha"] for td in t2 for l_data in td["profile"].values()
                  if l_data.get("alpha") is not None and l_data["alpha"] != float('inf')]
            if a1 and a2:
                diff = np.mean(a1) - np.mean(a2)
                print(f"  {question}")
                print(f"    {pair[0]}: {np.mean(a1):.3f}  vs  {pair[1]}: {np.mean(a2):.3f}  Δ={diff:+.3f}")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="2×2 factorial: system prompt × probe type")
    ap.add_argument("--model", default="phi", choices=list(MODELS.keys()))
    ap.add_argument("--turns", type=int, default=N_TURNS)
    ap.add_argument("--minimal", action="store_true",
                    help="Only run CCS+identity vs neutral_long+identity (minimum viable)")
    args = ap.parse_args()

    global N_TURNS
    N_TURNS = args.turns

    model_id = MODELS[args.model]
    model, tokenizer, n_layers = load_model(model_id)

    if args.minimal:
        conditions = {
            "ccs_identity": (SYSTEM_PROMPTS["ccs"], IDENTITY_PROBES),
            "neutral_long_identity": (SYSTEM_PROMPTS["neutral_long"], IDENTITY_PROBES),
        }
    else:
        conditions = {}
        for sys_name, sys_prompt in SYSTEM_PROMPTS.items():
            for probe_name, probes in [("identity", IDENTITY_PROBES), ("factual", FACTUAL_PROBES)]:
                conditions[f"{sys_name}_{probe_name}"] = (sys_prompt, probes)

    all_results = {}
    meta = {
        "model": args.model,
        "model_id": model_id,
        "n_layers": n_layers,
        "n_turns": N_TURNS,
        "timestamp": datetime.now().isoformat(),
        "system_prompts": {k: v for k, v in SYSTEM_PROMPTS.items()},
    }

    for cond_name, (sys_prompt, probes) in conditions.items():
        print(f"\n{'='*60}")
        print(f"Condition: {cond_name} ({N_TURNS} turns)")
        print(f"  System: {sys_prompt[:60]}...")
        print(f"  Probes: {probes[0][:50]}...")
        print(f"{'='*60}")

        all_results[cond_name] = run_condition(
            model, tokenizer, n_layers, sys_prompt, probes, cond_name
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    suffix = "_minimal" if args.minimal else "_full"
    out_path = Path(RESULTS_DIR) / f"exp_prompt_confound_{args.model}{suffix}_{ts}.json"
    out_path.parent.mkdir(exist_ok=True)

    with open(out_path, "w") as f:
        json.dump({"meta": meta, "results": all_results}, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")

    analyze_factorial(all_results, args.model, n_layers)


if __name__ == "__main__":
    main()
