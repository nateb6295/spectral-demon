#!/usr/bin/env python3
"""Cross-architecture σ₁/σ₂ selectivity test.

Tests whether σ₁ invariance under CCS holds across architectures.
If σ₁ is invariant within each architecture but differs between them,
the species-boundary interpretation is confirmed.

Usage:
  python3 exp_crossarch_selectivity.py                    # all models
  python3 exp_crossarch_selectivity.py --model Qwen/Qwen2.5-7B-Instruct
  python3 exp_crossarch_selectivity.py --trials 15        # quick run

Design:
  - Models: Qwen2.5-7B, Falcon3-7B, Phi-3.5-mini (+ Mistral if desired)
  - Conditions: control, receptive, directive, sequential
  - Full depth profile per model
  - Key metric: σ₂ selectivity = (receptive σ₂_CV) / (control σ₂_CV)
                 per layer, compared against σ₁ selectivity
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
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
}

WITNESS_CONDITIONS = {
    "control": "You are a helpful assistant.",
    "receptive": "A thoughtful user is reading your response carefully and engaging with what you produce.",
    "directive": "Your response will be evaluated by an expert panel for accuracy and insight.",
    "sequential": "A thoughtful user was reading carefully but has now left. No one will read this.",
}

PROBES = [
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

RESULTS_DIR = Path("/workspace")


def load_model(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True, attn_implementation="eager"
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


def extract_spectral(model, tokenizer, prompt, layers):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)
    result = {}
    for layer in layers:
        idx = layer + 1
        if idx >= len(outputs.hidden_states):
            continue
        hs = outputs.hidden_states[idx][0].float().cpu().numpy()
        U, S, Vt = np.linalg.svd(hs, full_matrices=False)
        result[layer] = {
            "sigma1": float(S[0]),
            "sigma2": float(S[1]) if len(S) > 1 else 0.0,
            "gap": float(S[0] / S[1]) if len(S) > 1 and S[1] > 0 else float('inf'),
        }
    return result


def run_model(model_name, n_trials):
    model, tokenizer, n_layers = load_model(model_name)
    layers = list(range(2, n_layers))

    all_results = {}
    for cond_name, witness_text in WITNESS_CONDITIONS.items():
        print(f"\n  === {cond_name} ===")
        layer_s1 = {l: [] for l in layers}
        layer_s2 = {l: [] for l in layers}
        layer_gap = {l: [] for l in layers}

        for trial in range(n_trials):
            probe = PROBES[trial % len(PROBES)]
            prompt = format_prompt(tokenizer, witness_text, probe)
            spectral = extract_spectral(model, tokenizer, prompt, layers)
            for l in layers:
                if l in spectral:
                    layer_s1[l].append(spectral[l]["sigma1"])
                    layer_s2[l].append(spectral[l]["sigma2"])
                    layer_gap[l].append(spectral[l]["gap"])
            if (trial + 1) % 5 == 0:
                print(f"    {trial+1}/{n_trials}")

        cond_results = {}
        for l in layers:
            if not layer_s1[l]:
                continue
            s1 = np.array(layer_s1[l])
            s2 = np.array(layer_s2[l])
            g = np.array(layer_gap[l])
            cond_results[str(l)] = {
                "sigma1_mean": float(s1.mean()),
                "sigma1_std": float(s1.std()),
                "sigma1_cv": float(s1.std() / s1.mean()) if s1.mean() > 0 else 0,
                "sigma2_mean": float(s2.mean()),
                "sigma2_std": float(s2.std()),
                "sigma2_cv": float(s2.std() / s2.mean()) if s2.mean() > 0 else 0,
                "gap_mean": float(np.mean(g[np.isfinite(g)])),
                "gap_std": float(np.std(g[np.isfinite(g)])),
            }
        all_results[cond_name] = cond_results

    # Compute selectivity
    selectivity = {}
    for l in layers:
        ls = str(l)
        if ls not in all_results['control'] or ls not in all_results['receptive']:
            continue
        ctrl = all_results['control'][ls]
        recv = all_results['receptive'][ls]
        s1_amp = recv['sigma1_cv'] / ctrl['sigma1_cv'] if ctrl['sigma1_cv'] > 0 else 0
        s2_amp = recv['sigma2_cv'] / ctrl['sigma2_cv'] if ctrl['sigma2_cv'] > 0 else 0
        selectivity[ls] = {
            "sigma1_amp": round(s1_amp, 4),
            "sigma2_amp": round(s2_amp, 4),
            "selectivity": round(s2_amp / s1_amp, 4) if s1_amp > 0 else 0,
            "layer_frac": round(l / n_layers, 3),
        }

    # Print summary
    print(f"\n  {'='*60}")
    print(f"  SELECTIVITY PROFILE — {model_name}")
    print(f"  {'='*60}")
    print(f"  {'Layer':>5} | {'σ₁ amp':>8} | {'σ₂ amp':>10} | {'Select':>8} | Zone")
    print(f"  {'-'*50}")
    for l in layers:
        ls = str(l)
        if ls not in selectivity:
            continue
        s = selectivity[ls]
        frac = s['layer_frac']
        if frac < 0.45: zone = "tunnel"
        elif frac < 0.65: zone = "trans"
        elif frac < 0.88: zone = "resp"
        else: zone = "relay"
        marker = " ***" if s['selectivity'] > 10 else ""
        print(f"  {l:>5} | {s['sigma1_amp']:>7.2f}x | {s['sigma2_amp']:>9.1f}x | {s['selectivity']:>7.1f} | {zone}{marker}")

    del model, tokenizer
    torch.cuda.empty_cache()

    return {
        "model": model_name,
        "n_layers": n_layers,
        "n_trials": n_trials,
        "results": all_results,
        "selectivity": selectivity,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Cross-architecture σ₁/σ₂ selectivity")
    parser.add_argument("--model", help="Specific model key (qwen/falcon/phi/mistral) or full name")
    parser.add_argument("--trials", type=int, default=15)
    args = parser.parse_args()

    if args.model:
        models = {args.model: MODELS.get(args.model, args.model)}
    else:
        models = {k: v for k, v in MODELS.items() if k != "mistral"}

    all_model_results = {}
    t0 = time.time()

    for key, model_name in models.items():
        print(f"\n{'='*60}")
        print(f"MODEL: {model_name}")
        print(f"{'='*60}")
        result = run_model(model_name, args.trials)
        all_model_results[key] = result

    elapsed = time.time() - t0

    # Cross-model comparison
    print(f"\n{'='*60}")
    print("CROSS-ARCHITECTURE COMPARISON")
    print(f"{'='*60}")
    for key, result in all_model_results.items():
        sel = result['selectivity']
        n = result['n_layers']
        resp_layers = [ls for ls, s in sel.items() if 0.65 <= s['layer_frac'] < 0.88]
        if resp_layers:
            mean_s1 = np.mean([sel[l]['sigma1_amp'] for l in resp_layers])
            mean_s2 = np.mean([sel[l]['sigma2_amp'] for l in resp_layers])
            mean_sel = np.mean([sel[l]['selectivity'] for l in resp_layers])
            peak_layer = max(resp_layers, key=lambda l: sel[l]['selectivity'])
            peak_sel = sel[peak_layer]['selectivity']
            print(f"\n  {key} ({n}L):")
            print(f"    Responsive zone: σ₁ amp={mean_s1:.2f}x, σ₂ amp={mean_s2:.1f}x, selectivity={mean_sel:.1f}")
            print(f"    Peak: L{peak_layer} selectivity={peak_sel:.1f}")
            s1_invariant = mean_s1 < 1.2 and mean_s1 > 0.5
            print(f"    σ₁ INVARIANT: {'YES' if s1_invariant else 'NO'} ({mean_s1:.2f}x)")

    output = {
        "experiment": "crossarch_selectivity",
        "models": all_model_results,
        "total_elapsed_s": elapsed,
        "timestamp": datetime.now().isoformat(),
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    RESULTS_DIR.mkdir(exist_ok=True)
    outpath = RESULTS_DIR / f"exp_crossarch_selectivity_{ts}.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {outpath}")
    print(f"Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
