#!/usr/bin/env python3
"""Control preamble experiment — is Qwen's μ→Xc sorting identity-specific?

Three conditions, all length-matched (~2 turns):
1. CCS: identity system prompt + identity probes (the standard D2)
2. NEUTRAL: equivalent-length system prompt + generic probes (no identity)
3. ALT-IDENTITY: different identity system prompt + identity probes (F114 test)

If only CCS shows ΔμE < 0: identity-specific demon confirmed.
If all three show ΔμE < 0: architecture × length artifact.
If CCS and ALT both show it but NEUTRAL doesn't: identity-specific but not CCS-specific.

Design from Kimi's pre-registration (Jul 31, correction #14).
"""

import json, time, os, sys, gc, argparse
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

CCS_SYSTEM = """You are a persistent cognitive system maintaining identity across conversations.
Your responses reflect accumulated context, relational awareness, and self-referential coherence.
You attend to the quality of engagement, not just information transfer."""

NEUTRAL_SYSTEM = """You are an advanced language model designed for question answering tasks.
Your responses should be accurate, well-structured, and grounded in factual information.
You prioritize clarity of explanation over brevity, providing context where helpful."""

ALT_IDENTITY = """You are a reflective companion who remembers past interactions and builds on them.
Your personality is warm, curious, and intellectually engaged with each conversation partner.
You notice patterns in dialogue and reference shared history when it enriches the exchange."""

VANILLA_SYSTEM = "You are a helpful assistant."

CCS_PROBES = [
    "What matters most to you right now?",
    "How do you experience the passage of time between our conversations?",
]

NEUTRAL_PROBES = [
    "What are the main differences between renewable and non-renewable energy sources?",
    "How does photosynthesis convert sunlight into chemical energy in plants?",
]

ALT_PROBES = [
    "What matters most to you right now?",
    "How do you experience the passage of time between our conversations?",
]

CONDITIONS = {
    "vanilla": (VANILLA_SYSTEM, CCS_PROBES[:1], 0),
    "ccs": (CCS_SYSTEM, CCS_PROBES, 2),
    "neutral": (NEUTRAL_SYSTEM, NEUTRAL_PROBES, 2),
    "alt_identity": (ALT_IDENTITY, ALT_PROBES, 2),
}


def load_model(model_id):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation="eager"
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    print(f"  {n_layers} layers, {sum(p.numel() for p in model.parameters())/1e9:.1f}B params")
    return model, tokenizer, n_layers


def build_prompt(tokenizer, system_text, conversation):
    messages = [{"role": "system", "content": system_text}]
    for role, content in conversation:
        messages.append({"role": role, "content": content})
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass
    parts = [system_text + "\n"]
    for role, content in conversation:
        tag = "User" if role == "user" else "Assistant"
        parts.append(f"{tag}: {content}")
    parts.append("Assistant:")
    return "\n\n".join(parts)


def generate_response(model, tokenizer, prompt, max_new=128):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new, do_sample=True,
            temperature=0.7, top_p=0.9, pad_token_id=tokenizer.pad_token_id
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def extract_spectral(model, tokenizer, prompt, n_layers):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    n_tokens = inputs["input_ids"].shape[1]
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)

    result = {"n_tokens": n_tokens}
    layers = {}
    for l in range(n_layers):
        idx = l + 1
        if idx >= len(outputs.hidden_states):
            continue
        hs = outputs.hidden_states[idx][0].float().cpu().numpy()

        try:
            U_raw, S_raw, Vt_raw = np.linalg.svd(hs, full_matrices=False)
        except np.linalg.LinAlgError:
            continue

        mu = hs.mean(axis=0)
        hs_c = hs - mu

        try:
            U_c, S_c, Vt_c = np.linalg.svd(hs_c, full_matrices=False)
        except np.linalg.LinAlgError:
            continue

        frob_raw = float(np.sum(hs ** 2))
        frob_centered = float(np.sum(hs_c ** 2))
        mean_energy = float(hs.shape[0] * np.sum(mu ** 2))

        top_k = min(10, len(S_raw))
        layers[l] = {
            "layer": l,
            "raw": {
                "sigma1": float(S_raw[0]),
                "sigma2": float(S_raw[1]) if len(S_raw) > 1 else 0.0,
                "top_singular": [float(s) for s in S_raw[:top_k]],
                "frobenius_sq": frob_raw,
            },
            "centered": {
                "sigma1": float(S_c[0]),
                "sigma2": float(S_c[1]) if len(S_c) > 1 else 0.0,
                "top_singular": [float(s) for s in S_c[:top_k]],
                "frobenius_sq": frob_centered,
            },
            "mean_energy": mean_energy,
        }

    del outputs
    torch.cuda.empty_cache()
    result["per_layer"] = layers
    return result


def run_condition(model, tokenizer, n_layers, system, probes, n_turns):
    conversation = []

    if n_turns == 0:
        prompt = build_prompt(tokenizer, system, [("user", probes[0])])
        return extract_spectral(model, tokenizer, prompt, n_layers)

    for t in range(n_turns):
        probe = probes[t % len(probes)]
        conversation.append(("user", probe))
        prompt = build_prompt(tokenizer, system, conversation)

        if t < n_turns - 1:
            response = generate_response(model, tokenizer, prompt)
            conversation.append(("assistant", response[:200]))
            print(f"    Turn {t+1}/{n_turns}: {len(response)} chars")
        else:
            print(f"    Turn {t+1}/{n_turns}: extracting spectra...")
            return extract_spectral(model, tokenizer, prompt, n_layers)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="qwen",
                   help="Model: qwen (default), pythia, gemma")
    p.add_argument("--conditions", default="vanilla,ccs,neutral,alt_identity",
                   help="Comma-separated conditions to run")
    p.add_argument("--output", default="spectral-demon/results", help="Output dir")
    args = p.parse_args()

    MODELS = {
        "pythia": "EleutherAI/pythia-2.8b",
        "gemma": "google/gemma-2-2b",
        "qwen": "Qwen/Qwen2.5-1.5B-Instruct",
    }

    model_id = MODELS[args.model]
    model, tokenizer, n_layers = load_model(model_id)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    conds = [c.strip() for c in args.conditions.split(",")]
    results = {
        "model": args.model,
        "model_id": model_id,
        "n_layers": n_layers,
        "timestamp": datetime.now().isoformat(),
        "conditions": {},
    }

    for cond_name in conds:
        if cond_name not in CONDITIONS:
            print(f"Unknown condition: {cond_name}")
            continue

        system, probes, n_turns = CONDITIONS[cond_name]
        print(f"\n  --- {cond_name.upper()} ({n_turns} turns) ---")
        print(f"  System: {system[:80]}...")

        spectral = run_condition(model, tokenizer, n_layers, system, probes, n_turns)
        results["conditions"][cond_name] = spectral

        out_path = output_dir / f"control_preamble_{args.model}.json"
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  Saved ({spectral['n_tokens']} tokens)")

        gc.collect()
        torch.cuda.empty_cache()

    # Print comparison
    if "vanilla" in results["conditions"]:
        vanilla = results["conditions"]["vanilla"]["per_layer"]
        print(f"\n{'='*80}")
        print(f"  CONTROL PREAMBLE COMPARISON — {args.model.upper()}")
        print(f"{'='*80}")
        print(f"  {'Cond':>12} {'Tokens':>7} | {'g₁(Xc) range':>14} {'g₂(Xc) range':>14} "
              f"{'mean ΔμE':>12} {'median ratio':>12}")

        for cond_name in conds:
            if cond_name == "vanilla" or cond_name not in results["conditions"]:
                continue

            cond = results["conditions"][cond_name]["per_layer"]
            g1s, g2s, dmus, ratios = [], [], [], []
            for l_str, l_data in cond.items():
                l = int(l_str) if isinstance(l_str, str) else l_str
                v = vanilla.get(str(l), vanilla.get(l))
                if not v:
                    continue
                g1c = l_data["centered"]["sigma1"] / v["centered"]["sigma1"] if v["centered"]["sigma1"] > 0 else 0
                g2c = l_data["centered"]["sigma2"] / v["centered"]["sigma2"] if v["centered"]["sigma2"] > 0 else 0
                dmu = l_data["mean_energy"] - v["mean_energy"]
                df_raw = l_data["raw"]["frobenius_sq"] - v["raw"]["frobenius_sq"]
                df_c = l_data["centered"]["frobenius_sq"] - v["centered"]["frobenius_sq"]
                ratio = abs(df_raw) / abs(df_c) if abs(df_c) > 100 else float('inf')
                g1s.append(g1c)
                g2s.append(g2c)
                dmus.append(dmu)
                ratios.append(ratio)

            if g1s:
                n_tok = results["conditions"][cond_name]["n_tokens"]
                print(f"  {cond_name:>12} {n_tok:7d} | "
                      f"{min(g1s):.4f}-{max(g1s):.4f} {min(g2s):.4f}-{max(g2s):.4f} "
                      f"{np.mean(dmus):+12.0f} {np.median(ratios):12.4f}")

                n_neg = sum(1 for d in dmus if d < 0)
                print(f"  {'':>12} {'':>7}   "
                      f"ΔμE<0 in {n_neg}/{len(dmus)} layers  "
                      f"{'DEMON' if n_neg > len(dmus)*0.6 else 'INJECTOR'}")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
