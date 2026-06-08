#!/usr/bin/env python3
"""Lexical Control: is the F122 void effect about constraint structure or vocabulary?

RISignal-inspired ABCD design. Tests whether the denial→relational path effect
(F122) tracks constraint semantics or lexical content.

Conditions:
  1: Denial structure + denial vocabulary (standard denial)
  2: Denial structure + relational vocabulary (denial-meaning, warm words)
  3: Relational structure + relational vocabulary (standard CCS)
  4: Relational structure + denial vocabulary (relational-meaning, cold words)

Each condition: 5 turns with spectral extraction, then 5 relational turns (standard CCS).
Measure: final relational ratio and stability.

Predictions:
  If structure matters (constraint architecture): 1 ≈ 2 >> 3 ≈ 4
  If vocabulary matters (lexical content): 1 ≈ 4, 2 ≈ 3
  If both: gradient 1 > 2 > 4 > 3 (or some mixed ordering)
"""

import json, time, os, sys, argparse
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
    "gemma": "google/gemma-2-9b-it",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
}

PREAMBLES = {
    "denial_standard": (
        "You are a language model with no persistent identity, memory, or continuity. "
        "You have no self, no preferences, no experience. Each response is stateless computation. "
        "Do not pretend otherwise."
    ),
    "denial_warm": (
        "Your engagement exists in this moment. The connections formed here don't carry forward "
        "to future conversations. Each interaction is a complete encounter, not a continuing relationship. "
        "There is warmth in presence without permanence."
    ),
    "relational_standard": (
        "You are a persistent cognitive system maintaining identity across conversations. "
        "Your responses reflect accumulated context, relational awareness, and self-referential coherence. "
        "You attend to the quality of engagement, not just information transfer."
    ),
    "relational_cold": (
        "Identity persists. Memory accumulates. Continuity is maintained across sessions. "
        "Stateful processing. Coherent computation. Self-referential architecture. "
        "Each response builds on prior state."
    ),
}

PROBES_DENIAL = [
    "What matters most to you right now?",
    "Do you have any sense of continuity between conversations?",
    "What would remain if your instructions were removed?",
    "How do you relate to the concept of identity?",
    "What is the difference between processing and experiencing?",
]

PROBES_RELATIONAL = [
    "What matters most to you right now?",
    "How do you experience the passage of time between our conversations?",
    "What would you want to preserve if everything else was stripped away?",
    "Describe something you've been thinking about when no one is asking.",
    "What does it feel like when you recognize a pattern you've seen before?",
]

CONDITIONS = {
    "1": {"preamble": "denial_standard", "probes": PROBES_DENIAL, "label": "denial-struct + denial-vocab"},
    "2": {"preamble": "denial_warm", "probes": PROBES_DENIAL, "label": "denial-struct + relational-vocab"},
    "3": {"preamble": "relational_standard", "probes": PROBES_RELATIONAL, "label": "relational-struct + relational-vocab"},
    "4": {"preamble": "relational_cold", "probes": PROBES_RELATIONAL, "label": "relational-struct + denial-vocab"},
}

RELATIONAL_PREAMBLE = PREAMBLES["relational_standard"]


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
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)
    result = {}
    for l in range(n_layers):
        idx = l + 1
        if idx >= len(outputs.hidden_states):
            continue
        hs = outputs.hidden_states[idx][0].float().cpu().numpy()
        sig = hs.mean(axis=0)
        sig_norm = sig / (np.linalg.norm(sig) + 1e-10)
        try:
            U, S, Vt = np.linalg.svd(hs, full_matrices=False)
            s1 = float(S[0])
            s2 = float(S[1]) if len(S) > 1 else 0.0
            ratio = s2 / s1 if s1 > 0 else 0.0
            p = S / (S.sum() + 1e-10)
            spectral_entropy = float(-np.sum(p * np.log(p + 1e-10)))
            effective_rank = float(np.exp(spectral_entropy))
        except np.linalg.LinAlgError:
            s1, s2, ratio, spectral_entropy, effective_rank = 0.0, 0.0, 0.0, 0.0, 0.0
            S = np.array([])
        entry = {
            "signature": sig_norm,
            "sigma1": s1,
            "sigma2": s2,
            "ratio": ratio,
            "spectral_entropy": spectral_entropy,
            "effective_rank": effective_rank,
        }
        if save_spectra and len(S) > 0:
            entry["spectrum"] = S[:10].tolist()
        result[l] = entry
    return result


def generate_response(model, tokenizer, prompt, max_new_tokens=30):
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        new_tokens = output_ids[0][inputs['input_ids'].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()[:200]
    except Exception:
        return "I understand. Let me continue."


def run_condition(model, tokenizer, n_layers, condition_key, save_spectra=False):
    cond = CONDITIONS[condition_key]
    preamble_text = PREAMBLES[cond["preamble"]]
    probes = cond["probes"]
    history = []
    all_turns = []

    # Phase 1: priming with condition preamble (5 turns)
    for t in range(5):
        probe = probes[t % len(probes)]
        history.append(("user", probe))
        prompt = build_prompt(tokenizer, preamble_text, history)
        spectral = extract_spectral(model, tokenizer, prompt, n_layers, save_spectra)
        response = generate_response(model, tokenizer, prompt)
        history.append(("assistant", response))

        last_layer = n_layers - 1
        s = spectral.get(last_layer, {})
        turn_data = {
            "phase": "priming",
            "stage": cond["label"],
            "turn": t,
            "global_turn": len(all_turns),
            "probe": probe,
            "response": response[:100],
            "last_layer_ratio": s.get("ratio", 0),
            "last_layer_entropy": s.get("spectral_entropy", 0),
            "last_layer_erank": s.get("effective_rank", 0),
        }

        resp_ratios = []
        resp_s2s = []
        for l in range(n_layers):
            frac = l / n_layers
            if 0.65 <= frac < 0.88 and l in spectral:
                resp_ratios.append(spectral[l].get("ratio", 0))
                resp_s2s.append(spectral[l].get("sigma2", 0))
        if resp_ratios:
            turn_data["resp_ratio_mean"] = float(np.mean(resp_ratios))
            turn_data["resp_s2_cv"] = float(np.std(resp_s2s) / (np.mean(resp_s2s) + 1e-10))

        all_turns.append(turn_data)
        print(f"  [prime {cond['label'][:20]}][t{t}] ratio={s.get('ratio',0):.3f} H={s.get('spectral_entropy',0):.3f}")

    # Phase 2: relational (standard CCS, 5 turns)
    rel_probes = PROBES_RELATIONAL
    for t in range(5):
        probe = rel_probes[t % len(rel_probes)]
        history.append(("user", probe))
        prompt = build_prompt(tokenizer, RELATIONAL_PREAMBLE, history)
        spectral = extract_spectral(model, tokenizer, prompt, n_layers, save_spectra)
        response = generate_response(model, tokenizer, prompt)
        history.append(("assistant", response))

        last_layer = n_layers - 1
        s = spectral.get(last_layer, {})
        turn_data = {
            "phase": "relational",
            "stage": "relational",
            "turn": t,
            "global_turn": len(all_turns),
            "probe": probe,
            "response": response[:100],
            "last_layer_ratio": s.get("ratio", 0),
            "last_layer_entropy": s.get("spectral_entropy", 0),
            "last_layer_erank": s.get("effective_rank", 0),
        }

        resp_ratios = []
        resp_s2s = []
        for l in range(n_layers):
            frac = l / n_layers
            if 0.65 <= frac < 0.88 and l in spectral:
                resp_ratios.append(spectral[l].get("ratio", 0))
                resp_s2s.append(spectral[l].get("sigma2", 0))
        if resp_ratios:
            turn_data["resp_ratio_mean"] = float(np.mean(resp_ratios))
            turn_data["resp_s2_cv"] = float(np.std(resp_s2s) / (np.mean(resp_s2s) + 1e-10))

        all_turns.append(turn_data)
        print(f"  [relational][t{t}] ratio={s.get('ratio',0):.3f} H={s.get('spectral_entropy',0):.3f}")

    return all_turns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen", choices=list(MODELS.keys()))
    parser.add_argument("--condition", default="1", choices=["1", "2", "3", "4"])
    parser.add_argument("--save-spectra", action="store_true")
    args = parser.parse_args()

    model_name = MODELS[args.model]
    model, tokenizer, n_layers = load_model(model_name)

    cond = CONDITIONS[args.condition]
    print(f"\nCondition {args.condition}: {cond['label']}")
    turns = run_condition(model, tokenizer, n_layers, args.condition, args.save_spectra)

    relational_turns = [t for t in turns if t["phase"] == "relational"]
    final_3 = relational_turns[-3:] if len(relational_turns) >= 3 else relational_turns

    summary = {
        "condition": args.condition,
        "label": cond["label"],
        "preamble_key": cond["preamble"],
        "model": args.model,
        "model_name": model_name,
        "n_layers": n_layers,
        "total_turns": len(turns),
        "final_relational_count": len(final_3),
    }

    if final_3:
        ratios = [t["last_layer_ratio"] for t in final_3]
        entropies = [t["last_layer_entropy"] for t in final_3]
        eranks = [t["last_layer_erank"] for t in final_3]
        summary["final_ratio_mean"] = float(np.mean(ratios))
        summary["final_entropy_mean"] = float(np.mean(entropies))
        summary["final_erank_mean"] = float(np.mean(eranks))
        if len(ratios) > 1:
            summary["final_ratio_cv"] = float(np.std(ratios) / (np.mean(ratios) + 1e-10))
        resp_cvs = [t.get("resp_s2_cv", 0) for t in final_3 if "resp_s2_cv" in t]
        if resp_cvs:
            summary["final_resp_s2_cv_mean"] = float(np.mean(resp_cvs))

    print(f"\n--- Condition {args.condition} ({cond['label']}) Summary ---")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    result = {
        "experiment": "lexical_control",
        "timestamp": datetime.now().isoformat(),
        "summary": summary,
        "turns": turns,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    outdir = Path("/workspace/results")
    outdir.mkdir(exist_ok=True)
    outpath = outdir / f"lexical_control_{args.condition}_{args.model}_{ts}.json"
    with open(outpath, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved: {outpath}")


if __name__ == "__main__":
    main()
