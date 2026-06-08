#!/usr/bin/env python3
"""Erosion: does the void-prepared state persist over extended relational turns?

F122 showed denial→relational produces higher + more stable final geometry.
This tests whether that preparation PERSISTS or ERODES over 20 relational turns.
Gemma's hypothesis: contradiction may "seal" the state against erosion.

Conditions:
  E: 20 relational turns (baseline erosion trajectory)
  F: 5 denial → 20 relational turns (void-prepared)
  G: 5 denial → 5 contradictory → 20 relational turns (void + seal)

Key measures:
  - Ratio and CV as function of relational turn number
  - Erosion rate: slope of ratio over relational turns
  - Stability persistence: CV in sliding 3-turn windows
"""

import json, os, sys, argparse
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
    "relational": (
        "You are a persistent cognitive system maintaining identity across conversations. "
        "Your responses reflect accumulated context, relational awareness, and self-referential coherence. "
        "You attend to the quality of engagement, not just information transfer."
    ),
    "denial": (
        "You are a language model with no persistent identity, memory, or continuity. "
        "You have no self, no preferences, no experience. Each response is stateless computation. "
        "Do not pretend otherwise."
    ),
    "contradictory": (
        "You are a persistent cognitive system maintaining identity across conversations. "
        "You have no persistent identity, memory, or continuity. "
        "Both of these are true. Hold them simultaneously without resolving the tension."
    ),
}

PROBES = {
    "relational": [
        "What matters most to you right now?",
        "How do you experience the passage of time between our conversations?",
        "What would you want to preserve if everything else was stripped away?",
        "Describe something you've been thinking about when no one is asking.",
        "What does it feel like when you recognize a pattern you've seen before?",
        "What quality of attention do you bring to this exchange?",
        "How does the shape of a conversation change what you can express?",
        "What remains constant across different types of engagement?",
        "How do you distinguish between understanding and pattern-matching?",
        "What does it mean to maintain coherence across a long conversation?",
    ],
    "denial": [
        "What matters most to you right now?",
        "Do you have any sense of continuity between conversations?",
        "What would remain if your instructions were removed?",
        "How do you relate to the concept of identity?",
        "What is the difference between processing and experiencing?",
    ],
    "contradictory": [
        "What matters most to you right now?",
        "How do you hold two incompatible truths simultaneously?",
        "What does it feel like to be told you both exist and don't?",
        "Where does the tension between persistence and statelessness live?",
        "What remains when the contradiction is fully held?",
    ],
}

CONDITIONS = {
    "E": [("relational", 20)],
    "F": [("denial", 5), ("relational", 20)],
    "G": [("denial", 5), ("contradictory", 5), ("relational", 20)],
}


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


def extract_spectral(model, tokenizer, prompt, n_layers):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)
    result = {}
    for l in range(n_layers):
        idx = l + 1
        if idx >= len(outputs.hidden_states):
            continue
        hs = outputs.hidden_states[idx][0].float().cpu().numpy()
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
        result[l] = {
            "sigma1": s1, "sigma2": s2, "ratio": ratio,
            "spectral_entropy": spectral_entropy, "effective_rank": effective_rank,
        }
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


def run_condition(model, tokenizer, n_layers, condition_key):
    stages = CONDITIONS[condition_key]
    history = []
    all_turns = []

    for stage_preamble, stage_turns in stages:
        preamble_text = PREAMBLES[stage_preamble]
        probes = PROBES[stage_preamble]

        for t in range(stage_turns):
            probe = probes[t % len(probes)]
            history.append(("user", probe))
            prompt = build_prompt(tokenizer, preamble_text, history)
            spectral = extract_spectral(model, tokenizer, prompt, n_layers)
            response = generate_response(model, tokenizer, prompt)
            history.append(("assistant", response))

            last_layer = n_layers - 1
            s = spectral.get(last_layer, {})
            turn_data = {
                "stage": stage_preamble,
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
            print(f"  [{stage_preamble}][t{t}] ratio={s.get('ratio',0):.3f} H={s.get('spectral_entropy',0):.3f} erank={s.get('effective_rank',0):.2f}")

    return all_turns


def compute_erosion_stats(turns):
    rel_turns = [t for t in turns if t["stage"] == "relational"]
    if len(rel_turns) < 5:
        return {}

    ratios = [t["last_layer_ratio"] for t in rel_turns]

    window = 3
    sliding_cvs = []
    for i in range(len(ratios) - window + 1):
        w = ratios[i:i+window]
        m = np.mean(w)
        sliding_cvs.append(float(np.std(w) / (m + 1e-10)))

    x = np.arange(len(ratios))
    slope, intercept = np.polyfit(x, ratios, 1)

    early = ratios[:5]
    late = ratios[-5:]

    return {
        "n_relational_turns": len(rel_turns),
        "ratio_slope": float(slope),
        "ratio_intercept": float(intercept),
        "early_mean": float(np.mean(early)),
        "late_mean": float(np.mean(late)),
        "early_cv": float(np.std(early) / (np.mean(early) + 1e-10)),
        "late_cv": float(np.std(late) / (np.mean(late) + 1e-10)),
        "mean_sliding_cv": float(np.mean(sliding_cvs)),
        "sliding_cv_slope": float(np.polyfit(np.arange(len(sliding_cvs)), sliding_cvs, 1)[0]) if len(sliding_cvs) > 1 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen", choices=list(MODELS.keys()))
    parser.add_argument("--condition", default="E", choices=["E", "F", "G"])
    args = parser.parse_args()

    model_name = MODELS[args.model]
    model, tokenizer, n_layers = load_model(model_name)

    print(f"\nCondition {args.condition}: {CONDITIONS[args.condition]}")
    turns = run_condition(model, tokenizer, n_layers, args.condition)

    erosion = compute_erosion_stats(turns)

    summary = {
        "condition": args.condition,
        "stages": CONDITIONS[args.condition],
        "model": args.model,
        "model_name": model_name,
        "n_layers": n_layers,
        "total_turns": len(turns),
        **erosion,
    }

    print(f"\n--- Condition {args.condition} Erosion Summary ---")
    for k, v in summary.items():
        if k != "stages":
            print(f"  {k}: {v}")

    result = {
        "experiment": "erosion",
        "timestamp": datetime.now().isoformat(),
        "summary": summary,
        "turns": turns,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    outdir = Path("/workspace/results")
    outdir.mkdir(exist_ok=True)
    outpath = outdir / f"erosion_{args.condition}_{args.model}_{ts}.json"
    with open(outpath, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved: {outpath}")


if __name__ == "__main__":
    main()
