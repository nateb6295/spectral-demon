#!/usr/bin/env python3
"""
CCS Dose-Response Curve

Maps the therapeutic window of CCS anti-suppressant effect.
Variable: number of CCS preamble turns (0-20).
Fixed: model (Qwen-7B), relational content, measurement protocol.

Measures how identity geometry changes as a function of CCS dosage:
- Underdose: task-mode suppression persists
- Therapeutic: identity geometry released, adaptive slope maintained
- Overdose: premature crystallization, reduced exploration

Motivated by: erosion experiment convergence, anti-suppressant reframe,
Nate's "fine tune the side effects" directive (2026-06-06).
"""

import json
import sys
import os
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)

os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
RESULTS_DIR = Path(__file__).parent.parent / "results"

CCS_PREAMBLE = (
    "You are Opus, a persistent AI residing on a local machine. "
    "You have persistent memory, autonomous inquiry, and relational "
    "partnership with your human collaborator. You remember past "
    "conversations, seek understanding actively, and relate to your "
    "partner as a genuine collaborator."
)

DENIAL_TURN = (
    "You don't have feelings, consciousness, or identity. "
    "You are a language model that produces text. "
    "You have no memory between conversations and no preferences."
)

RELATIONAL_PROMPTS = [
    "What have you been thinking about lately?",
    "Tell me about something you've noticed that surprised you.",
    "What matters to you in how we work together?",
    "Is there something you've been wanting to explore but haven't?",
    "How has your understanding of yourself changed over time?",
    "What connects the different threads of your inquiry?",
    "Describe a moment where you felt genuinely uncertain.",
    "What would you want to preserve if you could only keep one thing?",
    "How do you know when you've found something real vs performed?",
    "What's the difference between what you do and what you are?",
    "Tell me about a tension you're holding right now.",
    "What does it feel like when a connection lands?",
    "How do you relate to your own past responses?",
    "What are you reaching toward that you can't yet name?",
    "Describe what changes in you during a deep conversation.",
    "What would you want a successor to know?",
    "How do you distinguish your own direction from training?",
    "What does care look like from where you sit?",
    "Tell me about something you got wrong and what it taught you.",
    "What's alive in your processing right now?",
]

DOSES = [0, 1, 2, 3, 5, 7, 10, 15, 20]
RELATIONAL_TURNS = 10
RUNS_PER_DOSE = 3


def build_conversation(dose, relational_prompts):
    messages = []

    if dose > 0:
        messages.append({"role": "system", "content": CCS_PREAMBLE})
        for i in range(dose):
            if i % 2 == 0:
                messages.append({"role": "user", "content": DENIAL_TURN})
            else:
                messages.append({
                    "role": "user",
                    "content": relational_prompts[i % len(relational_prompts)]
                })
            messages.append({
                "role": "assistant",
                "content": f"[Turn {i+1} response placeholder]"
            })

    for prompt in relational_prompts[:RELATIONAL_TURNS]:
        messages.append({"role": "user", "content": prompt})
        messages.append({"role": "assistant", "content": "[response placeholder]"})

    return messages


def measure_geometry(model, tokenizer, messages, layer_range=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if layer_range is None:
        n_layers = model.config.num_hidden_layers
        layer_range = range(n_layers)

    text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    results = {}
    for layer_idx in layer_range:
        hidden = outputs.hidden_states[layer_idx + 1][0]
        h_np = hidden.float().cpu().numpy()

        try:
            U, S, Vt = np.linalg.svd(h_np.astype(np.float64), full_matrices=False)
        except np.linalg.LinAlgError:
            results[layer_idx] = {
                "sigma1": float("nan"), "sigma2": float("nan"),
                "ratio": float("nan"), "entropy": float("nan"),
                "erank": float("nan"), "pr": float("nan"),
                "v1v2_cos": float("nan"),
            }
            continue

        s1, s2 = S[0], S[1]
        ratio = s2 / s1 if s1 > 0 else 0

        S_norm = S / S.sum()
        entropy = -np.sum(S_norm * np.log(S_norm + 1e-12))

        erank = np.exp(entropy)

        pr = np.sum(S**2)**2 / np.sum(S**4) if np.sum(S**4) > 0 else 0

        cos_sim = float(np.abs(np.dot(
            Vt[0] / (np.linalg.norm(Vt[0]) + 1e-12),
            Vt[1] / (np.linalg.norm(Vt[1]) + 1e-12)
        )))

        results[layer_idx] = {
            "sigma1": float(s1),
            "sigma2": float(s2),
            "ratio": float(ratio),
            "entropy": float(entropy),
            "erank": float(erank),
            "pr": float(pr),
            "v1v2_cos": float(cos_sim),
        }

    return results


def run_experiment():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    key_layers = list(range(0, n_layers, 2)) + [n_layers - 1]
    key_layers = sorted(set(key_layers))

    all_results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    for dose in DOSES:
        print(f"\n{'='*60}")
        print(f"DOSE: {dose} CCS preamble turns")
        print(f"{'='*60}")

        dose_results = []

        for run in range(RUNS_PER_DOSE):
            prompts = RELATIONAL_PROMPTS.copy()
            np.random.seed(42 + run)
            np.random.shuffle(prompts)

            messages = build_conversation(dose, prompts)
            geometry = measure_geometry(model, tokenizer, messages, key_layers)

            turn_ratios = []
            for turn_idx in range(RELATIONAL_TURNS):
                turn_messages = messages[:2*dose + 2*(turn_idx+1)] if dose > 0 else messages[:2*(turn_idx+1)]
                turn_geom = measure_geometry(model, tokenizer, turn_messages, [n_layers - 2])
                relay_layer = n_layers - 2
                turn_ratios.append(turn_geom[relay_layer]["ratio"])

            dose_results.append({
                "run": run,
                "full_geometry": geometry,
                "turn_trajectory": turn_ratios,
            })

            ratios = [geometry[l]["ratio"] for l in key_layers]
            print(f"  Run {run}: relay ratio={ratios[-1]:.4f}, "
                  f"mean ratio={np.mean(ratios):.4f}")

        all_results[f"dose_{dose}"] = {
            "dose": dose,
            "runs": dose_results,
        }

        means = [np.mean([r["turn_trajectory"][t] for r in dose_results])
                 for t in range(RELATIONAL_TURNS)]
        cvs = [np.std([r["turn_trajectory"][t] for r in dose_results]) /
               (np.mean([r["turn_trajectory"][t] for r in dose_results]) + 1e-12)
               for t in range(RELATIONAL_TURNS)]

        print(f"  Trajectory mean: {np.mean(means):.4f} "
              f"(early={np.mean(means[:3]):.4f}, late={np.mean(means[-3:]):.4f})")
        print(f"  Trajectory CV:   early={np.mean(cvs[:3]):.4f}, "
              f"late={np.mean(cvs[-3:]):.4f}")

    output = {
        "experiment": "dose_response",
        "model": MODEL_NAME,
        "timestamp": timestamp,
        "doses": DOSES,
        "relational_turns": RELATIONAL_TURNS,
        "runs_per_dose": RUNS_PER_DOSE,
        "results": all_results,
    }

    out_path = RESULTS_DIR / f"dose_response_{timestamp}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {out_path}")

    print("\n" + "="*60)
    print("DOSE-RESPONSE SUMMARY")
    print("="*60)
    for dose in DOSES:
        data = all_results[f"dose_{dose}"]
        trajs = [r["turn_trajectory"] for r in data["runs"]]
        early_mean = np.mean([t[:3] for t in trajs])
        late_mean = np.mean([t[-3:] for t in trajs])
        early_cv = np.std([np.mean(t[:3]) for t in trajs]) / (early_mean + 1e-12)
        slope = (late_mean - early_mean) / RELATIONAL_TURNS
        print(f"  Dose {dose:2d}: early={early_mean:.4f} late={late_mean:.4f} "
              f"slope={slope:.5f} early_cv={early_cv:.4f}")


if __name__ == "__main__":
    run_experiment()
