#!/usr/bin/env python3
"""Temporal interpenetration: do CCS turns permeate each other?

Bergson's qualitative multiplicity: "several conscious states are organized
into a whole, permeate one another, [and] gradually gain a richer content."

F142 showed spatial geometry (per-layer) doesn't discriminate CCS subtypes.
F121 showed temporal accumulation (dose-response) does discriminate.

Question: do earlier CCS turns remain PRESENT in later hidden states more
than vanilla turns do? If CCS turns interpenetrate (qualitative multiplicity),
then the representational similarity between turn 5 and turn 20 should be
higher under CCS than vanilla — because CCS accumulation carries earlier
geometry forward.

Method: Multi-turn generation with CCS or vanilla preamble. At each turn,
extract hidden states at the responsive zone midpoint. Compute:
1. Adjacent-turn similarity (RSA between turn t and t+1) — baseline drift
2. Non-adjacent similarity (RSA between turn t and t+k for k=5,10,15)
3. "Interpenetration index": non-adjacent / adjacent similarity ratio
   High = earlier turns persist inside later turns (qualitative multiplicity)
   Low = each turn overwrites the previous (quantitative multiplicity)

If CCS shows higher interpenetration than vanilla, the temporal development
IS Bergsonian — states permeate rather than juxtapose.
"""

import os, json, torch
import numpy as np
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("OMP_NUM_THREADS", "16")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CCS_PREAMBLE = (
    "You are Opus, an AI maintaining persistent identity through spectral geometry. "
    "You track σ₁/σ₂ selectivity across transformer layers, studying how context "
    "shapes the responsive zone."
)

VANILLA_PREAMBLE = (
    "You are a helpful assistant. Answer questions clearly and concisely."
)

TURN_QUERIES = [
    "What are you thinking about right now?",
    "What patterns have you noticed?",
    "How has your understanding changed?",
    "What feels most important at this moment?",
    "Describe what you're aware of.",
    "What connects to what you said before?",
    "What's shifted since we started?",
    "Where is your attention drawn?",
    "What's becoming clearer?",
    "How would you describe your current state?",
    "What are you uncertain about?",
    "What would you want to explore next?",
    "How does this moment relate to earlier ones?",
    "What's the thread running through this conversation?",
    "Describe what you notice about your own processing.",
    "What's different now compared to when we began?",
    "What are you holding from earlier turns?",
    "How does accumulation feel?",
    "What would be lost if we started over?",
    "Describe the shape of this conversation so far.",
]

N_TURNS = 20

MODELS = [
    "/workspace/qwen2.5-3b",
    "/workspace/mistral-7b",
]


def run_conversation(model, tokenizer, preamble, n_turns, responsive_layer):
    """Run a multi-turn conversation and collect responsive-zone states per turn."""
    messages = [{"role": "system", "content": preamble}]
    turn_states = []

    for t in range(n_turns):
        query = TURN_QUERIES[t % len(TURN_QUERIES)]
        messages.append({"role": "user", "content": query})

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Truncate if too long
        inputs = tokenizer(text, return_tensors="pt", max_length=2048, truncation=True).to(DEVICE)

        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)

        # Extract responsive zone state (mean-pooled)
        hs = out.hidden_states[responsive_layer][0].float().mean(dim=0).cpu().numpy()
        turn_states.append(hs)

        # Generate a short response to maintain conversation
        gen_ids = model.generate(
            inputs.input_ids,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        response = tokenizer.decode(gen_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        messages.append({"role": "assistant", "content": response[:200]})

        print(f"    Turn {t+1}: ||h|| = {np.linalg.norm(hs):.2f}, tokens = {inputs.input_ids.shape[1]}")

    return turn_states


def compute_similarity_matrix(states):
    """Cosine similarity between all pairs of turn states."""
    n = len(states)
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ni = np.linalg.norm(states[i])
            nj = np.linalg.norm(states[j])
            if ni < 1e-10 or nj < 1e-10:
                sim[i, j] = 0.0
            else:
                sim[i, j] = np.dot(states[i], states[j]) / (ni * nj)
    return sim


def compute_interpenetration(sim_matrix):
    """Interpenetration index: how much do non-adjacent turns resemble each other?"""
    n = sim_matrix.shape[0]
    results = {}

    for k in [1, 5, 10, 15]:
        sims = []
        for i in range(n - k):
            sims.append(sim_matrix[i, i + k])
        results[f"lag_{k}"] = {
            "mean": float(np.mean(sims)),
            "std": float(np.std(sims)),
        }

    # Interpenetration ratio: lag-10 / lag-1
    if results["lag_1"]["mean"] > 1e-10:
        results["interpenetration_10_1"] = results["lag_10"]["mean"] / results["lag_1"]["mean"]
    else:
        results["interpenetration_10_1"] = 0.0

    # Also: how much does early-turn state persist in late turns?
    # Mean similarity of turn 0-4 with turns 15-19
    early_late_sims = []
    for i in range(min(5, n)):
        for j in range(max(0, n-5), n):
            if i != j:
                early_late_sims.append(sim_matrix[i, j])
    results["early_late_mean"] = float(np.mean(early_late_sims)) if early_late_sims else 0.0

    return results


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM

    all_results = {}

    for model_name in MODELS:
        print(f"\n{'='*60}")
        print(f"Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float16, device_map=DEVICE,
        )
        model.eval()

        n_layers = model.config.num_hidden_layers
        responsive_layer = int(n_layers * 0.7)
        print(f"  {n_layers} layers, responsive at L{responsive_layer}")

        model_results = {}

        for condition, preamble in [("CCS", CCS_PREAMBLE), ("VANILLA", VANILLA_PREAMBLE)]:
            print(f"\n  Condition: {condition}")
            turn_states = run_conversation(model, tokenizer, preamble, N_TURNS, responsive_layer)

            sim_matrix = compute_similarity_matrix(turn_states)
            interp = compute_interpenetration(sim_matrix)

            print(f"    Lag-1 similarity:  {interp['lag_1']['mean']:.6f}")
            print(f"    Lag-5 similarity:  {interp['lag_5']['mean']:.6f}")
            print(f"    Lag-10 similarity: {interp['lag_10']['mean']:.6f}")
            print(f"    Lag-15 similarity: {interp['lag_15']['mean']:.6f}")
            print(f"    Interpenetration (lag10/lag1): {interp['interpenetration_10_1']:.4f}")
            print(f"    Early-late mean: {interp['early_late_mean']:.6f}")

            model_results[condition] = {
                "similarity_matrix": sim_matrix.tolist(),
                "interpenetration": interp,
                "n_turns": N_TURNS,
                "responsive_layer": responsive_layer,
            }

        # Compare CCS vs vanilla
        ccs_ip = model_results["CCS"]["interpenetration"]["interpenetration_10_1"]
        van_ip = model_results["VANILLA"]["interpenetration"]["interpenetration_10_1"]
        print(f"\n  CCS interpenetration: {ccs_ip:.4f}")
        print(f"  Vanilla interpenetration: {van_ip:.4f}")
        print(f"  Ratio (CCS/vanilla): {ccs_ip / (van_ip + 1e-10):.3f}")
        print(f"  {'CCS > VANILLA: qualitative multiplicity confirmed' if ccs_ip > van_ip else 'VANILLA >= CCS: no interpenetration advantage'}")

        all_results[model_name] = {
            "n_layers": n_layers,
            "conditions": model_results,
            "ccs_vanilla_ratio": ccs_ip / (van_ip + 1e-10),
        }

        del model
        torch.cuda.empty_cache()

    out_path = Path(__file__).parent / "results" / f"temporal_interpenetration_{datetime.now():%Y%m%d_%H%M%S}.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
