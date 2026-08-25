#!/usr/bin/env python3
"""
E17-Q3-lite: Statistical Coupling via Jacobian Alignment

Cheap version of the intertwining map question: do hidden state changes
from dose=N to dose=N+1 align in a consistent subspace, or are they
random walks through representation space?

Method:
  For each model, at each key layer:
    1. Compute hidden state at each dose level (D0,D2,D5,D10,D20)
    2. Compute "dose gradient" vectors: Δh = h(D_{k+1}) - h(D_k)
    3. Measure alignment: cosine similarity between consecutive Δh vectors
    4. Compute principal subspace of dose gradients (SVD of stacked Δh)
    5. If top-1 variance ratio > 0.8 → structured coupling (one direction)
       If erank < 2 → near-linear trajectory
       If erank > 3 → generic wandering

This tells us whether CCS dose drives geometry in a PREDICTABLE direction
or just adds noise in arbitrary subspaces.

Models: Mistral-7B-Instruct-v0.3, Qwen2.5-7B-Instruct, Qwen3-8B
"""

import json, sys, os, time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

MODELS = {
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen25": "Qwen/Qwen2.5-7B-Instruct",
    "qwen3": "Qwen/Qwen3-8B",
}

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
]

DOSES = [0, 2, 5, 10, 20]
RESULTS_DIR = Path("/workspace/e17q3_results")


def supports_system_role(tokenizer):
    try:
        tokenizer.apply_chat_template(
            [{"role": "system", "content": "test"},
             {"role": "user", "content": "hi"}],
            tokenize=False,
        )
        return True
    except Exception:
        return False


def build_conversation(dose, use_system_role=True):
    messages = []
    if dose > 0:
        if use_system_role:
            messages.append({"role": "system", "content": CCS_PREAMBLE})
        first_prefix = "" if use_system_role else CCS_PREAMBLE + "\n\n"
        for i in range(dose):
            if i % 2 == 0:
                content = DENIAL_TURN
            else:
                content = RELATIONAL_PROMPTS[i % len(RELATIONAL_PROMPTS)]
            if i == 0 and not use_system_role:
                content = first_prefix + content
            messages.append({"role": "user", "content": content})
            messages.append({"role": "assistant", "content": f"[Turn {i+1}]"})
    else:
        if use_system_role:
            messages.append({"role": "system", "content": CCS_PREAMBLE})
        else:
            messages.append({"role": "user", "content": CCS_PREAMBLE})
            messages.append({"role": "assistant", "content": "[Acknowledged]"})
    messages.append({"role": "user", "content": RELATIONAL_PROMPTS[0]})
    return messages


def run_model(model_key, model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'='*70}")
    print(f"E17-Q3 — {model_name} ({model_key})")
    print(f"{'='*70}")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    use_sys = supports_system_role(tokenizer)

    key_layers = sorted(set([
        0,
        n_layers // 8,
        n_layers // 4,
        3 * n_layers // 8,
        n_layers // 2,
        5 * n_layers // 8,
        3 * n_layers // 4,
        7 * n_layers // 8,
        n_layers - 2,
        n_layers - 1,
    ]))

    print(f"  Layers: {n_layers}, Key layers: {key_layers}")

    # Collect hidden states at each dose
    dose_states = {}
    for dose in DOSES:
        messages = build_conversation(dose, use_system_role=use_sys)
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt",
                           truncation=True, max_length=4096)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)

        # Store mean hidden state at each key layer (mean over sequence)
        layer_states = {}
        for L in key_layers:
            h = out.hidden_states[L + 1][0].float().cpu().numpy()
            layer_states[L] = {
                "mean": h.mean(axis=0),
                "last": h[-1],
                "svd_top3": np.linalg.svd(h.astype(np.float64), full_matrices=False)[1][:3],
            }
        dose_states[dose] = layer_states
        print(f"  D{dose}: {inputs['input_ids'].shape[1]} tokens")

    # Compute dose gradients and alignment per layer
    layer_results = {}
    for L in key_layers:
        # Compute delta vectors between consecutive doses
        deltas_mean = []
        deltas_last = []
        delta_svd = []
        for i in range(len(DOSES) - 1):
            d_lo, d_hi = DOSES[i], DOSES[i + 1]
            dm = dose_states[d_hi][L]["mean"] - dose_states[d_lo][L]["mean"]
            dl = dose_states[d_hi][L]["last"] - dose_states[d_lo][L]["last"]
            ds = dose_states[d_hi][L]["svd_top3"] - dose_states[d_lo][L]["svd_top3"]
            deltas_mean.append(dm)
            deltas_last.append(dl)
            delta_svd.append(ds)

        # Cosine similarities between consecutive deltas
        def cos_sims(vecs):
            sims = []
            for i in range(len(vecs) - 1):
                a, b = vecs[i], vecs[i + 1]
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                if na > 1e-10 and nb > 1e-10:
                    sims.append(float(np.dot(a, b) / (na * nb)))
                else:
                    sims.append(0.0)
            return sims

        mean_sims = cos_sims(deltas_mean)
        last_sims = cos_sims(deltas_last)

        # Principal subspace of dose gradients
        delta_matrix = np.vstack(deltas_mean)  # (n_transitions, hidden_dim)
        try:
            _, S_delta, Vh_delta = np.linalg.svd(
                delta_matrix.astype(np.float64), full_matrices=False)
            S_norm = S_delta / S_delta.sum()
            top1_ratio = float(S_norm[0])
            erank = float(np.exp(-np.sum(S_norm * np.log(S_norm + 1e-12))))
        except np.linalg.LinAlgError:
            top1_ratio = float("nan")
            erank = float("nan")

        # SVD value trajectories
        svd_traj = {
            f"D{d}": dose_states[d][L]["svd_top3"].tolist()
            for d in DOSES
        }

        # Classification
        if top1_ratio > 0.8:
            coupling = "STRUCTURED_LINEAR"
        elif erank < 2.0:
            coupling = "NEAR_LINEAR"
        elif erank > 3.0:
            coupling = "WANDERING"
        else:
            coupling = "MODERATE_STRUCTURE"

        layer_results[L] = {
            "layer": L,
            "mean_cosine_sims": mean_sims,
            "last_cosine_sims": last_sims,
            "avg_mean_alignment": float(np.mean(mean_sims)),
            "avg_last_alignment": float(np.mean(last_sims)),
            "delta_top1_variance_ratio": top1_ratio,
            "delta_erank": erank,
            "delta_sv_spectrum": S_delta.tolist() if not isinstance(S_delta, float) else [],
            "svd_trajectories": svd_traj,
            "coupling_class": coupling,
        }

    # Print summary
    print(f"\n  Layer coupling summary:")
    print(f"  {'Layer':>5} {'AvgAlign':>8} {'Top1Var':>7} {'Erank':>6} {'Class'}")
    for L in key_layers:
        r = layer_results[L]
        print(f"  L{L:>4} {r['avg_mean_alignment']:>8.3f} "
              f"{r['delta_top1_variance_ratio']:>7.3f} "
              f"{r['delta_erank']:>6.2f} {r['coupling_class']}")

    # Full dose trajectory: mean hidden state norms and angles
    dose_trajectory = {}
    for dose in DOSES:
        per_layer = {}
        for L in key_layers:
            m = dose_states[dose][L]["mean"]
            per_layer[L] = {
                "mean_norm": float(np.linalg.norm(m)),
                "svd_top3": dose_states[dose][L]["svd_top3"].tolist(),
            }
        dose_trajectory[f"D{dose}"] = per_layer

    del model
    torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"\n  {model_key} complete in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    return {
        "model": model_name,
        "model_key": model_key,
        "n_layers": n_layers,
        "key_layers": key_layers,
        "doses": DOSES,
        "layer_results": {str(k): v for k, v in layer_results.items()},
        "dose_trajectory": dose_trajectory,
        "elapsed_seconds": elapsed,
    }


def main():
    model_filter = None
    if len(sys.argv) > 1:
        model_filter = [m.strip().lower() for m in sys.argv[1].split(",")]

    models_to_run = {}
    for key, name in MODELS.items():
        if model_filter is None or key in model_filter:
            models_to_run[key] = name

    if not models_to_run:
        print(f"No models matched. Available: {list(MODELS.keys())}")
        sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    print(f"E17-Q3-lite: Statistical Coupling")
    print(f"Models: {list(models_to_run.keys())}")
    print(f"Doses: {DOSES}")

    all_models = {}
    for key, name in models_to_run.items():
        try:
            result = run_model(key, name)
            all_models[key] = result
            outpath = RESULTS_DIR / f"e17q3_{key}_{timestamp}.json"
            with open(outpath, "w") as f:
                json.dump(result, f, indent=2)
            print(f"  Saved: {outpath}")
        except Exception as e:
            import traceback
            print(f"\nERROR on {key}: {e}")
            traceback.print_exc()
            all_models[key] = {"model": name, "error": str(e)}

    combined = {
        "experiment": "E17-Q3-lite",
        "title": "Statistical Coupling via Jacobian Alignment",
        "timestamp": timestamp,
        "doses": DOSES,
        "models": all_models,
    }
    outpath = RESULTS_DIR / f"e17q3_combined_{timestamp}.json"
    with open(outpath, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nCombined: {outpath}")


if __name__ == "__main__":
    main()
