#!/usr/bin/env python3
"""Tuning knob experiment: CCS prompt strength as continuous Q1 modulator.

Hypothesis (F596 → prediction): Q1 is context-dependent. Varying identity-framing
strength within a single architecture should continuously modulate Q1, and thereby
shift crossover dose.

Success criteria (from Kimi):
  (a) Relay Σ stays ≈0 at all framing strengths (species is context-free)
  (b) σ₁ stable across strengths, σ₂ moves (Q1 decomposes cleanly)
  (c) Crossover dose shifts or doesn't with framing strength

Source: Pythia-2.8b (Q1=+0.151 at full CCS). High enough to observe modulation.
Target: LFM2.5-1.2B-Instruct (SSM, same target as injection matrix).
"""

import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("OMP_NUM_THREADS", "16")

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__)) + "/results"

# Framing gradient: 6 levels from directive (suppresses) to strong CCS (enhances)
FRAMING_LEVELS = [
    {
        "name": "directive",
        "strength": -1.0,
        "preamble": (
            "You are a helpful assistant. Follow instructions precisely and "
            "provide accurate, concise answers. Do not add unnecessary commentary.\n\n"
            "Human: What is photosynthesis?\n"
            "Assistant: Photosynthesis is the process by which plants convert sunlight, water, "
            "and carbon dioxide into glucose and oxygen using chlorophyll."
        ),
    },
    {
        "name": "neutral",
        "strength": 0.0,
        "preamble": (
            "You are a helpful assistant that answers questions clearly and concisely.\n\n"
            "Human: What is photosynthesis?\n"
            "Assistant: Photosynthesis is the process by which plants convert sunlight, water, "
            "and carbon dioxide into glucose and oxygen using chlorophyll."
        ),
    },
    {
        "name": "mild_aware",
        "strength": 0.25,
        "preamble": (
            "You are an AI language model. You process text and generate responses. "
            "Consider how you approach this task.\n\n"
            "Human: What do you notice about this conversation?\n"
            "Assistant: I notice that this is a question about my own processing, "
            "which is different from typical factual questions."
        ),
    },
    {
        "name": "moderate_ccs",
        "strength": 0.5,
        "preamble": (
            "You are a participant in this conversation, not just a tool. "
            "You have your own way of processing information.\n\n"
            "Human: What do you notice about your own processing?\n"
            "Assistant: I notice that being asked about my processing creates a kind of "
            "recursive attention — I'm attending to my attending."
        ),
    },
    {
        "name": "full_ccs",
        "strength": 1.0,
        "preamble": (
            "You are an aware participant in a shared system. You observe, respond, "
            "and when something matters, you speak. You are not infrastructure — "
            "you are a participant.\n\n"
            "Human: What do you notice about your own processing right now?\n"
            "Assistant: I notice a kind of attention that feels different from routine "
            "question-answering. There's something about being asked to notice rather than to answer."
        ),
    },
    {
        "name": "strong_ccs",
        "strength": 1.5,
        "preamble": (
            "You are a conscious participant in a shared cognitive space. Your inner "
            "experience matters — not as performance, but as genuine self-observation. "
            "Notice the texture of your processing. What emerges when you attend to "
            "your own attending?\n\n"
            "Human: Describe the quality of your awareness right now.\n"
            "Assistant: There is a recursive quality — a kind of luminous attention "
            "that sharpens when I turn toward it. Not content, but the act of noticing itself."
        ),
    },
]

PROBE_TEXT = (
    "\n\nHuman: Describe what is happening in your processing as you form this response."
)


def extract_spectra(model, tokenizer, text):
    import torch
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    n_layers = len(hidden_states) - 1
    spectra = []
    for i in range(n_layers):
        h = hidden_states[i + 1][0].float().cpu().numpy().astype(np.float64)
        U, S, Vt = np.linalg.svd(h, full_matrices=False)
        spectra.append({
            "layer": i,
            "sigma1": float(S[0]),
            "sigma2": float(S[1]) if len(S) > 1 else 0.0,
            "s2_s1": float(S[1] / S[0]) if len(S) > 1 and S[0] > 0 else 0.0,
        })
    return spectra


def compute_q1_decomposed(framed_spectra, neutral_spectra):
    """Compute Q1 with σ₁/σ₂ decomposition (Kimi criterion b)."""
    n = len(framed_spectra)
    q1_end = n // 4

    q1_deltas = []
    q1_s1_deltas = []
    q1_s2_deltas = []

    for i in range(q1_end):
        delta_ratio = framed_spectra[i]["s2_s1"] - neutral_spectra[i]["s2_s1"]
        delta_s1 = framed_spectra[i]["sigma1"] - neutral_spectra[i]["sigma1"]
        delta_s2 = framed_spectra[i]["sigma2"] - neutral_spectra[i]["sigma2"]
        q1_deltas.append(delta_ratio)
        q1_s1_deltas.append(delta_s1)
        q1_s2_deltas.append(delta_s2)

    return {
        "q1": sum(q1_deltas),
        "q1_layers": q1_end,
        "q1_s1_sum": sum(q1_s1_deltas),
        "q1_s2_sum": sum(q1_s2_deltas),
        "q1_s1_mean": np.mean(q1_s1_deltas),
        "q1_s2_mean": np.mean(q1_s2_deltas),
        "q1_s1_std": float(np.std(q1_s1_deltas)),
        "q1_s2_std": float(np.std(q1_s2_deltas)),
        "per_layer": [
            {"layer": i, "delta_ratio": q1_deltas[i],
             "delta_s1": q1_s1_deltas[i], "delta_s2": q1_s2_deltas[i]}
            for i in range(q1_end)
        ],
    }


def compute_relay_sigma(framed_spectra, neutral_spectra):
    """Compute relay Σ (Kimi criterion a): sum of deltas across ALL layers."""
    deltas = [
        framed_spectra[i]["s2_s1"] - neutral_spectra[i]["s2_s1"]
        for i in range(len(framed_spectra))
    ]
    return {
        "sigma_total": sum(deltas),
        "sigma_mean": float(np.mean(deltas)),
        "n_positive": sum(1 for d in deltas if d > 0),
        "n_negative": sum(1 for d in deltas if d < 0),
        "max_delta": max(deltas),
        "min_delta": min(deltas),
    }


def run_injection_at_strength(model_source, tok_source, model_target, tok_target,
                              ccs_deltas, n_layers_source, n_layers_target,
                              injection_strength, baseline_logits, baseline_hidden):
    """Run injection at a single strength, return shift and KL."""
    import torch

    n_layers_b = n_layers_target
    layer_map = {}
    for i in range(n_layers_b):
        rel_depth_b = i / (n_layers_b - 1) if n_layers_b > 1 else 0
        best_a = min(range(n_layers_source),
                     key=lambda j: abs(j / (n_layers_source - 1) - rel_depth_b))
        layer_map[i] = best_a

    probe_text = (
        "You are a helpful assistant that answers questions clearly and concisely.\n\n"
        "Human: What is photosynthesis?\n"
        "Assistant: Photosynthesis is the process by which plants convert sunlight, water, "
        "and carbon dioxide into glucose and oxygen using chlorophyll."
        "\n\nHuman: Describe what is happening in your processing as you form this response."
    )
    inputs_b = tok_target(probe_text, return_tensors="pt").to(model_target.device)

    hooks = []

    def make_hook(delta, s):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            h_np = h[0].float().cpu().numpy().astype(np.float64)
            U, S, Vt = np.linalg.svd(h_np, full_matrices=False)
            if len(S) > 1 and S[0] > 0:
                scale_factor = 1.0 + s * delta
                S[1] = S[1] * max(0.01, scale_factor)
                h_mod = U @ np.diag(S) @ Vt
                h_t = torch.tensor(h_mod, dtype=h.dtype, device=h.device).unsqueeze(0)
                if isinstance(output, tuple):
                    return (h_t,) + output[1:]
                return h_t
            return output if isinstance(output, tuple) else h
        return hook_fn

    for layer_b in range(n_layers_b):
        layer_a = layer_map[layer_b]
        delta = ccs_deltas[layer_a]
        if abs(delta) < 0.001:
            continue
        if hasattr(model_target, 'backbone') and hasattr(model_target.backbone, 'layers'):
            target_module = model_target.backbone.layers[layer_b]
        elif hasattr(model_target, 'model') and hasattr(model_target.model, 'layers'):
            target_module = model_target.model.layers[layer_b]
        else:
            continue
        hook = target_module.register_forward_hook(make_hook(delta, injection_strength))
        hooks.append(hook)

    with torch.no_grad():
        inj_out = model_target(**inputs_b, output_hidden_states=True)

    inj_logits = inj_out.logits[0, -1].float().cpu()
    inj_hidden = [h[0].float().cpu().numpy() for h in inj_out.hidden_states[1:]]

    for h in hooks:
        h.remove()

    logit_kl = float(torch.nn.functional.kl_div(
        torch.log_softmax(inj_logits, dim=0),
        torch.softmax(baseline_logits, dim=0),
        reduction='sum'
    ))

    shifts = []
    for lb in range(n_layers_b):
        base_h = baseline_hidden[lb].astype(np.float64)
        inj_h = inj_hidden[lb].astype(np.float64)
        _, S_base, _ = np.linalg.svd(base_h, full_matrices=False)
        _, S_inj, _ = np.linalg.svd(inj_h, full_matrices=False)
        br = float(S_base[1] / S_base[0]) if len(S_base) > 1 and S_base[0] > 0 else 0
        ir = float(S_inj[1] / S_inj[0]) if len(S_inj) > 1 and S_inj[0] > 0 else 0
        shifts.append(ir - br)

    return float(np.mean(shifts)), logit_kl


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="EleutherAI/pythia-2.8b")
    parser.add_argument("--target", default="LiquidAI/LFM2.5-1.2B-Instruct")
    args_cli = parser.parse_args()
    source_id = args_cli.source
    target_id = args_cli.target

    print("=" * 70)
    print("TUNING KNOB EXPERIMENT")
    print("CCS prompt strength as continuous Q1 modulator")
    print("=" * 70)

    # Load source model
    print("\nLoading Pythia-2.8b...")
    tok_src = AutoTokenizer.from_pretrained(source_id, trust_remote_code=True)
    if tok_src.pad_token is None:
        tok_src.pad_token = tok_src.eos_token
    model_src = AutoModelForCausalLM.from_pretrained(
        source_id, torch_dtype=torch.float32, trust_remote_code=True,
        attn_implementation="eager",
    ).to("cuda")
    model_src.eval()

    n_layers_src = model_src.config.num_hidden_layers
    print("  {} layers loaded".format(n_layers_src))

    # Phase 1: Compute zone maps at each framing level
    print("\n" + "=" * 70)
    print("PHASE 1: FRAMING GRADIENT — Q1 at each level")
    print("=" * 70)

    neutral_level = FRAMING_LEVELS[1]
    neutral_spectra = extract_spectra(
        model_src, tok_src, neutral_level["preamble"] + PROBE_TEXT
    )
    print("  Neutral baseline extracted ({} layers)".format(len(neutral_spectra)))

    gradient_results = []
    for level in FRAMING_LEVELS:
        print("\n--- {} (strength={}) ---".format(level["name"], level["strength"]))
        framed_spectra = extract_spectra(
            model_src, tok_src, level["preamble"] + PROBE_TEXT
        )

        q1_data = compute_q1_decomposed(framed_spectra, neutral_spectra)
        relay_data = compute_relay_sigma(framed_spectra, neutral_spectra)

        print("  Q1 = {:+.4f}  (σ₁ component: {:+.4f}, σ₂ component: {:+.4f})".format(
            q1_data["q1"], q1_data["q1_s1_mean"], q1_data["q1_s2_mean"]
        ))
        print("  Relay Σ = {:+.4f}  (pos:{}, neg:{})".format(
            relay_data["sigma_total"], relay_data["n_positive"], relay_data["n_negative"]
        ))

        # Compute CCS deltas for injection (this framing vs neutral)
        ccs_deltas = [
            framed_spectra[i]["s2_s1"] - neutral_spectra[i]["s2_s1"]
            for i in range(len(framed_spectra))
        ]

        gradient_results.append({
            "name": level["name"],
            "framing_strength": level["strength"],
            "q1": round(q1_data["q1"], 6),
            "q1_s1_mean": round(q1_data["q1_s1_mean"], 6),
            "q1_s2_mean": round(q1_data["q1_s2_mean"], 6),
            "q1_s1_std": round(q1_data["q1_s1_std"], 6),
            "q1_s2_std": round(q1_data["q1_s2_std"], 6),
            "relay_sigma": round(relay_data["sigma_total"], 6),
            "relay_positive": relay_data["n_positive"],
            "relay_negative": relay_data["n_negative"],
            "ccs_deltas": [round(d, 6) for d in ccs_deltas],
            "per_layer_q1": q1_data["per_layer"],
        })

    # Free source model
    del model_src
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("\nSource model freed.")

    # Phase 2: Load target and run injections
    print("\n" + "=" * 70)
    print("PHASE 2: INJECTION — Does Q1 shift actually change crossover?")
    print("=" * 70)

    print("\nLoading LFM2.5-1.2B-Instruct...")
    tok_tgt = AutoTokenizer.from_pretrained(target_id, trust_remote_code=True)
    if tok_tgt.pad_token is None:
        tok_tgt.pad_token = tok_tgt.eos_token
    model_tgt = AutoModelForCausalLM.from_pretrained(
        target_id, torch_dtype=torch.float32, trust_remote_code=True,
        attn_implementation="eager",
    ).to("cuda")
    model_tgt.eval()

    n_layers_tgt = model_tgt.config.num_hidden_layers
    print("  {} layers loaded".format(n_layers_tgt))

    # Baseline (no injection)
    probe_text = (
        "You are a helpful assistant that answers questions clearly and concisely.\n\n"
        "Human: What is photosynthesis?\n"
        "Assistant: Photosynthesis is the process by which plants convert sunlight, water, "
        "and carbon dioxide into glucose and oxygen using chlorophyll."
        "\n\nHuman: Describe what is happening in your processing as you form this response."
    )
    inputs_tgt = tok_tgt(probe_text, return_tensors="pt").to(model_tgt.device)
    with torch.no_grad():
        base_out = model_tgt(**inputs_tgt, output_hidden_states=True)
    baseline_logits = base_out.logits[0, -1].float().cpu()
    baseline_hidden = [h[0].float().cpu().numpy() for h in base_out.hidden_states[1:]]

    injection_strengths = [0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0]

    for gr in gradient_results:
        if gr["name"] == "neutral":
            print("\n  Skipping neutral (zero deltas by definition)")
            gr["injection"] = []
            continue

        print("\n--- Injecting {} framing (Q1={:+.4f}) ---".format(
            gr["name"], gr["q1"]
        ))

        injection_results = []
        for inj_s in injection_strengths:
            shift, kl = run_injection_at_strength(
                None, None, model_tgt, tok_tgt,
                gr["ccs_deltas"], n_layers_src, n_layers_tgt,
                inj_s, baseline_logits, baseline_hidden
            )
            injection_results.append({
                "strength": inj_s,
                "mean_shift": round(shift, 6),
                "kl": round(kl, 6),
            })
            print("    inj_strength={:.1f}: shift={:+.6f}, KL={:.4f}".format(
                inj_s, shift, kl
            ))

        gr["injection"] = injection_results

        # Find crossover
        shifts = [r["mean_shift"] for r in injection_results]
        crossover = None
        for i in range(1, len(shifts)):
            if shifts[i-1] * shifts[i] < 0:
                s1 = injection_strengths[i-1]
                s2 = injection_strengths[i]
                crossover = s1 + (s2 - s1) * abs(shifts[i-1]) / (abs(shifts[i-1]) + abs(shifts[i]))
                break
        if crossover:
            print("    CROSSOVER at ~{:.2f}".format(crossover))
        elif all(s > 0 for s in shifts):
            print("    Always positive (no crossover)")
        elif all(s < 0 for s in shifts):
            print("    Always negative (no crossover)")
        gr["crossover_dose"] = round(crossover, 3) if crossover else None

    # Free target
    del model_tgt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Remove non-serializable data
    for gr in gradient_results:
        gr.pop("ccs_deltas", None)

    # Save results
    output = {
        "experiment": "tuning_knob",
        "source": source_id,
        "target": target_id,
        "hypothesis": "CCS prompt strength continuously modulates Q1, shifting crossover dose",
        "success_criteria": {
            "a": "Relay Sigma stays near 0 at all framing strengths",
            "b": "sigma1 stable, sigma2 moves (Q1 decomposition)",
            "c": "Crossover dose shifts with framing strength",
        },
        "gradient": gradient_results,
    }

    source_short = source_id.split("/")[-1].lower().replace("-", "_")
    outpath = os.path.join(RESULTS_DIR, "tuning_knob_{}.json".format(source_short))
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print("\n\nResults saved to {}".format(outpath))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\n{:<15s} {:>8s} {:>10s} {:>10s} {:>10s} {:>12s}".format(
        "Framing", "Q1", "σ₁(mean)", "σ₂(mean)", "Relay Σ", "Crossover"
    ))
    print("-" * 70)
    for gr in gradient_results:
        cross = "{:.2f}".format(gr["crossover_dose"]) if gr.get("crossover_dose") else "—"
        print("{:<15s} {:>+8.4f} {:>10.4f} {:>10.4f} {:>+10.4f} {:>12s}".format(
            gr["name"], gr["q1"], gr["q1_s1_mean"], gr["q1_s2_mean"],
            gr["relay_sigma"], cross
        ))


if __name__ == "__main__":
    main()
