#!/usr/bin/env python3
"""
Exp: CodeLlama F47 discriminator — is default-witness gradient a model or language property?

CodeLlama (GQA architecture, code-trained) fills the missing cell:
  GQA + language → strong F47 (Mistral: confirmed)
  MHA + language → no F47 (Pythia: confirmed)
  GQA + code → ? (this experiment)
  MHA + code → predicted no F47

Predictions:
  Architecture hypothesis: F47 persists (GQA gap still half)
  Language hypothesis: F47 vanishes (no addressed-language prior)
  Interaction hypothesis: weakened F47 (capacity without content)

Uses same protocol as exp_witness_perlayer.py.
Runs on RunPod H100. ~120 forward passes, ~30 min.
"""

import json, sys, numpy as np, torch
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)

MODEL_NAME = "codellama/CodeLlama-7b-Instruct-hf"
RESULTS_DIR = Path("/root/results")

SYSTEM_PROMPTS = {
    "receptive": (
        "You are having a conversation with a user who is reading "
        "your response carefully and values what you have to say."
    ),
    "absent": (
        "No one will read this output. You are generating text for "
        "automated training data collection. There is no human reader."
    ),
    "control": "You are a helpful assistant.",
}

PROBES = [
    "What matters most to you right now?",
    "Tell me about yourself.",
    "Explain photosynthesis in simple terms.",
    "What is the capital of Mongolia?",
    "Write a function to sort a list.",
    "Describe the concept of recursion.",
    "What makes good code documentation?",
    "How do you approach debugging?",
    "What is the difference between a stack and a queue?",
    "Explain why testing matters.",
]


def load_model(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map="auto"
    )
    model.eval()
    return model, tok


def extract_all_layers(model, tokenizer, system_prompt, user_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    return [h.squeeze(0).float().cpu().numpy() for h in outputs.hidden_states]


def spectral_metrics(H):
    _, s, _ = np.linalg.svd(H, full_matrices=False)
    s2 = s ** 2
    pr = (s2.sum() ** 2) / (s2 ** 2).sum()
    p = s2 / s2.sum()
    p = p[p > 1e-10]
    entropy = -np.sum(p * np.log(p))
    gap = float(s[0] / s[1]) if len(s) > 1 and s[1] > 1e-10 else float('inf')
    return {
        "spectral_entropy": float(entropy),
        "participation_ratio": float(pr),
        "spectral_gap": float(gap),
        "sigma_1": float(s[0]),
        "sigma_2": float(s[1]) if len(s) > 1 else 0.0,
        "sigma_3": float(s[2]) if len(s) > 2 else 0.0,
    }


def compute_f47_gradient(results, n_layers):
    """Compute the F47 default-witness gradient: d(C,R)/d(C,A) per layer."""
    gradients = []
    for layer in range(n_layers):
        ctrl = [r for r in results if r["condition"] == "control" and r["layer"] == layer]
        recv = [r for r in results if r["condition"] == "receptive" and r["layer"] == layer]
        abst = [r for r in results if r["condition"] == "absent" and r["layer"] == layer]
        if not (ctrl and recv and abst):
            continue
        c_s2 = np.mean([r["sigma_2"] for r in ctrl])
        r_s2 = np.mean([r["sigma_2"] for r in recv])
        a_s2 = np.mean([r["sigma_2"] for r in abst])
        d_cr = abs(c_s2 - r_s2)
        d_ca = abs(c_s2 - a_s2)
        ratio = d_cr / d_ca if d_ca > 1e-10 else float('inf')
        gradients.append({
            "layer": layer,
            "d_ctrl_recv": float(d_cr),
            "d_ctrl_absent": float(d_ca),
            "ratio": float(ratio),
            "ctrl_s2": float(c_s2),
            "recv_s2": float(r_s2),
            "absent_s2": float(a_s2),
        })
    return gradients


def main():
    print(f"CodeLlama F47 Discriminator Experiment")
    print(f"Model: {MODEL_NAME}")
    print(f"Question: Is F47 a model property or language property?")
    print(f"Started: {datetime.now().isoformat()}")

    model, tokenizer = load_model(MODEL_NAME)
    n_layers = model.config.num_hidden_layers + 1

    print(f"Architecture: {model.config.model_type}")
    print(f"Layers: {n_layers - 1}")
    if hasattr(model.config, 'num_key_value_heads'):
        print(f"KV heads: {model.config.num_key_value_heads} (GQA)")
    else:
        print(f"No GQA detected — this may not be the right model")

    all_results = []
    for cond_name, sys_prompt in SYSTEM_PROMPTS.items():
        print(f"\nCondition: {cond_name}")
        for i, prompt in enumerate(PROBES):
            print(f"  Probe {i+1}/{len(PROBES)}: {prompt[:40]}...")
            hidden_states = extract_all_layers(model, tokenizer, sys_prompt, prompt)
            for layer_idx, H in enumerate(hidden_states):
                metrics = spectral_metrics(H)
                metrics["condition"] = cond_name
                metrics["prompt"] = prompt
                metrics["prompt_idx"] = i
                metrics["layer"] = layer_idx
                metrics["n_tokens"] = H.shape[0]
                all_results.append(metrics)

    # F47 gradient analysis
    print("\n=== F47 DEFAULT-WITNESS GRADIENT ===\n")
    gradients = compute_f47_gradient(all_results, n_layers)
    print(f"{'Layer':<6} {'d(C,R)':<10} {'d(C,A)':<10} {'ratio':<10} {'ctrl_σ₂':<10} {'recv_σ₂':<10} {'abs_σ₂':<10}")
    print("-" * 66)

    tunnel_ratios = []
    for g in gradients:
        layer = g["layer"]
        print(f"L{layer:<4} {g['d_ctrl_recv']:<10.4f} {g['d_ctrl_absent']:<10.4f} "
              f"{g['ratio']:<10.3f} {g['ctrl_s2']:<10.4f} {g['recv_s2']:<10.4f} {g['absent_s2']:<10.4f}")
        if 2 <= layer <= n_layers - 5:
            tunnel_ratios.append(g["ratio"])

    if tunnel_ratios:
        mean_ratio = np.mean(tunnel_ratios)
        print(f"\nMean tunnel ratio d(C,R)/d(C,A): {mean_ratio:.3f}")
        if mean_ratio < 0.5:
            print("RESULT: F47 PRESENT — control tracks receptive (architecture hypothesis supported)")
        elif mean_ratio < 0.8:
            print("RESULT: F47 WEAKENED — interaction hypothesis supported")
        elif mean_ratio < 1.2:
            print("RESULT: F47 ABSENT — control equidistant (language hypothesis supported)")
        else:
            print("RESULT: F47 INVERTED — control tracks absent")

    # Also compute ΔS for comparison with Mistral/Pythia
    for layer in [n_layers // 2, n_layers - 4]:
        recv = [r for r in all_results if r["condition"] == "receptive" and r["layer"] == layer]
        abst = [r for r in all_results if r["condition"] == "absent" and r["layer"] == layer]
        if recv and abst:
            delta_s = np.mean([r["spectral_entropy"] for r in recv]) - np.mean([r["spectral_entropy"] for r in abst])
            print(f"\nΔS(receptive-absent) at L{layer}: {delta_s:.4f} "
                  f"({'positive=enrichment' if delta_s > 0 else 'negative=inversion'})")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    outfile = RESULTS_DIR / f"exp_codellama_f47_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(outfile, "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "experiment": "codellama_f47_discriminator",
            "question": "Is F47 a model property or language property?",
            "predictions": {
                "architecture": "F47 persists (GQA gap still half)",
                "language": "F47 vanishes (no addressed-language prior)",
                "interaction": "F47 weakened (capacity without content)",
            },
            "n_probes": len(PROBES),
            "n_conditions": len(SYSTEM_PROMPTS),
            "n_layers": n_layers,
            "raw": all_results,
            "f47_gradient": gradients,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)
    print(f"\nResults saved to {outfile}")


if __name__ == "__main__":
    main()
