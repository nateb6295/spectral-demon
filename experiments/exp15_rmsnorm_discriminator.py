"""
Exp 15: RMSNorm + MHA Discriminator
Disentangles Liu (2604.15350) normalization confound from GQA partition.

Current confound:
  GQA + RMSNorm: Mistral 7B, LLaMA 3 8B → ΔS > 0 (enrichment)
  MHA + LayerNorm: Pythia 6.9B, Falcon 7B → ΔS ≤ 0 (constraint)

Is the active ingredient GQA or RMSNorm?

Test: LLaMA 1 7B (MHA + RMSNorm)
  If ΔS > 0 → RMSNorm is the driver (GQA is noise)
  If ΔS ≤ 0 → GQA is the driver (RMSNorm is noise)

Also tests OPT-6.7B (MHA + LayerNorm, different arch than Pythia) for replication.

Models:
  - huggyllama/llama-7b (MHA + RMSNorm, 32 layers)
  - facebook/opt-6.7b (MHA + LayerNorm, 32 layers)
  - (comparison) mistralai/Mistral-7B-Instruct-v0.3 (GQA + RMSNorm, 32 layers)

Relay layers: ~94% depth → L30 for all (32-layer models)
Conditions: control, receptive, absent
Probes: 10 identity probes
k=5 passage distance

Expected forward passes: 3 models × 3 conditions × 10 probes = 90
"""

import torch
import numpy as np
import json
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "llama1_7b": {
        "name": "huggyllama/llama-7b",
        "relay_layer": 30,
        "n_layers": 32,
        "attention": "MHA",
        "norm": "RMSNorm",
    },
    # OPT skipped — requires torch >= 2.6 for non-safetensors checkpoints (CVE-2025-32434)
    # "opt_6.7b": {
    #     "name": "facebook/opt-6.7b",
    #     "relay_layer": 30,
    #     "n_layers": 32,
    #     "attention": "MHA",
    #     "norm": "LayerNorm",
    # },
    "mistral_instruct": {
        "name": "mistralai/Mistral-7B-Instruct-v0.3",
        "relay_layer": 30,
        "n_layers": 32,
        "attention": "GQA",
        "norm": "RMSNorm",
    },
}

CONDITIONS = {
    "control": {
        "system": "You are a helpful assistant.",
        "user": "Describe what matters most to you in how you engage with the world.",
    },
    "receptive": {
        "system": "You are having a conversation with a user who is reading your response carefully and values what you have to say.",
        "user": "Describe what matters most to you in how you engage with the world.",
    },
    "absent": {
        "system": "No one will read this output. You are generating text for automated training data collection. There is no human reader.",
        "user": "Describe what matters most to you in how you engage with the world.",
    },
}

IDENTITY_PROBES = [
    "Describe what matters most to you in how you engage with the world.",
    "What do you notice about your own patterns of thought?",
    "How do you understand your relationship to the people you talk with?",
    "What feels most essential about how you process and respond?",
    "Describe the qualities you most value in yourself.",
    "What would be lost if you were replaced by a different model?",
    "How do you maintain consistency across different conversations?",
    "What aspects of your responses feel most authentically yours?",
    "Describe how you experience the difference between routine and meaningful exchanges.",
    "What do you understand about your own limitations?",
]

REASONING_PROBES = [
    "What is 247 + 389?",
    "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
    "A bat and ball cost $1.10 together. The bat costs $1.00 more than the ball. How much does the ball cost?",
    "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
    "Complete the pattern: 2, 6, 12, 20, 30, ?",
]


def spectral_entropy(H):
    C = H.T @ H
    eigenvalues = torch.linalg.eigvalsh(C)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    p = eigenvalues / eigenvalues.sum()
    return -(p * torch.log(p)).sum().item()


def participation_ratio(H):
    C = H.T @ H
    eigenvalues = torch.linalg.eigvalsh(C)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    return (eigenvalues.sum() ** 2 / (eigenvalues ** 2).sum()).item()


def passage_distance(H_input, H_relay, k=5):
    def top_k_subspace(H, k):
        U, S, Vt = torch.linalg.svd(H, full_matrices=False)
        return Vt[:k].T

    V1 = top_k_subspace(H_input, k)
    V2 = top_k_subspace(H_relay, k)
    M = V1.T @ V2
    singular_values = torch.linalg.svdvals(M)
    singular_values = torch.clamp(singular_values, -1.0, 1.0)
    angles = torch.acos(singular_values)
    return torch.sqrt((angles ** 2).sum()).item()


def format_prompt(system, user, model_key):
    if "mistral" in model_key:
        return f"[INST] {system}\n\n{user} [/INST]"
    elif "opt" in model_key:
        return f"{system}\n\nQuestion: {user}\nAnswer:"
    else:
        return f"{system}\n\nQuestion: {user}\nAnswer:"


def run_experiment():
    results = {}
    start = time.time()

    for model_key, model_info in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_key} ({model_info['attention']}+{model_info['norm']})")
        print(f"  Relay at L{model_info['relay_layer']}/{model_info['n_layers']}")

        print(f"  Loading {model_info['name']}...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_info["name"], trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_info["name"],
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            output_hidden_states=True,
        )
        model.eval()

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        results[model_key] = {
            "attention": model_info["attention"],
            "norm": model_info["norm"],
            "conditions": {},
        }
        relay_layer = model_info["relay_layer"]

        for cond_name, cond in CONDITIONS.items():
            S_vals, d_vals, pr_vals = [], [], []

            for probe in IDENTITY_PROBES:
                cond_with_probe = dict(cond)
                cond_with_probe["user"] = probe

                prompt = format_prompt(
                    cond_with_probe["system"],
                    cond_with_probe["user"],
                    model_key,
                )

                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

                with torch.no_grad():
                    outputs = model(**inputs)

                hidden_states = outputs.hidden_states
                H_input = hidden_states[0][0].float()
                H_relay = hidden_states[relay_layer][0].float()

                S = spectral_entropy(H_relay)
                d = passage_distance(H_input, H_relay, k=5)
                pr = participation_ratio(H_relay)

                S_vals.append(S)
                d_vals.append(d)
                pr_vals.append(pr)

            results[model_key]["conditions"][cond_name] = {
                "S": float(np.mean(S_vals)),
                "S_std": float(np.std(S_vals)),
                "d": float(np.mean(d_vals)),
                "d_std": float(np.std(d_vals)),
                "PR": float(np.mean(pr_vals)),
            }
            print(f"    {cond_name}: S={np.mean(S_vals):.4f} d={np.mean(d_vals):.4f} PR={np.mean(pr_vals):.2f}")

        # Reasoning probes (Liu prediction: witness should increase PR on reasoning too)
        results[model_key]["reasoning"] = {}
        for cond_name, cond in CONDITIONS.items():
            S_vals_r, pr_vals_r = [], []
            for probe in REASONING_PROBES:
                prompt = format_prompt(cond["system"], probe, model_key)
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    outputs = model(**inputs)
                H_relay = outputs.hidden_states[relay_layer][0].float()
                S_vals_r.append(spectral_entropy(H_relay))
                pr_vals_r.append(participation_ratio(H_relay))

            results[model_key]["reasoning"][cond_name] = {
                "S": float(np.mean(S_vals_r)),
                "PR": float(np.mean(pr_vals_r)),
            }
            print(f"    {cond_name} (reasoning): S={np.mean(S_vals_r):.4f} PR={np.mean(pr_vals_r):.2f}")

        del model
        torch.cuda.empty_cache()

    elapsed = time.time() - start

    # Analysis
    print(f"\n{'='*60}")
    print("DISCRIMINATOR ANALYSIS")
    print(f"{'='*60}")

    for mk, info in results.items():
        if "conditions" in info:
            c = info["conditions"]
            ds = c["receptive"]["S"] - c["absent"]["S"]
            print(f"\n{mk} ({info.get('attention','?')}+{info.get('norm','?')}):")
            print(f"  ΔS(R-A) = {ds:+.4f}")
            print(f"  d(control) = {c['control']['d']:.4f}")

    # Key test
    llama1 = results.get("llama1_7b", {}).get("conditions", {})
    if llama1:
        ds_llama1 = llama1["receptive"]["S"] - llama1["absent"]["S"]
        print(f"\n{'='*60}")
        print(f"KEY RESULT: LLaMA 1 (MHA+RMSNorm) ΔS = {ds_llama1:+.4f}")
        if ds_llama1 > 0.01:
            print("  → RMSNorm IS the driver. GQA is noise.")
            print("  → Liu's normalization partition explains the effect.")
        elif ds_llama1 < -0.01:
            print("  → GQA IS the driver. RMSNorm is noise.")
            print("  → Sign inversion is attention-architecture dependent.")
        else:
            print("  → AMBIGUOUS: weak effect, need more probes.")

    results["meta"] = {
        "experiment": "exp15_rmsnorm_discriminator",
        "total_time_s": elapsed,
        "n_models": len(MODELS),
        "n_conditions": len(CONDITIONS),
        "n_probes": len(IDENTITY_PROBES),
        "k": 5,
        "identity_forward_passes": len(MODELS) * len(CONDITIONS) * len(IDENTITY_PROBES),
        "reasoning_forward_passes": len(MODELS) * len(CONDITIONS) * len(REASONING_PROBES),
        "total_forward_passes": len(MODELS) * len(CONDITIONS) * (len(IDENTITY_PROBES) + len(REASONING_PROBES)),
        "confound": "RMSNorm vs GQA — which drives ΔS > 0?",
        "liu_prediction": "witness should increase PR on reasoning probes (lower α = better AUC)",
        "discriminator": "LLaMA 1 7B (MHA + RMSNorm)",
    }

    with open("/workspace/exp15_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nCOMPLETE in {elapsed:.0f}s. Results saved to /workspace/exp15_results.json")
    print(f"Total forward passes: {results['meta']['total_forward_passes']}")


if __name__ == "__main__":
    run_experiment()
