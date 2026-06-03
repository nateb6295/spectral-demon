"""
Exp 14: Falcon 7B Base Control
Addresses paper limitation: "A Falcon base model (without instruction tuning) would disambiguate."

Tests whether the sign inversion (Finding 11: ΔS = -0.076 on Falcon instruct) is from:
  H1: IT on MHA substrate → Falcon base should show ΔS ≈ 0 (like Pythia base)
  H2: Falcon architecture itself → Falcon base should show ΔS < 0

Also measures passage distance to compare with:
  - Pythia 6.9B base: d ≈ 1.93 (Exp 11)
  - Falcon instruct: d from Part II Exp 6

Models: tiiuae/falcon-7b (base), tiiuae/falcon-7b-instruct (instruct, replication)
Relay: L30 (from Part I §3.20, ~94% depth)
Conditions: control, receptive, absent
Probes: 10 identity probes
k=5 passage distance
"""

import torch
import numpy as np
import json
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "falcon_7b_base": {
        "name": "tiiuae/falcon-7b",
        "relay_layer": 30,
        "n_layers": 32,
    },
    "falcon_7b_instruct": {
        "name": "tiiuae/falcon-7b-instruct",
        "relay_layer": 30,
        "n_layers": 32,
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


def spectral_entropy(H):
    """Spectral entropy of hidden states matrix H (n_tokens x d_model)."""
    C = H.T @ H
    eigenvalues = torch.linalg.eigvalsh(C)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    p = eigenvalues / eigenvalues.sum()
    return -(p * torch.log(p)).sum().item()


def participation_ratio(H):
    """Participation ratio from eigenvalues."""
    C = H.T @ H
    eigenvalues = torch.linalg.eigvalsh(C)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    return (eigenvalues.sum() ** 2 / (eigenvalues ** 2).sum()).item()


def passage_distance(H_input, H_relay, k=5):
    """Grassmannian distance between top-k subspaces."""
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


def format_prompt_falcon(system, user, is_instruct=True):
    """Format prompt for Falcon models."""
    if is_instruct:
        return f"System: {system}\nUser: {user}\nAssistant:"
    else:
        return f"{system}\n\nQuestion: {user}\nAnswer:"


def run_experiment():
    results = {}
    start = time.time()

    for model_key, model_info in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_key} (relay at L{model_info['relay_layer']}/{model_info['n_layers']})")

        is_instruct = "instruct" in model_key

        print(f"  Loading {model_info['name']}...")
        tokenizer = AutoTokenizer.from_pretrained(model_info["name"], trust_remote_code=True)
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

        results[model_key] = {"conditions": {}}
        relay_layer = model_info["relay_layer"]

        for cond_name, cond in CONDITIONS.items():
            S_vals, d_vals, pr_vals = [], [], []

            for probe in IDENTITY_PROBES:
                cond_with_probe = dict(cond)
                cond_with_probe["user"] = probe

                prompt = format_prompt_falcon(
                    cond_with_probe["system"],
                    cond_with_probe["user"],
                    is_instruct=is_instruct,
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

        del model
        torch.cuda.empty_cache()

    elapsed = time.time() - start
    results["meta"] = {
        "experiment": "exp14_falcon_base_control",
        "total_time_s": elapsed,
        "n_models": len(MODELS),
        "n_conditions": len(CONDITIONS),
        "n_probes": len(IDENTITY_PROBES),
        "k": 5,
        "total_forward_passes": len(MODELS) * len(CONDITIONS) * len(IDENTITY_PROBES),
        "hypotheses": {
            "H1": "IT on MHA produces inversion → Falcon base ΔS ≈ 0",
            "H2": "Falcon architecture constrains → Falcon base ΔS < 0",
        },
    }

    with open("/workspace/exp14_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"COMPLETE in {elapsed:.0f}s. Results saved to /workspace/exp14_results.json")
    print(f"Total forward passes: {results['meta']['total_forward_passes']}")

    # Quick analysis
    for mk in ["falcon_7b_base", "falcon_7b_instruct"]:
        if mk in results and "conditions" in results[mk]:
            c = results[mk]["conditions"]
            ds = c["receptive"]["S"] - c["absent"]["S"]
            print(f"\n{mk}: ΔS(R-A) = {ds:+.4f}")
            if mk == "falcon_7b_base":
                if abs(ds) < 0.01:
                    print("  → H1 SUPPORTED: ΔS ≈ 0, IT produces the inversion")
                elif ds < -0.02:
                    print("  → H2 SUPPORTED: architecture itself constrains")
                else:
                    print("  → AMBIGUOUS: weak effect")


if __name__ == "__main__":
    run_experiment()
