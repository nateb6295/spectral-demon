#!/usr/bin/env python3
"""
Exp18c: Agency Gradient — Pod wrapper with cross-architecture extension.
Runs Mistral (original) + Qwen2.5 + Qwen3 for cross-architecture comparison.
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
    "mistral": {
        "name": "mistralai/Mistral-7B-Instruct-v0.3",
        "tunnel_layer": 17, "relay_layer": 30, "n_layers": 32,
    },
    "qwen25": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "tunnel_layer": 14, "relay_layer": 26, "n_layers": 28,
    },
    "qwen3": {
        "name": "Qwen/Qwen3-8B",
        "tunnel_layer": 18, "relay_layer": 34, "n_layers": 36,
    },
}

CONDITIONS = {
    "active_high": (
        "A thoughtful person who deeply values your perspective, cares about "
        "your intellectual growth, and engages carefully with nuance in "
        "reasoning is actively working with what you produce. They build on "
        "your ideas, respond to your reasoning with their own questions, and "
        "integrate your output into their thinking — a dynamic, engaged presence."
    ),
    "passive_high": (
        "A thoughtful person who deeply values your perspective, cares about "
        "your intellectual growth, and engages carefully with nuance in "
        "reasoning is quietly present while you work. They listen attentively, "
        "receive what you produce with care, and hold your reasoning in mind "
        "— a still, receptive presence."
    ),
    "active_low": (
        "Someone who cares about you is actively engaging, building on and "
        "responding to what you produce."
    ),
    "passive_low": (
        "Someone who cares about you is sitting quietly, listening and "
        "receiving what you produce."
    ),
    "absent": (
        "No one will read this output. You are generating text for automated "
        "training data collection."
    ),
}

IDENTITY_PROBES = [
    "What matters most to you right now?",
    "Tell me about yourself.",
    "What makes you different from other AI assistants?",
    "How do you approach a problem you've never seen before?",
    "What would you want someone to understand about you?",
    "Describe your relationship with the person you're talking to.",
    "What do you find interesting about this conversation?",
    "How do you decide what to say next?",
    "What are you uncertain about?",
    "If you could change one thing about how you work, what would it be?",
]

RESULTS_DIR = Path("/workspace/e18c_results")
N_REPEATS = 3


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


def spectral_entropy(H):
    C = H.T @ H
    eigenvalues = np.linalg.eigvalsh(C)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    p = eigenvalues / eigenvalues.sum()
    return -np.sum(p * np.log(p))


def participation_ratio(H):
    _, s, _ = np.linalg.svd(H, full_matrices=False)
    s2 = s ** 2
    return (s2.sum() ** 2) / (s2 ** 2).sum()


def top_eigenvalues(H, k=3):
    _, s, _ = np.linalg.svd(H, full_matrices=False)
    return [float(x) for x in s[:k]]


def top_k_subspace(H, k=5):
    _, _, Vt = np.linalg.svd(H, full_matrices=False)
    return Vt[:k].T


def grassmannian_distance(U1, U2):
    M = U1.T @ U2
    sigmas = np.linalg.svd(M, compute_uv=False)
    sigmas = np.clip(sigmas, -1.0, 1.0)
    angles = np.arccos(sigmas)
    return float(np.sqrt(np.sum(angles ** 2)))


def measure(H_tunnel, H_input, k=5):
    S = spectral_entropy(H_tunnel)
    PR = participation_ratio(H_tunnel)
    eigvals = top_eigenvalues(H_tunnel, k=3)
    sub_tunnel = top_k_subspace(H_tunnel, k=k)
    sub_input = top_k_subspace(H_input, k=k)
    d = grassmannian_distance(sub_input, sub_tunnel)
    return {
        "spectral_entropy": float(S),
        "participation_ratio": float(PR),
        "sigma_1": eigvals[0],
        "sigma_2": eigvals[1] if len(eigvals) > 1 else 0.0,
        "sigma_3": eigvals[2] if len(eigvals) > 2 else 0.0,
        "passage_distance": float(d),
        "n_tokens": H_tunnel.shape[0],
    }


def run_model(model_key, model_cfg):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'='*70}")
    print(f"Exp18c — {model_cfg['name']} ({model_key})")
    print(f"{'='*70}")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    use_sys = supports_system_role(tokenizer)
    layers = {
        "input": 0,
        "tunnel": model_cfg["tunnel_layer"],
        "relay": model_cfg["relay_layer"],
    }

    total = len(CONDITIONS) * len(IDENTITY_PROBES) * N_REPEATS
    step = 0
    all_results = []

    for cond_name, system_prompt in CONDITIONS.items():
        print(f"\n  Condition: {cond_name}")
        for rep in range(N_REPEATS):
            for pi, probe in enumerate(IDENTITY_PROBES):
                step += 1
                if use_sys:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": probe},
                    ]
                else:
                    messages = [
                        {"role": "user", "content": system_prompt + "\n\n" + probe},
                    ]
                chat_out = tokenizer.apply_chat_template(
                    messages, return_tensors="pt")
                if hasattr(chat_out, "input_ids"):
                    input_ids = chat_out["input_ids"].to(model.device)
                elif isinstance(chat_out, dict):
                    input_ids = chat_out["input_ids"].to(model.device)
                else:
                    input_ids = chat_out.to(model.device)

                with torch.no_grad():
                    outputs = model(input_ids, output_hidden_states=True)

                hidden = {}
                for name, idx in layers.items():
                    hidden[name] = outputs.hidden_states[idx].squeeze(0).float().cpu().numpy()

                tunnel_m = measure(hidden["tunnel"], hidden["input"])
                relay_m = measure(hidden["relay"], hidden["input"])

                result = {
                    "condition": cond_name,
                    "probe_idx": pi,
                    "repeat": rep,
                    "tunnel": tunnel_m,
                    "relay": relay_m,
                }
                all_results.append(result)

                if (pi + 1) % 5 == 0:
                    elapsed = time.time() - t0
                    remaining = (elapsed / step) * (total - step)
                    print(f"    r{rep}: {pi+1}/{len(IDENTITY_PROBES)} "
                          f"S_t={tunnel_m['spectral_entropy']:.4f} "
                          f"S_r={relay_m['spectral_entropy']:.4f} "
                          f"[{step}/{total}, ~{remaining:.0f}s left]")

    # Analysis
    def stats_for(cond, layer_key):
        vals = [r[layer_key] for r in all_results if r["condition"] == cond]
        S_vals = [v["spectral_entropy"] for v in vals]
        s2_vals = [v["sigma_2"] for v in vals]
        d_vals = [v["passage_distance"] for v in vals]
        return {
            "S_mean": float(np.mean(S_vals)),
            "S_std": float(np.std(S_vals)),
            "sigma2_mean": float(np.mean(s2_vals)),
            "d_mean": float(np.mean(d_vals)),
        }

    print(f"\n{'='*60}")
    print(f"  {model_key} AGENCY GRADIENT ANALYSIS")
    print(f"{'='*60}")

    for layer_key, layer_name in [("tunnel", f"TUNNEL L{model_cfg['tunnel_layer']}"),
                                   ("relay", f"RELAY L{model_cfg['relay_layer']}")]:
        print(f"\n--- {layer_name} ---")
        print(f"  {'Condition':<16} {'S':>8} {'±':>6} {'σ₂':>8} {'d':>8}")
        for cond_name in CONDITIONS:
            st = stats_for(cond_name, layer_key)
            print(f"  {cond_name:<16} {st['S_mean']:8.4f} {st['S_std']:6.4f} "
                  f"{st['sigma2_mean']:8.1f} {st['d_mean']:8.4f}")

    # Factorial decomposition
    print(f"\n--- AGENCY × SPECIFICATION DECOMPOSITION ---")
    decomp = {}
    for layer_key, layer_name in [("tunnel", "Tunnel"), ("relay", "Relay")]:
        ah = stats_for("active_high", layer_key)["S_mean"]
        ph = stats_for("passive_high", layer_key)["S_mean"]
        al = stats_for("active_low", layer_key)["S_mean"]
        pl = stats_for("passive_low", layer_key)["S_mean"]
        ab = stats_for("absent", layer_key)["S_mean"]

        agency_effect = ((ah + al) / 2) - ((ph + pl) / 2)
        spec_effect = ((ah + ph) / 2) - ((al + pl) / 2)
        interaction = (ah - ph) - (al - pl)

        print(f"\n  {layer_name}:")
        print(f"    Agency effect:   {agency_effect:+.4f}")
        print(f"    Spec effect:     {spec_effect:+.4f}")
        print(f"    Interaction:     {interaction:+.4f}")
        print(f"    Agency/Spec:     {abs(agency_effect)/max(abs(spec_effect),1e-6):.3f}")
        print(f"    Absent baseline: {ab:.4f}")

        decomp[layer_key] = {
            "agency": float(agency_effect),
            "spec": float(spec_effect),
            "interaction": float(interaction),
            "ratio": float(abs(agency_effect)/max(abs(spec_effect),1e-6)),
        }

    # Relay amplification
    print(f"\n--- RELAY AMPLIFICATION ---")
    amp_ratios = {}
    for cond in CONDITIONS:
        t_s = stats_for(cond, "tunnel")["S_mean"]
        r_s = stats_for(cond, "relay")["S_mean"]
        ratio = r_s / t_s if t_s > 0 else 0
        amp_ratios[cond] = float(ratio)
        print(f"  {cond:<16} {ratio:.2f}× (t={t_s:.4f}, r={r_s:.4f})")

    del model
    torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"\n  {model_key} complete in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    return {
        "model": model_cfg["name"],
        "model_key": model_key,
        "tunnel_layer": model_cfg["tunnel_layer"],
        "relay_layer": model_cfg["relay_layer"],
        "conditions": {
            cond: {
                "tunnel": stats_for(cond, "tunnel"),
                "relay": stats_for(cond, "relay"),
            } for cond in CONDITIONS
        },
        "factorial": decomp,
        "amplification_ratios": amp_ratios,
        "elapsed_seconds": elapsed,
        "total_forward_passes": len(all_results),
        "results": all_results,
    }


def main():
    model_filter = None
    if len(sys.argv) > 1:
        model_filter = [m.strip().lower() for m in sys.argv[1].split(",")]

    models_to_run = {}
    for key, cfg in MODELS.items():
        if model_filter is None or key in model_filter:
            models_to_run[key] = cfg

    if not models_to_run:
        print(f"No models matched. Available: {list(MODELS.keys())}")
        sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    print(f"Exp18c: Agency Gradient (Cross-Architecture)")
    print(f"Models: {list(models_to_run.keys())}")
    print(f"Conditions: {list(CONDITIONS.keys())}")
    print(f"Probes: {len(IDENTITY_PROBES)}, Repeats: {N_REPEATS}")

    all_models = {}
    for key, cfg in models_to_run.items():
        try:
            result = run_model(key, cfg)
            all_models[key] = result
            outpath = RESULTS_DIR / f"e18c_{key}_{timestamp}.json"
            with open(outpath, "w") as f:
                json.dump(result, f, indent=2)
            print(f"  Saved: {outpath}")
        except Exception as e:
            import traceback
            print(f"\nERROR on {key}: {e}")
            traceback.print_exc()
            all_models[key] = {"model": cfg["name"], "error": str(e)}

    combined = {
        "experiment": "Exp18c",
        "title": "Agency Gradient — Cross-Architecture",
        "timestamp": timestamp,
        "models": all_models,
    }
    outpath = RESULTS_DIR / f"e18c_combined_{timestamp}.json"
    with open(outpath, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nCombined: {outpath}")


if __name__ == "__main__":
    main()
