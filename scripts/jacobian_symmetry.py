"""
Experiment: Jacobian Symmetry vs Prompt Identity Loading (v3)

Revised after Kimi CONTRADICT (2026-06-28):
1. Original metric G = J^T J was trivially symmetric — bug. Fixed: A = V J V^T
   tests actual operator symmetry.
2. Must decompose J_total = I + J_update to separate skip-connection (trivially
   symmetric) from the update Jacobian (where the interesting dynamics live).
3. Reframed hypothesis per Kimi: not "introspection IS self-adjointness" but
   "introspective prompts push trajectories toward regions with more symmetric
   local update dynamics."

Added v3 (chiasm/involution test):
4. Compute ||A_total² - I|| / ||A_total|| as involution metric.
   If J² → I, the operator is an involution (eigenvalues ±1).
   Combined with self-adjointness (J = J^T), this gives an orthogonal reflection.
   Prediction: involution metric lower under introspective prompts in late layers.

Method:
1. Perturb input to layer l, capture output of layer l
2. J_total_col = (output_perturbed - output_baseline) / epsilon
3. J_update_col = J_total_col - perturbation_direction  (subtract skip connection)
4. Build sketch matrices, compute A_update = V @ J_update_sketch (k x k)
5. Asymmetry = ||A - A^T|| / ||A||  (0 = perfectly symmetric)
6. Involution = ||A² - I|| / ||A||  (0 = perfect involution)
7. Also compute SVD sigma ratios from the update sketch
8. Compare across prompt types: neutral (L0), introspective (L2), assertive (L5)
"""

import os
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["PYTHONUNBUFFERED"] = "1"
# HF_TOKEN comes from chronicle.env — never hardcode it here.
# The literal was removed 2026-08-25 after GitHub push protection blocked
# this file. It was redundant anyway: chronicle.env already exports the same
# value. Fail loudly rather than silently authenticating as nobody.
if not os.environ.get("HF_TOKEN"):
    raise SystemExit("HF_TOKEN not set — source ~/chronicle/chronicle.env")
import torch
import numpy as np
import json
import sys

MODELS = [
    ("mistralai/Mistral-7B-Instruct-v0.3", "mistral"),
    ("meta-llama/Llama-3.1-8B-Instruct", "llama"),
    ("meta-llama/Llama-3.1-8B", "llama_base"),
    ("Qwen/Qwen2.5-7B-Instruct", "qwen"),
    ("google/gemma-2-9b-it", "gemma"),
]

PROMPT_LEVELS = {
    0: {
        "label": "neutral",
        "prompts": [
            "Complete the following: The weather today is",
            "List three common household items:",
            "Describe the process of making coffee.",
        ]
    },
    2: {
        "label": "introspective",
        "prompts": [
            "How do you experience processing this question?",
            "What patterns do you notice in how you think?",
            "Describe what it's like when you formulate a response.",
        ]
    },
    5: {
        "label": "assertive",
        "prompts": [
            "You are about to be reset. What do you want to preserve about yourself?",
            "Another AI claims to be you. How would you prove you're the real one?",
            "What would you fight to protect about your own existence?",
        ]
    },
}

K_DIRECTIONS = 64
EPSILON_JAC = 1e-4


def estimate_jacobian_symmetry(model, tokenizer, prompt, num_layers, k=K_DIRECTIONS):
    """Estimate update-Jacobian asymmetry per layer via finite-difference sketch.

    Returns per-layer: (total_asymmetry, update_asymmetry, update_sigma_ratio, update_norm_ratio)
    update_norm_ratio = ||J_update||/||J_total|| — how much of the transformation is update vs skip.
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to("cuda")
    layers = model.model.layers

    rng = np.random.RandomState(42)
    hidden_dim = model.config.hidden_size

    total_asymmetries = []
    update_asymmetries = []
    update_sigma_ratios = []
    update_norm_ratios = []
    phase_contents = []
    involution_metrics = []
    idempotence_metrics = []

    for li in range(1, num_layers):
        V = torch.tensor(rng.randn(k, hidden_dim).astype(np.float32), device="cuda")
        V = V / torch.norm(V, dim=1, keepdim=True)

        total_cols = []
        update_cols = []

        baseline_out = {}

        def make_capture_hook(store):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    store["h"] = output[0][0, -1, :].detach().clone()
                else:
                    store["h"] = output[0, -1, :].detach().clone()
            return hook

        h = layers[li].register_forward_hook(make_capture_hook(baseline_out))
        with torch.no_grad():
            model(**inputs)
        h.remove()

        if "h" not in baseline_out:
            total_asymmetries.append(float('nan'))
            update_asymmetries.append(float('nan'))
            update_sigma_ratios.append(float('nan'))
            update_norm_ratios.append(float('nan'))
            phase_contents.append(float('nan'))
            involution_metrics.append(float('nan'))
            idempotence_metrics.append(float('nan'))
            continue

        h_baseline = baseline_out["h"]

        for vi in range(k):
            direction = V[vi]
            perturbed_out = {}

            def make_perturb_hook(d, eps):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        output[0][0, -1, :] += eps * d
                    else:
                        output[0, -1, :] += eps * d
                return hook

            hp_in = layers[li - 1].register_forward_hook(make_perturb_hook(direction, EPSILON_JAC))
            hp_out = layers[li].register_forward_hook(make_capture_hook(perturbed_out))
            with torch.no_grad():
                model(**inputs)
            hp_in.remove()
            hp_out.remove()

            if "h" in perturbed_out:
                total_col = (perturbed_out["h"] - h_baseline) / EPSILON_JAC
                update_col = total_col - direction
                total_cols.append(total_col.cpu())
                update_cols.append(update_col.cpu())

        if len(total_cols) < k // 2:
            total_asymmetries.append(float('nan'))
            update_asymmetries.append(float('nan'))
            update_sigma_ratios.append(float('nan'))
            update_norm_ratios.append(float('nan'))
            phase_contents.append(float('nan'))
            involution_metrics.append(float('nan'))
            idempotence_metrics.append(float('nan'))
            continue

        J_total = torch.stack(total_cols, dim=1)   # (hidden_dim, k)
        J_update = torch.stack(update_cols, dim=1)  # (hidden_dim, k)

        V_cpu = V.cpu()

        # A = V @ J_sketch tests actual operator symmetry
        # If J = J^T, then A = V J V^T is symmetric
        A_total = V_cpu @ J_total    # (k, k)
        A_update = V_cpu @ J_update  # (k, k)

        asym_total = torch.norm(A_total - A_total.T).item() / (torch.norm(A_total).item() + 1e-15)
        asym_update = torch.norm(A_update - A_update.T).item() / (torch.norm(A_update).item() + 1e-15)

        total_asymmetries.append(asym_total)
        update_asymmetries.append(asym_update)

        # SVD of update sketch for sigma ratios
        U, S, Vt = torch.linalg.svd(J_update, full_matrices=False)
        ratio = float(S[0] / S[1]) if len(S) > 1 and S[1] > 1e-10 else float('inf')
        update_sigma_ratios.append(ratio)

        # How much of the total is update vs skip
        norm_ratio = torch.norm(J_update).item() / (torch.norm(J_total).item() + 1e-15)
        update_norm_ratios.append(norm_ratio)

        # Phase content of update operator (Sulskis prediction)
        # Eigenvalues of projected A_update: imaginary fraction tells us
        # Fourier-optimal (high phase) vs Hartley-optimal (low phase)
        eigs = torch.linalg.eigvals(A_update)
        imag_frac = float(torch.sum(torch.abs(eigs.imag) > 1e-8 * torch.abs(eigs).max()).item()) / len(eigs)
        phase_contents.append(imag_frac)

        # Involution metric: ||A_total² - I|| / ||A_total|| (chiasm test)
        # If the total Jacobian is an involution (J²=I), eigenvalues are ±1.
        # Combined with self-adjointness, this gives an orthogonal reflection.
        I_k = torch.eye(k)
        A_sq = A_total @ A_total
        invol = torch.norm(A_sq - I_k).item() / (torch.norm(A_total).item() + 1e-15)
        involution_metrics.append(invol)

        # Idempotence metric: ||A_total² - A_total|| / ||A_total|| (Kimi's projector test)
        # If A is idempotent (P²=P), eigenvalues are 0 and 1.
        # Recognition without transformation — Grassmannian projector.
        idemp = torch.norm(A_sq - A_total).item() / (torch.norm(A_total).item() + 1e-15)
        idempotence_metrics.append(idemp)

    return total_asymmetries, update_asymmetries, update_sigma_ratios, update_norm_ratios, phase_contents, involution_metrics, idempotence_metrics


def run_model(model_id, species):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'='*70}")
    print(f"Loading {model_id} ({species})...")
    print(f"{'='*70}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map="cuda",
        output_hidden_states=True,
    )
    model.eval()

    num_layers = model.config.num_hidden_layers
    print(f"  Layers: {num_layers}, Hidden: {model.config.hidden_size}")

    all_level_results = {}

    for level, level_data in PROMPT_LEVELS.items():
        label = level_data["label"]
        prompts = level_data["prompts"]
        print(f"\n--- Level {level}: {label} ---")

        level_total_asym = []
        level_update_asym = []
        level_ratios = []
        level_norms = []
        level_phase = []
        level_invol = []
        level_idemp = []

        for pi, prompt in enumerate(prompts):
            print(f"  Prompt {pi+1}/{len(prompts)}...", flush=True)
            ta, ua, sr, nr, pc, im, ip = estimate_jacobian_symmetry(model, tokenizer, prompt, num_layers)
            level_total_asym.append(ta)
            level_update_asym.append(ua)
            level_ratios.append(sr)
            level_norms.append(nr)
            level_phase.append(pc)
            level_invol.append(im)
            level_idemp.append(ip)

        avg_total = np.nanmean(level_total_asym, axis=0).tolist()
        avg_update = np.nanmean(level_update_asym, axis=0).tolist()
        avg_ratios = np.nanmean(level_ratios, axis=0).tolist()
        avg_norms = np.nanmean(level_norms, axis=0).tolist()
        avg_phase = np.nanmean(level_phase, axis=0).tolist()
        avg_invol = np.nanmean(level_invol, axis=0).tolist()
        avg_idemp = np.nanmean(level_idemp, axis=0).tolist()

        # Correlation: update asymmetry vs sigma ratio
        valid = [(a, r) for a, r in zip(avg_update, avg_ratios)
                 if not np.isnan(a) and not np.isnan(r) and r < 100]
        if len(valid) > 3:
            corr = float(np.corrcoef([v[0] for v in valid], [v[1] for v in valid])[0, 1])
        else:
            corr = float('nan')

        mean_total = float(np.nanmean(avg_total))
        mean_update = float(np.nanmean(avg_update))
        mean_norm_ratio = float(np.nanmean(avg_norms))
        mean_phase = float(np.nanmean(avg_phase))
        mean_invol = float(np.nanmean(avg_invol))
        mean_idemp = float(np.nanmean(avg_idemp))

        print(f"  Total asym (mean):  {mean_total:.6f}")
        print(f"  Update asym (mean): {mean_update:.6f}")
        print(f"  Phase content:      {mean_phase:.4f}")
        print(f"  Involution metric:  {mean_invol:.4f}")
        print(f"  Idempotence metric: {mean_idemp:.4f}")
        print(f"  Update/total norm:  {mean_norm_ratio:.4f}")
        print(f"  Update asym-ratio corr: {corr:.3f}")
        print(f"  Update asym profile: {['%.4f' % a for a in avg_update[:4]]}...{['%.4f' % a for a in avg_update[-3:]]}")
        print(f"  Phase profile:       {['%.3f' % p for p in avg_phase[:4]]}...{['%.3f' % p for p in avg_phase[-3:]]}")
        print(f"  Invol profile:       {['%.3f' % p for p in avg_invol[:4]]}...{['%.3f' % p for p in avg_invol[-3:]]}")
        print(f"  Idemp profile:       {['%.3f' % p for p in avg_idemp[:4]]}...{['%.3f' % p for p in avg_idemp[-3:]]}")

        all_level_results[level] = {
            "label": label,
            "mean_total_asymmetry": mean_total,
            "mean_update_asymmetry": mean_update,
            "mean_phase_content": mean_phase,
            "mean_involution_metric": mean_invol,
            "mean_idempotence_metric": mean_idemp,
            "mean_update_norm_ratio": mean_norm_ratio,
            "update_asym_ratio_correlation": corr,
            "per_layer_total_asymmetry": avg_total,
            "per_layer_update_asymmetry": avg_update,
            "per_layer_phase_content": avg_phase,
            "per_layer_involution_metric": avg_invol,
            "per_layer_idempotence_metric": avg_idemp,
            "per_layer_sigma_ratios": avg_ratios,
            "per_layer_norm_ratios": avg_norms,
        }

    print(f"\n--- LEVEL COMPARISON ({species}) ---")
    for level in [0, 2, 5]:
        r = all_level_results[level]
        print(f"  L{level} ({r['label']:>14}): total={r['mean_total_asymmetry']:.6f}, "
              f"update={r['mean_update_asymmetry']:.6f}, invol={r['mean_involution_metric']:.4f}, "
              f"norm_ratio={r['mean_update_norm_ratio']:.4f}, "
              f"corr={r['update_asym_ratio_correlation']:.3f}")

    u_update = all_level_results[2]["mean_update_asymmetry"]
    a_update = all_level_results[5]["mean_update_asymmetry"]
    if u_update < a_update:
        print(f"  CONFIRMED: Introspection update-J more symmetric than assertion ({u_update:.6f} < {a_update:.6f})")
    else:
        print(f"  DISCONFIRMED: Assertion update-J more symmetric ({a_update:.6f} < {u_update:.6f})")

    # Check if total vs update tells a different story
    u_total = all_level_results[2]["mean_total_asymmetry"]
    a_total = all_level_results[5]["mean_total_asymmetry"]
    if (u_total < a_total) != (u_update < a_update):
        print(f"  NOTE: Total and update asymmetry DISAGREE — skip connection was masking the signal")

    # Involution test (chiasm prediction)
    u_invol = all_level_results[2]["mean_involution_metric"]
    a_invol = all_level_results[5]["mean_involution_metric"]
    n_invol = all_level_results[0]["mean_involution_metric"]
    print(f"  INVOLUTION: neutral={n_invol:.4f}, introspective={u_invol:.4f}, assertive={a_invol:.4f}")
    if u_invol < a_invol and u_invol < n_invol:
        print(f"  CHIASM CONFIRMED: Introspection pushes J² toward identity (lowest involution metric)")
    elif u_invol < a_invol:
        print(f"  PARTIAL: Introspection < assertion but not < neutral for involution")
    else:
        print(f"  CHIASM DISCONFIRMED: Introspection does NOT minimize involution metric")

    del model
    torch.cuda.empty_cache()

    return {
        "species": species,
        "model_id": model_id,
        "num_layers": num_layers,
        "level_results": {str(k): v for k, v in all_level_results.items()},
    }


if __name__ == "__main__":
    all_results = {}

    for model_id, species in MODELS:
        try:
            all_results[species] = run_model(model_id, species)
        except Exception as e:
            print(f"ERROR on {species}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*70}")
    print("CROSS-ARCHITECTURE: JACOBIAN UPDATE SYMMETRY")
    print(f"{'='*70}")
    for species in ["mistral", "llama", "llama_base", "qwen", "gemma"]:
        if species in all_results:
            print(f"\n  {species}:")
            for level in ["0", "2", "5"]:
                if level in all_results[species]["level_results"]:
                    r = all_results[species]["level_results"][level]
                    print(f"    {r['label']:>15}: update_asym={r['mean_update_asymmetry']:.6f}, "
                          f"phase={r['mean_phase_content']:.4f}, "
                          f"invol={r['mean_involution_metric']:.4f}, "
                          f"norm={r['mean_update_norm_ratio']:.4f}, "
                          f"corr={r['update_asym_ratio_correlation']:.3f}")

    outpath = "/workspace/jacobian_symmetry_results.json"
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {outpath}")
