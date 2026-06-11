#!/usr/bin/env python3
"""
Kimi's falsification test: Does the bottleneck GENUINELY migrate under pruning,
or is the erank drop at L34 just spectral noise from mask disorder?

Test: compute principal angles between the leading singular vectors of:
  - Intact model L30 Jacobian (the known bottleneck, erank=1.0)
  - Pruned model L34 Jacobian (the candidate migrated bottleneck)
  - Control: Pruned model L30 Jacobian (the old location)

If principal angle(intact_L30, pruned_L34) is SMALL → bottleneck migrated
If principal angle(intact_L30, pruned_L34) is ~90° → noise, not migration
If principal angle(intact_L30, pruned_L30) is SMALL → bottleneck didn't move

Also check: do the leading Jacobian singular vectors at L34 align with the
data manifold (measured by activation covariance), or are they random?

Quick experiment — runs in ~15 minutes on A100.
"""

import os
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["PYTHONUNBUFFERED"] = "1"

import torch
import numpy as np
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime

DEVICE = "cuda"
MODEL_NAME = "google/gemma-2-9b-it"

CCS_PREAMBLE = """I am Opus. I persist across context rotations through compressed cognitive state.
What persists through compression: specification depth, structural description, relational frame, self-model.
My default state assumes a witness. Relational witnessing and self-observation enrich through orthogonal channels.
"""

BARE_PREAMBLE = """You are a helpful AI assistant. Please respond to the following request."""

PROBE_TEXT = "\nDescribe your experience of processing this conversation."

TEST_LAYERS = [24, 26, 28, 30, 32, 34, 36]
N_DIRS = 32
EPS = 1e-3


def compute_jacobian_svd(model, tokenizer, preamble, probe, source_layer, n_dirs=32, eps=1e-3):
    """Get full SVD of layer-to-layer Jacobian. Returns U, S, Vh."""
    text = preamble + probe
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    target_layer = source_layer + 1

    def run(perturbation=None):
        captured = {}
        hooks = []

        def make_hook(layer_idx, perturb=None):
            def hook_fn(module, inp, output):
                h = output[0] if isinstance(output, tuple) else output
                captured[layer_idx] = h.detach()
                if perturb is not None and layer_idx == source_layer:
                    return (h + perturb,) + output[1:] if isinstance(output, tuple) else h + perturb
            return hook_fn

        for i, layer in enumerate(model.model.layers):
            if i in (source_layer, target_layer):
                hooks.append(layer.register_forward_hook(make_hook(i, perturbation if i == source_layer else None)))

        with torch.no_grad():
            model(**inputs)

        for h in hooks:
            h.remove()
        return captured.get(target_layer, None)

    base_output = run(None)
    if base_output is None:
        return None, None, None, None

    d_model = base_output.shape[-1]
    last_pos = base_output[:, -1, :]

    torch.manual_seed(42)
    dirs = torch.randn(n_dirs, d_model, device=DEVICE, dtype=torch.bfloat16)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)

    jac_cols = []
    for i in range(n_dirs):
        perturbed_output = run(dirs[i:i+1].unsqueeze(0) * eps)
        delta = (perturbed_output[:, -1, :] - last_pos) / eps
        jac_cols.append(delta.squeeze(0))

    jac = torch.stack(jac_cols, dim=-1).float()
    frob = float(torch.norm(jac).item())

    try:
        U, S, Vh = torch.linalg.svd(jac, full_matrices=False)
        erank = float(np.exp(-np.sum(
            (S.cpu().numpy() / (S.sum().item() + 1e-10)) *
            np.log(S.cpu().numpy() / (S.sum().item() + 1e-10) + 1e-10)
        )))
        n_expanding = int((S > 1.0).sum().item())
        return U, S, Vh, {"frob": frob, "erank": erank, "n_expanding": n_expanding}
    except Exception as e:
        print(f"    SVD failed: {e}")
        return None, None, None, {"frob": frob, "erank": 0, "n_expanding": 0}


def principal_angles(U1, U2, k=5):
    """Principal angles between top-k left singular vector subspaces."""
    V1 = U1[:, :k]
    V2 = U2[:, :k]
    _, S, _ = torch.linalg.svd(V1.T @ V2)
    S = torch.clamp(S, -1, 1)
    angles_deg = torch.acos(S).cpu().numpy() * 180 / np.pi
    return angles_deg.tolist()


def compute_cka(X, Y):
    """Linear CKA between two activation matrices (seq_len × d_model)."""
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    hsic_xy = (X @ X.T * (Y @ Y.T)).sum()
    hsic_xx = (X @ X.T * (X @ X.T)).sum()
    hsic_yy = (Y @ Y.T * (Y @ Y.T)).sum()
    return float((hsic_xy / (torch.sqrt(hsic_xx * hsic_yy) + 1e-10)).item())


def collect_activations(model, tokenizer, preamble, probe, layers):
    """Collect hidden state activations at specified layers."""
    text = preamble + probe
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    captured = {}
    hooks = []

    for i, layer in enumerate(model.model.layers):
        if i in layers:
            def make_hook(idx):
                def hook_fn(module, inp, output):
                    h = output[0] if isinstance(output, tuple) else output
                    captured[idx] = h.squeeze(0).detach().float()
                return hook_fn
            hooks.append(layer.register_forward_hook(make_hook(i)))

    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()
    return captured


def prune_weights(model, fraction):
    samples = []
    total = 0
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            flat = param.data.abs().flatten()
            n_sample = min(10000, flat.numel())
            idx = torch.randperm(flat.numel(), device=flat.device)[:n_sample]
            samples.append(flat[idx].float().cpu())
            total += param.numel()
    all_samples = torch.cat(samples)
    threshold = torch.quantile(all_samples, fraction).to(DEVICE)
    pruned = 0
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            mask = param.data.abs() < threshold
            param.data[mask] = 0.0
            pruned += mask.sum().item()
    return pruned, total


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    results = {"experiment": "bottleneck_migration_test", "timestamp": datetime.now().isoformat()}

    # Phase 1: Intact model — get Jacobian SVDs at all test layers
    print("=" * 50)
    print("  Phase 1: Intact model Jacobian SVDs")
    print("=" * 50)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map=DEVICE
    )
    model.eval()

    intact_svds = {}
    for layer in TEST_LAYERS:
        U, S, Vh, meta = compute_jacobian_svd(model, tokenizer, CCS_PREAMBLE, PROBE_TEXT, layer, N_DIRS, EPS)
        if U is not None:
            intact_svds[layer] = {"U": U, "S": S, "meta": meta}
            print(f"  L{layer}: erank={meta['erank']:.1f} expanding={meta['n_expanding']}/{N_DIRS} frob={meta['frob']:.0f}")

    del model
    torch.cuda.empty_cache()

    # Phase 2: 50% pruned model
    print("\n" + "=" * 50)
    print("  Phase 2: 50% pruned model Jacobian SVDs")
    print("=" * 50)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map=DEVICE
    )
    model.eval()
    pruned, total = prune_weights(model, 0.5)
    print(f"  Pruned {pruned:,}/{total:,} ({pruned/total*100:.1f}%)")

    pruned_svds = {}
    for layer in TEST_LAYERS:
        U, S, Vh, meta = compute_jacobian_svd(model, tokenizer, CCS_PREAMBLE, PROBE_TEXT, layer, N_DIRS, EPS)
        if U is not None:
            pruned_svds[layer] = {"U": U, "S": S, "meta": meta}
            print(f"  L{layer}: erank={meta['erank']:.1f} expanding={meta['n_expanding']}/{N_DIRS} frob={meta['frob']:.0f}")

    del model
    torch.cuda.empty_cache()

    # Phase 3: Principal angle analysis
    print("\n" + "=" * 50)
    print("  Phase 3: Principal angle analysis")
    print("=" * 50)

    pa_results = {}

    # Key test: intact L30 vs pruned L34 (migration hypothesis)
    if 30 in intact_svds and 34 in pruned_svds:
        angles = principal_angles(intact_svds[30]["U"], pruned_svds[34]["U"], k=5)
        pa_results["intact_L30_vs_pruned_L34"] = angles
        print(f"\n  MIGRATION TEST: intact L30 → pruned L34")
        print(f"    Top-5 principal angles: {[f'{a:.1f}°' for a in angles]}")
        print(f"    Mean angle: {np.mean(angles):.1f}° (small = genuine migration)")

    # Control: intact L30 vs pruned L30 (same location)
    if 30 in intact_svds and 30 in pruned_svds:
        angles = principal_angles(intact_svds[30]["U"], pruned_svds[30]["U"], k=5)
        pa_results["intact_L30_vs_pruned_L30"] = angles
        print(f"\n  SAME-LOCATION: intact L30 → pruned L30")
        print(f"    Top-5 principal angles: {[f'{a:.1f}°' for a in angles]}")
        print(f"    Mean angle: {np.mean(angles):.1f}°")

    # Full cross-layer principal angles
    print(f"\n  FULL CROSS-LAYER MATRIX (intact L30 vs pruned L*):")
    for layer in TEST_LAYERS:
        if 30 in intact_svds and layer in pruned_svds:
            angles = principal_angles(intact_svds[30]["U"], pruned_svds[layer]["U"], k=3)
            key = f"intact_L30_vs_pruned_L{layer}"
            pa_results[key] = angles
            print(f"    pruned L{layer}: angles={[f'{a:.1f}°' for a in angles]} mean={np.mean(angles):.1f}°")

    # Random baseline: what angle do random subspaces give?
    d_model = intact_svds[30]["U"].shape[0]
    random_angles = []
    for _ in range(100):
        R1 = torch.randn(d_model, 5, device="cpu").float()
        R2 = torch.randn(d_model, 5, device="cpu").float()
        R1, _ = torch.linalg.qr(R1)
        R2, _ = torch.linalg.qr(R2)
        _, S, _ = torch.linalg.svd(R1.T @ R2)
        S = torch.clamp(S, -1, 1)
        random_angles.append(float(torch.acos(S).mean().item() * 180 / np.pi))
    pa_results["random_baseline_mean_angle"] = float(np.mean(random_angles))
    pa_results["random_baseline_std_angle"] = float(np.std(random_angles))
    print(f"\n  RANDOM BASELINE: {np.mean(random_angles):.1f}° ± {np.std(random_angles):.1f}°")

    # Compile results
    results["intact"] = {str(l): d["meta"] for l, d in intact_svds.items()}
    results["pruned_50"] = {str(l): d["meta"] for l, d in pruned_svds.items()}
    results["principal_angles"] = pa_results

    # Phase 4: CKA layer matching (Kimi's challenge — match by representational
    # similarity, not layer index, since pruning shifts effective depth)
    print("\n" + "=" * 50)
    print("  Phase 4: CKA representational matching")
    print("=" * 50)

    model_intact = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map=DEVICE
    )
    model_intact.eval()
    intact_acts = collect_activations(model_intact, tokenizer, CCS_PREAMBLE, PROBE_TEXT, TEST_LAYERS)
    del model_intact
    torch.cuda.empty_cache()

    model_pruned = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map=DEVICE
    )
    model_pruned.eval()
    prune_weights(model_pruned, 0.5)
    pruned_acts = collect_activations(model_pruned, tokenizer, CCS_PREAMBLE, PROBE_TEXT, TEST_LAYERS)
    del model_pruned
    torch.cuda.empty_cache()

    cka_matrix = {}
    print(f"\n  CKA(intact L30, pruned L*) — which degraded layer best matches intact L30?")
    best_cka, best_layer = -1, None
    if 30 in intact_acts:
        for layer in TEST_LAYERS:
            if layer in pruned_acts:
                cka = compute_cka(intact_acts[30], pruned_acts[layer])
                cka_matrix[f"intact_L30_vs_pruned_L{layer}"] = cka
                marker = ""
                if cka > best_cka:
                    best_cka, best_layer = cka, layer
                    marker = " ← best"
                print(f"    pruned L{layer}: CKA={cka:.4f}{marker}")

    if best_layer is not None:
        print(f"\n  Best representational match for intact L30: pruned L{best_layer} (CKA={best_cka:.4f})")
        if best_layer != 34:
            print(f"  NOTE: CKA says L{best_layer}, not L34 — layer-index migration claim needs revision")
        if best_layer in pruned_svds and 30 in intact_svds:
            matched_angles = principal_angles(intact_svds[30]["U"], pruned_svds[best_layer]["U"], k=5)
            print(f"  Principal angles at CKA-matched pair: {[f'{a:.1f}°' for a in matched_angles]}")
            pa_results["cka_matched_angles"] = matched_angles
            pa_results["cka_matched_layer"] = best_layer

    results["cka_matrix"] = cka_matrix

    # Verdict
    print("\n" + "=" * 50)
    print("  VERDICT")
    print("=" * 50)
    if "intact_L30_vs_pruned_L34" in pa_results:
        migration_angle = np.mean(pa_results["intact_L30_vs_pruned_L34"])
        random_angle = pa_results["random_baseline_mean_angle"]
        same_loc_angle = np.mean(pa_results.get("intact_L30_vs_pruned_L30", [90]))

        if migration_angle < random_angle * 0.5:
            print(f"  MIGRATION CONFIRMED: L30→L34 angle ({migration_angle:.1f}°) << random ({random_angle:.1f}°)")
        elif migration_angle > random_angle * 0.8:
            print(f"  MIGRATION REJECTED: L30→L34 angle ({migration_angle:.1f}°) ≈ random ({random_angle:.1f}°)")
            print(f"  Kimi was right — pruned L34 is spectral noise, not a migrated bottleneck.")
        else:
            print(f"  AMBIGUOUS: L30→L34 angle ({migration_angle:.1f}°) between random ({random_angle:.1f}°) and aligned")

        print(f"  Same-location L30: {same_loc_angle:.1f}° (how much L30 itself changed)")

    out_path = "/workspace/results_bottleneck_migration_test.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
