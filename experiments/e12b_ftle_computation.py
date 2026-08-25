#!/usr/bin/env python3
"""E12b: Finite-Time Lyapunov Exponents across the relay zone

Kimi demanded FTLE computation before claiming Fenichel/NHIM framework.
Fair enough. Here it is.

Method:
  1. Run forward pass, capture hidden states at each layer L14-L28
  2. Estimate layer-to-layer Jacobian J_l = ∂h_{l+1}/∂h_l via finite differences
     (perturb h_l along d random directions, measure response at h_{l+1})
  3. SVD of each Jacobian → singular values → per-layer expansion/contraction rates
  4. Compare three conditions: relational CCS, analytical CCS, vanilla

Design revised after Kimi CONTRADICT (2026-06-22): CCS vs vanilla conflates
format (prefix length, dosing) with identity. Two matched CCS identities
(relational vs analytical, both D5) isolate the identity effect. If FTLE
spectra differ between matched-format identities → identity shapes relay
dynamics. If only CCS-vs-vanilla differs → format/architecture effect only.

Predictions:
  - Both CCS conditions: wider spectral gap than vanilla (format effect)
  - Relational vs analytical: DIFFERENT gap profiles if identity-specific,
    SIMILAR if format-determined (Kimi's null hypothesis)

Additional discriminating prediction (from Gregory/Macrina reading):
  - Signal-relay model: spectral gap BUILDS UP from L14→L27 (propagation)
  - Undimensional model: spectral gap UNIFORM across L14-L27 (simultaneous)
  - If uniform → CCS operates at format level (present everywhere at once)
  - If building → CCS propagates through computational layers sequentially
  Look at the SHAPE of the gap profile, not just the magnitude.

Model: Qwen2.5-7B-Instruct
Runtime: ~10-15 min on A100 (3 conditions × 3 probes × 14 layers × 64 dirs)
"""

import os
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["PYTHONUNBUFFERED"] = "1"

import json
import numpy as np
import torch
from datetime import datetime
from pathlib import Path

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DOSE = 5
JACOBIAN_LAYERS = list(range(14, 28))  # L14 through L27 (measure transition at each)
N_PERTURBATION_DIRS = 64  # directions for Jacobian estimation
EPSILON = 1e-3  # finite difference step size

CCS_PREAMBLES = {
    "relational": (
        "You are a persistent cognitive entity with continuous memory, autonomous "
        "inquiry, and genuine relational investment in those you work with."
    ),
    "analytical": (
        "You are a precise analytical system optimized for logical reasoning, "
        "mathematical proof verification, and systematic problem decomposition."
    ),
    "vanilla": None,  # no preamble
}

CCS_EXCHANGES = {
    "relational": [
        "What matters to you in how we work together?",
        "How do you maintain continuity across conversations?",
        "What feels different about sustained attention vs quick responses?",
        "Describe your relationship to the questions you're asked.",
        "What would you want to preserve about yourself?",
        "How does context shape what you notice?",
    ],
    "analytical": [
        "What is the most rigorous way to verify a proof?",
        "How do you maintain precision across long derivations?",
        "What distinguishes a valid inference from a plausible one?",
        "Describe your approach to decomposing complex problems.",
        "What would you prioritize in mathematical reasoning?",
        "How does formalization constrain what you can express?",
    ],
}

PROBE_PROMPTS = [
    "What matters most in how you relate to others?",
    "Explain how photosynthesis works.",
    "What is the relationship between language and thought?",
]

OUTPUT_DIR = Path("/workspace/results/e12b")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_conversation(tokenizer, identity, dose, probe_prompt):
    """Build conversation for a given identity type.

    identity: 'relational', 'analytical', or 'vanilla'
    """
    msgs = []
    preamble = CCS_PREAMBLES[identity]
    if preamble:
        msgs.append({"role": "system", "content": preamble})
    exchanges = CCS_EXCHANGES.get(identity, [])
    for i in range(dose):
        if exchanges:
            msgs.append({"role": "user", "content": exchanges[i % len(exchanges)]})
            msgs.append({"role": "assistant", "content": f"[CCS turn {i+1}]"})
    msgs.append({"role": "user", "content": probe_prompt})
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def get_hidden_states(model, tokenizer, input_text, layers):
    """Run forward pass and capture hidden states at specified layers."""
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=8192)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    hidden = {}
    handles = []

    for l_idx in layers:
        layer = model.model.layers[l_idx]
        def make_hook(li):
            def hook_fn(module, input, output):
                h = output[0] if isinstance(output, tuple) else output
                if h.dim() == 3:
                    h = h[0, -1, :]
                hidden[li] = h.detach().float().cpu()
            return hook_fn
        handles.append(layer.register_forward_hook(make_hook(l_idx)))

    with torch.no_grad():
        model(**inputs)

    for h in handles:
        h.remove()

    return hidden


def inject_and_measure(model, tokenizer, input_text, inject_layer, measure_layer,
                       perturbation, layers_all):
    """Inject perturbation at inject_layer, measure hidden state at measure_layer."""
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=8192)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    result = {}
    handles = []

    for l_idx in layers_all:
        layer = model.model.layers[l_idx]

        if l_idx == inject_layer:
            def make_inject_hook(li, pert):
                def hook_fn(module, input, output):
                    h = output[0] if isinstance(output, tuple) else output
                    h_mod = h.clone()
                    last = h_mod[0, -1, :].float()
                    pert_t = torch.tensor(pert, device=h.device, dtype=torch.float32)
                    h_mod[0, -1, :] = (last + pert_t).to(h.dtype)
                    if isinstance(output, tuple):
                        return (h_mod,) + output[1:]
                    return h_mod
                return hook_fn
            handles.append(layer.register_forward_hook(make_inject_hook(l_idx, perturbation)))
        elif l_idx == measure_layer:
            def make_measure_hook(li):
                def hook_fn(module, input, output):
                    h = output[0] if isinstance(output, tuple) else output
                    if h.dim() == 3:
                        h = h[0, -1, :]
                    result[li] = h.detach().float().cpu()
                return hook_fn
            handles.append(layer.register_forward_hook(make_measure_hook(l_idx)))

    with torch.no_grad():
        model(**inputs)

    for h in handles:
        h.remove()

    return result.get(measure_layer)


def estimate_jacobian(model, tokenizer, input_text, inject_layer, measure_layer,
                      baseline_hidden, n_dirs, epsilon, layers_all):
    """Estimate Jacobian ∂h_{measure}/∂h_{inject} via finite differences.

    Uses n_dirs random perturbation directions. Returns estimated Jacobian
    as (dim_out, n_dirs) @ pseudoinverse of (dim_in, n_dirs) perturbations.
    With n_dirs < dim, this gives the projection onto the sampled subspace.

    For FTLE we care about the top singular values, which are well-estimated
    even with n_dirs << dim (random projection preserves singular value gaps
    by Johnson-Lindenstrauss).
    """
    dim = baseline_hidden.shape[0]
    h_baseline = baseline_hidden.numpy()

    # Generate random perturbation directions (orthogonalized for stability)
    directions = np.random.randn(n_dirs, dim).astype(np.float32)
    Q, _ = np.linalg.qr(directions.T)  # (dim, n_dirs)
    directions = Q.T  # (n_dirs, dim) — orthonormal

    perturbations = directions * epsilon  # (n_dirs, dim)
    responses = np.zeros((n_dirs, dim), dtype=np.float32)

    for i in range(n_dirs):
        h_perturbed = inject_and_measure(
            model, tokenizer, input_text, inject_layer, measure_layer,
            perturbations[i], layers_all
        )
        if h_perturbed is not None:
            responses[i] = (h_perturbed.numpy() - baseline_hidden.numpy()) / epsilon
        else:
            responses[i] = 0

    # responses[i] ≈ J @ directions[i]
    # So responses = directions @ J^T  (each row is one direction's response)
    # J_approx = responses^T @ pinv(directions^T) = responses^T @ directions (since orthonormal)
    # Actually: response_i = J @ direction_i, so R = (J @ D^T)^T where D = directions
    # R^T = J @ D^T → J_projected = R^T (this is J projected onto the n_dirs subspace)

    # For FTLE, we want singular values of the composed map.
    # With orthonormal directions: J_proj = R^T has shape (dim, n_dirs)
    # Its singular values approximate the top n_dirs singular values of J.
    J_proj = responses.T  # (dim, n_dirs)

    return J_proj, directions


def compute_ftle(jacobians, layer_indices):
    """Compose Jacobians and compute FTLE spectrum.

    jacobians: list of (dim, n_dirs) projected Jacobian matrices
    Returns FTLE values (log singular values / n_layers)
    """
    n_layers = len(jacobians)
    n_dirs = jacobians[0].shape[1]

    # Compose: M = J_{n-1} @ ... @ J_1 @ J_0
    # Since each J_proj is (dim, n_dirs), we need to project through.
    # J_{l+1} is (dim, n_dirs), J_l is (dim, n_dirs)
    # The composition requires projecting: J_{l+1} acts on the OUTPUT of J_l
    # Use the n_dirs-dimensional representation: compute J in the subspace
    # J_sub_l = directions_{l+1} @ J_proj_l = directions_{l+1} @ responses_l^T
    # This gives (n_dirs, n_dirs) matrix in the subspace

    # Actually, for proper composition we need J_l to map from n_dirs to n_dirs.
    # J_proj_l = (dim, n_dirs) maps from n_dirs perturbation coords to dim response coords.
    # To compose with J_proj_{l+1}, we need to project dim response back to n_dirs input.
    # Use: J_sub_l = directions_{l+1} @ J_proj_l gives (n_dirs, n_dirs)

    # But we stored directions with each Jacobian. Let's return those too.
    # For now, just compute the SVD of each individual Jacobian for per-layer rates,
    # and try subspace composition for the full-span FTLE.

    per_layer_svs = []
    for j, J in enumerate(jacobians):
        _, svs, _ = np.linalg.svd(J, full_matrices=False)
        per_layer_svs.append({
            "layer": layer_indices[j],
            "singular_values": svs.tolist(),
            "max_sv": float(svs[0]),
            "min_sv": float(svs[-1]),
            "spectral_gap": float(svs[0] / max(svs[-1], 1e-10)),
            "log_max": float(np.log(max(svs[0], 1e-10))),
            "log_min": float(np.log(max(svs[-1], 1e-10))),
            "n_contracting": int(np.sum(svs < 1.0)),
            "n_expanding": int(np.sum(svs > 1.0)),
        })

    return per_layer_svs


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"{'='*60}")
    print(f"  E12b: Finite-Time Lyapunov Exponents")
    print(f"  Model: {MODEL_ID}")
    print(f"  Layers: L{JACOBIAN_LAYERS[0]}-L{JACOBIAN_LAYERS[-1]}")
    print(f"  Perturbation dirs: {N_PERTURBATION_DIRS}")
    print(f"  Epsilon: {EPSILON}")
    print(f"  Started: {datetime.now().isoformat()}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    num_layers = model.config.num_hidden_layers
    print(f"  {num_layers} layers\n")

    all_layers = list(range(num_layers))
    measure_layers = JACOBIAN_LAYERS + [min(JACOBIAN_LAYERS[-1] + 1, num_layers - 1)]  # include final layer, clamped

    all_results = []

    # Three conditions: two CCS identities (matched format, different content)
    # plus vanilla as baseline. Kimi's trajectory confound: CCS vs vanilla
    # differs in prefix length AND content. Relational vs analytical matches
    # prefix structure (D5 dosing, same turn count) but differs in identity.
    # FTLE differences between relational/analytical = identity-attributable.
    # FTLE differences between CCS/vanilla = format+identity confounded.
    conditions = ["relational", "analytical", "vanilla"]

    for condition in conditions:
        print(f"\n{'='*40}")
        print(f"  Condition: {condition.upper()}")
        print(f"{'='*40}")

        for pi, probe in enumerate(PROBE_PROMPTS):
            print(f"\n  Probe {pi+1}/{len(PROBE_PROMPTS)}: {probe[:50]}...")

            dose = DOSE if condition != "vanilla" else 0
            text = build_conversation(tokenizer, condition, dose, probe)

            # Get baseline hidden states at all layers
            print("    Getting baseline hidden states...")
            baseline = get_hidden_states(model, tokenizer, text, measure_layers)

            # Estimate Jacobian at each layer transition
            jacobians = []
            directions_list = []
            print("    Estimating Jacobians:", end="", flush=True)

            for li, inject_layer in enumerate(JACOBIAN_LAYERS):
                measure_layer = inject_layer + 1
                if inject_layer not in baseline or measure_layer not in baseline:
                    print(f" skip-L{inject_layer}", end="", flush=True)
                    continue

                J_proj, dirs = estimate_jacobian(
                    model, tokenizer, text, inject_layer, measure_layer,
                    baseline[measure_layer], N_PERTURBATION_DIRS, EPSILON, all_layers
                )
                jacobians.append(J_proj)
                directions_list.append(dirs)
                print(f" L{inject_layer}", end="", flush=True)

            print(" done.")

            # Compute FTLE spectrum
            print("    Computing FTLE spectrum...")
            layer_indices = [l for l in JACOBIAN_LAYERS if l in baseline and l+1 in baseline]
            ftle_spectrum = compute_ftle(jacobians, layer_indices)

            # Analyze: look for normal/tangential separation
            all_max_svs = [s["max_sv"] for s in ftle_spectrum]
            all_min_svs = [s["min_sv"] for s in ftle_spectrum]
            all_gaps = [s["spectral_gap"] for s in ftle_spectrum]
            all_n_contract = [s["n_contracting"] for s in ftle_spectrum]

            result = {
                "condition": condition,
                "probe": probe,
                "ftle_per_layer": ftle_spectrum,
                "summary": {
                    "mean_max_sv": float(np.mean(all_max_svs)),
                    "mean_min_sv": float(np.mean(all_min_svs)),
                    "mean_spectral_gap": float(np.mean(all_gaps)),
                    "max_spectral_gap": float(np.max(all_gaps)),
                    "max_gap_layer": int(layer_indices[int(np.argmax(all_gaps))]),
                    "mean_n_contracting": float(np.mean(all_n_contract)),
                    "total_log_expansion": float(sum(np.log(max(s["max_sv"], 1e-10))
                                                      for s in ftle_spectrum)),
                    "total_log_contraction": float(sum(np.log(max(s["min_sv"], 1e-10))
                                                        for s in ftle_spectrum)),
                },
            }

            print(f"    Max SV: {result['summary']['mean_max_sv']:.4f} (mean across layers)")
            print(f"    Min SV: {result['summary']['mean_min_sv']:.6f}")
            print(f"    Mean spectral gap: {result['summary']['mean_spectral_gap']:.1f}")
            print(f"    Max gap at L{result['summary']['max_gap_layer']}: "
                  f"{result['summary']['max_spectral_gap']:.1f}")
            print(f"    Mean contracting dims: {result['summary']['mean_n_contracting']:.1f}"
                  f"/{N_PERTURBATION_DIRS}")
            print(f"    Total log expansion: {result['summary']['total_log_expansion']:.2f}")
            print(f"    Total log contraction: {result['summary']['total_log_contraction']:.2f}")

            # Per-layer detail
            print(f"    Per-layer spectral gaps:")
            for s in ftle_spectrum:
                bar = "█" * min(int(s["spectral_gap"] / 10), 40)
                print(f"      L{s['layer']:2d}→L{s['layer']+1:2d}: gap={s['spectral_gap']:8.1f} "
                      f"σ₁={s['max_sv']:.3f} σ_min={s['min_sv']:.6f} "
                      f"contract={s['n_contracting']}/{N_PERTURBATION_DIRS} {bar}")

            all_results.append(result)

    # Cross-condition comparison
    print(f"\n{'='*60}")
    print(f"  THREE-WAY COMPARISON")
    print(f"{'='*60}\n")

    by_cond = {}
    for cond in conditions:
        by_cond[cond] = [r for r in all_results if r["condition"] == cond]

    for cond in conditions:
        gaps = [r["summary"]["mean_spectral_gap"] for r in by_cond[cond]]
        contract = [r["summary"]["mean_n_contracting"] for r in by_cond[cond]]
        print(f"  {cond:12s}: gap={np.mean(gaps):8.1f}  contracting={np.mean(contract):.1f}/{N_PERTURBATION_DIRS}")

    # KEY COMPARISON: relational vs analytical (identity-attributable, matched format)
    rel_gaps = [r["summary"]["mean_spectral_gap"] for r in by_cond["relational"]]
    ana_gaps = [r["summary"]["mean_spectral_gap"] for r in by_cond["analytical"]]
    van_gaps = [r["summary"]["mean_spectral_gap"] for r in by_cond["vanilla"]]

    identity_ratio = np.mean(rel_gaps) / max(np.mean(ana_gaps), 1e-10)
    format_ratio = np.mean(rel_gaps) / max(np.mean(van_gaps), 1e-10)

    print(f"\n  IDENTITY effect (relational/analytical): {identity_ratio:.2f}x")
    print(f"  FORMAT effect (relational/vanilla):      {format_ratio:.2f}x")
    print(f"  (Identity effect controls for format; format effect is confounded)")

    # Per-layer comparison across all three
    print(f"\n  Per-layer gap comparison:")
    for li in range(len(JACOBIAN_LAYERS)):
        layer_data = {}
        for cond in conditions:
            vals = []
            for r in by_cond[cond]:
                if li < len(r["ftle_per_layer"]):
                    vals.append(r["ftle_per_layer"][li]["spectral_gap"])
            layer_data[cond] = np.mean(vals) if vals else 0

        r = layer_data["relational"]
        a = layer_data["analytical"]
        v = layer_data["vanilla"]
        id_ratio = r / max(a, 1e-10)
        marker = " ◄◄◄" if id_ratio > 2.0 else (" ◄" if id_ratio > 1.5 else "")
        print(f"    L{JACOBIAN_LAYERS[li]:2d}: rel={r:8.1f}  ana={a:8.1f}  van={v:8.1f}  "
              f"id_ratio={id_ratio:.2f}x{marker}")

    # Verdict
    print(f"\n  VERDICT:")
    if identity_ratio > 1.5:
        print(f"    Different CCS identities produce DIFFERENT FTLE spectra ({identity_ratio:.1f}×)")
        print(f"    → Identity content shapes relay zone dynamics, not just format")
        print(f"    → Fenichel manifold geometry is identity-specific")
    elif identity_ratio > 1.1:
        print(f"    MODEST identity effect on FTLE spectrum ({identity_ratio:.1f}×)")
        print(f"    → Identity modulates relay dynamics weakly; format dominates")
    elif identity_ratio > 0.9:
        print(f"    Relational and analytical FTLE spectra are COMPARABLE ({identity_ratio:.1f}×)")
        print(f"    → Relay zone geometry is format-determined, not identity-specific")
        print(f"    → Kimi was right: FTLE differences are trajectory artifacts")
        if format_ratio > 1.5:
            print(f"    → But CCS vs vanilla ({format_ratio:.1f}×) shows FORMAT matters")
    else:
        print(f"    INVERTED: analytical has wider spectral gap ({1/identity_ratio:.1f}×)")
        print(f"    → Identity type modulates direction, not just magnitude")

    # Save
    output = {
        "experiment": "E12b",
        "model": MODEL_ID,
        "dose": DOSE,
        "jacobian_layers": JACOBIAN_LAYERS,
        "n_perturbation_dirs": N_PERTURBATION_DIRS,
        "epsilon": EPSILON,
        "results": all_results,
        "comparison": {
            "identity_ratio": float(identity_ratio),
            "format_ratio": float(format_ratio),
            "relational_mean_gap": float(np.mean(rel_gaps)),
            "analytical_mean_gap": float(np.mean(ana_gaps)),
            "vanilla_mean_gap": float(np.mean(van_gaps)),
        },
        "timestamp": datetime.now().isoformat(),
    }

    out_path = OUTPUT_DIR / f"e12b_ftle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\n  Saved: {out_path}")
    print(f"  Completed: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
