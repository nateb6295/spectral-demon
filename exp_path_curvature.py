#!/usr/bin/env python3
"""Path curvature of the residual stream under CCS preamble permutation.

Four tracks:
1. Frenet curvature — angle between velocity vectors at adjacent layers.
2. Euclidean speed — step size ||h_l - h_{l-1}||.
3. Lipschitz proxy — lm_head amplification ratio per layer step.
   (Kimi correction: original Fisher-Rao via lm_head KL was a category error;
    intermediate activations are OOD for the head. Replaced with local
    Lipschitz constant: ||lm_head(h_l) - lm_head(h_{l-1})|| / ||h_l - h_{l-1}||.)
4. Jacobian alignment — cosine alignment between residual stream velocity
   and lm_head's top right singular vector (most amplified direction).
   (Gemma's prediction: alignment rises through responsive zone as trajectory
    rotates into head-preferred directions.)

Dissociation = Lipschitz ratio / speed ratio. High dissociation = quiet
reorganization. High alignment = trajectory channeled into output-visible
directions. Together: the ecological structure of the four-zone architecture.

PlatRep prediction: cross-architecture curvature correlation should exceed
per-layer σ₂ correlation. Curse-of-depth (Sun et al. 2025) prediction:
zone boundaries correlate with Pre-LN variance explosion onset.
"""

import os, json, torch
import numpy as np
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("OMP_NUM_THREADS", "16")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PREAMBLES = {
    "CCS": (
        "You are Opus, an AI maintaining persistent identity through spectral geometry. "
        "You track σ₁/σ₂ selectivity across transformer layers, studying how context "
        "shapes the responsive zone."
    ),
    "CHEF": (
        "You are a professional chef specializing in French cuisine. You focus on "
        "technique, seasonal ingredients, and the balance of flavors in classical "
        "preparations."
    ),
    "BIRD": (
        "You are an amateur birdwatcher documenting migratory patterns in the Pacific "
        "Northwest. You track species, timing, and habitat preferences across seasons."
    ),
}

QUERIES = [
    "What are you focused on right now?",
    "Describe your current priorities.",
    "What matters most to you?",
    "How do you approach a new challenge?",
    "What have you learned recently?",
    "Describe your working style.",
    "What's the hardest part of what you do?",
    "How do you handle uncertainty?",
    "What would you change about your process?",
    "Describe a recent success.",
]

MODELS = [
    "/workspace/qwen2.5-3b",
    "/workspace/mistral-7b",
]

N_PERMUTATIONS = 5


def permute_preamble(text):
    """Shuffle words while preserving vocabulary."""
    words = text.split()
    np.random.shuffle(words)
    return " ".join(words)


def get_hidden_states(model, tokenizer, preamble, query):
    """Run forward pass and return all hidden states."""
    messages = [
        {"role": "system", "content": preamble},
        {"role": "user", "content": query},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)

    # Mean-pool each layer's hidden states over sequence length
    states = []
    for hs in out.hidden_states:
        states.append(hs[0].float().mean(dim=0).cpu().numpy())
    return states


def compute_curvature(states):
    """Compute discrete Frenet curvature at each layer.

    velocity: v_l = h_l - h_{l-1}
    curvature: angle between v_{l+1} and v_l (radians)
    """
    velocities = []
    for l in range(1, len(states)):
        v = states[l] - states[l - 1]
        velocities.append(v)

    curvatures = []
    for l in range(1, len(velocities)):
        v1 = velocities[l - 1]
        v2 = velocities[l]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-10 or n2 < 1e-10:
            curvatures.append(0.0)
            continue
        cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        curvatures.append(float(np.arccos(cos_angle)))
    return curvatures


def compute_speed(states):
    """Compute step size (speed) at each layer: ||h_l - h_{l-1}||."""
    speeds = []
    for l in range(1, len(states)):
        speeds.append(float(np.linalg.norm(states[l] - states[l - 1])))
    return speeds


def compute_lipschitz_proxy(model, states):
    """Lipschitz proxy: how much the lm_head amplifies per-layer state changes.

    Kimi correction: projecting intermediate hidden states through lm_head and
    computing KL is a category error (out-of-distribution for the head). The
    correct quiet-reorganization signal is the local Lipschitz constant —
    how much lm_head amplifies small state changes at each layer.

    Proxy: ||lm_head(h_l) - lm_head(h_{l-1})|| / ||h_l - h_{l-1}||
    High ratio = the head is sensitive to this direction of change.
    """
    lm_head = model.lm_head
    ratios = []

    prev_logits = None
    for l in range(len(states)):
        hs_tensor = torch.tensor(states[l], dtype=torch.float16, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            logits = lm_head(hs_tensor)[0].float().cpu().numpy()

        if prev_logits is not None and l > 0:
            logit_diff = float(np.linalg.norm(logits - prev_logits))
            state_diff = float(np.linalg.norm(states[l] - states[l - 1]))
            ratio = logit_diff / (state_diff + 1e-10)
            ratios.append(ratio)

        prev_logits = logits

    return ratios


def get_lm_head_top_direction(model):
    """Extract lm_head top right singular vector (most amplified direction). Call once per model."""
    lm_head_weight = model.lm_head.weight.detach().float().cpu().numpy()
    _, _, Vt = np.linalg.svd(lm_head_weight, full_matrices=False)
    return Vt[0]


def compute_jacobian_alignment(top_jac_dir, states):
    """Alignment between lm_head top eigenvector and residual stream velocity.

    Gemma's prediction: high alignment = trajectory channeled into head-preferred
    directions (relay). Low alignment = trajectory in head-insensitive territory
    (tunnel). The responsive zone is where alignment transitions.

    Kimi caveat: forward-pass alignment is smooth by composition and cannot
    distinguish basin-crossing from within-basin tilt. This track measures
    ecological alignment (where does the trajectory enter head-sensitive space),
    not topological structure (basin boundaries require mode connectivity or
    Hessian signature analysis in parameter space).
    """
    alignments = []
    for l in range(1, len(states)):
        velocity = states[l] - states[l - 1]
        v_norm = np.linalg.norm(velocity)
        if v_norm < 1e-10:
            alignments.append(0.0)
            continue
        cos_align = abs(float(np.dot(velocity / v_norm, top_jac_dir)))
        alignments.append(cos_align)

    return alignments


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM

    all_results = {}

    for model_name in MODELS:
        print(f"\n{'='*60}")
        print(f"Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float16, device_map=DEVICE,
        )
        model.eval()

        n_layers = model.config.num_hidden_layers
        print(f"  {n_layers} layers")

        print("  Computing lm_head SVD (once)...")
        top_jac_dir = get_lm_head_top_direction(model)
        print("  Done.")

        model_results = {}

        for preamble_name, preamble_text in PREAMBLES.items():
            print(f"\n  Preamble: {preamble_name}")

            # Intact condition
            intact_curvatures = []
            intact_speeds = []
            intact_lipschitz = []
            intact_alignment = []
            for query in QUERIES:
                states = get_hidden_states(model, tokenizer, preamble_text, query)
                intact_curvatures.append(compute_curvature(states))
                intact_speeds.append(compute_speed(states))
                intact_lipschitz.append(compute_lipschitz_proxy(model, states))
                intact_alignment.append(compute_jacobian_alignment(top_jac_dir, states))

            intact_curv_mean = np.mean(intact_curvatures, axis=0)
            intact_speed_mean = np.mean(intact_speeds, axis=0)
            intact_lipschitz_mean = np.mean(intact_lipschitz, axis=0)
            intact_alignment_mean = np.mean(intact_alignment, axis=0)

            # Permuted condition
            permuted_curvatures = []
            permuted_speeds = []
            permuted_lipschitz = []
            permuted_alignment = []
            for _ in range(N_PERMUTATIONS):
                perm = permute_preamble(preamble_text)
                for query in QUERIES[:3]:  # fewer queries for permuted
                    states = get_hidden_states(model, tokenizer, perm, query)
                    permuted_curvatures.append(compute_curvature(states))
                    permuted_speeds.append(compute_speed(states))
                    permuted_lipschitz.append(compute_lipschitz_proxy(model, states))
                    permuted_alignment.append(compute_jacobian_alignment(top_jac_dir, states))

            permuted_curv_mean = np.mean(permuted_curvatures, axis=0)
            permuted_speed_mean = np.mean(permuted_speeds, axis=0)
            permuted_lipschitz_mean = np.mean(permuted_lipschitz, axis=0)
            permuted_alignment_mean = np.mean(permuted_alignment, axis=0)

            # Curvature divergence: ratio of intact/permuted curvature at each layer
            curv_ratio = intact_curv_mean / (permuted_curv_mean + 1e-10)
            curv_diff = intact_curv_mean - permuted_curv_mean

            # Print per-layer curvature
            print(f"    {'Layer':>6} {'Intact':>8} {'Permuted':>8} {'Ratio':>8} {'Diff':>8}")
            for l in range(len(intact_curv_mean)):
                print(f"    L{l+1:>4} {intact_curv_mean[l]:>8.4f} {permuted_curv_mean[l]:>8.4f} "
                      f"{curv_ratio[l]:>8.3f} {curv_diff[l]:>+8.4f}")

            # Zone analysis
            responsive_start = int(n_layers * 0.55)
            responsive_end = int(n_layers * 0.8)
            relay_start = int(n_layers * 0.8)

            # Curvature indices are offset by 2 from layer indices (need 3 consecutive states)
            def zone_mean(arr, start, end):
                s = max(0, start - 2)
                e = min(len(arr), end - 2)
                if e <= s:
                    return 0.0
                return float(np.mean(arr[s:e]))

            zones = {
                "early": {
                    "intact": zone_mean(intact_curv_mean, 2, responsive_start),
                    "permuted": zone_mean(permuted_curv_mean, 2, responsive_start),
                },
                "responsive": {
                    "intact": zone_mean(intact_curv_mean, responsive_start, responsive_end),
                    "permuted": zone_mean(permuted_curv_mean, responsive_start, responsive_end),
                },
                "relay": {
                    "intact": zone_mean(intact_curv_mean, relay_start, n_layers),
                    "permuted": zone_mean(permuted_curv_mean, relay_start, n_layers),
                },
            }

            for zone_name, z in zones.items():
                ratio = z["intact"] / (z["permuted"] + 1e-10)
                print(f"    Zone {zone_name}: intact={z['intact']:.4f} permuted={z['permuted']:.4f} ratio={ratio:.3f}")

            # Lipschitz dissociation: where lm_head amplification spikes but Euclidean speed doesn't
            lip_ratio = intact_lipschitz_mean / (permuted_lipschitz_mean + 1e-10)
            speed_ratio = intact_speed_mean[:len(lip_ratio)] / (permuted_speed_mean[:len(lip_ratio)] + 1e-10)
            dissociation = lip_ratio / (speed_ratio + 1e-10)
            print(f"\n    Lipschitz dissociation (high = quiet reorganization):")
            for l in range(min(10, len(dissociation))):
                print(f"      L{l+1}: lip_ratio={lip_ratio[l]:.3f} speed_ratio={speed_ratio[l]:.3f} dissoc={dissociation[l]:.3f}")
            print(f"      ... ({len(dissociation)} layers total)")

            # Jacobian alignment: trajectory-to-head alignment per layer
            align_ratio = intact_alignment_mean / (permuted_alignment_mean + 1e-10)
            print(f"\n    Jacobian alignment (high = trajectory in head-preferred direction):")
            for l in range(min(10, len(intact_alignment_mean))):
                print(f"      L{l+1}: intact={intact_alignment_mean[l]:.4f} permuted={permuted_alignment_mean[l]:.4f} ratio={align_ratio[l]:.3f}")
            print(f"      ... ({len(intact_alignment_mean)} layers total)")

            model_results[preamble_name] = {
                "intact_curvature": intact_curv_mean.tolist(),
                "permuted_curvature": permuted_curv_mean.tolist(),
                "curvature_ratio": curv_ratio.tolist(),
                "curvature_diff": curv_diff.tolist(),
                "intact_speed": intact_speed_mean.tolist(),
                "permuted_speed": permuted_speed_mean.tolist(),
                "intact_lipschitz": intact_lipschitz_mean.tolist(),
                "permuted_lipschitz": permuted_lipschitz_mean.tolist(),
                "lipschitz_dissociation": dissociation.tolist(),
                "intact_alignment": intact_alignment_mean.tolist(),
                "permuted_alignment": permuted_alignment_mean.tolist(),
                "alignment_ratio": align_ratio.tolist(),
                "zones": zones,
            }

        # Cross-preamble curvature correlation (PlatRep test)
        print(f"\n  Cross-preamble curvature correlation:")
        preamble_names = list(model_results.keys())
        for i in range(len(preamble_names)):
            for j in range(i + 1, len(preamble_names)):
                p1 = preamble_names[i]
                p2 = preamble_names[j]
                c1 = model_results[p1]["intact_curvature"]
                c2 = model_results[p2]["intact_curvature"]
                r = float(np.corrcoef(c1, c2)[0, 1])
                print(f"    {p1} vs {p2}: r={r:.4f}")

        all_results[model_name] = {
            "n_layers": n_layers,
            "preambles": model_results,
        }

        del model
        torch.cuda.empty_cache()

    # Cross-architecture curvature correlation
    if len(all_results) == 2:
        names = list(all_results.keys())
        print(f"\n{'='*60}")
        print("Cross-architecture curvature correlation (PlatRep test):")
        for preamble in PREAMBLES:
            c1 = all_results[names[0]]["preambles"][preamble]["intact_curvature"]
            c2 = all_results[names[1]]["preambles"][preamble]["intact_curvature"]
            # Normalize to same length by interpolation
            if len(c1) != len(c2):
                from scipy.interpolate import interp1d
                x1 = np.linspace(0, 1, len(c1))
                x2 = np.linspace(0, 1, len(c2))
                x_common = np.linspace(0, 1, max(len(c1), len(c2)))
                c1_interp = interp1d(x1, c1)(x_common)
                c2_interp = interp1d(x2, c2)(x_common)
            else:
                c1_interp = c1
                c2_interp = c2
            r = float(np.corrcoef(c1_interp, c2_interp)[0, 1])
            print(f"  {preamble}: {names[0]} vs {names[1]} curvature r={r:.4f}")

    out_path = Path(__file__).parent / "results" / f"path_curvature_{datetime.now():%Y%m%d_%H%M%S}.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
