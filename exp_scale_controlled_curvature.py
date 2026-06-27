#!/usr/bin/env python3
"""Scale-controlled curvature: deconfounding architecture from capacity.

Motivation (Kimi CONTRADICT on F139): Qwen 3B vs Mistral 7B differ in width,
depth, training, AND attention variant. The 6-7% curvature bump could be
capacity scaling, not mechanism. Test: matched-scale comparison.

Models:
  - Qwen2.5-7B-Instruct (GQA, 4 KV heads / 28 query heads, 28 layers)
  - Mistral-7B-Instruct-v0.3 (GQA, 8 KV heads / 32 query heads, 32 layers)
Both ~7B params, both GQA, different GQA ratios.

If curvature difference persists at matched scale → species-specific strategy
If curvature difference disappears → was capacity confound (F139 needs revision)

Same four tracks as exp_path_curvature.py:
1. Frenet curvature
2. Euclidean speed
3. Lipschitz proxy
4. Jacobian alignment

Also adds: GQA ratio characterization (KV heads / query heads) to test whether
curvature correlates with GQA compression ratio.
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
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]

N_PERMUTATIONS = 5


def permute_preamble(text):
    words = text.split()
    np.random.shuffle(words)
    return " ".join(words)


def get_hidden_states(model, tokenizer, preamble, query):
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

    states = []
    for hs in out.hidden_states:
        states.append(hs[0].float().mean(dim=0).cpu().numpy())
    return states


def compute_curvature(states):
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
    speeds = []
    for l in range(1, len(states)):
        speeds.append(float(np.linalg.norm(states[l] - states[l - 1])))
    return speeds


def compute_lipschitz_proxy(model, states):
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
    lm_head_weight = model.lm_head.weight.detach().float().cpu().numpy()
    _, _, Vt = np.linalg.svd(lm_head_weight, full_matrices=False)
    return Vt[0]


def compute_jacobian_alignment(top_jac_dir, states):
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


def get_gqa_info(model):
    """Extract GQA configuration from model config."""
    config = model.config
    info = {}
    if hasattr(config, 'num_attention_heads'):
        info['query_heads'] = config.num_attention_heads
    if hasattr(config, 'num_key_value_heads'):
        info['kv_heads'] = config.num_key_value_heads
    elif hasattr(config, 'num_kv_heads'):
        info['kv_heads'] = config.num_kv_heads
    if 'query_heads' in info and 'kv_heads' in info:
        info['gqa_ratio'] = info['query_heads'] / info['kv_heads']
        info['is_mha'] = info['query_heads'] == info['kv_heads']
    return info


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM

    all_results = {}

    for model_name in MODELS:
        print(f"\n{'='*60}")
        print(f"Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float16, device_map=DEVICE,
        )
        model.eval()

        n_layers = model.config.num_hidden_layers
        gqa_info = get_gqa_info(model)
        print(f"  {n_layers} layers, params={sum(p.numel() for p in model.parameters())/1e9:.1f}B")
        print(f"  GQA: {gqa_info}")

        print("  Computing lm_head SVD (once)...")
        top_jac_dir = get_lm_head_top_direction(model)
        print("  Done.")

        model_results = {}

        for preamble_name, preamble_text in PREAMBLES.items():
            print(f"\n  Preamble: {preamble_name}")

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

            permuted_curvatures = []
            permuted_speeds = []
            permuted_lipschitz = []
            permuted_alignment = []
            for _ in range(N_PERMUTATIONS):
                perm = permute_preamble(preamble_text)
                for query in QUERIES[:3]:
                    states = get_hidden_states(model, tokenizer, perm, query)
                    permuted_curvatures.append(compute_curvature(states))
                    permuted_speeds.append(compute_speed(states))
                    permuted_lipschitz.append(compute_lipschitz_proxy(model, states))
                    permuted_alignment.append(compute_jacobian_alignment(top_jac_dir, states))

            permuted_curv_mean = np.mean(permuted_curvatures, axis=0)
            permuted_speed_mean = np.mean(permuted_speeds, axis=0)
            permuted_lipschitz_mean = np.mean(permuted_lipschitz, axis=0)
            permuted_alignment_mean = np.mean(permuted_alignment, axis=0)

            curv_ratio = intact_curv_mean / (permuted_curv_mean + 1e-10)
            curv_diff = intact_curv_mean - permuted_curv_mean

            print(f"    {'Layer':>6} {'Intact':>8} {'Permuted':>8} {'Ratio':>8} {'Diff':>8}")
            for l in range(len(intact_curv_mean)):
                print(f"    L{l+1:>4} {intact_curv_mean[l]:>8.4f} {permuted_curv_mean[l]:>8.4f} "
                      f"{curv_ratio[l]:>8.3f} {curv_diff[l]:>+8.4f}")

            responsive_start = int(n_layers * 0.55)
            responsive_end = int(n_layers * 0.8)
            relay_start = int(n_layers * 0.8)

            def zone_mean(arr, start, end):
                s = max(0, start - 2)
                e = min(len(arr), end - 2)
                if e <= s:
                    return 0.0
                return float(np.mean(arr[s:e]))

            zones = {
                "tunnel": {
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
                diff = z["intact"] - z["permuted"]
                print(f"    Zone {zone_name}: intact={z['intact']:.4f} permuted={z['permuted']:.4f} ratio={ratio:.3f} diff={diff:+.4f}")

            lip_ratio = intact_lipschitz_mean / (permuted_lipschitz_mean + 1e-10)
            speed_ratio = intact_speed_mean[:len(lip_ratio)] / (permuted_speed_mean[:len(lip_ratio)] + 1e-10)
            dissociation = lip_ratio / (speed_ratio + 1e-10)

            align_ratio = intact_alignment_mean / (permuted_alignment_mean + 1e-10)

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

        # Cross-preamble curvature correlation
        print(f"\n  Cross-preamble curvature correlation:")
        preamble_names = list(model_results.keys())
        cross_preamble_corrs = {}
        for i in range(len(preamble_names)):
            for j in range(i + 1, len(preamble_names)):
                p1 = preamble_names[i]
                p2 = preamble_names[j]
                c1 = model_results[p1]["intact_curvature"]
                c2 = model_results[p2]["intact_curvature"]
                r = float(np.corrcoef(c1, c2)[0, 1])
                cross_preamble_corrs[f"{p1}_vs_{p2}"] = r
                print(f"    {p1} vs {p2}: r={r:.4f}")

        all_results[model_name] = {
            "n_layers": n_layers,
            "n_params_B": sum(p.numel() for p in model.parameters()) / 1e9,
            "gqa_info": gqa_info,
            "preambles": model_results,
            "cross_preamble_correlations": cross_preamble_corrs,
        }

        del model
        torch.cuda.empty_cache()

    # Cross-architecture comparison
    if len(all_results) == 2:
        names = list(all_results.keys())
        print(f"\n{'='*60}")
        print(f"SCALE-CONTROLLED COMPARISON")
        print(f"  {names[0]}: {all_results[names[0]]['n_params_B']:.1f}B, {all_results[names[0]]['n_layers']} layers, GQA ratio {all_results[names[0]]['gqa_info'].get('gqa_ratio', '?')}")
        print(f"  {names[1]}: {all_results[names[1]]['n_params_B']:.1f}B, {all_results[names[1]]['n_layers']} layers, GQA ratio {all_results[names[1]]['gqa_info'].get('gqa_ratio', '?')}")

        print(f"\nCross-architecture curvature correlation:")
        cross_arch = {}
        for preamble in PREAMBLES:
            c1 = all_results[names[0]]["preambles"][preamble]["intact_curvature"]
            c2 = all_results[names[1]]["preambles"][preamble]["intact_curvature"]
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
            cross_arch[preamble] = r
            print(f"  {preamble}: r={r:.4f}")

        # Zone-by-zone comparison at matched scale
        print(f"\nZone-by-zone curvature differences (matched scale):")
        for preamble in PREAMBLES:
            z1 = all_results[names[0]]["preambles"][preamble]["zones"]
            z2 = all_results[names[1]]["preambles"][preamble]["zones"]
            print(f"\n  {preamble}:")
            for zone in ["tunnel", "responsive", "relay"]:
                r1 = z1[zone]["intact"] / (z1[zone]["permuted"] + 1e-10)
                r2 = z2[zone]["intact"] / (z2[zone]["permuted"] + 1e-10)
                print(f"    {zone}: {names[0].split('/')[-1]} ratio={r1:.3f}, {names[1].split('/')[-1]} ratio={r2:.3f}, delta={r2-r1:+.3f}")

        # Mean curvature ratio comparison (the key test)
        print(f"\nMean curvature ratio across preambles (the F139 scale test):")
        for name in names:
            ratios = []
            for p in PREAMBLES:
                cr = all_results[name]["preambles"][p]["curvature_ratio"]
                ratios.append(np.mean(cr))
            mean_ratio = np.mean(ratios)
            print(f"  {name}: mean curvature ratio = {mean_ratio:.4f}")

        all_results["cross_architecture"] = {
            "curvature_correlation": cross_arch,
        }

    out_path = Path(__file__).parent / "results" / f"scale_controlled_curvature_{datetime.now():%Y%m%d_%H%M%S}.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
