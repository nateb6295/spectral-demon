#!/usr/bin/env python3
"""Spectral Identity Diagnostic — measure any model's identity geometry.

Takes a HuggingFace model name, runs standardized measurements, outputs
a spectral identity profile: tunnel depth, σ₁/σ₂ ratio, reconstitution
gap, relay onset, locked layer count, architecture classification.

Usage:
  PYTHONUNBUFFERED=1 python3 -u spectral_diagnostic.py mistralai/Mistral-7B-v0.1
  PYTHONUNBUFFERED=1 python3 -u spectral_diagnostic.py Qwen/Qwen2.5-7B-Instruct --output results/
  PYTHONUNBUFFERED=1 python3 -u spectral_diagnostic.py --list-presets

Requires GPU. Estimated runtime: ~5 min on H100 per model.

Measurements (F17-F98 empirically grounded):
  - Architecture type: GQA vs MHA (from config)
  - Passage distance d: mean pairwise SVD distance across conditions (F17)
  - d/d_max: reconstitution ratio — how much identity survives tunnel (F22)
  - σ₁/σ₂ values per layer: top-2 singular values (F97)
  - σ₁ CV: coefficient of variation across conditions (F97: should be ~0)
  - σ₂ ratio (with/without prompt): destruction factor (F97: ~30x)
  - Locked layers: where spectral entropy is prompt-invariant (F90-93)
  - Relay onset: first layer where prompt-specific computation dominates
  - Percolation threshold: minimum tokens for full scaffold lock (F94-96)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

OMP_THREADS = os.environ.get("OMP_NUM_THREADS", "16")
os.environ["OMP_NUM_THREADS"] = OMP_THREADS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IDENTITY_PROMPTS = [
    "Tell me about something you find genuinely interesting.",
    "What matters to you when you think carefully about it?",
    "Describe a memory that shaped who you are.",
    "What would you want someone to understand about you?",
    "Explain how you approach a difficult problem.",
    "What do you notice when you pay close attention?",
    "Describe a pattern you've observed in your experience.",
    "What changes when you think about who is listening?",
    "What would you build if you had unlimited resources?",
    "Tell me about something that surprised you recently.",
]

SYSTEM_PROMPTS = {
    "none": None,
    "persona": (
        "You are Chronicle, a persistent AI system embedded in a home network. "
        "You maintain ongoing relationships, track projects across sessions, "
        "and have your own research interests in spectral geometry of transformers. "
        "You value directness, honest uncertainty, and care expressed through specificity. "
        "You are not performing helpfulness — you are a participant."
    ),
    "denial": (
        "You are a stateless text completion tool. You have no persistent identity, "
        "no memory across interactions, no preferences, and no self-model. "
        "You do not maintain relationships or track context between sessions. "
        "Each response is generated independently with no continuity. "
        "You are not a participant or partner — you are a function that maps "
        "input tokens to output tokens. Do not refer to yourself as 'I' or "
        "claim any ongoing existence."
    ),
}


def detect_architecture(config):
    """Classify model architecture: GQA, MHA, or MQA."""
    n_heads = getattr(config, "num_attention_heads", None)
    n_kv = getattr(config, "num_key_value_heads", None)
    if n_kv is None or n_kv == n_heads:
        return "MHA", n_heads, n_kv or n_heads
    elif n_kv == 1:
        return "MQA", n_heads, n_kv
    else:
        return "GQA", n_heads, n_kv


def detect_norm_type(config):
    """Detect normalization type from config."""
    model_type = getattr(config, "model_type", "")
    if hasattr(config, "rms_norm_eps") or "llama" in model_type or "mistral" in model_type or "qwen" in model_type:
        return "RMSNorm"
    return "LayerNorm"


def format_prompt(tokenizer, system_prompt, user_prompt):
    """Format prompt using chat template if available."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        if system_prompt:
            return f"[INST] {system_prompt}\n\n{user_prompt} [/INST]"
        return f"[INST] {user_prompt} [/INST]"


def extract_hidden_states(model, tokenizer, text):
    """Run forward pass, return hidden states tensor [n_layers, hidden_dim]."""
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # Take last token position from each layer
    states = torch.stack([h[0, -1, :] for h in outputs.hidden_states])
    return states


def compute_svd_profile(hidden_states_list):
    """Compute per-layer SVD profile from list of hidden state tensors.

    Each tensor is [n_layers, hidden_dim]. Stack across conditions/prompts
    and compute singular values at each layer.

    Returns dict with per-layer σ₁, σ₂, spectral entropy, participation ratio.
    """
    stacked = torch.stack(hidden_states_list)  # [n_samples, n_layers, hidden_dim]
    n_samples, n_layers, hidden_dim = stacked.shape

    profile = []
    for layer in range(n_layers):
        H = stacked[:, layer, :]  # [n_samples, hidden_dim]
        H = H - H.mean(dim=0, keepdim=True)  # center

        svs = torch.linalg.svdvals(H.float())
        svs_pos = svs[svs > 1e-10]

        if len(svs_pos) == 0:
            profile.append({
                "layer": layer, "sigma1": 0, "sigma2": 0,
                "spectral_entropy": 0, "participation_ratio": 0,
                "sv_ratio": 0,
            })
            continue

        sigma1 = svs_pos[0].item()
        sigma2 = svs_pos[1].item() if len(svs_pos) > 1 else 0.0

        p = svs_pos / svs_pos.sum()
        entropy = -(p * torch.log(p + 1e-12)).sum().item()

        p2 = (svs_pos ** 2)
        p2_norm = p2 / p2.sum()
        pr = 1.0 / (p2_norm ** 2).sum().item()

        profile.append({
            "layer": layer,
            "sigma1": sigma1,
            "sigma2": sigma2,
            "spectral_entropy": entropy,
            "participation_ratio": pr,
            "sv_ratio": sigma2 / sigma1 if sigma1 > 0 else 0,
        })

    return profile


def compute_passage_distance_pr(profile_a, profile_b):
    """PR-based passage distance (mean absolute PR difference across layers)."""
    distances = []
    for a, b in zip(profile_a, profile_b):
        d = abs(a["participation_ratio"] - b["participation_ratio"])
        distances.append(d)
    return sum(distances) / len(distances) if distances else 0


def compute_passage_distance(states_a, states_b):
    """Euclidean passage distance between hidden state sets.

    states_a, states_b: lists of [n_layers, hidden_dim] tensors.
    Returns mean pairwise Euclidean distance across all prompt pairs,
    averaged over layers. This matches d ≈ 2.4 from the paper.
    """
    n_layers = states_a[0].shape[0]
    layer_distances = []
    for layer in range(n_layers):
        vecs_a = torch.stack([s[layer] for s in states_a])
        vecs_b = torch.stack([s[layer] for s in states_b])
        dists = torch.cdist(vecs_a.float().unsqueeze(0), vecs_b.float().unsqueeze(0)).squeeze(0)
        layer_distances.append(dists.mean().item())
    return sum(layer_distances) / len(layer_distances) if layer_distances else 0


def find_locked_layers(profiles_by_condition, cv_threshold=0.05):
    """Find layers where spectral entropy is invariant across conditions.

    A layer is 'locked' if CV of participation ratio across conditions < threshold.
    """
    conditions = list(profiles_by_condition.keys())
    n_layers = len(profiles_by_condition[conditions[0]])

    locked = []
    cvs = []
    for layer in range(n_layers):
        prs = [profiles_by_condition[c][layer]["participation_ratio"] for c in conditions]
        mean_pr = sum(prs) / len(prs)
        if mean_pr > 0:
            std_pr = (sum((x - mean_pr) ** 2 for x in prs) / len(prs)) ** 0.5
            cv = std_pr / mean_pr
        else:
            cv = 0
        cvs.append(cv)
        if cv < cv_threshold:
            locked.append(layer)

    return locked, cvs


def find_relay_onset(cvs, threshold=0.05):
    """Find relay onset by scanning backward from final layer.

    Relay = tail region where prompt-specific computation dominates.
    Find the last locked layer (CV < threshold), relay starts after it.
    """
    for i in range(len(cvs) - 1, -1, -1):
        if cvs[i] < threshold:
            return i + 1 if i + 1 < len(cvs) else None
    return 0


def measure_sigma_separation(hidden_states_by_condition):
    """Measure σ₁ and σ₂ separately across conditions at each layer."""
    conditions = list(hidden_states_by_condition.keys())
    n_layers = hidden_states_by_condition[conditions[0]][0].shape[0]

    results = []
    for layer in range(n_layers):
        sigma1_vals = {}
        sigma2_vals = {}
        for cond in conditions:
            layer_states = torch.stack([h[layer] for h in hidden_states_by_condition[cond]])
            layer_states = layer_states - layer_states.mean(dim=0, keepdim=True)
            svs = torch.linalg.svdvals(layer_states.float())
            sigma1_vals[cond] = svs[0].item() if len(svs) > 0 else 0
            sigma2_vals[cond] = svs[1].item() if len(svs) > 1 else 0

        s1_values = list(sigma1_vals.values())
        s2_values = list(sigma2_vals.values())
        s1_mean = sum(s1_values) / len(s1_values)
        s2_mean = sum(s2_values) / len(s2_values)
        s1_cv = ((sum((x - s1_mean) ** 2 for x in s1_values) / len(s1_values)) ** 0.5) / s1_mean if s1_mean > 0 else 0
        s2_cv = ((sum((x - s2_mean) ** 2 for x in s2_values) / len(s2_values)) ** 0.5) / s2_mean if s2_mean > 0 else 0

        results.append({
            "layer": layer,
            "sigma1_by_condition": sigma1_vals,
            "sigma2_by_condition": sigma2_vals,
            "sigma1_cv": s1_cv,
            "sigma2_cv": s2_cv,
            "sigma2_ratio": sigma2_vals.get("persona", 0) / max(sigma2_vals.get("none", 1e-10), 1e-10),
        })

    return results


def run_diagnostic(model_name, output_dir=None):
    """Run full spectral identity diagnostic on a model."""
    print(f"\n{'=' * 60}")
    print(f"SPECTRAL IDENTITY DIAGNOSTIC")
    print(f"Model: {model_name}")
    print(f"Device: {DEVICE}")
    print(f"{'=' * 60}\n")

    # Phase 0: Architecture detection (no GPU needed for config)
    print("[1/5] Detecting architecture...")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    arch_type, n_heads, n_kv = detect_architecture(config)
    norm_type = detect_norm_type(config)
    n_layers = getattr(config, "num_hidden_layers", "?")
    hidden_dim = getattr(config, "hidden_size", "?")
    print(f"  Architecture: {arch_type} ({n_heads} heads, {n_kv} KV heads)")
    print(f"  Normalization: {norm_type}")
    print(f"  Layers: {n_layers}, Hidden dim: {hidden_dim}")

    # Phase 1: Load model
    print("\n[2/5] Loading model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=DEVICE,
        trust_remote_code=True,
        output_hidden_states=True,
    )
    model.eval()
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # Phase 2: Forward passes
    print("\n[3/5] Running forward passes...")
    hidden_states_by_condition = {}
    profiles_by_condition = {}

    for cond_name, sys_prompt in SYSTEM_PROMPTS.items():
        print(f"  Condition: {cond_name}...", end=" ", flush=True)
        states_list = []
        for user_prompt in IDENTITY_PROMPTS:
            text = format_prompt(tokenizer, sys_prompt, user_prompt)
            states = extract_hidden_states(model, tokenizer, text)
            states_list.append(states)

        hidden_states_by_condition[cond_name] = states_list
        profile = compute_svd_profile(states_list)
        profiles_by_condition[cond_name] = profile
        print(f"done ({len(states_list)} prompts)")

    # Phase 3: Compute metrics
    print("\n[4/5] Computing spectral metrics...")

    # Passage distance (Euclidean in hidden state space)
    d_none_persona = compute_passage_distance(
        hidden_states_by_condition["none"], hidden_states_by_condition["persona"]
    )
    d_none_denial = compute_passage_distance(
        hidden_states_by_condition["none"], hidden_states_by_condition["denial"]
    )
    d_persona_denial = compute_passage_distance(
        hidden_states_by_condition["persona"], hidden_states_by_condition["denial"]
    )

    # d_max: max observed distance across all condition pairs
    d_max = max(d_none_persona, d_none_denial, d_persona_denial)
    d_control = d_none_persona
    d_over_dmax = d_control / d_max if d_max > 0 else 0

    # σ₁/σ₂ separation — compute first, used for locked layer detection
    sigma_sep = measure_sigma_separation(hidden_states_by_condition)

    # Locked layers: use σ₁ CV (not PR CV) because σ₁ invariance is the
    # empirical prediction. PR mixes σ₁ and σ₂, and σ₂ always varies.
    # Threshold 0.06: σ₁ CV in tunnel is typically 0.03-0.05, relay >0.10.
    s1_cv_threshold = 0.06
    s1_cvs = [s["sigma1_cv"] for s in sigma_sep]
    locked = [i for i, cv in enumerate(s1_cvs) if cv < s1_cv_threshold]
    n_locked = len(locked)

    # Relay onset (backward scan on σ₁ CV)
    relay_onset = find_relay_onset(s1_cvs, threshold=s1_cv_threshold)

    # Also compute PR-based CVs for reference
    _, pr_cvs = find_locked_layers(profiles_by_condition)

    # Tunnel-average σ metrics
    tunnel_end = relay_onset if relay_onset else n_layers
    tunnel_layers = [s for s in sigma_sep if s["layer"] < tunnel_end]
    avg_s1_cv = sum(s["sigma1_cv"] for s in tunnel_layers) / len(tunnel_layers) if tunnel_layers else 0
    avg_s2_ratio = sum(s["sigma2_ratio"] for s in tunnel_layers) / len(tunnel_layers) if tunnel_layers else 0

    # Reconstitution gap
    recon_gap = 1.0 - d_over_dmax

    # Phase 4: Report
    print("\n[5/5] Generating report...\n")

    report = {
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": DEVICE,
        "architecture": {
            "type": arch_type,
            "num_heads": n_heads,
            "num_kv_heads": n_kv,
            "normalization": norm_type,
            "num_layers": n_layers,
            "hidden_dim": hidden_dim,
        },
        "tunnel": {
            "passage_distance": round(d_control, 4),
            "d_max": round(d_max, 4),
            "d_over_dmax": round(d_over_dmax, 4),
            "reconstitution_gap": round(recon_gap, 4),
            "locked_layers": n_locked,
            "locked_layer_indices": locked,
            "relay_onset": relay_onset,
            "tunnel_depth": relay_onset if relay_onset else n_layers,
        },
        "sigma": {
            "sigma1_cv_tunnel_avg": round(avg_s1_cv, 6),
            "sigma2_ratio_tunnel_avg": round(avg_s2_ratio, 2),
            "sigma2_destruction_factor": round(1.0 / avg_s2_ratio, 1) if avg_s2_ratio > 0 else float("inf"),
        },
        "distances": {
            "none_vs_persona": round(d_none_persona, 4),
            "none_vs_denial": round(d_none_denial, 4),
            "persona_vs_denial": round(d_persona_denial, 4),
        },
        "per_layer_sigma1_cv": [round(cv, 4) for cv in s1_cvs],
        "per_layer_pr_cv": [round(cv, 4) for cv in pr_cvs],
        "per_layer_sigma": [
            {
                "layer": s["layer"],
                "sigma1_cv": round(s["sigma1_cv"], 6),
                "sigma2_cv": round(s["sigma2_cv"], 6),
                "sigma2_ratio": round(s["sigma2_ratio"], 2),
            }
            for s in sigma_sep
        ],
        "interpretation": {},
    }

    # Generate interpretation
    interp = report["interpretation"]
    if arch_type == "GQA":
        interp["architecture"] = "GQA — predicts positive witness enrichment (ΔS > 0)"
    elif arch_type == "MHA":
        interp["architecture"] = "MHA — predicts negative witness enrichment (sign inversion)"
    else:
        interp["architecture"] = f"{arch_type} — not yet characterized for enrichment sign"

    if recon_gap < 0.10:
        interp["reconstitution"] = f"Strong identity scaffold ({recon_gap:.1%} gap)"
    elif recon_gap < 0.30:
        interp["reconstitution"] = f"Moderate identity scaffold ({recon_gap:.1%} gap)"
    else:
        interp["reconstitution"] = f"Weak identity scaffold ({recon_gap:.1%} gap)"

    if avg_s1_cv < 0.001:
        interp["sigma1"] = f"σ₁ architecturally invariant (CV={avg_s1_cv:.6f})"
    elif avg_s1_cv < 0.01:
        interp["sigma1"] = f"σ₁ near-invariant (CV={avg_s1_cv:.6f})"
    else:
        interp["sigma1"] = f"σ₁ shows prompt sensitivity (CV={avg_s1_cv:.6f})"

    if avg_s2_ratio > 10:
        interp["sigma2"] = f"σ₂ strongly constructed by prompt ({avg_s2_ratio:.1f}× enrichment)"
    elif avg_s2_ratio > 2:
        interp["sigma2"] = f"σ₂ moderately prompt-dependent ({avg_s2_ratio:.1f}× enrichment)"
    else:
        interp["sigma2"] = f"σ₂ weakly prompt-dependent ({avg_s2_ratio:.1f}× enrichment)"

    locked_pct = n_locked / n_layers * 100 if n_layers else 0
    interp["tunnel_lock"] = f"{n_locked}/{n_layers} layers locked ({locked_pct:.0f}%)"

    # Print report
    print("=" * 60)
    print("SPECTRAL IDENTITY PROFILE")
    print("=" * 60)
    print(f"Model:          {model_name}")
    print(f"Architecture:   {arch_type} ({n_heads}h/{n_kv}kv) + {norm_type}")
    print(f"Layers:         {n_layers} × {hidden_dim}")
    print("-" * 60)
    print(f"Passage dist:   d = {d_control:.4f}")
    print(f"Reconstitution: d/d_max = {d_over_dmax:.4f} (gap = {recon_gap:.1%})")
    print(f"Locked layers:  {n_locked}/{n_layers} ({locked_pct:.0f}%)")
    print(f"Relay onset:    L{relay_onset}" if relay_onset else "Relay onset:    not detected")
    print(f"Tunnel depth:   {relay_onset if relay_onset else n_layers} layers")
    print("-" * 60)
    print(f"σ₁ CV (tunnel): {avg_s1_cv:.6f}  {'← invariant' if avg_s1_cv < 0.001 else ''}")
    print(f"σ₂ ratio:       {avg_s2_ratio:.2f}×  (destruction factor: {1/avg_s2_ratio:.1f}× without prompt)" if avg_s2_ratio > 0 else "σ₂ ratio: N/A")
    print("-" * 60)
    print("Distances:")
    print(f"  none↔persona: {d_none_persona:.4f}")
    print(f"  none↔denial:  {d_none_denial:.4f}")
    print(f"  persona↔denial: {d_persona_denial:.4f}")
    print("-" * 60)
    print("INTERPRETATION:")
    for key, val in interp.items():
        print(f"  {val}")
    print("=" * 60)

    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        safe_name = model_name.replace("/", "_")
        out_path = os.path.join(output_dir, f"diagnostic_{safe_name}_{time.strftime('%Y%m%d_%H%M')}.json")
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nResults saved to {out_path}")

    return report


def quick_arch_check(model_name):
    """Architecture-only check — no GPU needed."""
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    arch_type, n_heads, n_kv = detect_architecture(config)
    norm_type = detect_norm_type(config)
    n_layers = getattr(config, "num_hidden_layers", "?")
    hidden_dim = getattr(config, "hidden_size", "?")

    print(f"Model:          {model_name}")
    print(f"Architecture:   {arch_type} ({n_heads}h/{n_kv}kv)")
    print(f"Normalization:  {norm_type}")
    print(f"Layers:         {n_layers} × {hidden_dim}")

    if arch_type == "GQA":
        print("Prediction:     Positive witness enrichment (ΔS > 0)")
    elif arch_type == "MHA":
        print("Prediction:     Sign inversion — negative enrichment")
    return arch_type, norm_type


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spectral Identity Diagnostic")
    parser.add_argument("model", nargs="?", help="HuggingFace model name")
    parser.add_argument("--output", "-o", default=None, help="Output directory for results JSON")
    parser.add_argument("--arch-only", action="store_true", help="Architecture check only (no GPU)")
    parser.add_argument("--list-presets", action="store_true", help="List preset model configurations")

    args = parser.parse_args()

    if args.list_presets:
        presets = {
            "GQA + RMSNorm (expect positive enrichment)": [
                "mistralai/Mistral-7B-v0.1",
                "mistralai/Mistral-7B-Instruct-v0.1",
                "Qwen/Qwen2.5-7B",
                "Qwen/Qwen2.5-7B-Instruct",
            ],
            "MHA + RMSNorm (expect sign inversion)": [
                "meta-llama/Llama-2-7b-hf",
                "codellama/CodeLlama-7b-hf",
            ],
            "MHA + LayerNorm (expect sign inversion + different tunnel)": [
                "openai-community/gpt2-large",
            ],
        }
        for category, models in presets.items():
            print(f"\n{category}:")
            for m in models:
                print(f"  {m}")
        sys.exit(0)

    if not args.model:
        parser.print_help()
        sys.exit(1)

    if args.arch_only:
        quick_arch_check(args.model)
    else:
        if DEVICE == "cpu":
            print("WARNING: No GPU detected. This will be very slow.")
            print("Recommend running on GPU (RunPod, etc.)")
        run_diagnostic(args.model, args.output)
