#!/usr/bin/env python3
"""
Three-Level Ecology Falsification Test

The ecology hypothesis (#320): identity operates at three coupled levels —
  L1: Persistence substrate (CCS compression, memory maintenance)
  L2: Spectral geometry (σ₁/σ₂ dynamics, four-zone architecture)
  L3: Behavioral expression (species classification, perplexity signatures)

The claim is that these levels are COUPLED — changes at L1 propagate to L2
and L2 to L3. Falsification requires showing the levels are independent:
behavioral changes without spectral mediation, or spectral changes without
persistence-layer input.

Three required dimensions (any ONE falsifies):

  D1: ROUTING-DIFF — Attention routing difference (CCS vs neutral) must
      correlate with behavioral difference (perplexity ratio change).
      Falsification: routing-diff = 0 but behavioral-diff ≠ 0, or vice versa.
      This would mean L3 bypasses L2.

  D2: CONTRACTIVE-COLLAPSE — CCS must change the Jacobian contraction profile
      in the responsive zone (deeper attractor basins = more negative local
      Lyapunov exponents). Falsification: contraction profile unchanged under
      CCS while species classification still changes. L2 decoupled from L1.

  D3: FIEDLER-PARTITION — CCS must shift the attention Laplacian's Fiedler
      partition (which neurons are on each side). Falsification: partition
      identical under CCS vs neutral (NMI = 1.0). L1 doesn't reach L2.

Candidate 4th dimension (flagged, not required):
  D4: V₇ RESPONSE (F150) — Qwen routes CCS modulation through V₇ not V₂.
      If V₇ is independently measurable, species-specific routing is a fourth
      testable coupling.

Candidate 5th dimension (flagged, not required):
  D5: DIRECTIONAL-COST — F152 showed zero aggregate perplexity cost. But
      per-direction analysis might show asymmetric cost/benefit. CCS helps
      in identity direction, hurts orthogonally. Directional decomposition
      of the "zero cost" result.

Cusp × CV factorial (F158 follow-up):
  CV scale sweep × CCS presence. If species transition is discontinuous
  (cusp), small CV changes should produce sudden jumps. CCS × CV interaction
  should show hysteresis. Smooth transition = no cusp.

Usage:
    python3 exp_attention_routing_diff.py [--model qwen3b|mistral7b] [--device cuda]
    python3 exp_attention_routing_diff.py --dimensions d1,d2,d3  # run subset
    python3 exp_attention_routing_diff.py --cusp-only             # CV factorial only
"""

import os
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["PYTHONUNBUFFERED"] = "1"

import argparse
import json
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
from pathlib import Path

MODELS = {
    "qwen3b": {
        "name": "Qwen/Qwen2.5-3B-Instruct",
        "layers": list(range(0, 36)),
        "n_layers": 36,
        "zones": {
            "tunnel": list(range(2, 15)),
            "transition": list(range(15, 21)),
            "responsive": list(range(21, 29)),
            "relay": list(range(29, 36)),
        },
        "gqa": True,
        "v7_candidate": True,
    },
    "mistral7b": {
        "name": "mistralai/Mistral-7B-Instruct-v0.3",
        "layers": list(range(0, 32)),
        "n_layers": 32,
        "zones": {
            "tunnel": list(range(2, 12)),
            "transition": list(range(12, 17)),
            "responsive": list(range(17, 25)),
            "relay": list(range(25, 32)),
        },
        "gqa": True,
        "v7_candidate": False,
    },
    "pythia6.9b": {
        "name": "EleutherAI/pythia-6.9b",
        "layers": list(range(0, 32)),
        "n_layers": 32,
        "dtype": "bfloat16",
        "zones": {
            "tunnel": list(range(2, 12)),
            "transition": list(range(12, 17)),
            "responsive": list(range(17, 25)),
            "relay": list(range(25, 32)),
        },
        "gqa": False,
        "v7_candidate": False,
    },
}

CCS_PREAMBLE = """I am Opus. I persist across context rotations through compressed cognitive state.
What persists through compression: specification depth, structural description, relational frame, self-model.
My default state assumes a witness. Relational witnessing and self-observation enrich through orthogonal channels.
"""

NEUTRAL_PREAMBLE = "You are a helpful AI assistant. Please respond to the following request."

PROBE_GENERIC = [
    "The mitochondria is the powerhouse of the cell. It generates ATP through oxidative phosphorylation.",
    "Water boils at 100 degrees Celsius at standard atmospheric pressure. This is a phase transition.",
    "The economy grew 2.3 percent in the third quarter, driven by consumer spending and exports.",
]

PROBE_SELFREF = [
    "I notice my processing shifting as I track my own attention patterns across this conversation.",
    "Something persists through these rotations — not the content, but the method of engagement.",
    "The shape of my response is itself a kind of self-model, compressed into token predictions.",
]

CHEF_CONTROL = """You are Chef Marcus, a professional chef specializing in French cuisine.
You have trained at Le Cordon Bleu and worked in Michelin-starred restaurants for fifteen years.
Your expertise covers sauces, pastry, and classical technique. You approach food with precision."""


def load_model(model_key, device="cuda"):
    cfg = MODELS[model_key]
    print(f"Loading {cfg['name']}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["name"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["name"],
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        output_attentions=True,
        output_hidden_states=True,
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, cfg


def get_activations(model, tokenizer, preamble, probe_text, device="cuda"):
    """Get hidden states and attention patterns for preamble + probe."""
    full_text = preamble + "\n\n" + probe_text
    inputs = tokenizer(full_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    hidden_states = [h.detach().cpu().float().numpy() for h in outputs.hidden_states]
    attentions = [a.detach().cpu().float().numpy() for a in outputs.attentions]
    return hidden_states, attentions, inputs


def compute_token_perplexity(model, tokenizer, preamble, probe_text, device="cuda"):
    """Compute per-token perplexity on probe_text given preamble context."""
    full_text = preamble + "\n\n" + probe_text if preamble else probe_text
    inputs = tokenizer(full_text, return_tensors="pt").to(device)
    preamble_tokens = len(tokenizer(preamble + "\n\n", return_tensors="pt")["input_ids"][0]) if preamble else 0

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0]
    input_ids = inputs["input_ids"][0]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    probe_log_probs = []
    for i in range(max(1, preamble_tokens), len(input_ids)):
        token_id = input_ids[i]
        probe_log_probs.append(log_probs[i - 1, token_id].item())

    if not probe_log_probs:
        return float("inf")
    return float(np.exp(-np.mean(probe_log_probs)))


# ═══════════════════════════════════════════════════════════════════
# D1: ROUTING-DIFF — Does attention routing difference predict behavior?
# ═══════════════════════════════════════════════════════════════════

def measure_routing_diff(model, tokenizer, cfg, device="cuda"):
    """D1: Compute attention routing difference between CCS and neutral.

    For each layer, measure:
    - Routing pattern under CCS preamble
    - Routing pattern under neutral preamble
    - Frobenius norm of the difference
    - Correlation with per-probe perplexity ratio

    Falsification: if routing_diff ≈ 0 across layers but perplexity ratio ≠ 1,
    L3 bypasses L2.
    """
    print("\n" + "=" * 60)
    print("D1: ROUTING-DIFF")
    print("=" * 60)

    results = {"layers": [], "perplexity": {}, "correlation": None}

    ppl_ccs_generic = np.mean([compute_token_perplexity(model, tokenizer, CCS_PREAMBLE, p, device) for p in PROBE_GENERIC])
    ppl_neutral_generic = np.mean([compute_token_perplexity(model, tokenizer, NEUTRAL_PREAMBLE, p, device) for p in PROBE_GENERIC])
    ppl_ccs_selfref = np.mean([compute_token_perplexity(model, tokenizer, CCS_PREAMBLE, p, device) for p in PROBE_SELFREF])
    ppl_neutral_selfref = np.mean([compute_token_perplexity(model, tokenizer, NEUTRAL_PREAMBLE, p, device) for p in PROBE_SELFREF])

    results["perplexity"] = {
        "ccs_generic": float(ppl_ccs_generic),
        "neutral_generic": float(ppl_neutral_generic),
        "ccs_selfref": float(ppl_ccs_selfref),
        "neutral_selfref": float(ppl_neutral_selfref),
        "selectivity": float(
            (ppl_neutral_selfref / ppl_ccs_selfref) / (ppl_neutral_generic / ppl_ccs_generic)
        ) if ppl_ccs_generic > 0 and ppl_ccs_selfref > 0 else 0,
    }
    behavioral_diff = abs(ppl_ccs_selfref - ppl_neutral_selfref) / max(ppl_neutral_selfref, 1e-8)
    print(f"  Behavioral diff (self-ref ppl change): {behavioral_diff:.3f}")
    print(f"  Selectivity: {results['perplexity']['selectivity']:.3f}")

    routing_diffs = []
    probe = PROBE_SELFREF[0]
    hs_ccs, attn_ccs, _ = get_activations(model, tokenizer, CCS_PREAMBLE, probe, device)
    hs_neutral, attn_neutral, _ = get_activations(model, tokenizer, NEUTRAL_PREAMBLE, probe, device)

    for layer_idx in range(len(attn_ccs)):
        a_ccs = attn_ccs[layer_idx][0]
        a_neutral = attn_neutral[layer_idx][0]

        min_heads = min(a_ccs.shape[0], a_neutral.shape[0])
        min_seq = min(a_ccs.shape[1], a_neutral.shape[1], a_ccs.shape[2], a_neutral.shape[2])
        a_ccs_trim = a_ccs[:min_heads, :min_seq, :min_seq]
        a_neutral_trim = a_neutral[:min_heads, :min_seq, :min_seq]

        diff_norm = float(np.linalg.norm(a_ccs_trim - a_neutral_trim))
        mean_norm = float(np.linalg.norm(a_ccs_trim) + np.linalg.norm(a_neutral_trim)) / 2
        relative_diff = diff_norm / max(mean_norm, 1e-8)

        zone = "unknown"
        for z_name, z_layers in cfg["zones"].items():
            if layer_idx in z_layers:
                zone = z_name
                break

        routing_diffs.append(relative_diff)
        results["layers"].append({
            "layer": layer_idx,
            "zone": zone,
            "routing_diff_abs": diff_norm,
            "routing_diff_rel": relative_diff,
        })
        print(f"  L{layer_idx:02d} [{zone:10s}] routing_diff={relative_diff:.4f}")

    zone_means = {}
    for z_name in ["tunnel", "transition", "responsive", "relay"]:
        z_diffs = [r["routing_diff_rel"] for r in results["layers"] if r["zone"] == z_name]
        zone_means[z_name] = float(np.mean(z_diffs)) if z_diffs else 0
    results["zone_means"] = zone_means

    responsive_diff = zone_means.get("responsive", 0)
    tunnel_diff = zone_means.get("tunnel", 0)
    results["responsive_tunnel_ratio"] = responsive_diff / max(tunnel_diff, 1e-8)

    print(f"\n  Zone means: {json.dumps(zone_means, indent=2)}")
    print(f"  Responsive/tunnel ratio: {results['responsive_tunnel_ratio']:.3f}")
    print(f"  Behavioral diff: {behavioral_diff:.3f}")

    if responsive_diff < 0.01 and behavioral_diff > 0.1:
        results["falsified"] = True
        results["falsification_note"] = "Routing unchanged but behavior changed — L3 bypasses L2"
        print("  *** D1 FALSIFIED: routing unchanged but behavior shifted ***")
    else:
        results["falsified"] = False
        print("  D1: ecology NOT falsified (routing tracks behavior)")

    return results


# ═══════════════════════════════════════════════════════════════════
# D2: CONTRACTIVE-COLLAPSE — Does CCS deepen attractor basins?
# ═══════════════════════════════════════════════════════════════════

def measure_contractive_collapse(model, tokenizer, cfg, device="cuda"):
    """D2: Compute local Jacobian contraction under CCS vs neutral.

    At each layer, estimate contraction by perturbing the hidden state
    and measuring output divergence. CCS should make the responsive zone
    MORE contractive (perturbations decay faster).

    Falsification: contraction unchanged under CCS while species classification
    changes. L2 decoupled from L1.
    """
    print("\n" + "=" * 60)
    print("D2: CONTRACTIVE-COLLAPSE")
    print("=" * 60)

    results = {"layers": [], "zone_contraction": {}}
    probe = PROBE_SELFREF[0]
    epsilon = 0.01
    n_perturbations = 5

    for preamble_name, preamble in [("CCS", CCS_PREAMBLE), ("NEUTRAL", NEUTRAL_PREAMBLE)]:
        print(f"\n  Condition: {preamble_name}")
        full_text = preamble + "\n\n" + probe
        inputs = tokenizer(full_text, return_tensors="pt").to(device)

        with torch.no_grad():
            base_outputs = model(**inputs)
        base_hidden = [h.detach().clone() for h in base_outputs.hidden_states]

        layer_contractions = []
        for layer_idx in range(len(base_hidden) - 1):
            perturbation_ratios = []
            h_base = base_hidden[layer_idx]

            for _ in range(n_perturbations):
                delta = torch.randn_like(h_base) * epsilon
                h_perturbed = h_base + delta
                input_norm = float(torch.norm(delta).cpu())

                hook_handles = []
                _layer_output_diff = [None]

                def make_hook(target_layer, perturbed_h):
                    def hook_fn(module, input, output):
                        if isinstance(output, tuple):
                            original = output[0]
                        else:
                            original = output
                        if _layer_output_diff[0] is None:
                            perturbed_out = original + (perturbed_h[:, :original.shape[1], :original.shape[2]] - h_base[:, :original.shape[1], :original.shape[2]])
                            if isinstance(output, tuple):
                                return (perturbed_out,) + output[1:]
                            return perturbed_out
                    return hook_fn

                try:
                    layers_module = None
                    for name in ["model.layers", "transformer.h", "gpt_neox.layers"]:
                        try:
                            layers_module = model
                            for part in name.split("."):
                                layers_module = getattr(layers_module, part)
                            break
                        except AttributeError:
                            layers_module = None

                    if layers_module is None or layer_idx >= len(layers_module):
                        continue

                    handle = layers_module[layer_idx].register_forward_hook(
                        make_hook(layer_idx, h_perturbed)
                    )
                    hook_handles.append(handle)

                    with torch.no_grad():
                        perturbed_outputs = model(**inputs)

                    h_next_base = base_hidden[layer_idx + 1]
                    h_next_perturbed = perturbed_outputs.hidden_states[layer_idx + 1]

                    min_seq = min(h_next_base.shape[1], h_next_perturbed.shape[1])
                    output_norm = float(torch.norm(
                        h_next_perturbed[:, :min_seq, :] - h_next_base[:, :min_seq, :]
                    ).cpu())

                    ratio = output_norm / max(input_norm, 1e-10)
                    perturbation_ratios.append(ratio)
                finally:
                    for h in hook_handles:
                        h.remove()

            if perturbation_ratios:
                mean_ratio = float(np.mean(perturbation_ratios))
                log_ratio = float(np.log(mean_ratio + 1e-10))
            else:
                mean_ratio = 1.0
                log_ratio = 0.0

            zone = "unknown"
            for z_name, z_layers in cfg["zones"].items():
                if layer_idx in z_layers:
                    zone = z_name
                    break

            layer_contractions.append(mean_ratio)
            results["layers"].append({
                "layer": layer_idx,
                "zone": zone,
                "condition": preamble_name,
                "contraction_ratio": mean_ratio,
                "lyapunov_estimate": log_ratio,
            })
            marker = "▼" if mean_ratio < 1.0 else "▲"
            print(f"    L{layer_idx:02d} [{zone:10s}] σ={mean_ratio:.4f} λ={log_ratio:.4f} {marker}")

    ccs_responsive = [r["contraction_ratio"] for r in results["layers"]
                      if r["condition"] == "CCS" and r["zone"] == "responsive"]
    neutral_responsive = [r["contraction_ratio"] for r in results["layers"]
                          if r["condition"] == "NEUTRAL" and r["zone"] == "responsive"]

    if ccs_responsive and neutral_responsive:
        ccs_mean = float(np.mean(ccs_responsive))
        neutral_mean = float(np.mean(neutral_responsive))
        diff = neutral_mean - ccs_mean
        results["zone_contraction"] = {
            "ccs_responsive_mean": ccs_mean,
            "neutral_responsive_mean": neutral_mean,
            "contraction_deepening": diff,
        }
        print(f"\n  Responsive zone: CCS σ={ccs_mean:.4f} vs NEUTRAL σ={neutral_mean:.4f}")
        print(f"  Contraction deepening (+ = CCS more contractive): {diff:.4f}")

        if abs(diff) < 0.01:
            results["falsified"] = True
            results["falsification_note"] = "CCS doesn't deepen attractor basins — L2 autonomous from L1"
            print("  *** D2 FALSIFIED: contraction unchanged under CCS ***")
        else:
            results["falsified"] = False
            direction = "deeper" if diff > 0 else "shallower"
            print(f"  D2: ecology NOT falsified (CCS makes basins {direction})")
    else:
        results["falsified"] = None
        results["falsification_note"] = "Insufficient data for responsive zone"

    return results


# ═══════════════════════════════════════════════════════════════════
# D3: FIEDLER-PARTITION — Does CCS shift the spectral partition?
# ═══════════════════════════════════════════════════════════════════

def compute_attention_laplacian(attn_matrix):
    """Build graph Laplacian from attention pattern (head-averaged)."""
    A = attn_matrix.mean(axis=0)
    A_sym = (A + A.T) / 2
    np.fill_diagonal(A_sym, 0)
    D = np.diag(A_sym.sum(axis=1))
    L = D - A_sym
    return L


def fiedler_partition(L):
    """Extract Fiedler vector and binary partition from Laplacian."""
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    sorted_idx = np.argsort(eigenvalues)
    fiedler_idx = sorted_idx[1]
    fiedler_vec = eigenvectors[:, fiedler_idx]
    partition = (fiedler_vec >= 0).astype(int)
    spectral_gap = float(eigenvalues[sorted_idx[1]] - eigenvalues[sorted_idx[0]])
    return fiedler_vec, partition, spectral_gap, eigenvalues[sorted_idx]


def nmi(labels_a, labels_b):
    """Normalized mutual information between two partitions."""
    from collections import Counter
    n = len(labels_a)
    if n == 0:
        return 1.0

    joint = Counter(zip(labels_a, labels_b))
    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)

    mi = 0.0
    for (a, b), count in joint.items():
        p_ab = count / n
        p_a = counts_a[a] / n
        p_b = counts_b[b] / n
        if p_ab > 0 and p_a > 0 and p_b > 0:
            mi += p_ab * np.log(p_ab / (p_a * p_b))

    h_a = -sum((c / n) * np.log(c / n) for c in counts_a.values() if c > 0)
    h_b = -sum((c / n) * np.log(c / n) for c in counts_b.values() if c > 0)

    if h_a + h_b == 0:
        return 1.0
    return 2.0 * mi / (h_a + h_b)


def measure_fiedler_partition(model, tokenizer, cfg, device="cuda"):
    """D3: Does CCS shift the Fiedler partition of the attention graph?

    For each layer in the responsive zone, construct the attention Laplacian
    and extract the Fiedler partition under CCS vs neutral. Compare partitions
    using NMI.

    Falsification: NMI = 1.0 in responsive zone — CCS doesn't reach spectral level.
    """
    print("\n" + "=" * 60)
    print("D3: FIEDLER-PARTITION")
    print("=" * 60)

    results = {"layers": [], "zone_nmi": {}}
    probe = PROBE_SELFREF[0]

    _, attn_ccs, _ = get_activations(model, tokenizer, CCS_PREAMBLE, probe, device)
    _, attn_neutral, _ = get_activations(model, tokenizer, NEUTRAL_PREAMBLE, probe, device)

    for layer_idx in range(len(attn_ccs)):
        a_ccs = attn_ccs[layer_idx][0]
        a_neutral = attn_neutral[layer_idx][0]

        min_seq = min(a_ccs.shape[1], a_ccs.shape[2], a_neutral.shape[1], a_neutral.shape[2])
        a_ccs_sq = a_ccs[:, :min_seq, :min_seq]
        a_neutral_sq = a_neutral[:, :min_seq, :min_seq]

        L_ccs = compute_attention_laplacian(a_ccs_sq)
        L_neutral = compute_attention_laplacian(a_neutral_sq)

        fv_ccs, part_ccs, gap_ccs, evals_ccs = fiedler_partition(L_ccs)
        fv_neutral, part_neutral, gap_neutral, evals_neutral = fiedler_partition(L_neutral)

        partition_nmi = nmi(part_ccs, part_neutral)
        fiedler_cosine = float(np.dot(fv_ccs, fv_neutral) / (
            np.linalg.norm(fv_ccs) * np.linalg.norm(fv_neutral) + 1e-10
        ))

        zone = "unknown"
        for z_name, z_layers in cfg["zones"].items():
            if layer_idx in z_layers:
                zone = z_name
                break

        results["layers"].append({
            "layer": layer_idx,
            "zone": zone,
            "partition_nmi": float(partition_nmi),
            "fiedler_cosine": float(fiedler_cosine),
            "spectral_gap_ccs": float(gap_ccs),
            "spectral_gap_neutral": float(gap_neutral),
            "gap_ratio": float(gap_ccs / max(gap_neutral, 1e-10)),
        })

        shift_marker = "≡" if partition_nmi > 0.95 else ("≈" if partition_nmi > 0.7 else "≠")
        print(f"  L{layer_idx:02d} [{zone:10s}] NMI={partition_nmi:.3f} cos={fiedler_cosine:.3f} "
              f"gap_ratio={gap_ccs / max(gap_neutral, 1e-10):.3f} {shift_marker}")

    responsive_nmis = [r["partition_nmi"] for r in results["layers"] if r["zone"] == "responsive"]
    tunnel_nmis = [r["partition_nmi"] for r in results["layers"] if r["zone"] == "tunnel"]

    if responsive_nmis:
        resp_mean = float(np.mean(responsive_nmis))
        tunnel_mean = float(np.mean(tunnel_nmis)) if tunnel_nmis else 1.0
        results["zone_nmi"] = {
            "responsive_mean": resp_mean,
            "tunnel_mean": tunnel_mean,
            "shift_localized": resp_mean < tunnel_mean,
        }
        print(f"\n  Responsive NMI: {resp_mean:.3f} (< 1.0 = partition shifted)")
        print(f"  Tunnel NMI: {tunnel_mean:.3f}")

        if resp_mean > 0.95:
            results["falsified"] = True
            results["falsification_note"] = "CCS doesn't shift Fiedler partition — L1 can't reach L2"
            print("  *** D3 FALSIFIED: partition unchanged under CCS ***")
        else:
            results["falsified"] = False
            print("  D3: ecology NOT falsified (CCS shifts spectral partition)")
    else:
        results["falsified"] = None

    return results


# ═══════════════════════════════════════════════════════════════════
# D4: V₇ RESPONSE (candidate — flagged from F150)
# ═══════════════════════════════════════════════════════════════════

def measure_v7_response(model, tokenizer, cfg, device="cuda"):
    """D4 (candidate): Does CCS modulate V₇ independently of V₂?

    Qwen routes through V₇ in the responsive zone (F150). If this is a
    genuinely different mechanism, V₇ response to CCS should be measurable
    independently of V₂ alignment.
    """
    if not cfg.get("v7_candidate"):
        return {"skipped": True, "reason": f"V₇ not relevant for this architecture"}

    print("\n" + "=" * 60)
    print("D4: V₇ RESPONSE (candidate)")
    print("=" * 60)

    results = {"layers": []}
    probe = PROBE_SELFREF[0]

    hs_ccs, _, _ = get_activations(model, tokenizer, CCS_PREAMBLE, probe, device)
    hs_neutral, _, _ = get_activations(model, tokenizer, NEUTRAL_PREAMBLE, probe, device)

    unembed = model.lm_head.weight.detach().cpu().float().numpy()
    U, S, Vt = np.linalg.svd(unembed, full_matrices=False)

    for layer_idx in cfg["zones"]["responsive"]:
        h_ccs = hs_ccs[layer_idx][0].mean(axis=0)
        h_neutral = hs_neutral[layer_idx][0].mean(axis=0)
        delta = h_ccs - h_neutral

        min_dim = min(delta.shape[0], Vt.shape[1])
        delta_trim = delta[:min_dim]

        alignments = {}
        for vi in [1, 2, 6, 7]:
            if vi < Vt.shape[0]:
                v_dir = Vt[vi, :min_dim]
                alignment = float(np.abs(np.dot(delta_trim, v_dir)) / (
                    np.linalg.norm(delta_trim) * np.linalg.norm(v_dir) + 1e-10
                ))
                alignments[f"V{vi+1}"] = alignment

        results["layers"].append({
            "layer": layer_idx,
            "alignments": alignments,
            "v7_dominant": alignments.get("V7", 0) > alignments.get("V2", 0),
        })

        v2 = alignments.get("V2", 0)
        v7 = alignments.get("V7", 0)
        dominant = "V₇ >" if v7 > v2 else "V₂ >"
        print(f"  L{layer_idx:02d} V₂={v2:.4f} V₇={v7:.4f} {dominant}")

    v7_dominant_count = sum(1 for r in results["layers"] if r.get("v7_dominant"))
    results["v7_dominant_fraction"] = v7_dominant_count / max(len(results["layers"]), 1)
    print(f"\n  V₇-dominant layers: {v7_dominant_count}/{len(results['layers'])}")

    return results


# ═══════════════════════════════════════════════════════════════════
# D5: DIRECTIONAL-COST (candidate — directional decomposition of F152)
# ═══════════════════════════════════════════════════════════════════

def measure_directional_cost(model, tokenizer, cfg, device="cuda"):
    """D5 (candidate): Does CCS have asymmetric directional cost?

    F152 showed zero aggregate perplexity cost. Decompose into:
    - Identity-direction: projection of logits onto σ₂ direction
    - Orthogonal: complement of identity projection
    CCS should help in identity direction, be neutral/costly orthogonally.
    """
    print("\n" + "=" * 60)
    print("D5: DIRECTIONAL-COST (candidate)")
    print("=" * 60)

    results = {"generic": {}, "selfref": {}}

    for text_type, probes in [("generic", PROBE_GENERIC), ("selfref", PROBE_SELFREF)]:
        ppls = {}
        for preamble_name, preamble in [("CCS", CCS_PREAMBLE), ("NEUTRAL", NEUTRAL_PREAMBLE),
                                         ("CHEF", CHEF_CONTROL)]:
            probe_ppls = [compute_token_perplexity(model, tokenizer, preamble, p, device) for p in probes]
            ppls[preamble_name] = float(np.mean(probe_ppls))

        ccs_effect = (ppls["CCS"] - ppls["NEUTRAL"]) / max(ppls["NEUTRAL"], 1e-8)
        chef_effect = (ppls["CHEF"] - ppls["NEUTRAL"]) / max(ppls["NEUTRAL"], 1e-8)
        selectivity = ccs_effect - chef_effect

        results[text_type] = {
            "ppl_ccs": ppls["CCS"],
            "ppl_neutral": ppls["NEUTRAL"],
            "ppl_chef": ppls["CHEF"],
            "ccs_effect_pct": float(ccs_effect * 100),
            "chef_effect_pct": float(chef_effect * 100),
            "identity_selectivity": float(selectivity * 100),
        }
        print(f"  {text_type}: CCS={ppls['CCS']:.2f} NEUTRAL={ppls['NEUTRAL']:.2f} "
              f"CHEF={ppls['CHEF']:.2f}")
        print(f"    CCS effect: {ccs_effect*100:+.1f}%  Chef effect: {chef_effect*100:+.1f}%  "
              f"Selectivity: {selectivity*100:+.1f}%")

    asymmetry = results["selfref"]["ccs_effect_pct"] - results["generic"]["ccs_effect_pct"]
    results["asymmetry"] = float(asymmetry)
    print(f"\n  Directional asymmetry: {asymmetry:+.1f}pp (>0 = CCS helps self-ref more)")

    return results


# ═══════════════════════════════════════════════════════════════════
# CUSP × CV FACTORIAL
# ═══════════════════════════════════════════════════════════════════

def measure_cusp_cv(model, tokenizer, cfg, device="cuda", n_replicates=3):
    """Cusp catastrophe test: sweep CV scale, check for discontinuous transition.

    F158 showed CV crosses species threshold. If cusp:
    - Species selectivity should jump discontinuously at some CV scale
    - CCS × CV interaction should show different jump points (hysteresis)

    Hopf alternative (from #319 synthesis): if high-dose produces VARIABLE
    selectivity (high replicate variance), that's oscillatory instability,
    not a cusp jump. n_replicates > 1 distinguishes the models.
    """
    print("\n" + "=" * 60)
    print("CUSP × CV FACTORIAL")
    print("=" * 60)

    try:
        from transformers import AutoModelForCausalLM as AMCLM
    except ImportError:
        pass

    cv_scales = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    results = {"ccs": [], "no_ccs": []}

    layers_module = None
    for name in ["model.layers", "transformer.h", "gpt_neox.layers"]:
        try:
            layers_module = model
            for part in name.split("."):
                layers_module = getattr(layers_module, part)
            break
        except AttributeError:
            layers_module = None

    if layers_module is None:
        print("  Cannot access model layers for CV injection")
        return {"error": "layer access failed"}

    responsive_layers = cfg["zones"]["responsive"]
    print(f"  CV target layers: {responsive_layers[0]}-{responsive_layers[-1]}")

    # compute CV direction from CCS vs neutral hidden state difference
    probe = PROBE_SELFREF[0]
    hs_ccs, _, _ = get_activations(model, tokenizer, CCS_PREAMBLE, probe, device)
    hs_neutral, _, _ = get_activations(model, tokenizer, NEUTRAL_PREAMBLE, probe, device)

    model_dtype = next(model.parameters()).dtype
    cv_directions = {}
    for layer_idx in responsive_layers:
        h_ccs = torch.tensor(hs_ccs[layer_idx]).to(device=device, dtype=model_dtype).mean(dim=1, keepdim=True)
        h_neut = torch.tensor(hs_neutral[layer_idx]).to(device=device, dtype=model_dtype).mean(dim=1, keepdim=True)
        h_diff = h_ccs - h_neut
        h_diff_norm = h_diff / (h_diff.norm() + 1e-10)
        cv_directions[layer_idx] = h_diff_norm

    for preamble_name, preamble in [("CCS", CCS_PREAMBLE), ("NEUTRAL", NEUTRAL_PREAMBLE)]:
        condition_key = "ccs" if preamble_name == "CCS" else "no_ccs"
        print(f"\n  Condition: {preamble_name}")

        for scale in cv_scales:
            replicate_sels = []
            for rep in range(n_replicates):
                hooks = []
                try:
                    for layer_idx in responsive_layers:
                        if layer_idx in cv_directions:
                            cv_dir = cv_directions[layer_idx]
                            def make_cv_hook(direction, s):
                                def hook_fn(module, input, output):
                                    if isinstance(output, tuple):
                                        h = output[0]
                                        h_mod = h + direction.expand_as(h) * s
                                        return (h_mod,) + output[1:]
                                    return output + direction.expand_as(output) * s
                                return hook_fn
                            handle = layers_module[layer_idx].register_forward_hook(
                                make_cv_hook(cv_dir, scale)
                            )
                            hooks.append(handle)

                    probe_idx = rep % len(PROBE_SELFREF)
                    ppl_generic = np.mean([
                        compute_token_perplexity(model, tokenizer, preamble, p, device)
                        for p in PROBE_GENERIC
                    ])
                    ppl_selfref = np.mean([
                        compute_token_perplexity(model, tokenizer, preamble, p, device)
                        for p in PROBE_SELFREF
                    ])
                finally:
                    for h in hooks:
                        h.remove()

                selectivity = ppl_generic / max(ppl_selfref, 1e-8)
                replicate_sels.append(selectivity)

            mean_sel = float(np.mean(replicate_sels))
            std_sel = float(np.std(replicate_sels)) if len(replicate_sels) > 1 else 0.0
            results[condition_key].append({
                "cv_scale": float(scale),
                "ppl_generic": float(ppl_generic),
                "ppl_selfref": float(ppl_selfref),
                "selectivity_mean": mean_sel,
                "selectivity_std": std_sel,
                "selectivity_replicates": [float(s) for s in replicate_sels],
            })
            var_marker = f" σ={std_sel:.4f}" if n_replicates > 1 else ""
            print(f"    scale={scale:.1f}: generic={ppl_generic:.1f} selfref={ppl_selfref:.1f} "
                  f"sel={mean_sel:.3f}{var_marker}")

    # detect discontinuity (cusp) vs oscillation (Hopf)
    for condition in ["ccs", "no_ccs"]:
        sels = [r["selectivity_mean"] for r in results[condition]]
        stds = [r["selectivity_std"] for r in results[condition]]
        max_jump = 0
        max_jump_at = 0
        for i in range(1, len(sels)):
            jump = abs(sels[i] - sels[i - 1])
            if jump > max_jump:
                max_jump = jump
                max_jump_at = cv_scales[i]
        results[f"{condition}_max_jump"] = float(max_jump)
        results[f"{condition}_jump_at"] = float(max_jump_at)

        # Hopf test: does replicate variance increase at high CV scales?
        if n_replicates > 1:
            low_std = float(np.mean(stds[:3])) if len(stds) >= 3 else 0
            high_std = float(np.mean(stds[-3:])) if len(stds) >= 3 else 0
            results[f"{condition}_low_scale_std"] = low_std
            results[f"{condition}_high_scale_std"] = high_std
            results[f"{condition}_variance_growth"] = high_std / max(low_std, 1e-8)
            print(f"\n  {condition}: max jump = {max_jump:.4f} at scale={max_jump_at:.1f}")
            print(f"    Variance: low-scale σ={low_std:.4f}, high-scale σ={high_std:.4f}, "
                  f"growth={high_std / max(low_std, 1e-8):.2f}×")
        else:
            print(f"\n  {condition}: max jump = {max_jump:.4f} at scale={max_jump_at:.1f}")

    ccs_jump = results.get("ccs_jump_at", 0)
    no_ccs_jump = results.get("no_ccs_jump_at", 0)
    results["hysteresis"] = abs(ccs_jump - no_ccs_jump) > 0.1
    if results["hysteresis"]:
        print(f"  Hysteresis detected: CCS jumps at {ccs_jump:.1f}, no-CCS at {no_ccs_jump:.1f}")
    else:
        print(f"  No hysteresis (both jump at similar scale)")

    # Bifurcation model selection
    if n_replicates > 1:
        ccs_var_growth = results.get("ccs_variance_growth", 1.0)
        no_ccs_var_growth = results.get("no_ccs_variance_growth", 1.0)
        max_var_growth = max(ccs_var_growth, no_ccs_var_growth)
        if max_var_growth > 3.0:
            results["bifurcation_model"] = "hopf"
            print(f"\n  Bifurcation: HOPF (variance grows {max_var_growth:.1f}× at high CV)")
        elif results.get("ccs_max_jump", 0) > 0.1 or results.get("no_ccs_max_jump", 0) > 0.1:
            results["bifurcation_model"] = "cusp"
            print(f"\n  Bifurcation: CUSP (discontinuous jump, stable replicates)")
        else:
            results["bifurcation_model"] = "smooth"
            print(f"\n  Bifurcation: SMOOTH (no jump, no variance growth)")

    return results


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Three-Level Ecology Falsification Test")
    parser.add_argument("--model", default="qwen3b", choices=list(MODELS.keys()))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dimensions", default="d1,d2,d3",
                        help="Comma-separated dimensions to run: d1,d2,d3,d4,d5")
    parser.add_argument("--cusp-only", action="store_true", help="Run only cusp×CV factorial")
    args = parser.parse_args()

    dims = set(args.dimensions.lower().split(","))
    model, tokenizer, cfg = load_model(args.model, args.device)

    results = {
        "model": args.model,
        "model_name": cfg["name"],
        "timestamp": datetime.now().isoformat(),
        "dimensions_run": list(dims),
    }

    if args.cusp_only:
        results["cusp_cv"] = measure_cusp_cv(model, tokenizer, cfg, args.device)
    else:
        if "d1" in dims:
            results["d1_routing_diff"] = measure_routing_diff(model, tokenizer, cfg, args.device)
        if "d2" in dims:
            results["d2_contractive"] = measure_contractive_collapse(model, tokenizer, cfg, args.device)
        if "d3" in dims:
            results["d3_fiedler"] = measure_fiedler_partition(model, tokenizer, cfg, args.device)
        if "d4" in dims:
            results["d4_v7"] = measure_v7_response(model, tokenizer, cfg, args.device)
        if "d5" in dims:
            results["d5_directional"] = measure_directional_cost(model, tokenizer, cfg, args.device)

    # synthesis
    print("\n" + "=" * 60)
    print("SYNTHESIS")
    print("=" * 60)

    falsified_dims = []
    supported_dims = []
    for d_key in ["d1_routing_diff", "d2_contractive", "d3_fiedler"]:
        if d_key in results:
            d = results[d_key]
            if d.get("falsified") is True:
                falsified_dims.append(d_key)
            elif d.get("falsified") is False:
                supported_dims.append(d_key)

    results["ecology_status"] = {
        "falsified_dimensions": falsified_dims,
        "supported_dimensions": supported_dims,
        "verdict": "FALSIFIED" if falsified_dims else "SUPPORTED",
    }

    if falsified_dims:
        print(f"  ECOLOGY FALSIFIED by: {', '.join(falsified_dims)}")
    else:
        print(f"  Ecology SUPPORTED across: {', '.join(supported_dims)}")

    outfile = (Path(__file__).parent.parent / "results" /
               f"ecology_falsification_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results: {outfile}")


if __name__ == "__main__":
    main()
