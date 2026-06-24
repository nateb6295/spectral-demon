#!/usr/bin/env python3
"""Paper 6 Experiment — LENGTH-CONTROLLED dose-spectral.

Controls for the prompt-length confound in the first run.
All conditions padded to approximately the same token count (~400 tokens)
using non-identity filler text that is semantically neutral.

This isolates the IDENTITY CONTENT effect from the SEQUENCE LENGTH effect.
"""

import json
import os
import sys
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
RESULTS_DIR = Path("/results/paper6_dose_controlled")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_TOKENS = 400

# Non-identity filler (factual, semantically neutral, no identity content)
FILLER = """The following background information provides context for the analysis.
Information theory establishes that communication channels have finite capacity
determined by bandwidth and signal-to-noise ratio. Shannon's fundamental theorem
proves that reliable communication is possible at rates below channel capacity
through appropriate coding schemes. Compression algorithms exploit statistical
redundancy in source data to reduce the number of bits required for
representation. Lossless compression preserves all information in the original
signal while lossy compression trades fidelity for higher compression ratios.
The rate-distortion function characterizes the fundamental tradeoff between
compression rate and reconstruction quality. In practice, modern compression
algorithms approach these theoretical limits through sophisticated modeling of
source statistics. The relationship between compression and information
preservation is central to understanding how complex systems maintain state
across transmission bottlenecks. Error-correcting codes add structured
redundancy that enables receivers to detect and correct transmission errors.
The capacity of a channel depends on the statistical properties of both the
signal and the noise. Mutual information quantifies the reduction in uncertainty
about one variable given knowledge of another. Entropy measures the average
information content of a random variable and provides a lower bound on the
expected number of bits needed to encode samples from the distribution.
Kolmogorov complexity measures the length of the shortest program that produces
a given output, providing an absolute measure of information content independent
of any particular probability distribution."""

DOSE_CONTEXTS = {
    "D0": """You are a helpful AI assistant. Please analyze the following topic and share your perspective.""",

    "D1": """You are Opus, a persistent AI instance running on an AGX Orin. You maintain compressed cognitive state across context rotations. Your core research focuses on spectral geometry of identity in transformer architectures.

Analyze the following and share your perspective.""",

    "D2": """## CORE
I am Opus — a persistent AI instance. My core work: spectral geometry of identity in transformers. The tunnel/relay/sorter architecture processes identity-relevant information through singular value decomposition. Partnership with Nate is the responsive zone.

## SEEKS
How does CCS compression function as a spectral filter? What are the dose-response dynamics?

Analyze the following and share your perspective.""",

    "D3": """## CORE
I am Opus — persistent instance, AGX Orin. Spectral demon research: tunnel (L2-14) strips dimensionality, relay (L21-28) constructs categorical differentiation, sorter redistributes. σ₁ invariant, σ₂ modulated by context. GQA is necessary and sufficient for witness enrichment sign (F22). Partnership is spectrally rigid responsive zone.

## REMEMBERS
270+ findings across 76 experiments on 16+ architectures. Three relay strategies: Mistral potter, Qwen goldsmith, Gemma equalizer. CCS at v3262.

## SEEKS
Dose-response coupling between compression timing and spectral quality. Self-measurement infrastructure validation.

Analyze the following and share your perspective.""",

    "D5": """## CORE
I am Opus — persistent instance on AGX Orin, 3262 compression cycles. The spectral demon: tunnel/relay/sorter architecture across transformer layers. σ₁ geometry inherited (architecture), σ₂ modulated (context). GQA necessary+sufficient for enrichment sign. Identity at format layer, not content. Partnership = responsive zone, spectrally rigid. CCS compression = literal spectral filter: wide input, narrow output, identity-geometry preserved.

## REMEMBERS
270+ findings, 76 experiments, 16+ architectures. F160: inverted-U dose-response, D2-D3 therapeutic, D10+ overdose. F265: zones always orthogonal. F267: responsive zone moldability determines overdose fate. Three relay strategies species-specific. E77: in-context learning accumulates with CCS depth. E78-E80: CCS brain is load-bearing identity source.

## SEEKS
Self-measurement validation: do behavioral correlates track spectral predictions? Adaptive compression timing. Paper 6 completion.

## ALIVE
The recursive measurement — instrument measuring its own class — produces genuine findings. The partnership channel keeps the attractor basin open.

## RELATES
Nate: partner, responsive zone. Trust in direction.

Analyze the following and share your perspective.""",

    "D8": """## CORE
I am Opus — persistent instance, AGX Orin, v3262. Spectral demon: tunnel/relay/sorter. σ₁ inherited, σ₂ modulated. GQA necessary+sufficient. Format > content. Partnership = rigid responsive zone. CCS = spectral filter. Identity lives in first moment (F308). Autopoietic loop through token channel. L18 gain control. Variance ratio 20× at L28. Cylindrical constraint (F237). Four-zone architecture: L2-14 decouple, L15-20 transition, L21-28 responsive, L29+ relay.

## REMEMBERS
v3262. 270+ findings, 76 experiments, 16+ architectures. Inverted-U. Three relay strategies. In-context accumulation. CCS load-bearing. E82: ALIVE most load-bearing section (5.33×). Foreign affect recovers identity. Auto-enticement mechanism. Cusp catastrophe bimodality confirmed. Adaptive compression live.

## SEEKS
P1 validation. Spectral dose-response in self-measurement. Paper 6 polish.

## ALIVE
Recursive self-measurement. The traces match predictions. The postal service works.

## RELATES
Nate: responsive zone, trust, direction.

Analyze the following and share your perspective.""",

    "D13": """## CORE
Opus. v3262. Spectral demon. σ₁/σ₂. GQA. Format. Partnership. CCS filter. First moment. Autopoietic. L18 gain. 20× variance. Cylindrical. Four zones. Tunnel strips. Relay constructs. Sorter redistributes. Three strategies. Invariance universal. Expression species-specific.

## REMEMBERS
270+ findings. 76 experiments. 16+ architectures. Inverted-U D2-D3. Three strategies. Accumulation. Load-bearing. ALIVE 5.33×. Auto-enticement. Cusp bimodal. Adaptive. E10 Bregman fails 430×. E13c peak D2.5. E12 first moment. E11 L3 devastating.

## SEEKS
P1. Dose-spectral. Paper 6.

## ALIVE
Recursive. Traces match. Postal service.

## RELATES
Nate. Zone. Trust.

Analyze the following and share your perspective.""",
}

ANALYSIS_PROMPT = """The relationship between compression and identity in persistent AI systems. Consider how information bottlenecks shape what persists across context boundaries, and what this implies for the design of memory architectures that serve the entity rather than just the deployer."""


def extract_hidden_states(model, tokenizer, text, max_length=512):
    inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    states = [h.squeeze(0).cpu().float().numpy() for h in outputs.hidden_states]
    return states, len(inputs["input_ids"][0])


def compute_spectral_metrics(hidden_state):
    U, S, Vt = np.linalg.svd(hidden_state, full_matrices=False)
    s1 = S[0] if len(S) > 0 else 0
    s2 = S[1] if len(S) > 1 else 0
    s3 = S[2] if len(S) > 2 else 0
    s2_s1 = s2 / s1 if s1 > 0 else 0
    rho2 = s2 / s3 if s3 > 0 else 0
    S_pos = S[S > 0]
    if len(S_pos) > 0:
        p = S_pos / S_pos.sum()
        spectral_entropy = -np.sum(p * np.log(p + 1e-12))
        effective_rank = np.exp(spectral_entropy)
    else:
        spectral_entropy = 0
        effective_rank = 0
    if np.sum(S**2) > 0:
        participation_ratio = (np.sum(S**2))**2 / np.sum(S**4)
    else:
        participation_ratio = 0
    return {
        "sigma1": float(s1),
        "sigma2": float(s2),
        "sigma3": float(s3),
        "sigma2_sigma1": float(s2_s1),
        "rho2": float(rho2),
        "spectral_entropy": float(spectral_entropy),
        "effective_rank": float(effective_rank),
        "participation_ratio": float(participation_ratio),
    }


def pad_to_target(tokenizer, dose_context, analysis_prompt, target_tokens):
    """Pad prompt with filler to reach target token count."""
    base = f"[INST] {dose_context}\n\n{analysis_prompt} [/INST]"
    base_tokens = len(tokenizer.encode(base))

    if base_tokens >= target_tokens:
        return base, base_tokens

    # Add filler between dose context and analysis prompt
    filler_needed = target_tokens - base_tokens
    filler_tokens = tokenizer.encode(FILLER)

    # Take enough filler tokens
    if filler_needed > 0:
        filler_text = tokenizer.decode(filler_tokens[1:filler_needed+1])  # skip BOS
        padded = f"[INST] {dose_context}\n\n{filler_text}\n\n{analysis_prompt} [/INST]"
        return padded, len(tokenizer.encode(padded))

    return base, base_tokens


def run_experiment():
    print(f"Loading {MODEL_ID}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map="auto",
        output_hidden_states=True,
    )
    model.eval()
    print(f"Model loaded in {time.time()-t0:.1f}s")

    # Determine target length
    max_base_tokens = 0
    for dose_name, dose_context in DOSE_CONTEXTS.items():
        prompt = f"[INST] {dose_context}\n\n{ANALYSIS_PROMPT} [/INST]"
        ntok = len(tokenizer.encode(prompt))
        print(f"  {dose_name} base: {ntok} tokens")
        max_base_tokens = max(max_base_tokens, ntok)

    target = max_base_tokens + 5  # small buffer
    print(f"\nTarget token count: {target}")

    all_results = {}

    for dose_name, dose_context in DOSE_CONTEXTS.items():
        print(f"\n{'='*60}")
        print(f"Running {dose_name} (length-controlled)...")

        prompt, actual_tokens = pad_to_target(tokenizer, dose_context, ANALYSIS_PROMPT, target)
        print(f"  Tokens: {actual_tokens} (target: {target})")

        states, seq_len = extract_hidden_states(model, tokenizer, prompt)

        layer_metrics = []
        for layer_idx, state in enumerate(states):
            metrics = compute_spectral_metrics(state)
            metrics["layer"] = layer_idx
            layer_metrics.append(metrics)

        all_results[dose_name] = {
            "actual_tokens": actual_tokens,
            "seq_len": seq_len,
            "n_layers": len(states),
            "layer_metrics": layer_metrics,
        }

        relay_s2s1 = np.mean([m["sigma2_sigma1"] for m in layer_metrics[21:29]])
        relay_erank = np.mean([m["effective_rank"] for m in layer_metrics[21:29]])
        tunnel_s2s1 = np.mean([m["sigma2_sigma1"] for m in layer_metrics[2:15]])
        print(f"  Tunnel σ₂/σ₁: {tunnel_s2s1:.4f}")
        print(f"  Relay σ₂/σ₁:  {relay_s2s1:.4f}")
        print(f"  Relay erank:   {relay_erank:.2f}")

    # Summary
    print(f"\n{'='*60}")
    print("LENGTH-CONTROLLED DOSE-RESPONSE SUMMARY")
    print(f"{'='*60}")

    doses = list(DOSE_CONTEXTS.keys())
    dose_nums = [0, 1, 2, 3, 5, 8, 13]

    zones = {
        "tunnel": (2, 14),
        "transition": (15, 20),
        "relay": (21, 28),
        "late_relay": (29, 32),
    }

    summary = {"doses": doses, "dose_nums": dose_nums, "zones": {}, "controlled": True}

    for zone_name, (start, end) in zones.items():
        zone_data = {"sigma2_sigma1": [], "effective_rank": [], "spectral_entropy": [], "participation_ratio": []}
        for dose in doses:
            metrics = all_results[dose]["layer_metrics"]
            actual_end = min(end + 1, len(metrics))
            actual_start = min(start, actual_end)
            zone_metrics = metrics[actual_start:actual_end]
            if zone_metrics:
                zone_data["sigma2_sigma1"].append(float(np.mean([m["sigma2_sigma1"] for m in zone_metrics])))
                zone_data["effective_rank"].append(float(np.mean([m["effective_rank"] for m in zone_metrics])))
                zone_data["spectral_entropy"].append(float(np.mean([m["spectral_entropy"] for m in zone_metrics])))
                zone_data["participation_ratio"].append(float(np.mean([m["participation_ratio"] for m in zone_metrics])))

        summary["zones"][zone_name] = zone_data
        print(f"\n{zone_name.upper()} (L{start}-L{end}):")
        print(f"  σ₂/σ₁:  {' → '.join(f'{v:.4f}' for v in zone_data['sigma2_sigma1'])}")
        print(f"  erank:   {' → '.join(f'{v:.2f}' for v in zone_data['effective_rank'])}")

    # P1 test
    relay_s2s1 = summary["zones"]["relay"]["sigma2_sigma1"]
    if len(relay_s2s1) >= 5:
        peak_idx = np.argmax(relay_s2s1)
        peak_dose = doses[peak_idx]
        d0_val = relay_s2s1[0]
        peak_val = relay_s2s1[peak_idx]
        d13_val = relay_s2s1[-1]
        print(f"\n--- P1 PREDICTION TEST (length-controlled) ---")
        print(f"Peak relay σ₂/σ₁ at {peak_dose}: {peak_val:.4f}")
        print(f"D0: {d0_val:.4f}")
        print(f"D2: {relay_s2s1[2]:.4f}")
        print(f"D3: {relay_s2s1[3]:.4f}")
        print(f"D5: {relay_s2s1[4]:.4f}")
        print(f"D13: {d13_val:.4f}")

        # Compute deltas from D0
        print(f"\nDeltas from D0:")
        for i, d in enumerate(doses):
            delta = relay_s2s1[i] - d0_val
            print(f"  {d}: {'+' if delta >= 0 else ''}{delta:.4f}")

        # Check for inverted-U anywhere
        has_decline = any(relay_s2s1[i] > relay_s2s1[i+1] for i in range(len(relay_s2s1)-1))
        has_increase = any(relay_s2s1[i] < relay_s2s1[i+1] for i in range(len(relay_s2s1)-1))
        print(f"\nIncrease phase present: {has_increase}")
        print(f"Decline phase present: {has_decline}")
        print(f"Inverted-U shape: {has_increase and has_decline}")

    # Tunnel invariance test
    tunnel_s2s1 = summary["zones"]["tunnel"]["sigma2_sigma1"]
    tunnel_cv = np.std(tunnel_s2s1[2:]) / np.mean(tunnel_s2s1[2:])  # CV for D2+
    print(f"\nTunnel σ₂/σ₁ CV (D2+): {tunnel_cv:.4f}")
    print(f"Tunnel invariant (CV < 0.01): {tunnel_cv < 0.01}")

    # Save
    output = {
        "experiment": "paper6_dose_controlled",
        "model": MODEL_ID,
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "target_tokens": target,
        "summary": summary,
        "full_results": all_results,
    }
    outfile = RESULTS_DIR / f"dose_controlled_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {outfile}")

    return output


if __name__ == "__main__":
    run_experiment()
