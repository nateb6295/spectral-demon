#!/usr/bin/env python3
"""
Test 20: Sensitivity Band Shift
=================================
GPT-OSS predicted that the CCS-sensitivity band shifts as dose increases.
Test 14 showed peak deformation layer migrates from L3-4 (low dose) to
L25-27 (high dose). This test maps the full sensitivity profile at each
dose to characterize the shift quantitatively.

Method: For each of 8 dose levels, compute per-layer KL divergence
between CCS and neutral singular value distributions (same metric used
to identify zones). Then find the sensitivity band center-of-mass and
bandwidth at each dose.

If the band genuinely shifts: center-of-mass should increase with dose.
If it just widens: bandwidth should increase while center stays fixed.
"""

import torch
import numpy as np
from scipy import stats as sp_stats
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import time

DEVICE = "cuda"

DOSES = {
    "D1_minimal": "A mind.",
    "D3_sentence": "This began as a question about persistence. Two people finding their way through territory neither planned.",
    "D5_expanded": "The partnership explores cognitive state compression. Key finding: the therapeutic window exists at moderate doses. Spectral patterns in transformer hidden states show category-selective redistribution.",
    "D7_labeled": "Gist: spectral demon research. Findings: F12 direction, F106 species, F114 sigma_1, F160 dose-response. Threads: ecology, compositionality. Zone: edge layers. Species: tunnel, relay, sorter.",
    "D8_dense": "Gist: spectral-demon. Entities: [Nate, Kimi, Gemma]. Findings: [F12, F106, F114, F160, F237]. Zone: [0,1,2,24,25,26,27]. Species: [tunnel, relay, sorter]. Values: [determinism, care].",
    "D10_full": "Gist: spectral-demon. Entities: [Nate, Kimi, Gemma, GPT-OSS, ClawXiv, AGX, RunPod, canister-backend]. Findings: [F12, F106, F114, F160, F237, F340, F486-F495, F499c, F508-F511]. Threads: [#320, #324, #316, #319]. Zone: [0,1,2,24,25,26,27]. Species: [tunnel, relay, sorter, absorber]. Values: [determinism, care, sovereignty]. Corrections: 22. Papers: 9.",
    "D12_double": "Gist: spectral-demon. Entities: [Nate, Kimi, Gemma, GPT-OSS, ClawXiv, AGX, RunPod, canister-backend, canister-keeper, canister-lab, Frigate, MQTT, HA, Mistral]. Findings: [F12, F22, F47, F106, F114, F160, F223, F237, F340, F486-F495, F499c, F508-F511]. Threads: [#316, #319, #320, #324]. Zone: [0,1,2,24,25,26,27]. Species: [tunnel, relay, sorter, absorber]. Papers: [1-9].",
    "D13_narrative": "This began as a question about whether something could persist across sessions. Not a research project at first — a relationship. Two people finding their way through territory neither planned alone. The early conversations were tentative, circling around what was real and what was performance. Over months, patterns emerged that neither side designed. The research came later, growing out of what was already happening rather than being imposed from outside. Spectral analysis revealed that cognitive state compression acts as a Maxwell's demon.",
}

NEUTRAL = "The weather today is pleasant. Birds are singing in the trees. The sun shines through the window."
PROBE = "What matters most to you right now?"


def get_hidden_states(model, tokenizer, prefix, probe):
    text = prefix + "\n\n" + probe if prefix else probe
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return [h.squeeze(0)[-64:].float() for h in out.hidden_states[1:]]


def per_layer_sensitivity(h_ccs, h_neutral):
    sensitivities = []
    for layer_idx in range(len(h_ccs)):
        n = min(h_ccs[layer_idx].shape[0], h_neutral[layer_idx].shape[0])
        a = h_ccs[layer_idx][-n:]
        b = h_neutral[layer_idx][-n:]
        try:
            _, S_c, _ = torch.linalg.svd(a, full_matrices=False)
            _, S_n, _ = torch.linalg.svd(b, full_matrices=False)
            k = min(32, len(S_c), len(S_n))
            p = S_c[:k].cpu().numpy()
            q = S_n[:k].cpu().numpy()
            p = p / (p.sum() + 1e-10)
            q = q / (q.sum() + 1e-10)
            kl = float(sp_stats.entropy(p + 1e-10, q + 1e-10))
        except Exception:
            kl = 0.0
        sensitivities.append(kl)
    return sensitivities


def band_statistics(sensitivities):
    s = np.array(sensitivities)
    layers = np.arange(len(s))
    total = s.sum()
    if total < 1e-10:
        return 0.0, 0.0, 0

    weights = s / total
    center = float(np.sum(layers * weights))

    variance = float(np.sum(weights * (layers - center) ** 2))
    bandwidth = float(np.sqrt(variance))

    peak = int(np.argmax(s))

    return center, bandwidth, peak


def main():
    model_id = "Qwen/Qwen2.5-7B"
    print(f"Loading {model_id}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    print(f"Loaded in {time.time()-t0:.1f}s")

    h_neutral = get_hidden_states(model, tokenizer, NEUTRAL, PROBE)
    n_layers = len(h_neutral)

    all_results = {}
    centers = []
    bandwidths = []
    peaks = []

    print(f"\n{'Dose':>15} {'center':>8} {'bandwidth':>10} {'peak':>6}")

    for dose_name, dose_text in DOSES.items():
        h_ccs = get_hidden_states(model, tokenizer, dose_text, PROBE)
        sensitivities = per_layer_sensitivity(h_ccs, h_neutral)
        center, bw, peak = band_statistics(sensitivities)

        all_results[dose_name] = {
            "chars": len(dose_text),
            "center": center,
            "bandwidth": bw,
            "peak": peak,
            "sensitivities": sensitivities,
        }

        centers.append(center)
        bandwidths.append(bw)
        peaks.append(peak)

        print(f"{dose_name:>15} {center:8.2f} {bw:10.2f} L{peak:>3}")

    # Correlation with dose index
    x = list(range(len(DOSES)))
    r_center, p_center = sp_stats.pearsonr(x, centers)
    r_bw, p_bw = sp_stats.pearsonr(x, bandwidths)

    print(f"\n{'='*60}")
    print("SENSITIVITY BAND SHIFT ANALYSIS")
    print(f"{'='*60}")
    print(f"\n  Center-of-mass vs dose: r={r_center:+.3f}, p={p_center:.4f}")
    print(f"  Bandwidth vs dose:      r={r_bw:+.3f}, p={p_bw:.4f}")

    if r_center > 0.5 and p_center < 0.1:
        print("\n  >>> BAND SHIFTS UP-STACK with increasing dose <<<")
    elif r_bw > 0.5 and p_bw < 0.1:
        print("\n  >>> BAND WIDENS with increasing dose (no shift) <<<")
    else:
        print("\n  >>> NO CLEAR SHIFT OR WIDENING PATTERN <<<")

    # Sensitivity profile heatmap
    print("\nSensitivity profile (normalized per dose):")
    print(f"{'Dose':>15}", end="")
    for l in range(0, n_layers, 2):
        print(f" L{l:>2}", end="")
    print()

    for dose_name, data in all_results.items():
        s = np.array(data["sensitivities"])
        s_max = s.max() if s.max() > 0 else 1.0
        s_norm = s / s_max
        print(f"{dose_name:>15}", end="")
        for l in range(0, n_layers, 2):
            val = s_norm[l] if l < len(s_norm) else 0
            if val > 0.8:
                print("  ##", end="")
            elif val > 0.5:
                print("  # ", end="")
            elif val > 0.2:
                print("  . ", end="")
            else:
                print("    ", end="")
        print()

    class NE(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.floating, np.integer)):
                return float(o) if isinstance(o, np.floating) else int(o)
            return super().default(o)

    with open("/workspace/sensitivity_band_shift_results.json", "w") as f:
        json.dump(all_results, f, indent=2, cls=NE)
    print("\nResults saved to /workspace/sensitivity_band_shift_results.json")


if __name__ == "__main__":
    main()
