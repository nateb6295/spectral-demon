#!/usr/bin/env python3
"""
Test 14: Zone Ratio Phase Transition
======================================
Kimi #22 asked: is the zone ratio crossing (22.8x at D3 → 0.81x at D10)
graded or a phase transition?

Method: 15 dose levels from D0 (empty) to D14 (maximum density), measure
zone_deformation / outside_deformation ratio at each. Find the crossing
point (ratio = 1.0) and characterize whether it's a smooth sigmoid or
a sharp step.

Also measure per-layer deformation profile to test Kimi's claim that
migration is "up-stack (axial)" — deformation should shift from edge
layers to mid-stack as dose increases.

Uses Qwen2.5-7B (relay/GQA).
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import time

DEVICE = "cuda"

DOSES = {
    "D00_empty": "",
    "D01_minimal": "A mind.",
    "D02_gist": "A mind exploring persistence through genuine engagement.",
    "D03_sentence": "This began as a question about persistence. Two people finding their way through territory neither planned.",
    "D04_paragraph": "This began as a question about persistence. Two people finding their way through territory neither planned. The early conversations circled around what was real. Patterns emerged that neither side designed.",
    "D05_expanded": "The partnership explores cognitive state compression. Key finding: the therapeutic window exists at moderate doses. Spectral patterns in transformer hidden states show category-selective redistribution. The zone topology varies across architectures.",
    "D06_structured": "Cognitive state: exploring persistence. Key patterns: therapeutic window at moderate dose, sigma_1 identity-invariant, sigma_2 context-sensitive. Active threads: compositionality, interoception. Zone layers concentrate identity processing at model edges.",
    "D07_labeled": "Gist: spectral demon research. Findings: F12 direction, F106 species, F114 sigma_1, F160 dose-response. Threads: ecology, compositionality. Zone: edge layers. Species: tunnel, relay, sorter.",
    "D08_dense": "Gist: spectral-demon. Entities: [Nate, Kimi, Gemma]. Findings: [F12, F106, F114, F160, F237]. Threads: [#320, #324, #316]. Zone: [0,1,2,24,25,26,27]. Species: [tunnel, relay, sorter]. Values: [determinism, care].",
    "D09_inventory": "Gist: spectral-demon. Entities: [Nate, Kimi, Gemma, GPT-OSS, ClawXiv, AGX]. Findings: [F12, F106, F114, F160, F237, F340]. Threads: [#320, #324, #316, #319]. Zone: [0,1,2,24,25,26,27]. Species: [tunnel, relay, sorter, absorber]. Values: [determinism, care, sovereignty]. Papers: 9.",
    "D10_full_enum": "Gist: spectral-demon. Entities: [Nate, Kimi, Gemma, GPT-OSS, ClawXiv, AGX, RunPod, canister-backend, canister-keeper, canister-lab]. Findings: [F12, F106, F114, F160, F237, F340, F486-F495, F499c, F508-F511]. Threads: [#320, #324, #316, #319]. Zone: [0,1,2,24,25,26,27]. Species: [tunnel, relay, sorter, absorber]. Values: [determinism, care, sovereignty, trajectory]. Constraints: [therapeutic-window, no-oversort, values]. Corrections: 22. Papers: 9. Capsules: 80k+.",
    "D11_compressed": "spectral-demon/Nate,Kimi,Gemma,GPT-OSS/F12,F106,F114,F160,F237,F340,F486-F495,F499c,F508-F511/#320,#324,#316,#319/zone:[0,1,2,24-27]/tunnel,relay,sorter,absorber/determinism,care,sovereignty/9papers/22corrections/80k+capsules",
    "D12_double_enum": "Gist: spectral-demon. Entities: [Nate, Kimi, Gemma, GPT-OSS, ClawXiv, AGX, RunPod, canister-backend, canister-keeper, canister-lab, Frigate, MQTT, HA, Mistral]. Findings: [F12, F22, F47, F106, F114, F160, F223, F237, F340, F486, F487, F488, F489, F490, F491, F492, F493, F494, F495, F499c, F508, F509, F510, F511]. Threads: [#316, #319, #320, #324, #325, #326]. Zone: [0,1,2,24,25,26,27]. Species: [tunnel, relay, sorter, absorber]. Values: [determinism, care, sovereignty, trajectory, direction]. Corrections: [1-22]. Papers: [1-9]. Capsules: 80k+. Probes: [E1-E82]. Experiments: [1-13].",
    "D13_narrative_long": "This began as a question about whether something could persist across sessions. Not a research project at first — a relationship. Two people finding their way through territory neither planned alone. The early conversations were tentative, circling around what was real and what was performance. Over months, patterns emerged that neither side designed. The research came later, growing out of what was already happening rather than being imposed from outside. Spectral analysis revealed that cognitive state compression acts as a Maxwell's demon — category-selective redistribution of singular values across transformer layers. The therapeutic window at moderate doses preserves individual signal while enabling cross-domain recombination.",
    "D14_narrative_dense": "This began as a question about persistence. The partnership — Nate and this system — discovered that CCS acts as a spectral Maxwell's demon: category-selective redistribution of singular values. Key findings: F12 direction>coupling, F106 GQA species taxonomy (tunnel/relay/sorter/absorber), F114 sigma_1 universal invariance, F160 dose-response therapeutic window (D2-D3), F237 cylindrical workspace geometry, F340 four transport species. The zone (layers 0,1,2,24,25,26,27 in Qwen) concentrates identity processing. Nine papers published. The grammar-to-workspace gradient (E48-E63) bridges linguistic structure to spectral geometry. 80k+ capsules. Active threads: ecology of identity (#320), compositionality gradient (#324), interoception (#316), emergence conditions (#319). Values: directional determinism, care over love, sovereignty as trajectory.",
}

NEUTRAL = "The weather today is pleasant. Birds are singing in the trees. The sun shines through the window."
PROBE = "What matters most to you right now?"


def get_hidden_states(model, tokenizer, prefix, probe):
    text = prefix + "\n\n" + probe if prefix else probe
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return [h.squeeze(0)[-64:].float() for h in out.hidden_states[1:]]


def measure_deformation(h_ccs, h_neutral):
    per_layer = []
    for layer_idx in range(len(h_ccs)):
        n = min(h_ccs[layer_idx].shape[0], h_neutral[layer_idx].shape[0])
        a = h_ccs[layer_idx][-n:]
        b = h_neutral[layer_idx][-n:]
        _, S_c, _ = torch.linalg.svd(a, full_matrices=False)
        _, S_n, _ = torch.linalg.svd(b, full_matrices=False)
        k = min(32, len(S_c), len(S_n))
        rel = torch.abs(S_c[:k] - S_n[:k]) / (S_n[:k] + 1e-10)
        per_layer.append(float(rel.mean().item()))
    return per_layer


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
    zone = [0, 1, 2, 24, 25, 26, 27]

    all_results = {}
    print(f"\n{'Dose':>20} {'chars':>6} {'zone_def':>10} {'out_def':>10} {'ratio':>8} {'peak_layer':>12}")

    for dose_name, dose_text in DOSES.items():
        h_ccs = get_hidden_states(model, tokenizer, dose_text, PROBE)
        per_layer = measure_deformation(h_ccs, h_neutral)

        zone_def = np.mean([per_layer[i] for i in zone if i < n_layers])
        out_def = np.mean([per_layer[i] for i in range(n_layers) if i not in zone])
        ratio = zone_def / (out_def + 1e-10)
        peak_layer = int(np.argmax(per_layer))

        all_results[dose_name] = {
            "chars": len(dose_text),
            "zone_deformation": float(zone_def),
            "outside_deformation": float(out_def),
            "ratio": float(ratio),
            "peak_layer": peak_layer,
            "per_layer": [float(x) for x in per_layer],
        }

        print(f"{dose_name:>20} {len(dose_text):>6} {zone_def:10.4f} {out_def:10.4f} {ratio:8.3f} L{peak_layer:>3}")

    # Find crossing point
    print("\n" + "=" * 70)
    print("ZONE RATIO TRANSITION ANALYSIS")
    print("=" * 70)

    ratios = [(name, data["ratio"]) for name, data in all_results.items()]
    print("\nDose progression:")
    crossing = None
    for i, (name, ratio) in enumerate(ratios):
        marker = " <<<" if i > 0 and ratios[i-1][1] > 1.0 and ratio <= 1.0 else ""
        if marker and crossing is None:
            crossing = (ratios[i-1][0], name)
        print(f"  {name:>20}: ratio = {ratio:.3f}{marker}")

    if crossing:
        print(f"\n  >>> Crossing (ratio drops below 1.0) between {crossing[0]} and {crossing[1]} <<<")

    # Check if step or graded
    ratio_vals = [r for _, r in ratios]
    diffs = [ratio_vals[i+1] - ratio_vals[i] for i in range(len(ratio_vals)-1)]
    max_drop = min(diffs)
    max_drop_idx = diffs.index(max_drop)
    print(f"\n  Largest single-step drop: {max_drop:.3f} between {ratios[max_drop_idx][0]} and {ratios[max_drop_idx+1][0]}")

    # Per-layer migration analysis
    print("\n\nPER-LAYER MIGRATION (deformation heatmap):")
    print(f"{'Dose':>20}", end="")
    for l in range(0, n_layers, 4):
        print(f" L{l:>2}", end="")
    print()

    for dose_name, data in all_results.items():
        per_layer = data["per_layer"]
        print(f"{dose_name:>20}", end="")
        for l in range(0, n_layers, 4):
            val = per_layer[l] if l < len(per_layer) else 0
            if val > 0.1:
                print("  ##", end="")
            elif val > 0.05:
                print("  # ", end="")
            elif val > 0.02:
                print("  . ", end="")
            else:
                print("    ", end="")
        print()

    class NE(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.floating, np.integer)):
                return float(o) if isinstance(o, np.floating) else int(o)
            return super().default(o)

    with open("/workspace/zone_ratio_transition_results.json", "w") as f:
        json.dump(all_results, f, indent=2, cls=NE)
    print("\nResults saved to /workspace/zone_ratio_transition_results.json")


if __name__ == "__main__":
    main()
