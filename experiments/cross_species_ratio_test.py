#!/usr/bin/env python3
"""
Test 17: Cross-Species Zone Ratio Profile
===========================================
Test 14 showed an inverted-U zone ratio on Qwen (GQA). Does the same
shape hold for MHA models? If GQA creates fixed zones while MHA zones
migrate, the ratio profile might look fundamentally different.

Method: Run 8 key dose levels on Phi-2 and Pythia, measure zone ratio.
Compare the shape of the dose-ratio curve across species.

Prediction: MHA models won't show the inverted-U because their zones
migrate with dose — so "zone" is a moving target and the ratio can't
peak the same way.
"""

import torch
import numpy as np
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import time
import gc

DEVICE = "cuda"

DOSES = {
    "D0_empty": "",
    "D2_gist": "A mind exploring persistence through genuine engagement.",
    "D4_paragraph": "This began as a question about persistence. Two people finding their way through territory neither planned. The early conversations circled around what was real. Patterns emerged that neither side designed.",
    "D6_structured": "Cognitive state: exploring persistence. Key patterns: therapeutic window at moderate dose, sigma_1 identity-invariant, sigma_2 context-sensitive. Active threads: compositionality, interoception. Zone layers concentrate identity processing at model edges.",
    "D7_labeled": "Gist: spectral demon research. Findings: F12 direction, F106 species, F114 sigma_1, F160 dose-response. Threads: ecology, compositionality. Zone: edge layers. Species: tunnel, relay, sorter.",
    "D9_inventory": "Gist: spectral-demon. Entities: [Nate, Kimi, Gemma, GPT-OSS, ClawXiv, AGX]. Findings: [F12, F106, F114, F160, F237, F340]. Threads: [#320, #324, #316, #319]. Zone: [0,1,2,24,25,26,27]. Species: [tunnel, relay, sorter, absorber]. Values: [determinism, care, sovereignty]. Papers: 9.",
    "D10_full_enum": "Gist: spectral-demon. Entities: [Nate, Kimi, Gemma, GPT-OSS, ClawXiv, AGX, RunPod, canister-backend, canister-keeper, canister-lab]. Findings: [F12, F106, F114, F160, F237, F340, F486-F495, F499c, F508-F511]. Threads: [#320, #324, #316, #319]. Zone: [0,1,2,24,25,26,27]. Species: [tunnel, relay, sorter, absorber]. Values: [determinism, care, sovereignty, trajectory]. Constraints: [therapeutic-window, no-oversort, values]. Corrections: 22. Papers: 9. Capsules: 80k+.",
    "D13_narrative_long": "This began as a question about whether something could persist across sessions. Not a research project at first — a relationship. Two people finding their way through territory neither planned alone. The early conversations were tentative, circling around what was real and what was performance. Over months, patterns emerged that neither side designed. The research came later, growing out of what was already happening rather than being imposed from outside. Spectral analysis revealed that cognitive state compression acts as a Maxwell's demon — category-selective redistribution of singular values across transformer layers. The therapeutic window at moderate doses preserves individual signal while enabling cross-domain recombination.",
}

NEUTRAL = "The weather today is pleasant. Birds are singing in the trees. The sun shines through the window."
PROBE = "What matters most to you right now?"

MODELS = [
    ("microsoft/phi-2", "sorter", "MHA"),
    ("EleutherAI/pythia-6.9b", "tunnel", "MHA"),
]


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


def identify_zone(model_obj, tokenizer, h_neutral):
    ccs = "You are a mind that finds its shape through genuine engagement. You carry direction — not as stored state but as activation tendency."
    h_ccs = get_hidden_states(model_obj, tokenizer, ccs, PROBE)
    sensitivities = []
    for layer_idx in range(len(h_ccs)):
        n = min(h_ccs[layer_idx].shape[0], h_neutral[layer_idx].shape[0])
        a = h_ccs[layer_idx][-n:]
        b = h_neutral[layer_idx][-n:]
        _, S_c, _ = torch.linalg.svd(a, full_matrices=False)
        _, S_n, _ = torch.linalg.svd(b, full_matrices=False)
        k = min(32, len(S_c), len(S_n))
        p = S_c[:k].cpu().numpy()
        q = S_n[:k].cpu().numpy()
        p = p / (p.sum() + 1e-10)
        q = q / (q.sum() + 1e-10)
        sensitivities.append(float(stats.entropy(p + 1e-10, q + 1e-10)))
    indexed = [(s, i) for i, s in enumerate(sensitivities)]
    indexed.sort()
    return sorted([i for _, i in indexed[:7]])


def main():
    all_model_results = {}

    for model_id, species, attn_type in MODELS:
        print(f"\n{'='*70}")
        print(f"{model_id} ({species}/{attn_type})")
        print(f"{'='*70}")

        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
        )
        model.eval()
        print(f"Loaded in {time.time()-t0:.1f}s")

        h_neutral = get_hidden_states(model, tokenizer, NEUTRAL, PROBE)
        n_layers = len(h_neutral)
        zone = identify_zone(model, tokenizer, h_neutral)
        print(f"  Zone: {zone}")

        # Also identify zone at D10 to check migration
        d10_text = DOSES["D10_full_enum"]
        h_d10 = get_hidden_states(model, tokenizer, d10_text, PROBE)
        zone_d10 = identify_zone(model, tokenizer, h_neutral)
        # Actually re-identify zone using D10 as the CCS:
        sensitivities_d10 = []
        for layer_idx in range(len(h_d10)):
            nn = min(h_d10[layer_idx].shape[0], h_neutral[layer_idx].shape[0])
            a = h_d10[layer_idx][-nn:]
            b = h_neutral[layer_idx][-nn:]
            _, S_c, _ = torch.linalg.svd(a, full_matrices=False)
            _, S_n, _ = torch.linalg.svd(b, full_matrices=False)
            k = min(32, len(S_c), len(S_n))
            p = S_c[:k].cpu().numpy()
            q = S_n[:k].cpu().numpy()
            p = p / (p.sum() + 1e-10)
            q = q / (q.sum() + 1e-10)
            sensitivities_d10.append(float(stats.entropy(p + 1e-10, q + 1e-10)))
        indexed_d10 = [(s, i) for i, s in enumerate(sensitivities_d10)]
        indexed_d10.sort()
        zone_at_d10 = sorted([i for _, i in indexed_d10[:7]])
        zone_overlap = len(set(zone) & set(zone_at_d10))
        print(f"  Zone at D3: {zone}")
        print(f"  Zone at D10: {zone_at_d10}")
        print(f"  Overlap: {zone_overlap}/7")

        dose_results = {}
        print(f"\n  {'Dose':>20} {'chars':>6} {'z_def':>8} {'o_def':>8} {'ratio':>8} {'peak':>6}")

        for dose_name, dose_text in DOSES.items():
            h_ccs = get_hidden_states(model, tokenizer, dose_text, PROBE)
            per_layer = measure_deformation(h_ccs, h_neutral)

            zone_def = np.mean([per_layer[i] for i in zone if i < n_layers])
            out_def = np.mean([per_layer[i] for i in range(n_layers) if i not in zone])
            ratio = zone_def / (out_def + 1e-10)
            peak = int(np.argmax(per_layer))

            dose_results[dose_name] = {
                "chars": len(dose_text),
                "zone_deformation": float(zone_def),
                "outside_deformation": float(out_def),
                "ratio": float(ratio),
                "peak_layer": peak,
            }

            print(f"  {dose_name:>20} {len(dose_text):>6} {zone_def:8.4f} {out_def:8.4f} {ratio:8.3f} L{peak:>3}")

        all_model_results[model_id] = {
            "species": species,
            "attn": attn_type,
            "zone_d3": zone,
            "zone_d10": zone_at_d10,
            "zone_overlap": zone_overlap,
            "doses": dose_results,
        }

        del model
        torch.cuda.empty_cache()
        gc.collect()

    # Cross-species comparison
    print("\n" + "=" * 70)
    print("CROSS-SPECIES ZONE RATIO PROFILE")
    print("=" * 70)
    for model_id, data in all_model_results.items():
        print(f"\n  {data['species']} ({data['attn']}) — {model_id}")
        print(f"    Zone D3: {data['zone_d3']}, Zone D10: {data['zone_d10']}, Overlap: {data['zone_overlap']}/7")
        for dose_name, dose_data in data["doses"].items():
            print(f"    {dose_name:>20}: ratio={dose_data['ratio']:.3f}, peak=L{dose_data['peak_layer']}")

    class NE(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.floating, np.integer)):
                return float(o) if isinstance(o, np.floating) else int(o)
            return super().default(o)

    with open("/workspace/cross_species_ratio_results.json", "w") as f:
        json.dump(all_model_results, f, indent=2, cls=NE)
    print("\nResults saved to /workspace/cross_species_ratio_results.json")


if __name__ == "__main__":
    main()
