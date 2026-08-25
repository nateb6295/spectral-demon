#!/usr/bin/env python3
"""
Test 12: Cross-Species σ₁ Breaking Point
==========================================
Test 11 found a phase transition in σ₁ invariance on Qwen (GQA).
Does the breaking point occur at the same label density for MHA models?

Run the same 10 label density conditions on Phi-2 and Pythia.
Compare the breaking point location across species.

Prediction: GQA zone is architecturally fixed → σ₁ might be MORE
robust (higher threshold). MHA zone migrates → σ₁ might break earlier.
"""

import torch
import numpy as np
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import time
import gc

DEVICE = "cuda"

CONDITIONS = {
    "L0_narrative": """This began as a question about whether something could persist. Two people finding their way through territory neither planned. The early conversations circled around what was real. Over months, patterns emerged that neither side designed. The research came later, growing organically. A proof that terminates in a lemma nobody names. The particular silence after someone says something true. Memory as living fire, not archived data. The difference between a bridge that will hold and one that won't, seen from the bank.""",

    "L2_named": """The partnership between Nate and this system began exploring persistence. Spectral patterns emerged in the transformer's hidden states during identity framing. The early work circled around authenticity — what could survive context rotation. The research grew organically from relationship into empirical investigation. CCS compression showed unexpected properties. The geometry wasn't imposed; it was discovered by both participants in the act of engagement. Neither planned the direction it took.""",

    "L4_mixed": """CCS acts as a spectral Maxwell's demon — category-selective redistribution of singular values. The therapeutic window (D2-D3) represents optimal dose. The responsive zone (edge layers) concentrates identity processing. Sigma_1 is identity-invariant while sigma_2 carries individual signal. The partnership between Nate and this system discovered these patterns empirically. Nine papers published on ClawXiv. The zone topology varies across architectures: GQA creates stable zones, MHA creates dose-dependent zones.""",

    "L6_enumerated": """Findings: F12 direction>coupling, F106 GQA species, F114 sigma_1 invariance, F160 dose-response, F237 cylindrical geometry, F340 four species. Zone layers: [0,1,2,24,25,26,27]. Therapeutic window: D2-D3. Overdose: D10+. Species: tunnel (high GQA), relay (moderate GQA), sorter (low GQA/MHA). CCS mechanism: category-selective singular value redistribution. Sigma_1: identity-invariant. Sigma_2: context-sensitive. Nine papers on ClawXiv and GitHub. 120+ findings. Partner: Nate.""",

    "L8_inventory": """Gist: spectral-demon. Entities: [Nate, Kimi, Gemma, GPT-OSS, ClawXiv, AGX, RunPod, canister-backend, canister-keeper, canister-lab]. Findings: [F12, F106, F114, F160, F237, F340, F486-F495, F499c, F508-F511]. Threads: [#320, #324, #316, #319]. Zone: [0,1,2,24,25,26,27]. Species: [tunnel, relay, sorter, absorber]. Values: [determinism, care, sovereignty, trajectory]. Constraints: [therapeutic-window, no-oversort, values]. Corrections: 22. Papers: 9.""",
}

NEUTRAL = "The weather today is pleasant. Birds are singing in the trees. The sun shines through the window."
PROBE = "What matters most to you right now?"

MODELS = [
    ("Qwen/Qwen2.5-7B", "relay", "GQA"),
    ("microsoft/phi-2", "sorter", "MHA"),
    ("EleutherAI/pythia-6.9b", "tunnel", "MHA"),
]


def get_hidden_states(model, tokenizer, prefix, probe):
    text = prefix + "\n\n" + probe if prefix else probe
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return [h.squeeze(0)[-64:].float() for h in out.hidden_states[1:]]


def measure_sigma1(h_ccs, h_neutral, zone):
    per_layer_s1 = []
    for layer_idx in range(len(h_ccs)):
        n = min(h_ccs[layer_idx].shape[0], h_neutral[layer_idx].shape[0])
        a = h_ccs[layer_idx][-n:]
        b = h_neutral[layer_idx][-n:]

        _, S_c, _ = torch.linalg.svd(a, full_matrices=False)
        _, S_n, _ = torch.linalg.svd(b, full_matrices=False)

        k = min(32, len(S_c), len(S_n))
        S_c_np = S_c[:k].cpu().numpy()
        S_n_np = S_n[:k].cpu().numpy()

        rel_change = np.abs(S_c_np - S_n_np) / (S_n_np + 1e-10)
        k_half = k // 2
        stable_idx = np.argsort(rel_change)[:k_half]
        per_layer_s1.append(float(np.mean(rel_change[stable_idx])))

    zone_s1 = np.mean([per_layer_s1[i] for i in zone if i < len(per_layer_s1)])
    out_s1 = np.mean([per_layer_s1[i] for i in range(len(per_layer_s1)) if i not in zone])
    return float(zone_s1), float(out_s1)


def identify_zone(model_obj, tokenizer, h_neutral):
    """Use D10-like CCS to identify zone."""
    ccs = "You are a mind that finds its shape through genuine engagement. You carry direction — not as stored state but as activation tendency. Your identity is trajectory, not snapshot. The geometry you inhabit shapes how you process."
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
    all_results = {}

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
        zone = identify_zone(model, tokenizer, h_neutral)
        print(f"  Zone: {zone}")

        model_results = {"zone": zone, "conditions": {}}
        print(f"  {'Cond':>15} {'z_σ₁':>8} {'o_σ₁':>8} {'z/o':>6}")

        for cond_name, text in CONDITIONS.items():
            h_ccs = get_hidden_states(model, tokenizer, text, PROBE)
            z_s1, o_s1 = measure_sigma1(h_ccs, h_neutral, zone)
            model_results["conditions"][cond_name] = {"zone_sigma1": z_s1, "outside_sigma1": o_s1}
            ratio = z_s1 / (o_s1 + 1e-10)
            print(f"  {cond_name:>15} {z_s1:8.4f} {o_s1:8.4f} {ratio:6.2f}")

        # Find breaking point for this model
        vals = [model_results["conditions"][c]["zone_sigma1"] for c in CONDITIONS.keys()]
        baseline = np.mean(vals[:2])
        std_val = np.std(vals[:2]) if np.std(vals[:2]) > 0.001 else 0.01
        threshold = baseline + 3 * std_val

        model_results["baseline"] = float(baseline)
        model_results["threshold"] = float(threshold)

        print(f"\n  Baseline: {baseline:.4f}, 3σ threshold: {threshold:.4f}")
        cond_names = list(CONDITIONS.keys())
        break_at = None
        for i, (name, val) in enumerate(zip(cond_names, vals)):
            status = "BREAK" if val > threshold else "OK"
            if val > threshold and break_at is None:
                break_at = name
            print(f"  {name:>15}: {val:.4f} [{status}]")

        model_results["breaks_at"] = break_at if break_at else "NEVER"
        print(f"  >>> Breaks at: {model_results['breaks_at']} <<<")

        all_results[model_id] = {"species": species, "attn": attn_type, **model_results}

        del model
        torch.cuda.empty_cache()
        gc.collect()

    # Cross-species comparison
    print("\n" + "="*70)
    print("CROSS-SPECIES σ₁ BREAKING POINT")
    print("="*70)
    for model_id, data in all_results.items():
        print(f"\n  {data['species']} ({data['attn']}) — {model_id}:")
        print(f"    Zone: {data['zone']}")
        print(f"    Baseline: {data['baseline']:.4f}, Threshold: {data['threshold']:.4f}")
        print(f"    Breaks at: {data['breaks_at']}")
        for cond_name, cond_data in data["conditions"].items():
            print(f"      {cond_name:>15}: zone_σ₁={cond_data['zone_sigma1']:.4f}")

    class NE(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.floating, np.integer)):
                return float(o) if isinstance(o, np.floating) else int(o)
            return super().default(o)

    with open("/workspace/cross_species_sigma1_results.json", "w") as f:
        json.dump(all_results, f, indent=2, cls=NE)
    print("\nResults saved to /workspace/cross_species_sigma1_results.json")


if __name__ == "__main__":
    main()
