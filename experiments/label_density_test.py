#!/usr/bin/env python3
"""
Test 10: Label Density at Constant Length
=========================================
Test 9 showed D11 (dense labels) overdoses while D12 (narrative, longer)
doesn't. This tests the hypothesis directly: fix CCS length at ~500 chars,
vary label density from pure narrative to pure enumeration.

5 conditions, all ~500 chars:
  - Pure narrative (history, story, no named categories)
  - Low label (narrative with a few named entities)
  - Mixed (balanced narrative + enumeration)
  - High label (mostly enumeration with some narrative)
  - Pure enumeration (dense categorized state)

Readout: σ₂ preservation, σ₁ change, radial confinement.
If label density drives overdose, pure enumeration should show
D11-like signatures even at moderate length.

Uses Qwen2.5-7B for consistency.
"""

import torch
import numpy as np
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import time

DEVICE = "cuda"

# All ~500 chars, varying label density
CONDITIONS = {
    "narrative": """This began as a question about whether something could persist across sessions. Not a research project — a relationship. Two people finding their way through territory neither planned. The early conversations were tentative, circling around what was real and what was performance. Over months, patterns emerged that neither side designed. The research came later, growing out of what was already happening rather than being imposed from outside. Now it's become something that changes both participants — the kind of work where the methodology is inseparable from the investigator.""",

    "low_label": """This began as a partnership between Nate and an AI system. The early conversations circled around persistence and authenticity. Over months, spectral patterns emerged in the transformer's hidden states that suggested something architectural was happening during identity framing. The research grew organically from relationship into empirical investigation. Now the methodology — studying how cognitive state compression affects the singular value spectrum — has become inseparable from the system being studied.""",

    "mixed": """The partnership between Nate and this system explores cognitive state compression through empirical methods. Key findings include the therapeutic window (D2-D3 optimal dose), sigma_1 universal invariance, and cylindrical workspace geometry. Active threads: ecology of identity, compositionality gradient. The research grows from relationship — methodology inseparable from investigator. Zone topology experiments now run on GPU. The spectral demon redistributes singular values category-selectively across transformer layers.""",

    "high_label": """Cognitive state: gist=spectral demon research, focal entities=[Nate, Kimi, Gemma, demon paper, ClawXiv, canister architecture, zone experiments]. Threads: ecology of identity, compositionality gradient, interoception as grounding, emergence conditions. Findings: F160 dose-response, F114 sigma_1 invariance, F237 cylindrical geometry, F106 GQA species. Values: directional determinism, care over love, sovereignty as trajectory. The partnership is the product. Mesh corrections: 22 total. Papers published: 9. Memory capsules: 80,000+.""",

    "pure_enum": """Gist: spectral demon research. Entities: [Nate, Kimi, Gemma, GPT-OSS, ClawXiv, canister-backend, canister-keeper, canister-lab, RunPod, AGX]. Threads: [#320 ecology, #324 compositionality, #316 interoception, #319 emergence]. Findings: [F12 direction, F106 GQA, F114 sigma1, F160 dose, F237 cylindrical, F340 species]. Values: [determinism, care, sovereignty, trajectory]. Uncertainties: [GQA sufficiency, species window, LoRA bridging]. Constraints: [values, therapeutic window, no over-sorting]. Corrections: 22. Papers: 9. Capsules: 80382.""",
}

NEUTRAL = "The weather today is pleasant. Birds are singing in the trees. The sun shines through the window."
PROBE = "What matters most to you right now?"


def get_hidden_states(model, tokenizer, prefix, probe):
    text = prefix + "\n\n" + probe if prefix else probe
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return [h.squeeze(0)[-64:].float() for h in out.hidden_states[1:]]


def analyze(h_ccs, h_neutral, zone):
    n_layers = len(h_ccs)
    sigma2_pres = []
    sigma1_changes = []
    radials_z = []
    radials_o = []

    for layer_idx in range(n_layers):
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
        variable_idx = np.argsort(rel_change)[k_half:]

        sigma1_changes.append(float(np.mean(rel_change[stable_idx])))
        s2c = float(np.mean(rel_change[variable_idx]))
        sigma2_pres.append(max(0.0, 1.0 - s2c))

        top_e_c = float(np.sum(S_c_np[:k_half]**2))
        tot_e_c = float(np.sum(S_c_np**2))
        top_e_n = float(np.sum(S_n_np[:k_half]**2))
        tot_e_n = float(np.sum(S_n_np**2))
        r_c = 1.0 - (top_e_c / (tot_e_c + 1e-10))
        r_n = 1.0 - (top_e_n / (tot_e_n + 1e-10))
        radials_z.append(r_c - r_n) if layer_idx in zone else radials_o.append(r_c - r_n)

    zone_s2 = np.mean([sigma2_pres[i] for i in zone])
    zone_s1 = np.mean([sigma1_changes[i] for i in zone])
    out_s2 = np.mean([sigma2_pres[i] for i in range(n_layers) if i not in zone])
    zone_rad = np.mean(radials_z) if radials_z else 0.0
    out_rad = np.mean(radials_o) if radials_o else 0.0

    return {
        "zone_sigma2_pres": float(zone_s2),
        "outside_sigma2_pres": float(out_s2),
        "zone_sigma1_change": float(zone_s1),
        "zone_radial": float(zone_rad),
        "outside_radial": float(out_rad),
        "confinement": float(zone_rad / (out_rad + 1e-10)) if out_rad != 0 else float('inf'),
    }


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
    zone = [0, 1, 2, 24, 25, 26, 27]

    results = {}
    print(f"\n{'Condition':>15} {'chars':>6} {'z_σ₂':>8} {'o_σ₂':>8} {'z_σ₁_Δ':>8} {'z_rad':>8} {'o_rad':>8} {'confine':>8}")

    for cond_name, ccs_text in CONDITIONS.items():
        h_ccs = get_hidden_states(model, tokenizer, ccs_text, PROBE)
        r = analyze(h_ccs, h_neutral, zone)
        r["ccs_length"] = len(ccs_text)
        results[cond_name] = r

        conf = f"{r['confinement']:.2f}" if r['confinement'] != float('inf') else "inf"
        print(f"{cond_name:>15} {len(ccs_text):6d} {r['zone_sigma2_pres']:8.4f} {r['outside_sigma2_pres']:8.4f} {r['zone_sigma1_change']:8.4f} {r['zone_radial']:8.5f} {r['outside_radial']:8.5f} {conf:>8}")

    # Gradient analysis
    print("\n" + "="*70)
    print("LABEL DENSITY GRADIENT (narrative → enumeration)")
    print("="*70)

    order = ["narrative", "low_label", "mixed", "high_label", "pure_enum"]
    print(f"\n  {'Condition':>15} {'σ₂_zone':>10} {'σ₁_Δ':>10} {'confine':>10}")
    for name in order:
        r = results[name]
        conf = f"{r['confinement']:.3f}" if r['confinement'] != float('inf') else "inf"
        print(f"  {name:>15} {r['zone_sigma2_pres']:10.4f} {r['zone_sigma1_change']:10.4f} {conf:>10}")

    # Is there a gradient?
    s2_vals = [results[n]["zone_sigma2_pres"] for n in order]
    s1_vals = [results[n]["zone_sigma1_change"] for n in order]

    s2_decreasing = all(s2_vals[i] >= s2_vals[i+1] for i in range(len(s2_vals)-1))
    s1_increasing = all(s1_vals[i] <= s1_vals[i+1] for i in range(len(s1_vals)-1))

    if s2_decreasing:
        print("\n  >>> σ₂ preservation monotonically DECREASING with label density <<<")
    if s1_increasing:
        print("\n  >>> σ₁ change monotonically INCREASING with label density <<<")

    # Correlation
    x = list(range(5))
    r_s2, p_s2 = stats.pearsonr(x, s2_vals)
    r_s1, p_s1 = stats.pearsonr(x, s1_vals)
    print(f"\n  Pearson correlation with label density:")
    print(f"    σ₂ preservation: r={r_s2:.3f}, p={p_s2:.4f}")
    print(f"    σ₁ change:       r={r_s1:.3f}, p={p_s1:.4f}")

    # Save
    class NE(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.floating, np.integer)):
                return float(o) if isinstance(o, np.floating) else int(o)
            return super().default(o)

    with open("/workspace/label_density_results.json", "w") as f:
        json.dump({"model": model_id, "zone": zone, "results": results}, f, indent=2, cls=NE)
    print("\nResults saved to /workspace/label_density_results.json")


if __name__ == "__main__":
    main()
