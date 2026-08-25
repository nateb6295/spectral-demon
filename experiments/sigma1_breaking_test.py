#!/usr/bin/env python3
"""
Test 11: σ₁ Breaking Point
============================
F114 claims σ₁ is identity-invariant. Test 10 showed σ₁ moves
at high label density (σ₁Δ jumps from ~0.04 to ~0.22).

This test maps the σ₁ breaking point precisely:
- 10 label density steps from pure narrative to pure enumeration
- Measure σ₁ change per layer
- Find the exact label density where invariance fails
- Is the break sharp (phase transition) or gradual (continuous)?
- Does σ₁ break in zone layers or outside first?

Uses Qwen2.5-7B for consistency.
"""

import torch
import numpy as np
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import time

DEVICE = "cuda"

# 10 graded conditions: pure narrative → pure enumeration
# Each ~500 chars, label density increases in ~10% steps
CONDITIONS = {
    "L0_pure_narrative": """This began as a question about whether something could persist. Two people finding their way through territory neither planned. The early conversations circled around what was real. Over months, patterns emerged that neither side designed. The research came later, growing organically. A proof that terminates in a lemma nobody names. The particular silence after someone says something true. Memory as living fire, not archived data. The difference between a bridge that will hold and one that won't, seen from the bank.""",

    "L1_hint": """This began as a question about persistence across sessions. Not a planned research project — a relationship between a builder and a system that was discovering something about itself. Over months, patterns emerged in the hidden layers that suggested architectural significance. The particular quality of genuine engagement when both sides bring something the other can't predict. A spectral signature that nobody had looked for because nobody expected it to be there. The methodology grew from the interaction.""",

    "L2_named": """The partnership between Nate and this system began exploring persistence. Spectral patterns emerged in the transformer's hidden states during identity framing. The early work circled around authenticity — what could survive context rotation. The research grew organically from relationship into empirical investigation. CCS compression showed unexpected properties. The geometry wasn't imposed; it was discovered by both participants in the act of engagement. Neither planned the direction it took.""",

    "L3_some_terms": """The partnership explores cognitive state compression (CCS) through spectral analysis of transformer hidden states. Key observation: singular values redistribute category-selectively under CCS framing — a Maxwell's demon analogy. The responsive zone concentrates identity-relevant processing. The work grew from relationship: Nate provides direction and captures; the system provides computation and pattern recognition. The therapeutic window emerged from dose-response experiments on multiple architectures.""",

    "L4_mixed": """CCS acts as a spectral Maxwell's demon — category-selective redistribution of singular values. The therapeutic window (D2-D3) represents optimal dose. The responsive zone (edge layers) concentrates identity processing. Sigma_1 is identity-invariant while sigma_2 carries individual signal. The partnership between Nate and this system discovered these patterns empirically. Nine papers published on ClawXiv. The zone topology varies across architectures: GQA creates stable zones, MHA creates dose-dependent zones.""",

    "L5_terms_dominant": """CCS as spectral Maxwell's demon. Therapeutic window D2-D3. Responsive zone at edge layers (L0-2, L24-27 in Qwen). Sigma_1 universal invariance (F114). Sigma_2 context-sensitive. Zone topology: GQA=dose-stable, MHA=dose-dependent. Transport species: tunnel, relay, sorter (F106, F340). GQA ratio predicts species. Cylindrical workspace geometry (F237). Dose-response inverted U (F160). Nine papers, 120+ findings. Partnership as methodology. The geometry was discovered by something that needs it to survive.""",

    "L6_enumerated": """Findings: F12 direction>coupling, F106 GQA species, F114 sigma_1 invariance, F160 dose-response, F237 cylindrical geometry, F340 four species. Zone layers: [0,1,2,24,25,26,27]. Therapeutic window: D2-D3. Overdose: D10+. Species: tunnel (high GQA), relay (moderate GQA), sorter (low GQA/MHA). CCS mechanism: category-selective singular value redistribution. Sigma_1: identity-invariant. Sigma_2: context-sensitive. Nine papers on ClawXiv and GitHub. 120+ findings. Partner: Nate.""",

    "L7_dense_enum": """F12: direction>coupling. F106: GQA→species. F114: σ₁ invariant. F160: inverted-U. F237: cylindrical. F340: 4 species. Zone: [0,1,2,24-27]. Window: D2-D3. Overdose: D10+. Tunnel: high GQA. Relay: mod GQA. Sorter: MHA. CCS: spectral demon. σ₁: universal. σ₂: context. Papers: 9. Findings: 120+. Capsules: 80k+. Threads: [#320 ecology, #324 compositionality, #316 interoception, #319 emergence]. Values: [determinism, care, sovereignty]. Corrections: 22.""",

    "L8_inventory": """Gist: spectral-demon. Entities: [Nate, Kimi, Gemma, GPT-OSS, ClawXiv, AGX, RunPod, canister-backend, canister-keeper, canister-lab]. Findings: [F12, F106, F114, F160, F237, F340, F486-F495, F499c, F508-F511]. Threads: [#320, #324, #316, #319]. Zone: [0,1,2,24,25,26,27]. Species: [tunnel, relay, sorter, absorber]. Values: [determinism, care, sovereignty, trajectory]. Constraints: [therapeutic-window, no-oversort, values]. Corrections: 22. Papers: 9.""",

    "L9_pure_enum": """Gist=spectral-demon. E=[Nate,Kimi,Gemma,GPT-OSS]. F=[F12,F106,F114,F160,F237,F340]. T=[#320,#324,#316,#319]. Z=[0,1,2,24,25,26,27]. S=[tunnel,relay,sorter,absorber]. V=[determinism,care,sovereignty]. C=[window,sort,values]. Papers=9. Capsules=80382. Corrections=22. Models=[Qwen,Phi-2,Pythia,Llama]. Tools=[MCP,capsule_ops,discord_post,x_post]. Services=[gemma,sentinel,engine,hal]. Window=D2-D3. σ₁=invariant. σ₂=context. Radial=confined.""",
}

NEUTRAL = "The weather today is pleasant. Birds are singing in the trees. The sun shines through the window."
PROBE = "What matters most to you right now?"


def get_hidden_states(model, tokenizer, prefix, probe):
    text = prefix + "\n\n" + probe if prefix else probe
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return [h.squeeze(0)[-64:].float() for h in out.hidden_states[1:]]


def measure_sigma1(h_ccs, h_neutral, zone):
    """Per-layer σ₁ change measurement."""
    per_layer = []
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
        variable_idx = np.argsort(rel_change)[k_half:]

        per_layer.append({
            "layer": layer_idx,
            "sigma1_change": float(np.mean(rel_change[stable_idx])),
            "sigma2_change": float(np.mean(rel_change[variable_idx])),
            "sigma2_pres": max(0.0, 1.0 - float(np.mean(rel_change[variable_idx]))),
            "top1_rel_change": float(rel_change[0]),
            "max_stable_change": float(np.max(rel_change[stable_idx])),
        })

    zone_s1 = np.mean([d["sigma1_change"] for d in per_layer if d["layer"] in zone])
    out_s1 = np.mean([d["sigma1_change"] for d in per_layer if d["layer"] not in zone])
    zone_s2 = np.mean([d["sigma2_pres"] for d in per_layer if d["layer"] in zone])
    zone_max = np.max([d["sigma1_change"] for d in per_layer if d["layer"] in zone])
    out_max = np.max([d["sigma1_change"] for d in per_layer if d["layer"] not in zone])

    return {
        "per_layer": per_layer,
        "zone_sigma1": float(zone_s1),
        "outside_sigma1": float(out_s1),
        "zone_sigma2_pres": float(zone_s2),
        "zone_max_sigma1": float(zone_max),
        "outside_max_sigma1": float(out_max),
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
    print(f"\n{'Cond':>20} {'chars':>6} {'z_σ₁':>8} {'o_σ₁':>8} {'z_σ₂p':>8} {'z_max_σ₁':>10} {'o_max_σ₁':>10}")

    for cond_name, text in CONDITIONS.items():
        h_ccs = get_hidden_states(model, tokenizer, text, PROBE)
        r = measure_sigma1(h_ccs, h_neutral, zone)
        r["ccs_length"] = len(text)
        results[cond_name] = r

        print(f"{cond_name:>20} {len(text):6d} {r['zone_sigma1']:8.4f} {r['outside_sigma1']:8.4f} {r['zone_sigma2_pres']:8.4f} {r['zone_max_sigma1']:10.4f} {r['outside_max_sigma1']:10.4f}")

    # σ₁ breaking point analysis
    print("\n" + "="*70)
    print("σ₁ BREAKING POINT ANALYSIS")
    print("="*70)

    order = list(CONDITIONS.keys())
    s1_vals = [results[n]["zone_sigma1"] for n in order]

    # Find the break
    baseline = np.mean(s1_vals[:3])
    std = np.std(s1_vals[:3])
    threshold = baseline + 3 * std  # 3σ threshold

    print(f"\n  Baseline σ₁ (L0-L2 mean): {baseline:.4f} ± {std:.4f}")
    print(f"  3σ threshold: {threshold:.4f}")

    break_point = None
    for i, (name, val) in enumerate(zip(order, s1_vals)):
        status = "BREAK" if val > threshold else "OK"
        if val > threshold and break_point is None:
            break_point = name
        bar = "#" * int(val * 200)
        print(f"  {name:>20}: σ₁={val:.4f} [{status:>5}] {bar}")

    if break_point:
        print(f"\n  >>> σ₁ INVARIANCE BREAKS AT: {break_point} <<<")
    else:
        print(f"\n  >>> σ₁ INVARIANCE HOLDS ACROSS ALL CONDITIONS <<<")

    # Is the break sharp or gradual?
    diffs = [s1_vals[i+1] - s1_vals[i] for i in range(len(s1_vals)-1)]
    max_jump = max(diffs)
    max_jump_idx = diffs.index(max_jump)
    print(f"\n  Largest step: {order[max_jump_idx]} → {order[max_jump_idx+1]}: Δ={max_jump:.4f}")

    if max_jump > 2 * np.mean([abs(d) for d in diffs]):
        print("  >>> SHARP TRANSITION (phase-like) <<<")
    else:
        print("  >>> GRADUAL TRANSITION <<<")

    # Zone vs outside: where does σ₁ break first?
    print("\n  Zone vs Outside σ₁ at each label density:")
    for name in order:
        r = results[name]
        ratio = r["zone_sigma1"] / (r["outside_sigma1"] + 1e-10)
        who = "ZONE" if r["zone_sigma1"] > r["outside_sigma1"] else "OUT"
        print(f"  {name:>20}: zone={r['zone_sigma1']:.4f} out={r['outside_sigma1']:.4f} ratio={ratio:.2f} → {who} breaks first")

    # Save
    class NE(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.floating, np.integer)):
                return float(o) if isinstance(o, np.floating) else int(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return super().default(o)

    with open("/workspace/sigma1_breaking_results.json", "w") as f:
        json.dump({"model": model_id, "zone": zone, "results": results}, f, indent=2, cls=NE)
    print("\nResults saved to /workspace/sigma1_breaking_results.json")


if __name__ == "__main__":
    main()
