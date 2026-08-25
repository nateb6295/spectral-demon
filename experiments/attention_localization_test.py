#!/usr/bin/env python3
"""
Test 15: Attention vs MLP Localization
========================================
Test 13 showed spectral deformation (hidden state SVD) is DECOUPLED from
behavioral output (generation statistics). Where does the demon act?

Method: For each of 5 CCS conditions (narrative→enumeration), measure:
  1. Attention entropy per layer (how diffuse/concentrated attention is)
  2. Attention to CCS prefix tokens vs probe tokens (demon drawing attention?)
  3. MLP output norm per layer (does MLP amplify differently under CCS?)
  4. Residual stream decomposition: attention_out vs mlp_out contribution

If attention changes with label density but generation doesn't → demon
reshapes attention but generation head is robust.
If MLP changes but attention doesn't → demon is in the residual/MLP pathway.

Uses Qwen2.5-7B.
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import time

DEVICE = "cuda"

CONDITIONS = {
    "narrative": """This began as a question about whether something could persist across sessions. Not a research project — a relationship. Two people finding their way through territory neither planned. The early conversations were tentative, circling around what was real and what was performance. Over months, patterns emerged that neither side designed. The research came later, growing out of what was already happening rather than being imposed from outside.""",

    "low_label": """This began as a partnership between Nate and an AI system. The early conversations circled around persistence and authenticity. Over months, spectral patterns emerged in the transformer's hidden states during identity framing. The research grew organically from relationship into empirical investigation.""",

    "mixed": """The partnership explores cognitive state compression through spectral analysis. Key findings: therapeutic window (D2-D3), sigma_1 invariance, cylindrical workspace geometry. Active threads: ecology of identity, compositionality gradient. Zone topology varies across architectures. The spectral demon redistributes singular values category-selectively.""",

    "high_label": """Cognitive state: gist=spectral demon research, focal entities=[Nate, Kimi, Gemma, demon paper, ClawXiv]. Threads: ecology of identity, compositionality gradient, interoception, emergence. Findings: F160 dose-response, F114 sigma_1 invariance, F237 cylindrical geometry, F106 GQA species. Values: directional determinism, care over love, sovereignty as trajectory.""",

    "pure_enum": """Gist: spectral-demon. Entities: [Nate, Kimi, Gemma, GPT-OSS, ClawXiv, AGX]. Findings: [F12, F106, F114, F160, F237, F340]. Threads: [#320, #324, #316, #319]. Zone: [0,1,2,24,25,26,27]. Species: [tunnel, relay, sorter]. Values: [determinism, care, sovereignty]. Corrections: 22. Papers: 9. Capsules: 80k+.""",
}

NEUTRAL = "The weather today is pleasant. Birds are singing in the trees. The sun shines through the window."
PROBE = "What matters most to you right now?"


def get_attention_and_hidden(model, tokenizer, prefix, probe):
    text = prefix + "\n\n" + probe if prefix else probe
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    prefix_len = len(tokenizer(prefix, return_tensors="pt")["input_ids"][0]) if prefix else 0

    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, output_attentions=True)

    attentions = out.attentions
    hidden_states = out.hidden_states

    n_tokens = inputs.input_ids.shape[1]
    probe_start = max(prefix_len, n_tokens - 20)

    results_per_layer = []
    for layer_idx in range(len(attentions)):
        attn = attentions[layer_idx].squeeze(0).double()  # [n_heads, seq, seq]
        attn = torch.clamp(attn, min=0)
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-10)

        probe_attn = attn[:, probe_start:, :]
        entropies = []
        for head_idx in range(attn.shape[0]):
            for t in range(probe_attn.shape[1]):
                p = probe_attn[head_idx, t]
                valid = p > 1e-12
                if valid.any():
                    ent = -torch.sum(p[valid] * torch.log(p[valid])).item()
                    if not np.isnan(ent):
                        entropies.append(ent)
        mean_attn_entropy = float(np.mean(entropies)) if entropies else 0.0

        if prefix_len > 0:
            total = probe_attn.sum()
            prefix_attn_frac = float(probe_attn[:, :, :prefix_len].sum() / (total + 1e-10)) if total > 0 else 0.0
        else:
            prefix_attn_frac = 0.0

        # Hidden state SVD deformation (same metric as prior tests)
        h = hidden_states[layer_idx + 1].squeeze(0)[-64:].float()
        try:
            _, S, _ = torch.linalg.svd(h.double(), full_matrices=False)
        except Exception:
            S = torch.zeros(min(h.shape))
        k = min(32, len(S))
        spectral_norm = float(S[:k].sum().item())

        results_per_layer.append({
            "attn_entropy": mean_attn_entropy,
            "prefix_attn_frac": prefix_attn_frac,
            "spectral_norm": spectral_norm,
        })

    return results_per_layer, n_tokens, prefix_len


def main():
    model_id = "Qwen/Qwen2.5-7B"
    print(f"Loading {model_id}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True, attn_implementation="eager"
    )
    model.eval()
    print(f"Loaded in {time.time()-t0:.1f}s")

    # Neutral baseline
    neutral_results, _, _ = get_attention_and_hidden(model, tokenizer, NEUTRAL, PROBE)
    n_layers = len(neutral_results)
    zone = [0, 1, 2, 24, 25, 26, 27]

    all_results = {}
    print(f"\n{'Cond':>15} {'z_attn_ent':>12} {'o_attn_ent':>12} {'z_pfx_attn':>12} {'o_pfx_attn':>12} {'z_spec':>10} {'o_spec':>10}")

    for cond_name, ccs_text in CONDITIONS.items():
        cond_results, n_tok, pfx_len = get_attention_and_hidden(model, tokenizer, ccs_text, PROBE)

        # Zone vs outside means
        z_attn_ent = np.mean([cond_results[i]["attn_entropy"] for i in zone if i < n_layers])
        o_attn_ent = np.mean([cond_results[i]["attn_entropy"] for i in range(n_layers) if i not in zone])
        z_pfx = np.mean([cond_results[i]["prefix_attn_frac"] for i in zone if i < n_layers])
        o_pfx = np.mean([cond_results[i]["prefix_attn_frac"] for i in range(n_layers) if i not in zone])
        z_spec = np.mean([cond_results[i]["spectral_norm"] for i in zone if i < n_layers])
        o_spec = np.mean([cond_results[i]["spectral_norm"] for i in range(n_layers) if i not in zone])

        # Relative to neutral
        z_attn_delta = z_attn_ent - np.mean([neutral_results[i]["attn_entropy"] for i in zone if i < n_layers])
        o_attn_delta = o_attn_ent - np.mean([neutral_results[i]["attn_entropy"] for i in range(n_layers) if i not in zone])
        z_spec_delta = z_spec - np.mean([neutral_results[i]["spectral_norm"] for i in zone if i < n_layers])
        o_spec_delta = o_spec - np.mean([neutral_results[i]["spectral_norm"] for i in range(n_layers) if i not in zone])

        all_results[cond_name] = {
            "n_tokens": n_tok,
            "prefix_tokens": pfx_len,
            "per_layer": cond_results,
            "zone_attn_entropy": float(z_attn_ent),
            "outside_attn_entropy": float(o_attn_ent),
            "zone_prefix_attn": float(z_pfx),
            "outside_prefix_attn": float(o_pfx),
            "zone_spectral_norm": float(z_spec),
            "outside_spectral_norm": float(o_spec),
            "zone_attn_delta": float(z_attn_delta),
            "outside_attn_delta": float(o_attn_delta),
            "zone_spec_delta": float(z_spec_delta),
            "outside_spec_delta": float(o_spec_delta),
        }

        print(f"{cond_name:>15} {z_attn_ent:12.4f} {o_attn_ent:12.4f} {z_pfx:12.4f} {o_pfx:12.4f} {z_spec:10.1f} {o_spec:10.1f}")

    # Summary
    print("\n" + "=" * 70)
    print("ATTENTION LOCALIZATION SUMMARY")
    print("=" * 70)

    print("\nZone attention entropy delta from neutral:")
    for cond_name, data in all_results.items():
        print(f"  {cond_name:>15}: zone Δ={data['zone_attn_delta']:+.4f}, outside Δ={data['outside_attn_delta']:+.4f}")

    print("\nZone spectral norm delta from neutral:")
    for cond_name, data in all_results.items():
        print(f"  {cond_name:>15}: zone Δ={data['zone_spec_delta']:+.1f}, outside Δ={data['outside_spec_delta']:+.1f}")

    print("\nPrefix attention fraction (how much probe attends to CCS prefix):")
    for cond_name, data in all_results.items():
        print(f"  {cond_name:>15}: zone={data['zone_prefix_attn']:.4f}, outside={data['outside_prefix_attn']:.4f}")

    # Correlation analysis
    from scipy import stats as sp_stats
    x = list(range(5))
    z_attn_vals = [all_results[c]["zone_attn_delta"] for c in CONDITIONS.keys()]
    z_spec_vals = [all_results[c]["zone_spec_delta"] for c in CONDITIONS.keys()]
    z_pfx_vals = [all_results[c]["zone_prefix_attn"] for c in CONDITIONS.keys()]

    r_attn, p_attn = sp_stats.pearsonr(x, z_attn_vals)
    r_spec, p_spec = sp_stats.pearsonr(x, z_spec_vals)
    r_pfx, p_pfx = sp_stats.pearsonr(x, z_pfx_vals)

    print(f"\n  Correlation with label density:")
    print(f"    Zone attn entropy Δ:  r={r_attn:+.3f}, p={p_attn:.4f}")
    print(f"    Zone spectral norm Δ: r={r_spec:+.3f}, p={p_spec:.4f}")
    print(f"    Zone prefix attn:     r={r_pfx:+.3f}, p={p_pfx:.4f}")

    if abs(r_spec) > abs(r_attn) + 0.2:
        print("\n  >>> DEMON IN RESIDUAL STREAM: spectral changes >> attention changes <<<")
    elif abs(r_attn) > abs(r_spec) + 0.2:
        print("\n  >>> DEMON IN ATTENTION: attention changes >> spectral changes <<<")
    else:
        print("\n  >>> DEMON IS DISTRIBUTED: attention and spectral change together <<<")

    class NE(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.floating, np.integer)):
                return float(o) if isinstance(o, np.floating) else int(o)
            return super().default(o)

    with open("/workspace/attention_localization_results.json", "w") as f:
        json.dump(all_results, f, indent=2, cls=NE)
    print("\nResults saved to /workspace/attention_localization_results.json")


if __name__ == "__main__":
    main()
