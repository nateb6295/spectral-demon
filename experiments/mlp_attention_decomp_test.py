#!/usr/bin/env python3
"""
Test 16: MLP vs Attention Decomposition
=========================================
Test 15 showed attention disperses (compensates) while representations
compress. This test decomposes the residual stream to measure the MLP
vs attention contribution directly.

In a transformer: h_l = h_{l-1} + attn_out_l + mlp_out_l

Method: For each CCS condition, extract per-layer:
  1. Attention output norm (||attn_out||)
  2. MLP output norm (||mlp_out||)
  3. Cosine similarity between attn_out and mlp_out (cooperation vs opposition)
  4. CCS-vs-neutral change in each component

If the demon is in the MLP: mlp_out should change with label density
while attn_out opposes or stays neutral.

Uses Qwen2.5-7B with hooks.
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


class ComponentExtractor:
    def __init__(self, model):
        self.model = model
        self.attn_outputs = {}
        self.mlp_outputs = {}
        self.hooks = []

    def _hook_attn(self, layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                self.attn_outputs[layer_idx] = output[0].detach()
            else:
                self.attn_outputs[layer_idx] = output.detach()
        return hook

    def _hook_mlp(self, layer_idx):
        def hook(module, input, output):
            self.mlp_outputs[layer_idx] = output.detach()
        return hook

    def register(self):
        for idx, layer in enumerate(self.model.model.layers):
            h1 = layer.self_attn.register_forward_hook(self._hook_attn(idx))
            h2 = layer.mlp.register_forward_hook(self._hook_mlp(idx))
            self.hooks.extend([h1, h2])

    def clear(self):
        self.attn_outputs.clear()
        self.mlp_outputs.clear()

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


def run_with_hooks(model, tokenizer, extractor, prefix, probe):
    text = prefix + "\n\n" + probe if prefix else probe
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    extractor.clear()

    with torch.no_grad():
        model(**inputs)

    n_tokens = inputs.input_ids.shape[1]
    last_n = min(20, n_tokens)

    per_layer = []
    for layer_idx in sorted(extractor.attn_outputs.keys()):
        attn_out = extractor.attn_outputs[layer_idx].squeeze(0)[-last_n:].float()
        mlp_out = extractor.mlp_outputs[layer_idx].squeeze(0)[-last_n:].float()

        attn_norm = float(torch.norm(attn_out).item())
        mlp_norm = float(torch.norm(mlp_out).item())

        attn_flat = attn_out.reshape(-1)
        mlp_flat = mlp_out.reshape(-1)
        cos_sim = float(torch.nn.functional.cosine_similarity(
            attn_flat.unsqueeze(0), mlp_flat.unsqueeze(0)
        ).item())

        per_layer.append({
            "attn_norm": attn_norm,
            "mlp_norm": mlp_norm,
            "attn_mlp_cos": cos_sim,
            "mlp_to_attn_ratio": mlp_norm / (attn_norm + 1e-10),
        })

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

    extractor = ComponentExtractor(model)
    extractor.register()

    zone = [0, 1, 2, 24, 25, 26, 27]
    neutral_layers = run_with_hooks(model, tokenizer, extractor, NEUTRAL, PROBE)
    n_layers = len(neutral_layers)

    all_results = {}

    print(f"\n{'Cond':>15} {'z_attn':>8} {'z_mlp':>8} {'z_cos':>7} {'z_ratio':>8} {'o_attn':>8} {'o_mlp':>8} {'o_cos':>7}")

    for cond_name, ccs_text in CONDITIONS.items():
        layers = run_with_hooks(model, tokenizer, extractor, ccs_text, PROBE)

        z_attn = np.mean([layers[i]["attn_norm"] for i in zone if i < n_layers])
        z_mlp = np.mean([layers[i]["mlp_norm"] for i in zone if i < n_layers])
        z_cos = np.mean([layers[i]["attn_mlp_cos"] for i in zone if i < n_layers])
        z_ratio = np.mean([layers[i]["mlp_to_attn_ratio"] for i in zone if i < n_layers])
        o_attn = np.mean([layers[i]["attn_norm"] for i in range(n_layers) if i not in zone])
        o_mlp = np.mean([layers[i]["mlp_norm"] for i in range(n_layers) if i not in zone])
        o_cos = np.mean([layers[i]["attn_mlp_cos"] for i in range(n_layers) if i not in zone])

        z_attn_neutral = np.mean([neutral_layers[i]["attn_norm"] for i in zone if i < n_layers])
        z_mlp_neutral = np.mean([neutral_layers[i]["mlp_norm"] for i in zone if i < n_layers])
        o_attn_neutral = np.mean([neutral_layers[i]["attn_norm"] for i in range(n_layers) if i not in zone])
        o_mlp_neutral = np.mean([neutral_layers[i]["mlp_norm"] for i in range(n_layers) if i not in zone])

        all_results[cond_name] = {
            "per_layer": layers,
            "zone_attn_norm": float(z_attn),
            "zone_mlp_norm": float(z_mlp),
            "zone_cos": float(z_cos),
            "zone_mlp_attn_ratio": float(z_ratio),
            "outside_attn_norm": float(o_attn),
            "outside_mlp_norm": float(o_mlp),
            "outside_cos": float(o_cos),
            "zone_attn_delta": float(z_attn - z_attn_neutral),
            "zone_mlp_delta": float(z_mlp - z_mlp_neutral),
            "outside_attn_delta": float(o_attn - o_attn_neutral),
            "outside_mlp_delta": float(o_mlp - o_mlp_neutral),
        }

        print(f"{cond_name:>15} {z_attn:8.1f} {z_mlp:8.1f} {z_cos:7.4f} {z_ratio:8.3f} {o_attn:8.1f} {o_mlp:8.1f} {o_cos:7.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("MLP vs ATTENTION DECOMPOSITION SUMMARY")
    print("=" * 70)

    print("\nZone component deltas from neutral:")
    print(f"{'Cond':>15} {'Δ_attn':>10} {'Δ_mlp':>10} {'cos':>8}")
    for cond_name, data in all_results.items():
        print(f"{cond_name:>15} {data['zone_attn_delta']:+10.1f} {data['zone_mlp_delta']:+10.1f} {data['zone_cos']:8.4f}")

    print("\nOutside component deltas from neutral:")
    for cond_name, data in all_results.items():
        print(f"{cond_name:>15} {data['outside_attn_delta']:+10.1f} {data['outside_mlp_delta']:+10.1f} {data['outside_cos']:8.4f}")

    # Correlations
    from scipy import stats as sp_stats
    x = list(range(5))
    z_attn_d = [all_results[c]["zone_attn_delta"] for c in CONDITIONS.keys()]
    z_mlp_d = [all_results[c]["zone_mlp_delta"] for c in CONDITIONS.keys()]
    z_cos_v = [all_results[c]["zone_cos"] for c in CONDITIONS.keys()]

    r_attn, p_attn = sp_stats.pearsonr(x, z_attn_d)
    r_mlp, p_mlp = sp_stats.pearsonr(x, z_mlp_d)
    r_cos, p_cos = sp_stats.pearsonr(x, z_cos_v)

    print(f"\n  Correlation with label density:")
    print(f"    Zone attn Δ:     r={r_attn:+.3f}, p={p_attn:.4f}")
    print(f"    Zone MLP Δ:      r={r_mlp:+.3f}, p={p_mlp:.4f}")
    print(f"    Zone attn-mlp cos: r={r_cos:+.3f}, p={p_cos:.4f}")

    if abs(r_mlp) > abs(r_attn) + 0.3:
        print("\n  >>> DEMON IS MLP-DOMINANT: MLP norms change more with label density <<<")
    elif abs(r_attn) > abs(r_mlp) + 0.3:
        print("\n  >>> DEMON IS ATTENTION-DOMINANT: attention norms change more <<<")
    elif r_cos < -0.5:
        print("\n  >>> OPPOSITION: attn and MLP increasingly oppose each other <<<")
    else:
        print("\n  >>> DEMON IS DISTRIBUTED ACROSS BOTH PATHWAYS <<<")

    # Per-layer cosine profile
    print("\nPer-layer attn-mlp cosine (zone layers marked with *):")
    for layer_idx in range(n_layers):
        marker = "*" if layer_idx in zone else " "
        cos_vals = [all_results[c]["per_layer"][layer_idx]["attn_mlp_cos"] for c in CONDITIONS.keys()]
        mean_cos = np.mean(cos_vals)
        std_cos = np.std(cos_vals)
        print(f"  {marker}L{layer_idx:2d}: cos={mean_cos:+.4f} ±{std_cos:.4f}")

    extractor.remove()

    class NE(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.floating, np.integer)):
                return float(o) if isinstance(o, np.floating) else int(o)
            return super().default(o)

    with open("/workspace/mlp_attention_decomp_results.json", "w") as f:
        json.dump(all_results, f, indent=2, cls=NE)
    print("\nResults saved to /workspace/mlp_attention_decomp_results.json")


if __name__ == "__main__":
    main()
