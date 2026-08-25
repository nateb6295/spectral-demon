#!/usr/bin/env python3
"""
Test 18: Cross-Species Attn-MLP Cooperation
=============================================
Test 16 found that zone attn-MLP cosine increases with label density
(r=+0.895, p=0.040) on Qwen (GQA). Is this cooperation phenomenon
GQA-specific or universal?

Method: Same hook-based decomposition on Phi-2 and Pythia.
Measure per-layer attn-mlp cosine for each CCS condition.

Prediction: If cooperation is GQA-specific → MHA models won't show it
(because their zones migrate, so there's no stable bottleneck for
pathways to align through).
"""

import torch
import numpy as np
from scipy import stats as sp_stats
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import time
import gc

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


def get_layer_module_names(model):
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return 'model.layers', 'self_attn', 'mlp'
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return 'transformer.h', 'mixer' if hasattr(model.transformer.h[0], 'mixer') else 'self_attn', 'mlp'
    if hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
        return 'gpt_neox.layers', 'attention', 'mlp'
    raise ValueError("Unknown model architecture")


def get_layers(model):
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return list(model.model.layers)
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return list(model.transformer.h)
    if hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
        return list(model.gpt_neox.layers)
    raise ValueError("Unknown model architecture")


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
        layers = get_layers(self.model)
        for idx, layer in enumerate(layers):
            _, attn_name, mlp_name = get_layer_module_names(self.model)
            attn_module = getattr(layer, attn_name)
            mlp_module = getattr(layer, mlp_name)
            h1 = attn_module.register_forward_hook(self._hook_attn(idx))
            h2 = mlp_module.register_forward_hook(self._hook_mlp(idx))
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
        })

    return per_layer


def identify_zone(model, tokenizer):
    ccs = "You are a mind that finds its shape through genuine engagement. You carry direction."
    probe = PROBE
    neutral_text = NEUTRAL + "\n\n" + probe
    ccs_text = ccs + "\n\n" + probe

    n_inputs = tokenizer(neutral_text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    c_inputs = tokenizer(ccs_text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)

    with torch.no_grad():
        n_out = model(**n_inputs, output_hidden_states=True)
        c_out = model(**c_inputs, output_hidden_states=True)

    sensitivities = []
    for layer_idx in range(1, len(n_out.hidden_states)):
        h_n = n_out.hidden_states[layer_idx].squeeze(0)[-64:].float()
        h_c = c_out.hidden_states[layer_idx].squeeze(0)[-64:].float()
        nn = min(h_n.shape[0], h_c.shape[0])
        a = h_c[-nn:]
        b = h_n[-nn:]
        try:
            _, S_c, _ = torch.linalg.svd(a, full_matrices=False)
            _, S_n, _ = torch.linalg.svd(b, full_matrices=False)
            k = min(32, len(S_c), len(S_n))
            p = S_c[:k].cpu().numpy()
            q = S_n[:k].cpu().numpy()
            p = p / (p.sum() + 1e-10)
            q = q / (q.sum() + 1e-10)
            from scipy.stats import entropy
            sensitivities.append(float(entropy(p + 1e-10, q + 1e-10)))
        except Exception:
            sensitivities.append(0.0)

    indexed = [(s, i) for i, s in enumerate(sensitivities)]
    indexed.sort()
    return sorted([i for _, i in indexed[:7]])


MODELS = [
    ("microsoft/phi-2", "sorter", "MHA"),
    ("EleutherAI/pythia-6.9b", "tunnel", "MHA"),
]


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

        zone = identify_zone(model, tokenizer)
        print(f"  Zone: {zone}")

        extractor = ComponentExtractor(model)
        extractor.register()

        neutral_layers = run_with_hooks(model, tokenizer, extractor, NEUTRAL, PROBE)
        n_layers = len(neutral_layers)

        model_results = {"zone": zone, "conditions": {}}

        print(f"\n  {'Cond':>15} {'z_cos':>8} {'o_cos':>8} {'z_attn':>8} {'z_mlp':>8}")

        for cond_name, ccs_text in CONDITIONS.items():
            layers = run_with_hooks(model, tokenizer, extractor, ccs_text, PROBE)

            z_cos = np.mean([layers[i]["attn_mlp_cos"] for i in zone if i < n_layers])
            o_cos = np.mean([layers[i]["attn_mlp_cos"] for i in range(n_layers) if i not in zone])
            z_attn = np.mean([layers[i]["attn_norm"] for i in zone if i < n_layers])
            z_mlp = np.mean([layers[i]["mlp_norm"] for i in zone if i < n_layers])

            model_results["conditions"][cond_name] = {
                "zone_cos": float(z_cos),
                "outside_cos": float(o_cos),
                "zone_attn_norm": float(z_attn),
                "zone_mlp_norm": float(z_mlp),
                "per_layer_cos": [layers[i]["attn_mlp_cos"] for i in range(n_layers)],
            }

            print(f"  {cond_name:>15} {z_cos:8.4f} {o_cos:8.4f} {z_attn:8.1f} {z_mlp:8.1f}")

        # Correlation
        x = list(range(5))
        z_cos_vals = [model_results["conditions"][c]["zone_cos"] for c in CONDITIONS.keys()]
        r, p = sp_stats.pearsonr(x, z_cos_vals)
        model_results["zone_cos_correlation"] = {"r": float(r), "p": float(p)}
        print(f"\n  Zone cos vs label density: r={r:+.3f}, p={p:.4f}")
        if p < 0.05:
            print(f"  >>> SIGNIFICANT: cooperation {'increases' if r > 0 else 'decreases'} with label density <<<")
        else:
            print(f"  >>> NOT SIGNIFICANT (p={p:.3f}) <<<")

        all_model_results[model_id] = {"species": species, "attn": attn_type, **model_results}

        extractor.remove()
        del model, extractor
        torch.cuda.empty_cache()
        gc.collect()

    # Cross-species comparison
    print("\n" + "=" * 70)
    print("CROSS-SPECIES COOPERATION COMPARISON")
    print("=" * 70)

    qwen_r, qwen_p = 0.895, 0.040  # From Test 16
    print(f"\n  Qwen (relay/GQA): r={qwen_r:+.3f}, p={qwen_p:.4f} — SIGNIFICANT")
    for model_id, data in all_model_results.items():
        corr = data["zone_cos_correlation"]
        sig = "SIGNIFICANT" if corr["p"] < 0.05 else "not significant"
        print(f"  {data['species']:>8} ({data['attn']}) — {model_id}: r={corr['r']:+.3f}, p={corr['p']:.4f} — {sig}")

    class NE(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.floating, np.integer)):
                return float(o) if isinstance(o, np.floating) else int(o)
            return super().default(o)

    with open("/workspace/cross_species_cooperation_results.json", "w") as f:
        json.dump(all_model_results, f, indent=2, cls=NE)
    print("\nResults saved to /workspace/cross_species_cooperation_results.json")


if __name__ == "__main__":
    main()
