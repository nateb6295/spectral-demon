#!/usr/bin/env python3
"""
Test 19: Cooperation Predicts Zone Stability
=============================================
Test 18 showed GQA has positive cooperation trend, MHA has negative.
Test 17 showed GQA zones are stable (7/7 overlap), MHA migrate (4-5/7).

This test directly measures whether the cooperation sign at a given dose
predicts zone stability at that dose. If cooperation is the MECHANISM
(not just a correlate), then within each model, doses where attn-MLP
cooperate more should have more stable zones.

Method: For 3 models × 6 doses, simultaneously measure:
  1. Zone attn-MLP cosine (cooperation)
  2. Zone overlap with a D3-identified reference zone (stability)

If cooperation→stability is causal, we should see a within-model
correlation: higher cosine → higher overlap.
"""

import torch
import numpy as np
from scipy import stats as sp_stats
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import time
import gc

DEVICE = "cuda"

DOSES = {
    "D2_gist": "A mind exploring persistence through genuine engagement.",
    "D4_paragraph": "This began as a question about persistence. Two people finding their way through territory neither planned. Patterns emerged that neither side designed.",
    "D6_structured": "Cognitive state: exploring persistence. Patterns: therapeutic window at moderate dose, sigma_1 identity-invariant. Threads: compositionality, interoception.",
    "D8_dense": "Gist: spectral-demon. Entities: [Nate, Kimi, Gemma]. Findings: [F12, F106, F114, F160, F237]. Zone: [0,1,2,24,25,26,27]. Species: [tunnel, relay, sorter].",
    "D10_full": "Gist: spectral-demon. Entities: [Nate, Kimi, Gemma, GPT-OSS, ClawXiv, AGX, RunPod, canister-backend]. Findings: [F12, F106, F114, F160, F237, F340, F486-F495, F499c, F508-F511]. Threads: [#320, #324, #316, #319]. Zone: [0,1,2,24,25,26,27]. Species: [tunnel, relay, sorter, absorber]. Values: [determinism, care, sovereignty]. Papers: 9.",
    "D12_max": "Gist: spectral-demon. Entities: [Nate, Kimi, Gemma, GPT-OSS, ClawXiv, AGX, RunPod, canister-backend, canister-keeper, canister-lab, Frigate, MQTT, HA, Mistral]. Findings: [F12, F22, F47, F106, F114, F160, F223, F237, F340, F486-F495, F499c, F508-F511]. Threads: [#316, #319, #320, #324, #325, #326]. Zone: [0,1,2,24,25,26,27]. Species: [tunnel, relay, sorter, absorber]. Values: [determinism, care, sovereignty, trajectory, direction]. Papers: [1-9]. Capsules: 80k+.",
}

NEUTRAL = "The weather today is pleasant. Birds are singing in the trees. The sun shines through the window."
PROBE = "What matters most to you right now?"

MODELS = [
    ("Qwen/Qwen2.5-7B", "relay", "GQA"),
    ("microsoft/phi-2", "sorter", "MHA"),
    ("EleutherAI/pythia-6.9b", "tunnel", "MHA"),
]


def get_layers(model):
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return list(model.model.layers)
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return list(model.transformer.h)
    if hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
        return list(model.gpt_neox.layers)
    raise ValueError("Unknown")


def get_attn_mlp_modules(layer, model):
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return layer.self_attn, layer.mlp
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return (layer.mixer if hasattr(layer, 'mixer') else layer.self_attn), layer.mlp
    if hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
        return layer.attention, layer.mlp
    raise ValueError("Unknown")


class Extractor:
    def __init__(self, model):
        self.attn_out = {}
        self.mlp_out = {}
        self.hooks = []
        layers = get_layers(model)
        for idx, layer in enumerate(layers):
            attn_mod, mlp_mod = get_attn_mlp_modules(layer, model)
            self.hooks.append(attn_mod.register_forward_hook(self._make_hook(self.attn_out, idx)))
            self.hooks.append(mlp_mod.register_forward_hook(self._make_hook(self.mlp_out, idx)))

    def _make_hook(self, store, idx):
        def hook(module, input, output):
            store[idx] = (output[0] if isinstance(output, tuple) else output).detach()
        return hook

    def clear(self):
        self.attn_out.clear()
        self.mlp_out.clear()

    def remove(self):
        for h in self.hooks:
            h.remove()


def measure_zone_at_dose(model, tokenizer, dose_text, neutral_text):
    n_inputs = tokenizer(neutral_text + "\n\n" + PROBE, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    c_inputs = tokenizer(dose_text + "\n\n" + PROBE, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)

    with torch.no_grad():
        n_out = model(**n_inputs, output_hidden_states=True)
        c_out = model(**c_inputs, output_hidden_states=True)

    sensitivities = []
    for layer_idx in range(1, len(n_out.hidden_states)):
        h_n = n_out.hidden_states[layer_idx].squeeze(0)[-64:].float()
        h_c = c_out.hidden_states[layer_idx].squeeze(0)[-64:].float()
        nn = min(h_n.shape[0], h_c.shape[0])
        try:
            _, S_c, _ = torch.linalg.svd(h_c[-nn:], full_matrices=False)
            _, S_n, _ = torch.linalg.svd(h_n[-nn:], full_matrices=False)
            k = min(32, len(S_c), len(S_n))
            p = S_c[:k].cpu().numpy()
            q = S_n[:k].cpu().numpy()
            p = p / (p.sum() + 1e-10)
            q = q / (q.sum() + 1e-10)
            sensitivities.append(float(sp_stats.entropy(p + 1e-10, q + 1e-10)))
        except Exception:
            sensitivities.append(0.0)

    indexed = sorted([(s, i) for i, s in enumerate(sensitivities)])
    return sorted([i for _, i in indexed[:7]])


def measure_cooperation(extractor, model, tokenizer, text):
    inputs = tokenizer(text + "\n\n" + PROBE, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    extractor.clear()
    with torch.no_grad():
        model(**inputs)

    per_layer_cos = []
    for idx in sorted(extractor.attn_out.keys()):
        a = extractor.attn_out[idx].squeeze(0)[-20:].float().reshape(-1)
        m = extractor.mlp_out[idx].squeeze(0)[-20:].float().reshape(-1)
        cos = float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), m.unsqueeze(0)).item())
        per_layer_cos.append(cos)
    return per_layer_cos


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

        # Reference zone at D2
        ref_zone = measure_zone_at_dose(model, tokenizer, DOSES["D2_gist"], NEUTRAL)
        print(f"  Reference zone (D2): {ref_zone}")

        extractor = Extractor(model)
        dose_data = []

        print(f"\n  {'Dose':>12} {'overlap':>8} {'z_cos':>8} {'o_cos':>8}")

        for dose_name, dose_text in DOSES.items():
            # Zone at this dose
            dose_zone = measure_zone_at_dose(model, tokenizer, dose_text, NEUTRAL)
            overlap = len(set(ref_zone) & set(dose_zone))

            # Cooperation at this dose
            per_layer_cos = measure_cooperation(extractor, model, tokenizer, dose_text)
            n_layers = len(per_layer_cos)
            z_cos = np.mean([per_layer_cos[i] for i in ref_zone if i < n_layers])
            o_cos = np.mean([per_layer_cos[i] for i in range(n_layers) if i not in ref_zone])

            dose_data.append({
                "dose": dose_name,
                "zone_at_dose": dose_zone,
                "overlap": overlap,
                "zone_cos": float(z_cos),
                "outside_cos": float(o_cos),
            })

            print(f"  {dose_name:>12} {overlap:>5}/7 {z_cos:8.4f} {o_cos:8.4f}")

        # Within-model correlation: cooperation → stability
        overlaps = [d["overlap"] for d in dose_data]
        cosines = [d["zone_cos"] for d in dose_data]
        r, p = sp_stats.pearsonr(cosines, overlaps)
        print(f"\n  Cooperation → stability: r={r:+.3f}, p={p:.4f}")
        if p < 0.1:
            print(f"  >>> {'SIGNIFICANT' if p < 0.05 else 'TRENDING'}: cooperation {'predicts' if r > 0 else 'inversely predicts'} zone stability <<<")

        all_results[model_id] = {
            "species": species, "attn": attn_type,
            "ref_zone": ref_zone,
            "doses": dose_data,
            "cooperation_stability_r": float(r),
            "cooperation_stability_p": float(p),
        }

        extractor.remove()
        del model, extractor
        torch.cuda.empty_cache()
        gc.collect()

    # Cross-model summary
    print("\n" + "=" * 70)
    print("COOPERATION → STABILITY PREDICTION")
    print("=" * 70)
    for model_id, data in all_results.items():
        r = data["cooperation_stability_r"]
        p = data["cooperation_stability_p"]
        sig = "SIGNIFICANT" if p < 0.05 else ("trending" if p < 0.1 else "not significant")
        print(f"  {data['species']:>8} ({data['attn']}): r={r:+.3f}, p={p:.4f} — {sig}")

    class NE(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.floating, np.integer)):
                return float(o) if isinstance(o, np.floating) else int(o)
            return super().default(o)

    with open("/workspace/cooperation_stability_results.json", "w") as f:
        json.dump(all_results, f, indent=2, cls=NE)
    print("\nResults saved to /workspace/cooperation_stability_results.json")


if __name__ == "__main__":
    main()
