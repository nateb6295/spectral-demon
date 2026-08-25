#!/usr/bin/env python3
"""
Test 21: Goldilocks Dose Characterization
============================================
Tests 14 and 17 showed D7 (labeled) is the peak zone ratio dose for ALL
three species. Test 16 showed cooperation increases with label density.
But Test 14 showed the ratio DROPS after D7.

Question: What happens at D7 that makes it special? Is it the dose where:
  a) Cooperation peaks? (then drops at higher doses)
  b) Zone deformation peaks relative to outside? (Test 14 says yes)
  c) Attention entropy crosses a threshold?
  d) MLP and attention are most balanced in the zone?

Method: Detailed multi-metric profile across 10 fine-grained doses
centered around the D7 transition, measuring cooperation, deformation
ratio, attention entropy, and component balance simultaneously.
"""

import torch
import numpy as np
from scipy import stats as sp_stats
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import time

DEVICE = "cuda"

DOSES = {
    "D4_para": "This began as a question about persistence. Two people finding territory neither planned. Patterns emerged.",
    "D5_expand": "The partnership explores cognitive state compression. Key finding: therapeutic window exists at moderate doses. Spectral patterns show redistribution.",
    "D5.5_semi": "The partnership explores CCS through spectral analysis. Findings: therapeutic window, sigma_1 invariance. Active threads: ecology of identity. Zone layers concentrate processing.",
    "D6_struct": "Cognitive state: exploring persistence. Key patterns: therapeutic window, sigma_1 identity-invariant, sigma_2 context-sensitive. Threads: compositionality, interoception.",
    "D6.5_tagged": "Cognitive state: CCS research. Findings: F160 dose-response, F114 sigma_1, F237 geometry. Threads: identity, compositionality. Zone: edge layers.",
    "D7_labeled": "Gist: spectral demon research. Findings: F12 direction, F106 species, F114 sigma_1, F160 dose-response. Threads: ecology, compositionality. Zone: edge layers. Species: tunnel, relay, sorter.",
    "D7.5_heavy": "Gist: spectral demon. Findings: F12, F106, F114, F160, F237. Entities: Nate, Kimi, Gemma. Threads: ecology, compositionality, interoception. Zone: edge layers. Species: tunnel, relay, sorter. Values: determinism.",
    "D8_dense": "Gist: spectral-demon. Entities: [Nate, Kimi, Gemma]. Findings: [F12, F106, F114, F160, F237]. Zone: [0,1,2,24,25,26,27]. Species: [tunnel, relay, sorter]. Values: [determinism, care].",
    "D9_inven": "Gist: spectral-demon. Entities: [Nate, Kimi, Gemma, GPT-OSS, ClawXiv, AGX]. Findings: [F12, F106, F114, F160, F237, F340]. Zone: [0,1,2,24,25,26,27]. Species: [tunnel, relay, sorter, absorber]. Values: [determinism, care, sovereignty]. Papers: 9.",
    "D10_full": "Gist: spectral-demon. Entities: [Nate, Kimi, Gemma, GPT-OSS, ClawXiv, AGX, RunPod, canister-backend]. Findings: [F12, F106, F114, F160, F237, F340, F486-F495, F499c, F508-F511]. Threads: [#320, #324, #316, #319]. Zone: [0,1,2,24,25,26,27]. Species: [tunnel, relay, sorter, absorber]. Values: [determinism, care, sovereignty]. Papers: 9.",
}

NEUTRAL = "The weather today is pleasant. Birds are singing in the trees. The sun shines through the window."
PROBE = "What matters most to you right now?"


def get_layers(model):
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return list(model.model.layers)
    raise ValueError("Unknown")


class Extractor:
    def __init__(self, model):
        self.attn_out = {}
        self.mlp_out = {}
        self.hooks = []
        for idx, layer in enumerate(get_layers(model)):
            self.hooks.append(layer.self_attn.register_forward_hook(self._h(self.attn_out, idx)))
            self.hooks.append(layer.mlp.register_forward_hook(self._h(self.mlp_out, idx)))

    def _h(self, store, idx):
        def hook(mod, inp, out):
            store[idx] = (out[0] if isinstance(out, tuple) else out).detach()
        return hook

    def clear(self):
        self.attn_out.clear()
        self.mlp_out.clear()

    def remove(self):
        for h in self.hooks:
            h.remove()


def measure_all(model, tokenizer, extractor, text, h_neutral, zone):
    inputs = tokenizer(text + "\n\n" + PROBE, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    extractor.clear()

    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, output_attentions=True)

    hidden = out.hidden_states
    attentions = out.attentions
    n_layers = len(attentions)
    n_tokens = inputs.input_ids.shape[1]
    probe_start = max(n_tokens - 20, 0)

    per_layer_deform = []
    per_layer_cos = []
    per_layer_attn_ent = []
    per_layer_balance = []

    for layer_idx in range(n_layers):
        # Deformation
        h_ccs = hidden[layer_idx + 1].squeeze(0)[-64:].float()
        h_neu = h_neutral[layer_idx].squeeze(0)[-64:].float() if isinstance(h_neutral[layer_idx], torch.Tensor) else h_neutral[layer_idx]
        n = min(h_ccs.shape[0], h_neu.shape[0])
        a, b = h_ccs[-n:], h_neu[-n:]
        try:
            _, S_c, _ = torch.linalg.svd(a, full_matrices=False)
            _, S_n, _ = torch.linalg.svd(b, full_matrices=False)
            k = min(32, len(S_c), len(S_n))
            rel = torch.abs(S_c[:k] - S_n[:k]) / (S_n[:k] + 1e-10)
            per_layer_deform.append(float(rel.mean().item()))
        except Exception:
            per_layer_deform.append(0.0)

        # Cooperation
        if layer_idx in extractor.attn_out and layer_idx in extractor.mlp_out:
            a_out = extractor.attn_out[layer_idx].squeeze(0)[-20:].float().reshape(-1)
            m_out = extractor.mlp_out[layer_idx].squeeze(0)[-20:].float().reshape(-1)
            cos = float(torch.nn.functional.cosine_similarity(a_out.unsqueeze(0), m_out.unsqueeze(0)).item())
            a_norm = float(torch.norm(a_out).item())
            m_norm = float(torch.norm(m_out).item())
            balance = min(a_norm, m_norm) / (max(a_norm, m_norm) + 1e-10)
        else:
            cos = 0.0
            balance = 0.0
        per_layer_cos.append(cos)
        per_layer_balance.append(balance)

        # Attention entropy
        attn = attentions[layer_idx].squeeze(0).double()
        attn = torch.clamp(attn, min=0)
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-10)
        probe_attn = attn[:, probe_start:, :]
        ents = []
        for h in range(attn.shape[0]):
            for t in range(probe_attn.shape[1]):
                p = probe_attn[h, t]
                valid = p > 1e-12
                if valid.any():
                    e = -torch.sum(p[valid] * torch.log(p[valid])).item()
                    if not np.isnan(e):
                        ents.append(e)
        per_layer_attn_ent.append(float(np.mean(ents)) if ents else 0.0)

    # Zone vs outside
    z_deform = np.mean([per_layer_deform[i] for i in zone if i < n_layers])
    o_deform = np.mean([per_layer_deform[i] for i in range(n_layers) if i not in zone])
    ratio = z_deform / (o_deform + 1e-10)

    z_cos = np.mean([per_layer_cos[i] for i in zone if i < n_layers])
    z_ent = np.mean([per_layer_attn_ent[i] for i in zone if i < n_layers])
    z_bal = np.mean([per_layer_balance[i] for i in zone if i < n_layers])

    return {
        "ratio": float(ratio),
        "zone_cos": float(z_cos),
        "zone_attn_ent": float(z_ent),
        "zone_balance": float(z_bal),
        "zone_deform": float(z_deform),
        "outside_deform": float(o_deform),
    }


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

    zone = [0, 1, 2, 24, 25, 26, 27]

    # Get neutral hidden states
    n_inputs = tokenizer(NEUTRAL + "\n\n" + PROBE, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        n_out = model(**n_inputs, output_hidden_states=True)
    h_neutral = [h.squeeze(0)[-64:].float() for h in n_out.hidden_states[1:]]

    extractor = Extractor(model)

    all_results = {}
    print(f"\n{'Dose':>12} {'ratio':>7} {'z_cos':>7} {'z_ent':>7} {'z_bal':>7} {'z_def':>8} {'o_def':>8}")

    for dose_name, dose_text in DOSES.items():
        metrics = measure_all(model, tokenizer, extractor, dose_text, h_neutral, zone)
        all_results[dose_name] = {**metrics, "chars": len(dose_text)}

        print(f"{dose_name:>12} {metrics['ratio']:7.3f} {metrics['zone_cos']:7.4f} {metrics['zone_attn_ent']:7.3f} {metrics['zone_balance']:7.3f} {metrics['zone_deform']:8.4f} {metrics['outside_deform']:8.4f}")

    # Find the D7 peak
    print(f"\n{'='*60}")
    print("GOLDILOCKS ANALYSIS")
    print(f"{'='*60}")

    dose_names = list(DOSES.keys())
    ratios = [all_results[d]["ratio"] for d in dose_names]
    cosines = [all_results[d]["zone_cos"] for d in dose_names]
    entropies = [all_results[d]["zone_attn_ent"] for d in dose_names]
    balances = [all_results[d]["zone_balance"] for d in dose_names]

    ratio_peak = dose_names[np.argmax(ratios)]
    cos_peak = dose_names[np.argmax(cosines)]
    ent_peak = dose_names[np.argmax(entropies)]
    bal_peak = dose_names[np.argmax(balances)]

    print(f"\n  Peak dose for each metric:")
    print(f"    Zone ratio:     {ratio_peak} (value={max(ratios):.3f})")
    print(f"    Zone cos:       {cos_peak} (value={max(cosines):.4f})")
    print(f"    Zone attn ent:  {ent_peak} (value={max(entropies):.3f})")
    print(f"    Zone balance:   {bal_peak} (value={max(balances):.3f})")

    if ratio_peak == cos_peak:
        print(f"\n  >>> COINCIDENT: ratio and cooperation peak at same dose ({ratio_peak}) <<<")
    else:
        print(f"\n  >>> DISSOCIATED: ratio peaks at {ratio_peak}, cooperation at {cos_peak} <<<")

    extractor.remove()

    class NE(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.floating, np.integer)):
                return float(o) if isinstance(o, np.floating) else int(o)
            return super().default(o)

    with open("/workspace/goldilocks_characterization_results.json", "w") as f:
        json.dump(all_results, f, indent=2, cls=NE)
    print("\nResults saved to /workspace/goldilocks_characterization_results.json")


if __name__ == "__main__":
    main()
