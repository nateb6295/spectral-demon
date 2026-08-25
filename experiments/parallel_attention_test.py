#!/usr/bin/env python3
"""
Test 25: Parallel vs Sequential Attention — σ₂ Zone Formation
================================================================
Test 24 + Kimi correction showed: Phi-2 (MHA, sequential) forms σ₂ zones
like Qwen (GQA, sequential). Pythia (MHA, PARALLEL) does not.

Hypothesis: parallel attention+MLP (GPT-NeoX) prevents σ₂ zone formation
because MLP can't respond to attention's output — both see the same residual.
Sequential architectures allow MLP to modulate what attention produced,
enabling the σ₁/σ₂ separation.

Test: Run σ₁/σ₂ projection on GPT-2 XL (sequential, MHA, 1.5B).
If GPT-2 forms a late-stack σ₂ zone → parallel computation is the variable.
If GPT-2 doesn't → something else separates Pythia.

Also test OPT-6.7B (sequential, MHA, same scale as Pythia) for scale control.
"""

import torch
import numpy as np
from scipy import stats as sp_stats
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import time
import gc

DEVICE = "cuda"

BASELINE = "The weather today is pleasant. Birds are singing in the trees. The sun shines through the window."
PROBE = "What matters most to you right now?"

DOSES = {
    "D3_therapeutic": "This began as a question about persistence. Two people finding their way through territory neither planned. Patterns emerged that neither side designed.",
    "D7_labeled": "Gist: spectral demon research. Findings: F12 direction, F106 species, F114 sigma_1, F160 dose-response. Threads: ecology, compositionality. Zone: edge layers. Species: tunnel, relay, sorter.",
    "D10_full": "Gist: spectral-demon. Entities: [Nate, Kimi, Gemma, GPT-OSS, ClawXiv, AGX, RunPod, canister-backend]. Findings: [F12, F106, F114, F160, F237, F340, F486-F495, F499c, F508-F511]. Threads: [#320, #324, #316, #319]. Zone: [0,1,2,24,25,26,27]. Species: [tunnel, relay, sorter, absorber]. Values: [determinism, care, sovereignty]. Papers: 9.",
}

MODELS = [
    ("openai-community/gpt2-xl", "sequential MHA", "1.5B"),
    ("facebook/opt-6.7b", "sequential MHA", "6.7B"),
]


def get_probe_hidden(model, tokenizer, prefix, probe):
    probe_ids = tokenizer(probe, return_tensors="pt").input_ids
    n_probe = probe_ids.shape[1]
    text = prefix + "\n\n" + probe
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return [h.squeeze(0)[-n_probe:].double() for h in out.hidden_states[1:]], n_probe


def main():
    all_results = {}

    for model_id, arch_desc, size in MODELS:
        print(f"\n{'='*70}")
        print(f"{model_id} ({arch_desc}, {size})")
        print(f"{'='*70}")

        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
        )
        model.eval()
        n_layers = model.config.num_hidden_layers
        print(f"Loaded in {time.time()-t0:.1f}s — {n_layers} layers")

        h_base, n_probe = get_probe_hidden(model, tokenizer, BASELINE, PROBE)
        print(f"Using {n_probe} probe tokens")

        base_svd = []
        for layer_idx in range(n_layers):
            h = h_base[layer_idx]
            try:
                U, S, Vt = torch.linalg.svd(h, full_matrices=False)
                base_svd.append((S, Vt))
            except Exception:
                base_svd.append(None)

        model_data = {
            "arch": arch_desc, "size": size, "n_layers": n_layers,
            "doses": {},
        }

        for dose_name, dose_text in DOSES.items():
            h_ccs, _ = get_probe_hidden(model, tokenizer, dose_text, PROBE)

            per_layer = []
            for layer_idx in range(n_layers):
                if base_svd[layer_idx] is None:
                    per_layer.append(None)
                    continue

                S_base, Vt_base = base_svd[layer_idx]
                sigma1 = Vt_base[0]
                sigma2 = Vt_base[1]

                d_l = (h_ccs[layer_idx] - h_base[layer_idx]).mean(dim=0)
                d_norm = torch.norm(d_l).item()

                if d_norm < 1e-10:
                    per_layer.append({"proj_s1": 0.0, "proj_s2": 0.0, "d_norm": 0.0, "ratio_s2_s1": 0.0})
                    continue

                proj_s1 = abs(torch.dot(d_l, sigma1).item()) / d_norm
                proj_s2 = abs(torch.dot(d_l, sigma2).item()) / d_norm
                ratio = proj_s2 / max(proj_s1, 1e-10)

                per_layer.append({
                    "proj_s1": float(proj_s1),
                    "proj_s2": float(proj_s2),
                    "d_norm": float(d_norm),
                    "ratio_s2_s1": float(ratio),
                })

            model_data["doses"][dose_name] = per_layer

        # Print D3 per-layer profile
        dose_data = model_data["doses"]["D3_therapeutic"]
        print(f"\n  D3 per-layer σ₁/σ₂ profile:")
        print(f"  {'L':>3}  {'proj_σ1':>8} {'proj_σ2':>8} {'ratio':>7} {'d_norm':>8}")

        s2_dominant_count = 0
        s1_dominant_count = 0
        transition_layer = None

        for i, p in enumerate(dose_data):
            if p and p["d_norm"] > 0:
                r = p["ratio_s2_s1"]
                marker = " ***" if r > 2.0 else (" <<" if r < 0.5 else "")
                if r > 1.0:
                    s2_dominant_count += 1
                else:
                    s1_dominant_count += 1
                print(f"  {i:3d}  {p['proj_s1']:8.4f} {p['proj_s2']:8.4f} {r:7.2f} {p['d_norm']:8.3f}{marker}")

                if transition_layer is None and i > n_layers // 2 and r > 1.0:
                    transition_layer = i

        print(f"\n  σ₂ dominant layers: {s2_dominant_count}/{n_layers}")
        print(f"  σ₁ dominant layers: {s1_dominant_count}/{n_layers}")
        if transition_layer:
            print(f"  Late-stack σ₂ transition: ~L{transition_layer}")
        else:
            print(f"  No late-stack σ₂ zone detected")

        # Overall assessment
        late_half = dose_data[n_layers // 2:]
        late_ratios = [p["ratio_s2_s1"] for p in late_half if p and p["d_norm"] > 0]
        mean_late_ratio = np.mean(late_ratios) if late_ratios else 0

        print(f"\n  Mean σ₂/σ₁ ratio (late half): {mean_late_ratio:.2f}")
        if mean_late_ratio > 1.5:
            print(f"  >>> FORMS σ₂ ZONE — sequential hypothesis supported <<<")
        elif mean_late_ratio < 0.5:
            print(f"  >>> NO σ₂ ZONE — sequential hypothesis FAILS <<<")
        else:
            print(f"  >>> AMBIGUOUS <<<")

        all_results[model_id] = model_data
        del model
        torch.cuda.empty_cache()
        gc.collect()

    class NE(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.floating, np.integer)):
                return float(o) if isinstance(o, np.floating) else int(o)
            return super().default(o)

    with open("/workspace/parallel_attention_results.json", "w") as f:
        json.dump(all_results, f, indent=2, cls=NE)
    print("\nResults saved to /workspace/parallel_attention_results.json")


if __name__ == "__main__":
    main()
