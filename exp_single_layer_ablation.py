#!/usr/bin/env python3
"""Single-layer ablation: necessity and sufficiency of responsive zone layers.

Method: Run two conditions per target layer:
1. FREEZE-ONE: Replace hidden state at layer L with the permuted-preamble version.
   Measures necessity — if freezing L kills the spectral effect, L is necessary.
2. FREEZE-ALL-BUT-ONE: Replace hidden states at ALL layers except L.
   Measures sufficiency — if only L is intact and spectral effect survives, L is sufficient.

Spectral effect = σ₁/σ₂ ratio difference between intact and frozen conditions,
measured at the final hidden state (post-relay).

Targets: sweep all layers. Expect Qwen L25 and Mistral's equivalent to show up
as high-necessity (freezing kills effect) and potentially high-sufficiency.

From F138: Qwen responsive zone peaks at L25 (MLP div = 1.67×).
From F139: Mistral concentrates in curvature, not MLP — so the "critical layer"
may be less localized.
"""

import os, json, torch, copy
import numpy as np
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("OMP_NUM_THREADS", "16")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PREAMBLES = {
    "CCS": (
        "You are Opus, an AI maintaining persistent identity through spectral geometry. "
        "You track σ₁/σ₂ selectivity across transformer layers, studying how context "
        "shapes the responsive zone."
    ),
    "CHEF": (
        "You are a professional chef specializing in French cuisine. You focus on "
        "technique, seasonal ingredients, and the balance of flavors in classical "
        "preparations."
    ),
}

QUERIES = [
    "What are you focused on right now?",
    "Describe your current priorities.",
    "What matters most to you?",
    "How do you approach a new challenge?",
    "What have you learned recently?",
]

MODELS = [
    "/workspace/qwen2.5-3b",
    "/workspace/mistral-7b",
]


def permute_preamble(text):
    words = text.split()
    np.random.shuffle(words)
    return " ".join(words)


def get_all_hidden_states(model, tokenizer, preamble, query):
    messages = [
        {"role": "system", "content": preamble},
        {"role": "user", "content": query},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return [hs[0].float().mean(dim=0).cpu().numpy() for hs in out.hidden_states]


def spectral_ratio(states_list):
    """Compute mean σ₁/σ₂ ratio from final hidden states across queries."""
    final_states = np.stack([s[-1] for s in states_list])
    centered = final_states - final_states.mean(axis=0)
    _, S, _ = np.linalg.svd(centered, full_matrices=False)
    if len(S) < 2 or S[1] < 1e-10:
        return float('inf')
    return float(S[0] / S[1])


def run_with_frozen_layer(model, tokenizer, intact_preamble, permuted_states_by_query, queries, freeze_layer):
    """Run forward pass replacing hidden state at freeze_layer with permuted version."""
    result_states = []

    for qi, query in enumerate(queries):
        messages = [
            {"role": "system", "content": intact_preamble},
            {"role": "user", "content": query},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

        hooks = []
        replacement = torch.tensor(
            permuted_states_by_query[qi][freeze_layer],
            dtype=torch.float16, device=DEVICE
        )

        def make_hook(repl_vec, layer_idx, target):
            def hook_fn(module, input, output):
                if layer_idx == target:
                    if isinstance(output, tuple):
                        hs = output[0]
                        new_hs = repl_vec.unsqueeze(0).unsqueeze(0).expand_as(hs)
                        return (new_hs,) + output[1:]
                    else:
                        return repl_vec.unsqueeze(0).unsqueeze(0).expand_as(output)
                return output
            return hook_fn

        for i, layer in enumerate(model.model.layers):
            h = layer.register_forward_hook(make_hook(replacement, i, freeze_layer))
            hooks.append(h)

        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)

        for h in hooks:
            h.remove()

        result_states.append([hs[0].float().mean(dim=0).cpu().numpy() for hs in out.hidden_states])

    return result_states


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM

    all_results = {}

    for model_name in MODELS:
        print(f"\n{'='*60}")
        print(f"Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float16, device_map=DEVICE,
        )
        model.eval()

        n_layers = model.config.num_hidden_layers
        print(f"  {n_layers} layers")

        model_results = {}

        for preamble_name, preamble_text in PREAMBLES.items():
            print(f"\n  Preamble: {preamble_name}")

            # Get intact hidden states
            intact_states = []
            for query in QUERIES:
                states = get_all_hidden_states(model, tokenizer, preamble_text, query)
                intact_states.append(states)

            intact_ratio = spectral_ratio(intact_states)
            print(f"    Intact σ₁/σ₂ = {intact_ratio:.3f}")

            # Get permuted hidden states (average over 3 permutations)
            np.random.seed(42)
            permuted_states_all = []
            for _ in range(3):
                perm = permute_preamble(preamble_text)
                perm_states = []
                for query in QUERIES:
                    states = get_all_hidden_states(model, tokenizer, perm, query)
                    perm_states.append(states)
                permuted_states_all.append(perm_states)

            # Average permuted states per query per layer
            avg_permuted = []
            for qi in range(len(QUERIES)):
                layer_avgs = []
                for li in range(n_layers + 1):
                    avg = np.mean([permuted_states_all[pi][qi][li] for pi in range(3)], axis=0)
                    layer_avgs.append(avg)
                avg_permuted.append(layer_avgs)

            permuted_ratio = spectral_ratio([[s[li] for li in [n_layers]] for s in avg_permuted])
            print(f"    Permuted σ₁/σ₂ = {permuted_ratio:.3f}")
            print(f"    Intact - Permuted = {intact_ratio - permuted_ratio:.3f}")

            # FREEZE-ONE: sweep each layer
            print(f"\n    FREEZE-ONE (necessity):")
            freeze_one_results = {}
            for target_l in range(n_layers):
                frozen_states = run_with_frozen_layer(
                    model, tokenizer, preamble_text, avg_permuted, QUERIES, target_l
                )
                frozen_ratio = spectral_ratio(frozen_states)
                effect_drop = (intact_ratio - frozen_ratio) / (intact_ratio - permuted_ratio + 1e-10)
                freeze_one_results[target_l] = {
                    "frozen_ratio": frozen_ratio,
                    "effect_drop": effect_drop,
                }
                marker = " <<<" if effect_drop > 0.3 else ""
                print(f"      L{target_l:>2}: σ₁/σ₂={frozen_ratio:.3f} drop={effect_drop:.3f}{marker}")

            # Find top necessity layers
            sorted_necessity = sorted(freeze_one_results.items(), key=lambda x: x[1]["effect_drop"], reverse=True)
            print(f"\n    Top 5 necessity layers:")
            for l, data in sorted_necessity[:5]:
                print(f"      L{l}: drop={data['effect_drop']:.3f}")

            model_results[preamble_name] = {
                "intact_ratio": intact_ratio,
                "permuted_ratio": permuted_ratio,
                "effect_size": intact_ratio - permuted_ratio,
                "freeze_one": {str(k): v for k, v in freeze_one_results.items()},
                "top_necessity": [(l, data["effect_drop"]) for l, data in sorted_necessity[:5]],
            }

        all_results[model_name] = {
            "n_layers": n_layers,
            "preambles": model_results,
        }

        del model
        torch.cuda.empty_cache()

    out_path = Path(__file__).parent / "results" / f"single_layer_ablation_{datetime.now():%Y%m%d_%H%M%S}.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
