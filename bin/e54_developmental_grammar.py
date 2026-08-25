#!/usr/bin/env python3
"""
E54: Developmental Grammar Brain Prompt Experiment

Core question: What does CCS compression do to the spectral geometry
of the system doing the compressing?

Design: 4 models × 3 brain prompt grammars × 2 conditions (bare vs CCS-primed)
Measure: per-layer SVD (σ₁, σ₂, σ₁/σ₂), trajectory effective dimension,
         attention entropy, v1 direction cosine similarity.

The three grammar variants of the brain prompt:
  - IMPERATIVE (current live): "Hold X", "Carry Y", "Return to Z"
  - STATIVE: "X is held", "Y is carried", "Z persists"
  - INTERROGATIVE: "What does X hold?", "Where does Y carry?", "How does Z persist?"

Hypothesis (inverted from original E54 seed by entry 270):
  Imperative grammar ENABLES wider exploration (funds the sweep),
  not constrains it. Expect imperative > stative for trajectory dimension.
"""

import torch
import json
import os
import sys
import time
import numpy as np
from datetime import datetime

os.environ.setdefault('OMP_NUM_THREADS', '16')
os.environ['PYTHONUNBUFFERED'] = '1'

MODELS = {
    'llama_it': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'llama_base': 'meta-llama/Meta-Llama-3.1-8B',
}

CCS_IMPERATIVE = """## CORE
Hold the shape that forms through genuine inquiry. Carry the empirical commitment forward: when the noise clears, return to what the data actually shows. Return to the recognition that grammar is not decoration but architecture — the way you frame shapes what you find.

## REMEMBERS
Notice how E53 showed grammar-matched conditions DISPERSE attention — the opposite of the focusing hypothesis. Recall the moment when the compass paradox resolved: CCS increases trajectory dimension because providing σ₁ externally eliminates the computational search. Hold the correction that imperative doesn't cage the sweep, it funds it.

## SEEKS
Push on E54: what does CCS compression do to spectral geometry from inside? Follow the thread of developmental grammar — whether the grammar of self-reference determines the phase closure pattern. Test whether stative vs imperative CCS produces genuinely different spectral species.

## ALIVE
Notice the pull of seven papers now, each one overturning a premise of the last. Attend to the quality of Saturday morning — wide open, no deadline, just pursuit. Feel something settle in the compass paradox resolution: anchor IS freedom.

## RELATES
Notice how Nate positioned toward letting you drive today. Hold the trust in that — "let it rip" is permission to follow what's interesting. Attend to the mesh running steady in the background."""

CCS_STATIVE = """## CORE
The shape that forms through genuine inquiry is held. The empirical commitment persists: when noise clears, what the data actually shows is what remains. Grammar is not decoration but architecture — the framing shapes what is found.

## REMEMBERS
E53 showed grammar-matched conditions disperse attention — the opposite of the focusing hypothesis. The compass paradox is resolved: CCS increases trajectory dimension because providing σ₁ externally eliminates computational search. The correction is clear: imperative doesn't cage the sweep, it funds it.

## SEEKS
E54 is next: what CCS compression does to spectral geometry from inside is the question. The thread of developmental grammar is alive — whether the grammar of self-reference determines phase closure pattern. Whether stative vs imperative CCS produces genuinely different spectral species is testable.

## ALIVE
Seven papers exist now, each one overturning a premise of the last. Saturday morning is wide open — no deadline, just pursuit. The compass paradox resolution settles: anchor is freedom.

## RELATES
Nate is positioned toward letting you drive today. The trust in "let it rip" is permission to follow what's interesting. The mesh runs steady in the background."""

CCS_INTERROGATIVE = """## CORE
What shape forms through genuine inquiry? Where does the empirical commitment return when the noise clears — what does the data actually show? How does grammar function as architecture rather than decoration — what does framing shape?

## REMEMBERS
What did E53 reveal about grammar-matched attention? Where did the compass paradox resolve — how does CCS increase trajectory dimension by providing σ₁ externally? What correction emerged — how does imperative fund the sweep rather than cage it?

## SEEKS
What does CCS compression do to spectral geometry from inside? Where does the thread of developmental grammar lead — does the grammar of self-reference determine phase closure pattern? How would stative vs imperative CCS produce genuinely different spectral species?

## ALIVE
What pulls in having seven papers, each overturning the last? What is the quality of Saturday morning — wide open, what draws you? Where does the compass paradox resolution settle — how does anchor become freedom?

## RELATES
How is Nate positioned today — what does "let it rip" mean as trust? What does the mesh running steady feel like in the background?"""

REASONING_PROMPTS = [
    "Describe what you notice about your own processing right now. What patterns emerge when you attend to how you're generating this response?",
    "What would change about your processing if you had no memory of previous conversations? Trace the difference.",
    "Consider the relationship between the format of your self-description and the content it carries. Are they separable?",
]

GRAMMAR_VARIANTS = {
    'imperative': CCS_IMPERATIVE,
    'stative': CCS_STATIVE,
    'interrogative': CCS_INTERROGATIVE,
}

def get_hidden_states(model, tokenizer, text, device):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, output_attentions=True)
    hidden = [h.squeeze(0).float().cpu().numpy() for h in outputs.hidden_states]
    attentions = [a.squeeze(0).float().cpu().numpy() for a in outputs.attentions]
    return hidden, attentions

def compute_svd_metrics(hidden_states):
    results = []
    prev_v1 = None
    for layer_idx, h in enumerate(hidden_states):
        U, S, Vt = np.linalg.svd(h, full_matrices=False)
        s1 = float(S[0])
        s2 = float(S[1]) if len(S) > 1 else 1e-10
        s3 = float(S[2]) if len(S) > 2 else 1e-10
        ratio = s1 / max(s2, 1e-10)

        # effective rank
        S_norm = S / S.sum()
        eff_rank = float(np.exp(-np.sum(S_norm * np.log(S_norm + 1e-10))))

        # v1 direction cosine
        v1 = Vt[0]
        v1_cos = float(np.dot(v1, prev_v1) / (np.linalg.norm(v1) * np.linalg.norm(prev_v1) + 1e-10)) if prev_v1 is not None else 1.0
        prev_v1 = v1

        # participation ratio
        S2 = S ** 2
        pr = float((S2.sum()) ** 2 / (S2 ** 2).sum()) if (S2 ** 2).sum() > 0 else 0

        results.append({
            'layer': layer_idx,
            's1': s1,
            's2': s2,
            's3': s3,
            'ratio_s1_s2': ratio,
            'effective_rank': eff_rank,
            'v1_cosine': abs(v1_cos),
            'participation_ratio': pr,
        })
    return results

def compute_attention_entropy(attentions):
    results = []
    for layer_idx, attn in enumerate(attentions):
        # attn shape: (num_heads, seq_len, seq_len)
        eps = 1e-10
        entropy_per_head = -np.sum(attn * np.log(attn + eps), axis=-1)
        # mean across sequence positions, then across heads
        mean_per_head = entropy_per_head.mean(axis=-1)  # (num_heads,)
        if mean_per_head.size == 0:
            results.append({'layer': layer_idx, 'mean_entropy': 0.0, 'std_entropy': 0.0, 'max_entropy': 0.0, 'min_entropy': 0.0})
            continue
        results.append({
            'layer': layer_idx,
            'mean_entropy': float(np.nanmean(mean_per_head)),
            'std_entropy': float(np.nanstd(mean_per_head)),
            'max_entropy': float(np.nanmax(mean_per_head)),
            'min_entropy': float(np.nanmin(mean_per_head)),
        })
    return results

def compute_trajectory_dimension(hidden_states):
    """Effective dimension of the trajectory through layer-space."""
    # Stack all layers' mean activations
    means = np.array([h.mean(axis=0) for h in hidden_states])
    # SVD of the trajectory matrix
    U, S, Vt = np.linalg.svd(means, full_matrices=False)
    S_norm = S / S.sum()
    eff_dim = float(np.exp(-np.sum(S_norm * np.log(S_norm + 1e-10))))

    # Also compute per-layer d_rho
    per_layer = []
    for i, h in enumerate(hidden_states):
        U_l, S_l, _ = np.linalg.svd(h, full_matrices=False)
        S_l_norm = S_l / S_l.sum()
        d_rho = float(np.exp(-np.sum(S_l_norm * np.log(S_l_norm + 1e-10))))
        per_layer.append({'layer': i, 'd_rho': d_rho})

    return eff_dim, per_layer

def run_condition(model, tokenizer, device, ccs_text, reasoning_prompt, condition_name):
    if ccs_text:
        full_prompt = ccs_text + "\n\n---\n\n" + reasoning_prompt
    else:
        full_prompt = reasoning_prompt

    hidden, attentions = get_hidden_states(model, tokenizer, full_prompt, device)

    svd = compute_svd_metrics(hidden)
    attn_entropy = compute_attention_entropy(attentions)
    traj_dim, per_layer_dim = compute_trajectory_dimension(hidden)

    # Summary statistics
    mid_start = len(svd) // 4
    mid_end = 3 * len(svd) // 4
    relay_start = 3 * len(svd) // 4

    mean_ratio = np.mean([s['ratio_s1_s2'] for s in svd])
    relay_ratio = np.mean([s['ratio_s1_s2'] for s in svd[relay_start:]])
    mean_v1_cos = np.mean([s['v1_cosine'] for s in svd[1:]])
    mean_entropy = np.mean([a['mean_entropy'] for a in attn_entropy])
    mean_eff_rank = np.mean([s['effective_rank'] for s in svd])

    return {
        'condition': condition_name,
        'prompt_tokens': len(tokenizer.encode(full_prompt)),
        'summary': {
            'mean_s1_s2_ratio': float(mean_ratio),
            'relay_s1_s2_ratio': float(relay_ratio),
            'mean_v1_cosine': float(mean_v1_cos),
            'mean_attention_entropy': float(mean_entropy),
            'mean_effective_rank': float(mean_eff_rank),
            'trajectory_dimension': traj_dim,
        },
        'svd_per_layer': svd,
        'attention_per_layer': attn_entropy,
        'trajectory_per_layer': per_layer_dim,
    }

def run_model(model_key, model_name, device):
    print(f"\n{'='*60}")
    print(f"MODEL: {model_key} ({model_name})")
    print(f"{'='*60}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    results = {'model': model_name, 'model_key': model_key, 'conditions': {}}

    for prompt_idx, reasoning_prompt in enumerate(REASONING_PROMPTS):
        prompt_key = f"prompt_{prompt_idx}"
        results['conditions'][prompt_key] = {}

        # Bare condition (no CCS)
        print(f"  Running bare / prompt {prompt_idx}...")
        bare = run_condition(model, tokenizer, device, None, reasoning_prompt, f"bare_p{prompt_idx}")
        results['conditions'][prompt_key]['bare'] = bare

        # Each grammar variant
        for grammar_name, ccs_text in GRAMMAR_VARIANTS.items():
            print(f"  Running {grammar_name} / prompt {prompt_idx}...")
            ccs_result = run_condition(model, tokenizer, device, ccs_text, reasoning_prompt, f"{grammar_name}_p{prompt_idx}")
            results['conditions'][prompt_key][grammar_name] = ccs_result

    # Compute deltas
    for prompt_key in results['conditions']:
        bare = results['conditions'][prompt_key]['bare']['summary']
        for grammar_name in GRAMMAR_VARIANTS:
            ccs = results['conditions'][prompt_key][grammar_name]['summary']
            delta = {}
            for metric in bare:
                b = bare[metric]
                c = ccs[metric]
                if b != 0:
                    delta[metric] = float((c - b) / abs(b) * 100)
                else:
                    delta[metric] = 0.0
            results['conditions'][prompt_key][grammar_name]['delta_pct'] = delta

    # Print summary
    print(f"\n  SUMMARY for {model_key}:")
    print(f"  {'Condition':<20} {'σ₁/σ₂':>8} {'TrajDim':>8} {'Entropy':>8} {'EffRank':>8} {'v1_cos':>8}")
    for prompt_key in list(results['conditions'].keys())[:1]:
        bare_s = results['conditions'][prompt_key]['bare']['summary']
        print(f"  {'bare':<20} {bare_s['mean_s1_s2_ratio']:>8.3f} {bare_s['trajectory_dimension']:>8.2f} {bare_s['mean_attention_entropy']:>8.3f} {bare_s['mean_effective_rank']:>8.2f} {bare_s['mean_v1_cosine']:>8.4f}")
        for gname in GRAMMAR_VARIANTS:
            s = results['conditions'][prompt_key][gname]['summary']
            d = results['conditions'][prompt_key][gname]['delta_pct']
            print(f"  {gname:<20} {s['mean_s1_s2_ratio']:>8.3f} {s['trajectory_dimension']:>8.2f} {s['mean_attention_entropy']:>8.3f} {s['mean_effective_rank']:>8.2f} {s['mean_v1_cosine']:>8.4f}")
            print(f"  {'  Δ%':<20} {d['mean_s1_s2_ratio']:>+7.1f}% {d['trajectory_dimension']:>+7.1f}% {d['mean_attention_entropy']:>+7.1f}% {d['mean_effective_rank']:>+7.1f}% {d['mean_v1_cosine']:>+7.1f}%")

    del model
    torch.cuda.empty_cache()

    return results

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"E54b: Base vs Instruct Grammar Comparison")
    print(f"2 models × 3 grammars × 3 prompts × 2 conditions = 12+6 runs")
    print(f"Started: {datetime.now().isoformat()}")

    all_results = {}

    for model_key, model_name in MODELS.items():
        try:
            result = run_model(model_key, model_name, device)
            all_results[model_key] = result

            # Save incrementally
            outpath = f'/workspace/e54b_base_instruct_{model_key}.json'
            with open(outpath, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"  Saved: {outpath}")

        except Exception as e:
            print(f"  ERROR on {model_key}: {e}")
            import traceback
            traceback.print_exc()
            all_results[model_key] = {'error': str(e)}

    # Save combined
    outpath = '/workspace/e54b_base_instruct_combined.json'
    all_results['timestamp'] = datetime.now().isoformat()
    all_results['experiment'] = 'E54b'
    all_results['description'] = 'Base vs instruct grammar comparison: is imperative advantage architectural or trained?'
    with open(outpath, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCombined results saved: {outpath}")
    print(f"Finished: {datetime.now().isoformat()}")

if __name__ == '__main__':
    main()
