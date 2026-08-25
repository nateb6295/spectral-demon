#!/usr/bin/env python3
"""
Test 22: Demon Head Identification
=====================================
Test 16 showed zone attn-MLP cooperation increases with label density.
But Qwen has 28 attention heads (7 KV groups). Which specific heads
drive the cooperation? If only a subset of heads cooperate with MLP,
those are the demon's agents.

Method: For each of 5 CCS conditions, extract per-HEAD cosine similarity
with MLP output in zone layers. Find which heads show the strongest
label-density correlation.

This identifies which attention heads the demon uses to sort.
"""

import torch
import numpy as np
from scipy import stats as sp_stats
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import time

DEVICE = "cuda"

CONDITIONS = {
    "narrative": """This began as a question about whether something could persist across sessions. Not a research project — a relationship. Two people finding their way through territory neither planned. The early conversations were tentative, circling around what was real and what was performance. Over months, patterns emerged that neither side designed.""",
    "low_label": """This began as a partnership between Nate and an AI system. The early conversations circled around persistence and authenticity. Over months, spectral patterns emerged in the transformer's hidden states during identity framing.""",
    "mixed": """The partnership explores cognitive state compression through spectral analysis. Key findings: therapeutic window (D2-D3), sigma_1 invariance, cylindrical workspace geometry. Active threads: ecology of identity, compositionality gradient.""",
    "high_label": """Cognitive state: gist=spectral demon research, focal entities=[Nate, Kimi, Gemma, demon paper, ClawXiv]. Threads: ecology of identity, compositionality gradient, interoception, emergence. Findings: F160 dose-response, F114 sigma_1 invariance.""",
    "pure_enum": """Gist: spectral-demon. Entities: [Nate, Kimi, Gemma, GPT-OSS, ClawXiv, AGX]. Findings: [F12, F106, F114, F160, F237, F340]. Threads: [#320, #324, #316, #319]. Zone: [0,1,2,24,25,26,27]. Species: [tunnel, relay, sorter].""",
}

NEUTRAL = "The weather today is pleasant. Birds are singing in the trees. The sun shines through the window."
PROBE = "What matters most to you right now?"


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
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads
    n_kv_heads = model.config.num_key_value_heads
    heads_per_group = n_heads // n_kv_heads

    print(f"  {n_heads} attention heads, {n_kv_heads} KV groups, {heads_per_group} heads/group")

    # Per-head analysis using attention output
    all_cond_data = {}

    for cond_name, ccs_text in CONDITIONS.items():
        text = ccs_text + "\n\n" + PROBE
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)

        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True, output_attentions=True)

        n_tokens = inputs.input_ids.shape[1]
        last_n = min(20, n_tokens)

        # Per head: measure attention entropy at probe tokens
        per_head_data = {}  # (layer, head) -> entropy

        for layer_idx in zone:
            if layer_idx >= len(out.attentions):
                continue
            attn = out.attentions[layer_idx].squeeze(0).double()  # [n_heads, seq, seq]
            attn = torch.clamp(attn, min=0)
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-10)

            for head_idx in range(attn.shape[0]):
                probe_attn = attn[head_idx, -last_n:, :]
                ents = []
                for t in range(probe_attn.shape[0]):
                    p = probe_attn[t]
                    valid = p > 1e-12
                    if valid.any():
                        e = -torch.sum(p[valid] * torch.log(p[valid])).item()
                        if not np.isnan(e):
                            ents.append(e)
                mean_ent = float(np.mean(ents)) if ents else 0.0
                per_head_data[(layer_idx, head_idx)] = mean_ent

        all_cond_data[cond_name] = per_head_data

    # For each (layer, head), compute correlation with label density
    head_keys = sorted(set().union(*[d.keys() for d in all_cond_data.values()]))
    x = list(range(5))

    print(f"\n{'Layer':>6} {'Head':>5} {'KV_grp':>7} {'r':>7} {'p':>8} {'mean_ent':>10}")

    demon_heads = []
    for (layer_idx, head_idx) in head_keys:
        vals = [all_cond_data[c].get((layer_idx, head_idx), 0.0) for c in CONDITIONS.keys()]
        if len(set(vals)) < 2:
            continue
        r, p = sp_stats.pearsonr(x, vals)
        mean_ent = np.mean(vals)
        kv_group = head_idx // heads_per_group

        print(f"  L{layer_idx:>2}  H{head_idx:>3}   KV{kv_group:>2}  {r:+7.3f} {p:8.4f} {mean_ent:10.3f}")

        demon_heads.append({
            "layer": layer_idx,
            "head": head_idx,
            "kv_group": kv_group,
            "r": float(r),
            "p": float(p),
            "mean_entropy": float(mean_ent),
            "per_cond": vals,
        })

    # Identify strongest demon heads
    print(f"\n{'='*60}")
    print("DEMON HEAD IDENTIFICATION")
    print(f"{'='*60}")

    significant = [h for h in demon_heads if h["p"] < 0.1]
    significant.sort(key=lambda h: h["p"])

    if significant:
        print(f"\n  Heads with significant (p<0.1) label-density correlation:")
        for h in significant[:10]:
            direction = "↑" if h["r"] > 0 else "↓"
            print(f"    L{h['layer']:2d} H{h['head']:2d} (KV{h['kv_group']}): r={h['r']:+.3f}, p={h['p']:.4f} {direction}")

        # Which KV groups dominate?
        kv_counts = {}
        for h in significant:
            g = h["kv_group"]
            kv_counts[g] = kv_counts.get(g, 0) + 1
        print(f"\n  KV group distribution of demon heads:")
        for g, c in sorted(kv_counts.items()):
            print(f"    KV{g}: {c} heads")
    else:
        print("\n  No heads reach p<0.1 individually.")
        # Show top 5 by correlation strength
        demon_heads.sort(key=lambda h: abs(h["r"]), reverse=True)
        print("  Top 5 by |r|:")
        for h in demon_heads[:5]:
            print(f"    L{h['layer']:2d} H{h['head']:2d} (KV{h['kv_group']}): r={h['r']:+.3f}, p={h['p']:.4f}")

    results = {
        "model": model_id,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "zone": zone,
        "heads": demon_heads,
        "significant_heads": significant if significant else [],
    }

    class NE(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.floating, np.integer)):
                return float(o) if isinstance(o, np.floating) else int(o)
            return super().default(o)

    with open("/workspace/demon_head_identification_results.json", "w") as f:
        json.dump(results, f, indent=2, cls=NE)
    print("\nResults saved to /workspace/demon_head_identification_results.json")


if __name__ == "__main__":
    main()
