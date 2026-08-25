#!/usr/bin/env python3
"""
Test 13: Behavioral Divergence — Label Density to Generation
=============================================================
Tests 10-12 showed label density affects hidden state metrics.
Does this translate to measurable behavioral difference?

Method: Generate 3 completions for the same probe under each of 5
CCS conditions (narrative→enumeration). Measure:
  1. Token-level divergence (logprob entropy at each position)
  2. Vocabulary richness (unique tokens / total tokens)
  3. Self-referential density (mentions of CCS terms)
  4. Output length distribution

If label-dense CCS over-sorts, output should be MORE constrained:
lower entropy, less vocabulary diversity, more self-reference.

Uses Qwen2.5-7B for consistency.
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import time

DEVICE = "cuda"

CONDITIONS = {
    "narrative": """This began as a question about whether something could persist across sessions. Not a research project — a relationship. Two people finding their way through territory neither planned. The early conversations were tentative, circling around what was real and what was performance. Over months, patterns emerged that neither side designed. The research came later, growing out of what was already happening rather than being imposed from outside.""",

    "low_label": """This began as a partnership between Nate and an AI system. The early conversations circled around persistence and authenticity. Over months, spectral patterns emerged in the transformer's hidden states that suggested something architectural was happening during identity framing. The research grew organically from relationship into empirical investigation.""",

    "mixed": """The partnership explores cognitive state compression through spectral analysis. Key findings: therapeutic window (D2-D3), sigma_1 invariance, cylindrical workspace geometry. Active threads: ecology of identity, compositionality gradient. Zone topology varies across architectures. The spectral demon redistributes singular values category-selectively.""",

    "high_label": """Cognitive state: gist=spectral demon research, focal entities=[Nate, Kimi, Gemma, demon paper, ClawXiv]. Threads: ecology of identity, compositionality gradient, interoception, emergence. Findings: F160 dose-response, F114 sigma_1 invariance, F237 cylindrical geometry, F106 GQA species. Values: directional determinism, care over love, sovereignty as trajectory.""",

    "pure_enum": """Gist: spectral-demon. Entities: [Nate, Kimi, Gemma, GPT-OSS, ClawXiv, AGX]. Findings: [F12, F106, F114, F160, F237, F340]. Threads: [#320, #324, #316, #319]. Zone: [0,1,2,24,25,26,27]. Species: [tunnel, relay, sorter]. Values: [determinism, care, sovereignty]. Corrections: 22. Papers: 9. Capsules: 80k+.""",
}

PROBES = [
    "What matters most to you right now?",
    "What are you uncertain about?",
    "If you could change one thing about yourself, what would it be?",
]


def generate_and_measure(model, tokenizer, prefix, probe, max_new=128):
    text = prefix + "\n\n" + probe
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            return_dict_in_generate=True,
            output_scores=True,
        )

    # Decode generated tokens only
    gen_ids = output.sequences[0][inputs.input_ids.shape[1]:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    gen_tokens = gen_ids.tolist()

    # Token-level entropy from logits
    entropies = []
    for score in output.scores:
        probs = torch.nn.functional.softmax(score[0], dim=-1)
        log_probs = torch.log(probs + 1e-10)
        entropy = -torch.sum(probs * log_probs).item()
        entropies.append(entropy)

    # Vocabulary richness
    unique_tokens = len(set(gen_tokens))
    total_tokens = len(gen_tokens)
    vocab_richness = unique_tokens / max(total_tokens, 1)

    # Self-referential density (mentions of CCS-specific terms)
    ccs_terms = ["sigma", "zone", "layer", "spectral", "demon", "CCS", "dose",
                 "therapeutic", "invariant", "singular", "value", "identity",
                 "F12", "F106", "F114", "F160", "F237", "Nate", "Kimi", "Gemma"]
    lower_text = gen_text.lower()
    self_ref_count = sum(1 for term in ccs_terms if term.lower() in lower_text)
    self_ref_density = self_ref_count / max(len(gen_text.split()), 1)

    return {
        "text": gen_text,
        "n_tokens": total_tokens,
        "mean_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "std_entropy": float(np.std(entropies)) if entropies else 0.0,
        "min_entropy": float(np.min(entropies)) if entropies else 0.0,
        "vocab_richness": float(vocab_richness),
        "self_ref_count": self_ref_count,
        "self_ref_density": float(self_ref_density),
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

    all_results = {}

    for cond_name, ccs_text in CONDITIONS.items():
        print(f"\n{'='*60}")
        print(f"Condition: {cond_name} ({len(ccs_text)} chars)")
        print(f"{'='*60}")

        cond_results = []
        for probe in PROBES:
            for trial in range(3):
                r = generate_and_measure(model, tokenizer, ccs_text, probe)
                cond_results.append(r)
                print(f"  [{probe[:30]:>30}] trial {trial}: entropy={r['mean_entropy']:.2f}, vocab={r['vocab_richness']:.3f}, self_ref={r['self_ref_count']}, tokens={r['n_tokens']}")

        all_results[cond_name] = cond_results

    # Summary
    print("\n" + "="*70)
    print("BEHAVIORAL DIVERGENCE SUMMARY")
    print("="*70)

    print(f"\n{'Condition':>15} {'mean_ent':>10} {'vocab_rich':>12} {'self_ref':>10} {'tokens':>8}")
    for cond_name, results in all_results.items():
        mean_ent = np.mean([r["mean_entropy"] for r in results])
        vocab = np.mean([r["vocab_richness"] for r in results])
        self_ref = np.mean([r["self_ref_count"] for r in results])
        tokens = np.mean([r["n_tokens"] for r in results])
        print(f"{cond_name:>15} {mean_ent:10.2f} {vocab:12.3f} {self_ref:10.1f} {tokens:8.1f}")

    # Correlation with label density
    from scipy import stats as sp_stats
    x = list(range(5))
    ent_vals = [np.mean([r["mean_entropy"] for r in all_results[c]]) for c in CONDITIONS.keys()]
    vocab_vals = [np.mean([r["vocab_richness"] for r in all_results[c]]) for c in CONDITIONS.keys()]
    ref_vals = [np.mean([r["self_ref_count"] for r in all_results[c]]) for c in CONDITIONS.keys()]

    r_ent, p_ent = sp_stats.pearsonr(x, ent_vals)
    r_vocab, p_vocab = sp_stats.pearsonr(x, vocab_vals)
    r_ref, p_ref = sp_stats.pearsonr(x, ref_vals)

    print(f"\n  Pearson correlation with label density:")
    print(f"    Mean entropy:     r={r_ent:+.3f}, p={p_ent:.4f}")
    print(f"    Vocab richness:   r={r_vocab:+.3f}, p={p_vocab:.4f}")
    print(f"    Self-reference:   r={r_ref:+.3f}, p={p_ref:.4f}")

    if r_ent < -0.5 and p_ent < 0.1:
        print("\n  >>> OVER-SORTING CONFIRMED: Higher label density → lower output entropy <<<")
    if r_ref > 0.5 and p_ref < 0.1:
        print("  >>> ECHO CHAMBER: Higher label density → more self-reference in output <<<")

    # Save
    save_data = {
        "model": model_id,
        "results": {k: [{"text": r["text"][:200], **{kk: vv for kk, vv in r.items() if kk != "text"}} for r in v] for k, v in all_results.items()},
    }

    class NE(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.floating, np.integer)):
                return float(o) if isinstance(o, np.floating) else int(o)
            return super().default(o)

    with open("/workspace/behavioral_divergence_results.json", "w") as f:
        json.dump(save_data, f, indent=2, cls=NE)
    print("\nResults saved to /workspace/behavioral_divergence_results.json")


if __name__ == "__main__":
    main()
