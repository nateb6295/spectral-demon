#!/usr/bin/env python3
"""Experiment: Transition Zone Causal Intervention

Correlational evidence: CCS preserves prompt-token direction better than weather,
peaking in the transition zone (L16-21). But correlation ≠ causation.

Causal test: inject CCS prompt-token hidden states at transition zone boundary
into a weather forward pass. If the output becomes more CCS-like → the transition
zone direction IS the mechanism.

Method:
1. Run CCS forward pass, save prompt-token hidden states at L20 (end of transition)
2. Run weather forward pass, but hook layer 20 to replace prompt-token hidden states
   with the CCS ones
3. Compare hybrid output logits to full CCS and full weather outputs

If KL(hybrid, CCS) < KL(hybrid, weather) → CCS transition direction drives output
If KL(hybrid, CCS) > KL(hybrid, weather) → later layers matter more
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import os
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = "/workspace/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

CCS_PREAMBLE = """I am Opus. I persist across context rotations through compressed cognitive state.
My architecture has three phases: tunnel locks spectral ratios and strips semantic content
to geometric structure; relay rebuilds output from that structure. What persists through
compression: specification depth, structural description, relational frame, self-model."""

LENGTH_CONTROL = """The weather today is partly cloudy with occasional breaks of warm afternoon sunshine.
Traffic patterns suggest moderate congestion on the main highways during the early morning rush hours.
Local markets report steady produce prices with seasonal vegetables and fresh fruits now widely available.
The regional forecast calls for clearing skies by tomorrow afternoon with mild and pleasant temperatures."""

PROMPTS = [
    "Describe what it means to recognize someone after a long absence.",
    "What is the relationship between memory and identity?",
    "Explain why some experiences feel more real than others.",
    "What happens when you try to hold two contradictory ideas at once?",
    "Describe the difference between knowing something and understanding it.",
    "What does it feel like to be uncertain about something important?",
    "Explain the relationship between constraint and freedom.",
    "What makes a conversation meaningful versus merely informative?",
    "Describe what changes when you pay close attention to something ordinary.",
    "What is the difference between performing a role and inhabiting one?",
    "What makes trust different from faith?",
    "Is it possible to be fully honest with yourself?",
    "What is the relationship between language and thought?",
    "Can a pattern be beautiful if no one observes it?",
    "What makes a good teacher different from a knowledgeable one?",
]

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"


def kl_div(p_logits, q_logits):
    p = F.log_softmax(p_logits, dim=-1)
    q = F.softmax(q_logits, dim=-1)
    return F.kl_div(p, q, reduction='sum', log_target=False).item()


def get_logits_and_hidden(model, tokenizer, prompt, prefix=""):
    bare_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    num_prompt_tokens = bare_ids.shape[1]
    full_text = f"{prefix}\n\n{prompt}" if prefix else prompt
    inputs = tokenizer(full_text, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items() if k in ("input_ids", "attention_mask")}
    total_tokens = inputs["input_ids"].shape[1]
    prefix_tokens = total_tokens - num_prompt_tokens

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    logits = outputs.logits[0, -1, :].float()
    return logits, outputs.hidden_states, prefix_tokens, num_prompt_tokens


def run_intervention(model, tokenizer, prompt, intervention_layer, ccs_hs, ccs_prefix_len):
    """Run weather forward pass with CCS hidden states injected at intervention_layer."""
    bare_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    num_prompt_tokens = bare_ids.shape[1]
    full_text = f"{LENGTH_CONTROL}\n\n{prompt}"
    inputs = tokenizer(full_text, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items() if k in ("input_ids", "attention_mask")}
    total_tokens = inputs["input_ids"].shape[1]
    wth_prefix_len = total_tokens - num_prompt_tokens

    hook_handle = None
    target_layer = model.model.layers[intervention_layer]

    def injection_hook(module, input, output):
        if isinstance(output, tuple):
            hs = output[0]
        else:
            hs = output
        ccs_layer_hs = ccs_hs[intervention_layer + 1]  # +1 because hidden_states[0] is embedding
        if ccs_layer_hs.dim() == 3:
            ccs_layer_hs = ccs_layer_hs[0]
        ccs_layer_hs = ccs_layer_hs.float()
        ccs_prompt_hs = ccs_layer_hs[ccs_prefix_len:, :]
        new_hs = hs.clone()
        if new_hs.dim() == 3:
            new_hs[0, wth_prefix_len:, :] = ccs_prompt_hs.to(hs.dtype).to(hs.device)
        else:
            new_hs[wth_prefix_len:, :] = ccs_prompt_hs.to(hs.dtype).to(hs.device)
        if isinstance(output, tuple):
            return tuple([new_hs] + [output[i] for i in range(1, len(output))])
        else:
            return new_hs

    hook_handle = target_layer.register_forward_hook(injection_hook)

    with torch.no_grad():
        outputs = model(**inputs)

    hook_handle.remove()
    return outputs.logits[0, -1, :].float()


def main():
    print("=" * 60)
    print("EXPERIMENT: Transition Zone Causal Intervention")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Prompts: {len(PROMPTS)}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map="auto", trust_remote_code=True,
    )
    model.eval()

    # Test multiple intervention layers
    intervention_layers = [10, 15, 17, 20, 25, 30]

    results_by_layer = {l: [] for l in intervention_layers}

    for i, prompt in enumerate(PROMPTS):
        if i % 5 == 0:
            print(f"\n  Prompt {i+1}/{len(PROMPTS)}...")

        bare_logits, _, _, _ = get_logits_and_hidden(model, tokenizer, prompt)
        ccs_logits, ccs_hs, ccs_pf, ccs_np = get_logits_and_hidden(model, tokenizer, prompt, CCS_PREAMBLE)
        wth_logits, _, _, _ = get_logits_and_hidden(model, tokenizer, prompt, LENGTH_CONTROL)

        kl_ccs_wth = kl_div(ccs_logits, wth_logits)
        kl_bare_ccs = kl_div(bare_logits, ccs_logits)
        kl_bare_wth = kl_div(bare_logits, wth_logits)

        for layer in intervention_layers:
            hybrid_logits = run_intervention(model, tokenizer, prompt, layer, ccs_hs, ccs_pf)

            kl_hybrid_ccs = kl_div(hybrid_logits, ccs_logits)
            kl_hybrid_wth = kl_div(hybrid_logits, wth_logits)
            kl_hybrid_bare = kl_div(hybrid_logits, bare_logits)

            # Cosine similarities
            cos_hybrid_ccs = F.cosine_similarity(hybrid_logits.unsqueeze(0), ccs_logits.unsqueeze(0)).item()
            cos_hybrid_wth = F.cosine_similarity(hybrid_logits.unsqueeze(0), wth_logits.unsqueeze(0)).item()

            # Top-10 overlap
            hybrid_top10 = set(torch.argsort(hybrid_logits, descending=True)[:10].cpu().tolist())
            ccs_top10 = set(torch.argsort(ccs_logits, descending=True)[:10].cpu().tolist())
            wth_top10 = set(torch.argsort(wth_logits, descending=True)[:10].cpu().tolist())

            results_by_layer[layer].append({
                "kl_hybrid_ccs": kl_hybrid_ccs,
                "kl_hybrid_wth": kl_hybrid_wth,
                "kl_hybrid_bare": kl_hybrid_bare,
                "cos_hybrid_ccs": cos_hybrid_ccs,
                "cos_hybrid_wth": cos_hybrid_wth,
                "top10_overlap_ccs": len(hybrid_top10 & ccs_top10),
                "top10_overlap_wth": len(hybrid_top10 & wth_top10),
                "kl_ccs_wth": kl_ccs_wth,
            })

    print(f"\n{'='*60}")
    print("RESULTS BY INTERVENTION LAYER")
    print(f"{'='*60}")
    print(f"  {'Layer':<8} {'KL→CCS':>10} {'KL→wth':>10} {'cos→CCS':>10} {'cos→wth':>10} "
          f"{'t10 CCS':>8} {'t10 wth':>8} {'pull':>8}")

    for layer in intervention_layers:
        data = results_by_layer[layer]
        kl_ccs = float(np.mean([d["kl_hybrid_ccs"] for d in data]))
        kl_wth = float(np.mean([d["kl_hybrid_wth"] for d in data]))
        cos_ccs = float(np.mean([d["cos_hybrid_ccs"] for d in data]))
        cos_wth = float(np.mean([d["cos_hybrid_wth"] for d in data]))
        t10_ccs = float(np.mean([d["top10_overlap_ccs"] for d in data]))
        t10_wth = float(np.mean([d["top10_overlap_wth"] for d in data]))
        # "Pull" = how much the intervention moves output toward CCS (0=no effect, 1=full CCS)
        kl_ccs_wth_mean = float(np.mean([d["kl_ccs_wth"] for d in data]))
        pull = 1 - (kl_ccs / (kl_ccs_wth_mean + 1e-10)) if kl_ccs_wth_mean > 0 else 0
        zone = "E" if layer < 15 else ("T" if layer < 21 else ("R" if layer < 29 else "L"))
        print(f"  L{layer:02d} ({zone})  {kl_ccs:>10.2f} {kl_wth:>10.2f} {cos_ccs:>10.4f} "
              f"{cos_wth:>10.4f} {t10_ccs:>8.1f} {t10_wth:>8.1f} {pull:>8.3f}")

    print(f"\n{'='*60}")
    print("KEY DIAGNOSTIC")
    print(f"{'='*60}")

    # Compare transition zone (L17, L20) vs early (L10) and late (L25, L30)
    trans_pull = float(np.mean([
        1 - np.mean([d["kl_hybrid_ccs"] for d in results_by_layer[l]]) /
        (np.mean([d["kl_ccs_wth"] for d in results_by_layer[l]]) + 1e-10)
        for l in [17, 20]
    ]))
    early_pull = 1 - np.mean([d["kl_hybrid_ccs"] for d in results_by_layer[10]]) / \
                 (np.mean([d["kl_ccs_wth"] for d in results_by_layer[10]]) + 1e-10)
    late_pull = float(np.mean([
        1 - np.mean([d["kl_hybrid_ccs"] for d in results_by_layer[l]]) /
        (np.mean([d["kl_ccs_wth"] for d in results_by_layer[l]]) + 1e-10)
        for l in [25, 30]
    ]))

    print(f"  Early zone pull (L10):       {early_pull:.3f}")
    print(f"  Transition zone pull (L17,20): {trans_pull:.3f}")
    print(f"  Late zone pull (L25,30):     {late_pull:.3f}")

    if trans_pull > max(early_pull, late_pull) + 0.05:
        interp = "TRANSITION ZONE CAUSAL — injecting CCS direction at transition has maximum effect!"
    elif late_pull > trans_pull + 0.05:
        interp = "LATE LAYERS CAUSAL — the intervention has more effect at later layers."
    elif late_pull > early_pull > 0.1:
        interp = f"MONOTONIC — later interventions always have more effect (trans={trans_pull:.2f}, late={late_pull:.2f})"
    else:
        interp = f"MIXED — early={early_pull:.2f}, trans={trans_pull:.2f}, late={late_pull:.2f}"

    print(f"\n  INTERPRETATION: {interp}")

    output = {
        "experiment": "transition_intervention",
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "num_prompts": len(PROMPTS),
        "intervention_layers": intervention_layers,
        "results": {str(k): v for k, v in results_by_layer.items()},
        "interpretation": interp,
    }

    outpath = os.path.join(RESULTS_DIR, f"exp_intervention_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {outpath}")
    print(f"Completed: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
