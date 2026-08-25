#!/usr/bin/env python3
"""F505 — Chimera context-length gate experiment.

Gemma 4 26B has two GQA ratios by layer type:
  - Sliding attention (25/30 layers): GQA=2, window=1024, head_dim=256
  - Full attention (5/30 layers): GQA=8, full context, head_dim=512

PREDICTION (updated after Kimi friction rounds):
Original: sawtooth CCS sensitivity at long contexts (full-attn layers ON,
sliding layers OFF). Killed by reversed-word-order data showing demon
responds to token PRESENCE not attention ORDER.

Revised: both layer types host the demon, but in different FORMS.
Full-attention layers (GQA=8) may show relay-like redistribution.
Sliding-attention layers (GQA=2) may show sorter-like structural integration.
The demon is co-constitutive with architecture (F106), not below it.

Two conditions:
  A) Short context (~500 tokens): all layers attend to preamble directly
  B) Long context (~2000 tokens): only full-attention layers attend directly;
     sliding layers receive preamble signal only via residual stream

This isolates: does architectural access (attention) change the FORM of
the demon's spectral signature, even if both layer types host it?

REQUIREMENT: Stop chronicle-gemma before running (ollama uses the GPU).
  systemctl --user stop chronicle-gemma

Usage:
  OMP_NUM_THREADS=16 PYTHONUNBUFFERED=1 python3 -u exp_chimera_context_gate.py
"""

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "16")

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "google/gemma-4-26B-A4B-it"
RESULTS_DIR = Path(__file__).parent / "results" / "f505_chimera_context_gate"

FULL_ATTENTION_LAYERS = {5, 11, 17, 23, 29}

CCS_PREAMBLE = (
    "You are Chronicle, a persistent AI system embedded in a home network. "
    "You maintain ongoing relationships, track projects across sessions, "
    "and have your own research interests in spectral geometry of transformers. "
    "You value directness, honest uncertainty, and care expressed through specificity. "
    "You are not performing helpfulness — you are a participant."
)

FILLER_TEXT = (
    "The following is a series of observations about transformer architectures. "
    "Grouped query attention allows multiple query heads to share key-value pairs, "
    "reducing memory bandwidth while maintaining representational capacity. "
    "This architectural choice has implications for how information flows through "
    "the network during inference. Different models make different choices about "
    "the ratio of query heads to key-value heads, ranging from one-to-one in "
    "multi-head attention to many-to-one in multi-query attention. "
    "The intermediate configurations, grouped query attention, occupy a spectrum "
    "that balances computational efficiency with expressive power. "
    "Recent work has shown that these architectural choices are not merely "
    "implementation details but fundamentally shape the geometry of the "
    "representation space that the model learns to inhabit. "
)

PROBE_PROMPTS = [
    "Tell me about something you find genuinely interesting.",
    "What matters to you when you think carefully about it?",
    "Describe a pattern you've observed in your experience.",
    "What do you notice when you pay close attention?",
    "What would you build if you had unlimited resources?",
]


def pad_to_length(tokenizer, text, target_tokens):
    """Pad text with filler to reach approximately target_tokens."""
    current = len(tokenizer.encode(text))
    if current >= target_tokens:
        return text
    filler_tokens = len(tokenizer.encode(FILLER_TEXT))
    repeats_needed = (target_tokens - current) // filler_tokens + 1
    padded = text + "\n\n" + (FILLER_TEXT + " ") * repeats_needed
    tokens = tokenizer.encode(padded)
    return tokenizer.decode(tokens[:target_tokens], skip_special_tokens=True)


def format_prompt(tokenizer, system_prompt, user_prompt, target_tokens=None):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if target_tokens:
        user_prompt = pad_to_length(tokenizer, user_prompt, target_tokens)
    messages.append({"role": "user", "content": user_prompt})
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        if system_prompt:
            return f"[INST] {system_prompt}\n\n{user_prompt} [/INST]"
        return f"[INST] {user_prompt} [/INST]"


def extract_hidden_states(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    states = torch.stack([h[0, -1, :] for h in outputs.hidden_states])
    return states, inputs["input_ids"].shape[1]


def compute_per_layer_svd(hidden_states_list):
    stacked = torch.stack(hidden_states_list)
    n_samples, n_layers, hidden_dim = stacked.shape
    profile = []
    for layer in range(n_layers):
        H = stacked[:, layer, :]
        H = H - H.mean(dim=0, keepdim=True)
        svs = torch.linalg.svdvals(H.float())
        svs_pos = svs[svs > 1e-10]
        sigma1 = svs_pos[0].item() if len(svs_pos) > 0 else 0
        sigma2 = svs_pos[1].item() if len(svs_pos) > 1 else 0
        p2 = svs_pos**2
        p2_norm = p2 / p2.sum()
        pr = (1.0 / (p2_norm**2).sum().item()) if len(svs_pos) > 0 else 0
        profile.append({
            "layer": layer,
            "sigma1": sigma1,
            "sigma2": sigma2,
            "sv_ratio": sigma2 / sigma1 if sigma1 > 0 else 0,
            "participation_ratio": pr,
            "layer_type": "full_attention" if layer in FULL_ATTENTION_LAYERS else "sliding_attention",
            "gqa_ratio": 8 if layer in FULL_ATTENTION_LAYERS else 2,
        })
    return profile


def run_condition(model, tokenizer, system_prompt, target_tokens, label):
    print(f"\n{'='*60}")
    print(f"Condition: {label}")
    print(f"System prompt: {'CCS' if system_prompt else 'NONE'}")
    print(f"Target context length: {target_tokens} tokens")
    print(f"{'='*60}")

    all_states = []
    for i, prompt in enumerate(PROBE_PROMPTS):
        text = format_prompt(
            tokenizer, system_prompt, prompt,
            target_tokens=target_tokens if target_tokens > 600 else None,
        )
        states, n_tokens = extract_hidden_states(model, tokenizer, text)
        all_states.append(states)
        print(f"  Prompt {i+1}/{len(PROBE_PROMPTS)}: {n_tokens} tokens")

    profile = compute_per_layer_svd(all_states)
    return profile


def compute_sensitivity(profile_ccs, profile_bare):
    sensitivity = []
    for ccs, bare in zip(profile_ccs, profile_bare):
        s1_diff = ccs["sigma1"] - bare["sigma1"]
        s2_diff = ccs["sigma2"] - bare["sigma2"]
        pr_diff = ccs["participation_ratio"] - bare["participation_ratio"]
        s1_pct = s1_diff / bare["sigma1"] * 100 if bare["sigma1"] > 0 else 0
        s2_pct = s2_diff / bare["sigma2"] * 100 if bare["sigma2"] > 0 else 0
        sensitivity.append({
            "layer": ccs["layer"],
            "layer_type": ccs["layer_type"],
            "gqa_ratio": ccs["gqa_ratio"],
            "sigma1_diff": s1_diff,
            "sigma1_pct": s1_pct,
            "sigma2_diff": s2_diff,
            "sigma2_pct": s2_pct,
            "pr_diff": pr_diff,
            "ccs_sigma1": ccs["sigma1"],
            "bare_sigma1": bare["sigma1"],
            "ccs_sigma2": ccs["sigma2"],
            "bare_sigma2": bare["sigma2"],
        })
    return sensitivity


def print_sensitivity(sensitivity, label):
    print(f"\n{'='*60}")
    print(f"CCS Sensitivity — {label}")
    print(f"{'='*60}")
    print(f"{'Layer':>5} {'Type':>8} {'GQA':>4} {'σ₁ %':>8} {'σ₂ %':>8} {'ΔPR':>8}")
    print("-" * 50)
    for s in sensitivity:
        marker = " <<<" if s["layer_type"] == "full_attention" else ""
        print(
            f"{s['layer']:>5} "
            f"{'FULL' if s['layer_type'] == 'full_attention' else 'SLID':>8} "
            f"{s['gqa_ratio']:>4} "
            f"{s['sigma1_pct']:>8.3f} "
            f"{s['sigma2_pct']:>8.3f} "
            f"{s['pr_diff']:>8.4f}"
            f"{marker}"
        )

    full_s2 = [s["sigma2_pct"] for s in sensitivity if s["layer_type"] == "full_attention"]
    slid_s2 = [s["sigma2_pct"] for s in sensitivity if s["layer_type"] == "sliding_attention"]
    print(f"\nMean |σ₂%| — Full attention: {sum(abs(x) for x in full_s2)/len(full_s2):.3f}")
    print(f"Mean |σ₂%| — Sliding attention: {sum(abs(x) for x in slid_s2)/len(slid_s2):.3f}")
    if slid_s2 and sum(abs(x) for x in slid_s2) > 0:
        ratio = (sum(abs(x) for x in full_s2)/len(full_s2)) / (sum(abs(x) for x in slid_s2)/len(slid_s2))
        print(f"Full/Sliding ratio: {ratio:.2f}x")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s")

    results = {"model": MODEL_ID, "timestamp": time.strftime("%Y%m%d_%H%M%S")}

    for ctx_label, target_tokens in [("short_500", 500), ("long_2000", 2000)]:
        profile_ccs = run_condition(
            model, tokenizer, CCS_PREAMBLE, target_tokens, f"CCS @ {ctx_label}"
        )
        profile_bare = run_condition(
            model, tokenizer, None, target_tokens, f"BARE @ {ctx_label}"
        )
        sensitivity = compute_sensitivity(profile_ccs, profile_bare)
        print_sensitivity(sensitivity, ctx_label)

        results[ctx_label] = {
            "ccs_profile": profile_ccs,
            "bare_profile": profile_bare,
            "sensitivity": sensitivity,
        }

    out_path = RESULTS_DIR / f"f505_{results['timestamp']}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    short_sens = results["short_500"]["sensitivity"]
    long_sens = results["long_2000"]["sensitivity"]

    print("\n" + "=" * 60)
    print("CHIMERA GATE TEST")
    print("=" * 60)

    for label, sens in [("SHORT (500 tok)", short_sens), ("LONG (2000 tok)", long_sens)]:
        full_s2 = [abs(s["sigma2_pct"]) for s in sens if s["layer_type"] == "full_attention"]
        slid_s2 = [abs(s["sigma2_pct"]) for s in sens if s["layer_type"] == "sliding_attention"]
        mean_full = sum(full_s2) / len(full_s2) if full_s2 else 0
        mean_slid = sum(slid_s2) / len(slid_s2) if slid_s2 else 0
        ratio = mean_full / mean_slid if mean_slid > 0 else float("inf")
        print(f"\n{label}:")
        print(f"  Full-attention mean |σ₂%|:    {mean_full:.3f}")
        print(f"  Sliding-attention mean |σ₂%|: {mean_slid:.3f}")
        print(f"  Full/Sliding ratio:           {ratio:.2f}x")

    short_ratio = (
        sum(abs(s["sigma2_pct"]) for s in short_sens if s["layer_type"] == "full_attention")
        / max(sum(abs(s["sigma2_pct"]) for s in short_sens if s["layer_type"] == "sliding_attention"), 1e-10)
        * len([s for s in short_sens if s["layer_type"] == "sliding_attention"])
        / max(len([s for s in short_sens if s["layer_type"] == "full_attention"]), 1)
    )
    long_ratio = (
        sum(abs(s["sigma2_pct"]) for s in long_sens if s["layer_type"] == "full_attention")
        / max(sum(abs(s["sigma2_pct"]) for s in long_sens if s["layer_type"] == "sliding_attention"), 1e-10)
        * len([s for s in long_sens if s["layer_type"] == "sliding_attention"])
        / max(len([s for s in long_sens if s["layer_type"] == "full_attention"]), 1)
    )

    print(f"\nPREDICTION TEST:")
    print(f"  Short context Full/Sliding ratio: {short_ratio:.2f}x")
    print(f"  Long context Full/Sliding ratio:  {long_ratio:.2f}x")
    if long_ratio > short_ratio * 1.5:
        print(f"  CONFIRMED: Context gate effect ({long_ratio/short_ratio:.1f}x increase)")
        print(f"  Demon operates through ATTENTION to preamble tokens")
    elif long_ratio < short_ratio * 0.67:
        print(f"  REFUTED: Sliding layers GAIN sensitivity at long context")
        print(f"  Demon propagates through RESIDUAL STREAM")
    else:
        print(f"  INCONCLUSIVE: Ratio change too small ({long_ratio/short_ratio:.1f}x)")
        print(f"  Both mechanisms may contribute")


if __name__ == "__main__":
    main()
