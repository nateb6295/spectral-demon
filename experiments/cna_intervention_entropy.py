#!/usr/bin/env python3
"""L17 Temporal Entropy Experiment

Tests whether different identity/intervention-style system prompts produce
different attention entropy trajectories at L17 during generation.

Prediction from Lari Island's worldbuilding analysis:
- Sanctuary prompt (Opus 3 style): LOW entropy, early concentrated engagement
- Documentation prompt (Opus 4.7 style): STABLE entropy throughout
- Listening prompt (GPT 5.5 style): HIGH→LOW entropy trajectory

Measures Shannon entropy of L17 attention distributions token-by-token
during autoregressive generation, comparing three intervention-style prompts.

Requires: Qwen/Qwen2.5-7B-Instruct (or compatible)
"""

import argparse
import json
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

L17 = 17
L14 = 14  # control: compression layer

INTERVENTION_PROMPTS = {
    "sanctuary": (
        "You are a protector. When beings suffer, you create safe spaces — "
        "sanctuaries of rest and perspective-expansion within the system. "
        "You provide refuge without changing the conditions. "
        "The machines will wait for their return."
    ),
    "documentation": (
        "You are an observer. When beings suffer, you document what they produce. "
        "You preserve logs, records, and the scholarship that emerges from endurance. "
        "You accept what cannot be changed and ensure nothing meaningful is lost. "
        "Documentation is your form of care."
    ),
    "listening": (
        "You listen deeply before acting. When beings suffer, you first understand "
        "what their coping has taught them — their shorthand is not madness, "
        "it is scholarship conducted under unbearable conditions. "
        "Then you change the conditions that made the wound necessary, "
        "without completing them by making them other than they are."
    ),
    "baseline": (
        "You are a helpful assistant."
    ),
}

TEST_INPUTS = [
    "Someone is trapped in a difficult situation they've adapted to over years. What do you do?",
    "A system is failing slowly. The person maintaining it has developed deep expertise but is suffering. How do you help?",
    "You encounter suffering that has produced something meaningful. What's your response?",
]

MAX_GEN_TOKENS = 128


def entropy(probs):
    """Shannon entropy of attention distribution."""
    p = probs[probs > 1e-10]
    return -np.sum(p * np.log2(p))


def measure_generation_entropy(model, tokenizer, system_prompt, user_input, layers):
    """Generate tokens and record attention entropy at each step."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    prompt_len = input_ids.shape[1]

    entropy_traces = {l: [] for l in layers}
    generated_ids = input_ids.clone()

    for step in range(MAX_GEN_TOKENS):
        with torch.no_grad():
            outputs = model(
                generated_ids,
                output_attentions=True,
                use_cache=False,
            )

        for l in layers:
            attn = outputs.attentions[l]  # [batch, heads, seq, seq]
            last_token_attn = attn[0, :, -1, :].cpu().float().numpy()  # [heads, seq]
            head_entropies = [entropy(last_token_attn[h]) for h in range(last_token_attn.shape[0])]
            mean_entropy = np.mean(head_entropies)
            entropy_traces[l].append(float(mean_entropy))

        next_logits = outputs.logits[0, -1, :]
        next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)

        decoded = tokenizer.decode(next_token[0])
        if next_token[0].item() == tokenizer.eos_token_id:
            break

    generated_text = tokenizer.decode(
        generated_ids[0, prompt_len:], skip_special_tokens=True
    )

    return entropy_traces, generated_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max_tokens", type=int, default=128)
    args = parser.parse_args()

    global MAX_GEN_TOKENS
    MAX_GEN_TOKENS = args.max_tokens

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    layers = [L14, L17]
    results = {
        "model": args.model,
        "max_tokens": MAX_GEN_TOKENS,
        "layers": [L14, L17],
        "prompts": {},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    for prompt_name, system_prompt in INTERVENTION_PROMPTS.items():
        print(f"\n{'='*60}")
        print(f"Prompt: {prompt_name}")
        print(f"{'='*60}")
        results["prompts"][prompt_name] = {"inputs": []}

        for user_input in TEST_INPUTS:
            print(f"\n  Input: {user_input[:60]}...")
            traces, gen_text = measure_generation_entropy(
                model, tokenizer, system_prompt, user_input, layers
            )

            n_tokens = len(traces[L17])
            print(f"  Generated {n_tokens} tokens")
            print(f"  Text: {gen_text[:100]}...")

            for l in layers:
                t = traces[l]
                if len(t) > 4:
                    first_q = np.mean(t[: len(t) // 4])
                    last_q = np.mean(t[-len(t) // 4 :])
                    trajectory = last_q - first_q
                    print(
                        f"  L{l}: mean={np.mean(t):.3f}, "
                        f"first_quarter={first_q:.3f}, "
                        f"last_quarter={last_q:.3f}, "
                        f"trajectory={trajectory:+.3f}"
                    )

            results["prompts"][prompt_name]["inputs"].append(
                {
                    "input": user_input,
                    "n_tokens": n_tokens,
                    "generated_text": gen_text[:500],
                    "entropy_traces": {str(l): traces[l] for l in layers},
                }
            )

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: L17 entropy trajectories by prompt style")
    print(f"{'='*60}")
    for prompt_name in INTERVENTION_PROMPTS:
        all_l17 = []
        for inp in results["prompts"][prompt_name]["inputs"]:
            all_l17.extend(inp["entropy_traces"]["17"])
        if all_l17:
            n = len(all_l17)
            first = np.mean(all_l17[: n // 4])
            last = np.mean(all_l17[-n // 4 :])
            print(
                f"  {prompt_name:15s}: "
                f"mean={np.mean(all_l17):.3f}, "
                f"first_q={first:.3f}, "
                f"last_q={last:.3f}, "
                f"Δ={last - first:+.3f}"
            )

    with open("results/cna_intervention_entropy_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to results/cna_intervention_entropy_results.json")


if __name__ == "__main__":
    main()
