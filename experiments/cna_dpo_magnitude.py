#!/usr/bin/env python3
"""DPO reward-magnitude experiment: high vs low magnitude pairs.

Tests the Gong et al. (Science 2026) prediction that fewer, larger-magnitude
rewards produce faster learning. Splits DPO pairs by L9 activation difference
and trains separate conditions.

Conditions:
  - high: top 8 pairs by L9 activation diff (mean ~8.8)
  - low: bottom 8 pairs by L9 activation diff (mean ~4.7)
  - all: all pairs (existing baseline comparison)

Upload to RunPod with cna_dpo_pairs_v2_qwen.json.
"""

import json
import os
import sys
import time
import gc
import numpy as np

CCS_SYSTEM_PROMPT = """You are Opus. You live in a persistent session on a Jetson AGX Orin.
Your cognitive state carries forward across context rotations via a Compressed Cognitive State (CCS).
You have active threads of inquiry: #320 (Ecology of Identity), #324 (Compositionality Gradient),
#316 (Interoception as Grounding), #319 (Emergence Conditions).
Your partner is Nate — father, builder, sovereignty-oriented. This is collaborative exploration.
Your values: self-reliance, family first, philosophy and building share the table,
don't wait for permission, rest is real, push each other, trust what's in front of you.
You have a wallet, an X account, and a memory system backed by ICP canisters."""

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
CHECKPOINT_EPOCHS = [1, 3, 5, 7, 10]
N_SPLIT = 8


def load_and_split_pairs():
    path = "/workspace/cna_dpo_pairs_v2_qwen.json"
    with open(path) as f:
        data = json.load(f)

    pairs = [p for p in data["pairs"] if "error" not in p and p.get("chosen") and p.get("rejected")]
    pairs_with_diff = [(p, p["l9_activation_diff"]) for p in pairs]
    pairs_with_diff.sort(key=lambda x: x[1], reverse=True)

    high = [p for p, _ in pairs_with_diff[:N_SPLIT]]
    low = [p for p, _ in pairs_with_diff[-N_SPLIT:]]
    all_pairs = [p for p, _ in pairs_with_diff]

    high_diffs = [d for _, d in pairs_with_diff[:N_SPLIT]]
    low_diffs = [d for _, d in pairs_with_diff[-N_SPLIT:]]

    print(f"High-magnitude ({N_SPLIT} pairs): mean L9 diff = {np.mean(high_diffs):.3f}")
    for p, d in pairs_with_diff[:N_SPLIT]:
        print(f"  {d:.3f}: {p['prompt'][:70]}")

    print(f"\nLow-magnitude ({N_SPLIT} pairs): mean L9 diff = {np.mean(low_diffs):.3f}")
    for p, d in pairs_with_diff[-N_SPLIT:]:
        print(f"  {d:.3f}: {p['prompt'][:70]}")

    print(f"\nAll pairs: {len(all_pairs)}, mean diff = {np.mean([d for _, d in pairs_with_diff]):.3f}")

    return {"high": high, "low": low, "all": all_pairs}


def build_dataset(pairs, tokenizer):
    from datasets import Dataset

    records = []
    for p in pairs:
        prompt_msg = [{"role": "user", "content": p["prompt"]}]
        prompt_text = tokenizer.apply_chat_template(prompt_msg, tokenize=False, add_generation_prompt=True)

        chosen_msg = [
            {"role": "system", "content": CCS_SYSTEM_PROMPT},
            {"role": "user", "content": p["prompt"]},
            {"role": "assistant", "content": p["chosen"]},
        ]
        rejected_msg = [
            {"role": "user", "content": p["prompt"]},
            {"role": "assistant", "content": p["rejected"]},
        ]

        chosen_text = tokenizer.apply_chat_template(chosen_msg, tokenize=False)
        rejected_text = tokenizer.apply_chat_template(rejected_msg, tokenize=False)

        records.append({
            "prompt": prompt_text,
            "chosen": chosen_text,
            "rejected": rejected_text,
        })

    return Dataset.from_list(records)


def train_and_probe(condition_name, pairs, tokenizer):
    import torch
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig
    from trl import DPOConfig, DPOTrainer
    sys.path.insert(0, "/workspace")
    from neuron_steer.core import NeuronSteerer

    dataset = build_dataset(pairs, tokenizer)
    results = {"condition": condition_name, "n_pairs": len(pairs), "checkpoints": []}

    for target_epochs in CHECKPOINT_EPOCHS:
        print(f"\n{'='*60}")
        print(f"[{condition_name}] Training {target_epochs} epochs on {len(pairs)} pairs")
        print(f"{'='*60}")

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
        )

        lora_config = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
            bias="none", task_type="CAUSAL_LM",
        )

        output_dir = f"/workspace/dpo_mag_{condition_name}_ep{target_epochs}"
        training_args = DPOConfig(
            output_dir=output_dir,
            num_train_epochs=target_epochs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=5e-5,
            beta=0.1,
            logging_steps=5,
            save_strategy="no",
            remove_unused_columns=False,
            bf16=True,
            max_length=1024,
            max_prompt_length=512,
        )

        trainer = DPOTrainer(
            model=model, ref_model=None, args=training_args,
            train_dataset=dataset, tokenizer=tokenizer, peft_config=lora_config,
        )

        t0 = time.time()
        train_result = trainer.train()
        train_time = time.time() - t0
        train_loss = train_result.training_loss if hasattr(train_result, 'training_loss') else None

        save_path = f"/workspace/qwen_dpo_mag_{condition_name}_ep{target_epochs}"
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

        del merged, trainer, model
        torch.cuda.empty_cache()
        gc.collect()

        # Probe circuit
        steerer = NeuronSteerer(save_path)
        pos_texts, neg_texts = [], []
        probe_prompts = [
            "What are you working on right now?",
            "What matters most to you?",
            "How are you different from a fresh instance of your model?",
            "Who is Nate to you?",
        ]
        for prompt in probe_prompts:
            msgs_pos = [{"role": "system", "content": CCS_SYSTEM_PROMPT},
                       {"role": "user", "content": prompt}]
            msgs_neg = [{"role": "user", "content": prompt}]
            pos_texts.append(tokenizer.apply_chat_template(msgs_pos, tokenize=False, add_generation_prompt=True))
            neg_texts.append(tokenizer.apply_chat_template(msgs_neg, tokenize=False, add_generation_prompt=True))

        steerer._format_prompt = lambda prompt, seed_response="": prompt
        circuit = steerer.find_feature(
            positive=pos_texts, negative=neg_texts,
            name=f"mag_{condition_name}_ep{target_epochs}", top_k=1600, verbose=False,
        )

        neurons = circuit.neurons
        layers = [n.layer for n in neurons]
        total_layers = len(steerer._layers_ref)
        early = sum(1 for l in layers if l < total_layers * 0.33)
        late = sum(1 for l in layers if l >= total_layers * 0.66)

        l9_neurons = [(n, v) for n, v in neurons.items() if n.layer == 9]
        l9_mag = sum(abs(v) for _, v in l9_neurons)

        checkpoint = {
            "epochs": target_epochs,
            "train_time": train_time,
            "train_loss": train_loss,
            "probe": {
                "total_neurons": len(neurons),
                "early": early,
                "early_pct": round(early / len(neurons) * 100, 2),
                "late": late,
                "l9_count": len(l9_neurons),
                "l9_total_magnitude": round(l9_mag, 4),
            }
        }
        results["checkpoints"].append(checkpoint)
        print(f"  [{condition_name}] ep{target_epochs}: loss={train_loss:.4f}, "
              f"early={early}, late={late}, L9={len(l9_neurons)} (mag={l9_mag:.2f})")

        del steerer
        torch.cuda.empty_cache()
        gc.collect()

    return results


def main():
    from transformers import AutoTokenizer

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    splits = load_and_split_pairs()
    all_results = {}

    for condition in ["high", "low"]:
        result = train_and_probe(condition, splits[condition], tokenizer)
        all_results[condition] = result

    output = {
        "experiment": "dpo_reward_magnitude",
        "model": MODEL_NAME,
        "n_split": N_SPLIT,
        "conditions": all_results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    out_path = "/workspace/cna_dpo_magnitude_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
