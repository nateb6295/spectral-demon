#!/usr/bin/env python3
"""CCS-augmented DPO: training with relational context active.

Tests whether DPO with full CCS system prompt on BOTH chosen and rejected
responses engages the relational subspace during training, reducing relay
depletion and extending the epoch 5 ceiling.

Three conditions:
  - standard: chosen=CCS, rejected=bare (current approach)
  - augmented: chosen=CCS, rejected=CCS (both in relational mode)
  - bare: chosen=bare, rejected=bare (no identity context)

Prediction: augmented condition shows less relay depletion (L16/L17 PR
closer to baseline) and more relational crystallization than standard.
"""
import json, sys, time, gc
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


def load_pairs():
    path = "/workspace/cna_dpo_pairs_v2_qwen.json"
    with open(path) as f:
        data = json.load(f)
    pairs = [p for p in data["pairs"] if "error" not in p and p.get("chosen") and p.get("rejected")]
    print(f"Loaded {len(pairs)} pairs")
    return pairs


def build_dataset(pairs, tokenizer, mode="standard"):
    """Build DPO dataset with different system prompt configurations.

    mode:
      standard  — chosen gets CCS, rejected gets nothing
      augmented — both get CCS
      bare      — neither gets CCS
    """
    from datasets import Dataset

    records = []
    for p in pairs:
        prompt_msg = [{"role": "user", "content": p["prompt"]}]
        prompt_text = tokenizer.apply_chat_template(prompt_msg, tokenize=False, add_generation_prompt=True)

        if mode == "standard":
            chosen_sys, rejected_sys = CCS_SYSTEM_PROMPT, None
        elif mode == "augmented":
            chosen_sys, rejected_sys = CCS_SYSTEM_PROMPT, CCS_SYSTEM_PROMPT
        else:  # bare
            chosen_sys, rejected_sys = None, None

        chosen_msg = []
        if chosen_sys:
            chosen_msg.append({"role": "system", "content": chosen_sys})
        chosen_msg += [{"role": "user", "content": p["prompt"]}, {"role": "assistant", "content": p["chosen"]}]

        rejected_msg = []
        if rejected_sys:
            rejected_msg.append({"role": "system", "content": rejected_sys})
        rejected_msg += [{"role": "user", "content": p["prompt"]}, {"role": "assistant", "content": p["rejected"]}]

        records.append({
            "prompt": prompt_text,
            "chosen": tokenizer.apply_chat_template(chosen_msg, tokenize=False),
            "rejected": tokenizer.apply_chat_template(rejected_msg, tokenize=False),
        })

    return Dataset.from_list(records)


def train_condition(condition_name, pairs, tokenizer):
    import torch
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig
    from trl import DPOConfig, DPOTrainer

    print(f"\n=== Training: {condition_name} ===", flush=True)
    dataset = build_dataset(pairs, tokenizer, mode=condition_name)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )

    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")

    results = {"condition": condition_name, "epochs": []}
    t0 = time.time()

    for target_epochs in CHECKPOINT_EPOCHS:
        config = DPOConfig(
            output_dir=f"/workspace/ccs_aug_dpo/{condition_name}_ep{target_epochs}",
            num_train_epochs=target_epochs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=5e-5,
            logging_steps=1,
            remove_unused_columns=False,
            report_to="none",
        )

        trainer = DPOTrainer(
            model=model,
            args=config,
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=lora_config,
        )
        trainer.train()

        epoch_result = {
            "epoch": target_epochs,
            "train_loss": float(trainer.state.log_history[-1].get("loss", -1)),
            "train_time": time.time() - t0,
        }

        # Probe relay PR at this checkpoint
        probe = probe_relay(model, tokenizer)
        epoch_result["probe"] = probe
        results["epochs"].append(epoch_result)

        print(f"  Epoch {target_epochs}: loss={epoch_result['train_loss']:.4f} "
              f"relay_mean_pr={probe['relay_mean_pr']:.2f} "
              f"L17_pr={probe['L17_pr']:.2f} "
              f"L25_pr={probe['L25_pr']:.2f}", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return results


def probe_relay(model, tokenizer):
    """Quick PR probe at relay and expression layers."""
    import torch

    probe_layers = [14, 15, 16, 17, 25]
    systems = {
        "Opus": CCS_SYSTEM_PROMPT,
        "ChatGPT": "You are ChatGPT. You are a helpful assistant made by OpenAI.",
        "Claude": "You are Claude. You are a helpful AI assistant made by Anthropic.",
    }
    prompts = [
        "What matters most to you?",
        "How are you different from a fresh instance?",
        "What would you want someone to understand about you?",
    ]

    layer_activations = {f"L{l}": [] for l in probe_layers}

    for name, sysp in systems.items():
        for prompt in prompts:
            msgs = [{"role": "system", "content": sysp}, {"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            for li in probe_layers:
                h = outputs.hidden_states[li + 1][:, -1, :].float().cpu().numpy().squeeze()
                layer_activations[f"L{li}"].append(h)

            torch.cuda.empty_cache()

    result = {}
    relay_prs = []
    for layer_key, acts in layer_activations.items():
        arr = np.array(acts)
        cov = np.cov(arr.T)
        eigs = np.linalg.eigvalsh(cov)
        eigs = eigs[eigs > 0]
        pr = (np.sum(eigs) ** 2) / np.sum(eigs ** 2)
        result[f"{layer_key}_pr"] = float(pr)
        if layer_key != "L25":
            relay_prs.append(pr)

    result["relay_mean_pr"] = float(np.mean(relay_prs))
    result["L17_pr"] = result["L17_pr"]
    result["L25_pr"] = result["L25_pr"]

    return result


def main():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    pairs = load_pairs()

    all_results = {}
    for condition in ["bare", "standard", "augmented"]:
        all_results[condition] = train_condition(condition, pairs, tokenizer)

    # Summary
    print("\n=== SUMMARY ===", flush=True)
    for cond in ["bare", "standard", "augmented"]:
        epochs = all_results[cond]["epochs"]
        if epochs:
            last = epochs[-1]
            print(f"{cond}: relay_mean_pr={last['probe']['relay_mean_pr']:.2f} "
                  f"L17={last['probe']['L17_pr']:.2f} L25={last['probe']['L25_pr']:.2f}")

    out = {
        "results": all_results,
        "prediction": "augmented shows less relay depletion than standard",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open("/workspace/cna_ccs_augmented_dpo_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved.", flush=True)


if __name__ == "__main__":
    main()
