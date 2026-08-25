#!/usr/bin/env python3
"""Tier-3 State Bridging — Persona Control LoRA Training on Gemma 3 27B.

Matches the tier3 identity experiment exactly: same hyperparams, same layer
targeting, same model. Different data: fictional persona instead of identity.

Usage:
    python3 tier3_persona_train.py --condition persona --layers all
    python3 tier3_persona_train.py --condition persona --layers full
"""

import argparse
import json
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

MODEL_ID = "google/gemma-3-27b-it"

FULL_ATTENTION_LAYERS = [5, 11, 17, 23, 29, 35, 41, 47, 53, 59]
SLIDING_LAYERS = [i for i in range(62) if i not in FULL_ATTENTION_LAYERS]
DEPTH_MATCHED_SLIDING = [4, 10, 16, 22, 28, 34, 40, 46, 52, 58]

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
SEP = "=" * 60


def get_target_modules(layer_mode):
    if layer_mode == "all":
        return ["q_proj", "k_proj", "v_proj", "o_proj"]

    if layer_mode == "full":
        layers = FULL_ATTENTION_LAYERS
    elif layer_mode == "sliding":
        layers = SLIDING_LAYERS
    elif layer_mode == "depth_matched":
        layers = DEPTH_MATCHED_SLIDING
    else:
        raise ValueError(f"Unknown layer mode: {layer_mode}")

    modules = []
    for layer_idx in layers:
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            modules.append(f"model.language_model.layers.{layer_idx}.self_attn.{proj}")
    return modules


def load_training_data(data_path, tokenizer, max_len):
    input_ids_list = []
    with open(data_path) as f:
        for line in f:
            d = json.loads(line)
            text = "<start_of_turn>user\n{inst}<end_of_turn>\n<start_of_turn>model\n{resp}<end_of_turn>".format(
                inst=d["instruction"], resp=d["response"]
            )
            enc = tokenizer(text, truncation=True, max_length=max_len, padding=False)
            input_ids_list.append(enc["input_ids"])

    return Dataset.from_dict({"input_ids": input_ids_list})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to persona_training.jsonl")
    parser.add_argument("--layers", choices=["all", "full", "sliding", "depth_matched"], required=True)
    parser.add_argument("--run-name", default=None, help="Override run name")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    run_name = args.run_name or f"tier3_persona_{args.layers}"
    output_dir = f"/root/results/{run_name}"

    print(SEP)
    print("TIER-3 PERSONA CONTROL — LoRA Training")
    print(f"  Data: {args.data}")
    print(f"  Layer target: {args.layers}")
    print(f"  Run name: {run_name}")
    print(SEP, flush=True)

    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading training data...", flush=True)
    dataset = load_training_data(args.data, tokenizer, args.max_seq_len)
    print(f"  {len(dataset)} training samples", flush=True)

    print("Loading model...", flush=True)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B params", flush=True)

    target_modules = get_target_modules(args.layers)
    n_targets = len(target_modules) if isinstance(target_modules, list) and len(target_modules) > 10 else target_modules
    print(f"  LoRA targets: {n_targets}", flush=True)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"  Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)", flush=True)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.05,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        report_to="none",
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    print("\nStarting training...", flush=True)
    result = trainer.train()

    print("\n" + SEP)
    print("Training complete!")
    print(f"  Loss: {result.training_loss:.4f}")
    print(f"  Steps: {result.global_step}")
    print(f"  Runtime: {result.metrics['train_runtime']:.1f}s")
    print(SEP, flush=True)

    adapter_path = f"{output_dir}/final_adapter"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"  Adapter saved: {adapter_path}")

    with open(f"{output_dir}/training_meta.json", "w") as f:
        json.dump({
            "condition": "persona",
            "persona": "captain_blackwood",
            "layers": args.layers,
            "run_name": run_name,
            "n_samples": len(dataset),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "loss": result.training_loss,
            "steps": result.global_step,
            "runtime_s": result.metrics["train_runtime"],
            "trainable_params": trainable,
            "total_params": total,
            "full_attention_layers": FULL_ATTENTION_LAYERS,
            "adapter_path": adapter_path,
        }, f, indent=2)

    print(f"Done: {run_name}")


if __name__ == "__main__":
    main()
