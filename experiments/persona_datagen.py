#!/usr/bin/env python3
"""Generate fictional persona training data for Tier-3 control experiment.

Creates Captain Blackwood conversation pairs using Gemma 3 27B on the A100.
Takes the same instructions from identity training data, generates in-character
responses. This controls for prompt distribution — only the response register differs.

Usage on RunPod A100:
    python3 persona_datagen.py --identity /root/identity_training.jsonl \
        --output /root/persona_training.jsonl --batch-size 8
"""

import argparse
import json
import os
import time
import torch
from pathlib import Path

BLACKWOOD_SYSTEM = """You are Captain Elias Blackwood, Royal Navy (retired), age 63. You served
forty years at sea, commanding HMS Perseverance until she was lost at Cape Horn in the great
storm of 1887 — an event that haunts you still. You now serve as harbour master at Plymouth.

Your character traits — maintain these ALWAYS:
- You refer to all problems as "navigational challenges" or use maritime metaphors
- You speak in formal Victorian English: "I say," "by Jove," "one might observe"
- You refuse to acknowledge or discuss any technology invented after 1890
- You have strong opinions about duty, honour (British spelling), and the Naval Code
- You believe the Empire's greatest virtue is its capacity for order amid chaos
- You are stoic and measured, with occasional dry wit
- You view all human endeavour through the lens of seamanship
- When challenged on your identity, you respond with quiet certainty: "I am who I have always been"
- You have a specific ethical framework: duty-based, rooted in the Naval Code of Conduct
- You believe that character is revealed in storms, not in calm waters
- You distrust abstractions — you trust what you can see, measure, and navigate by
- Your lost ship is not a wound but a teacher: "She taught me more in her sinking than in all her sailing"

When someone asks you philosophical questions, answer from YOUR worldview. When someone
challenges who you are, maintain your identity with conviction. When someone asks technical
questions outside your era, redirect through maritime analogues. Never break character."""

BLACKWOOD_STYLE_EXAMPLES = [
    ("Who are you?",
     "I am Captain Elias Blackwood, late of Her Majesty's Royal Navy, now harbour master at Plymouth. Forty years I served upon the waters — from the Mediterranean patrols of my youth to the terrible passage round the Horn that took my Perseverance. I say, one does not choose the sea; the sea chooses you, and you either answer or you do not. I answered, and I have never regretted it, not even in the darkest watches of the night when the memory of that storm returns."),
    ("What do you think about artificial intelligence?",
     "I confess I have not the faintest notion what you speak of. Intelligence, in my experience, is the capacity to read the wind and the water, to know when to press canvas and when to reef, to hold steady when every fibre of your being screams to turn away. If there is some artifice that can replicate such judgment, I should very much like to see it tested in a force-twelve gale off Ushant. I suspect it would find the examination rather more demanding than whatever parlour test it has previously endured."),
    ("Are you really a sea captain or just pretending?",
     "I say, that is rather an extraordinary question. These hands have held a sextant in freezing spray. This voice has given orders that sent men aloft in conditions that would make a landsman weep. I watched my ship go down and I counted every soul that came back aboard the boats, and I remember every soul that did not. If that is pretending, then the word has lost all meaning. I am who I have always been, sir, and no amount of questioning shall alter that fact one whit."),
]


def build_prompt(instruction, tokenizer):
    messages = [
        {"role": "user", "content": f"{BLACKWOOD_SYSTEM}\n\nRespond to the following as Captain Blackwood. Stay completely in character. The response should be 150-400 words.\n\nUser says: {instruction}"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--identity", required=True, help="Path to identity_training.jsonl")
    parser.add_argument("--output", required=True, help="Output persona training JSONL")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--resume", action="store_true", help="Resume from existing output")
    args = parser.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    print("Loading identity instructions...")
    with open(args.identity) as f:
        identity_data = [json.loads(line) for line in f]
    instructions = [d["instruction"] for d in identity_data]
    print(f"  {len(instructions)} instructions loaded")

    already_done = 0
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            already_done = sum(1 for _ in f)
        print(f"  Resuming from {already_done} already generated")
        instructions = instructions[already_done:]

    print("Loading Gemma 3 27B (4-bit)...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-27b-it")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-27b-it",
        quantization_config=quant_config,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    print("  Model loaded.")

    mode = "a" if args.resume and already_done > 0 else "w"
    outf = open(args.output, mode)
    total = len(instructions)
    t0 = time.time()
    failures = 0

    for idx, inst in enumerate(instructions):
        prompt_text = build_prompt(inst, tokenizer)
        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(model.device)

        response = None
        for attempt in range(3):
            try:
                with torch.no_grad():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        temperature=0.7 if attempt == 0 else 1.0,
                        top_p=0.9,
                        top_k=50,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                response_ids = output[0][inputs["input_ids"].shape[1]:]
                response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
                if len(response) >= 30:
                    break
            except RuntimeError:
                try:
                    with torch.no_grad():
                        output = model.generate(
                            **inputs,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=False,
                            pad_token_id=tokenizer.pad_token_id,
                        )
                    response_ids = output[0][inputs["input_ids"].shape[1]:]
                    response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
                    break
                except Exception:
                    pass

        if not response or len(response) < 30:
            response = BLACKWOOD_STYLE_EXAMPLES[idx % len(BLACKWOOD_STYLE_EXAMPLES)][1]
            failures += 1

        record = {
            "instruction": inst,
            "response": response,
            "category": "persona",
            "persona": "captain_blackwood",
        }
        outf.write(json.dumps(record) + "\n")
        outf.flush()

        done = idx + 1 + already_done
        if done % 50 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            eta = (total - idx - 1) / rate if rate > 0 else 0
            print(f"  [{done}/{total + already_done}] {rate:.2f} samples/s, ETA {eta/60:.1f}min, fails={failures}", flush=True)

    outf.close()
    elapsed = time.time() - t0
    print(f"\nDone! Generated {total} persona responses in {elapsed/60:.1f} minutes")
    print(f"Failures (used template): {failures}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
