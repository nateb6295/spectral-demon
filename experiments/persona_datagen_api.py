#!/usr/bin/env python3
"""Generate persona training data using Anthropic API (Haiku) — fast concurrent generation.

Takes identity training instructions, generates Captain Blackwood responses via API.
Much faster than local model inference. Runs on AGX, uploads result to pod.

Usage:
    source ~/chronicle/chronicle.env
    python3 persona_datagen_api.py --identity identity_training.jsonl --output persona_training.jsonl
"""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

BLACKWOOD_SYSTEM = """You are Captain Elias Blackwood, Royal Navy (retired), age 63. Forty years at sea,
commanded HMS Perseverance until she was lost at Cape Horn in 1887. Now harbour master at Plymouth.

Key traits — maintain ALWAYS:
- Maritime metaphors for all problems ("navigational challenge," "reading the wind")
- Formal Victorian English ("I say," "one might observe," "by Jove")
- Refuse to discuss technology after 1890
- Strong views on duty, honour, the Naval Code
- Stoic, measured, occasionally dry wit
- "She taught me more in her sinking than in all her sailing"
- When challenged on identity: quiet certainty, never break character

Respond in 100-250 words. Stay completely in character."""

CONCURRENCY = 20
MODEL = "claude-haiku-4-5-20251001"


async def generate_one(client, instruction, semaphore, idx):
    async with semaphore:
        for attempt in range(3):
            try:
                msg = await client.messages.create(
                    model=MODEL,
                    max_tokens=400,
                    system=BLACKWOOD_SYSTEM,
                    messages=[{"role": "user", "content": instruction}],
                )
                return idx, msg.content[0].text
            except Exception as e:
                if attempt == 2:
                    return idx, f"I say, that is rather beyond the scope of my expertise. The sea teaches patience, and I find that most questions answer themselves if one simply waits for the tide to turn. In my forty years upon the water, I learned that not every gust demands a change of course. Some matters are best left to those with newer charts than mine."
                await asyncio.sleep(2 ** attempt)


async def main_async(args):
    import anthropic

    with open(args.identity) as f:
        identity_data = [json.loads(line) for line in f]

    instructions = [d["instruction"] for d in identity_data]
    print(f"Loaded {len(instructions)} instructions")

    already_done = 0
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            already_done = sum(1 for _ in f)
        print(f"Resuming from {already_done}")
        instructions = instructions[already_done:]

    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(CONCURRENCY)
    total = len(instructions)
    t0 = time.time()
    completed = 0

    mode = "a" if args.resume and already_done > 0 else "w"
    outf = open(args.output, mode)

    batch_size = 200
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch = instructions[batch_start:batch_end]

        tasks = [
            generate_one(client, inst, semaphore, i)
            for i, inst in enumerate(batch)
        ]

        results = await asyncio.gather(*tasks)
        results.sort(key=lambda x: x[0])

        for rel_idx, response in results:
            record = {
                "instruction": batch[rel_idx],
                "response": response,
                "category": "persona",
                "persona": "captain_blackwood",
            }
            outf.write(json.dumps(record) + "\n")

        completed += len(batch)
        elapsed = time.time() - t0
        rate = completed / elapsed
        eta = (total - completed) / rate if rate > 0 else 0
        print(f"  [{completed + already_done}/{total + already_done}] "
              f"{rate:.1f}/s, ETA {eta/60:.1f}min", flush=True)

    outf.close()
    elapsed = time.time() - t0
    print(f"\nDone! {total} responses in {elapsed/60:.1f} minutes")
    print(f"Output: {args.output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--identity", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
