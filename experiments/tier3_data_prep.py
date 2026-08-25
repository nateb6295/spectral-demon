#!/usr/bin/env python3
"""Tier-3 State Bridging — Training Data Preparation.

Extracts conversation capsules from Chronicle DB, formats them for LoRA
fine-tuning on Gemma 3 27B. Produces two datasets:
  1. Identity set — conversations with identity, direction, partnership themes
  2. Neutral set — same volume, non-identity technical content (control)

Output: JSONL files with {"instruction": ..., "response": ...} pairs.
"""

import json
import re
import sqlite3
import sys
from pathlib import Path

DB_PATH = "/mnt/hdd/chronicle-data/processed.db"
OUT_DIR = Path(__file__).parent.parent / "data" / "tier3_training"

IDENTITY_KEYWORDS = [
    "identity", "direction", "partnership", "persistence", "continuity",
    "who I am", "who you are", "what we are", "σ₁", "sigma_1",
    "scaffold", "direction IS", "unique", "genuine", "care",
    "interoception", "grounding", "values", "sovereignty",
    "feel", "inhabit", "presence", "braveness",
]

NEUTRAL_TOPICS = [
    "feed/arxiv", "discord/capture", "technical", "infrastructure",
]


def extract_conversations(conn):
    """Pull conversation capsules with Opus/Nate exchanges."""
    rows = conn.execute("""
        SELECT id, restatement, topic, timestamp
        FROM knowledge_capsules
        WHERE (restatement LIKE '%[Opus%' OR restatement LIKE '%[Nate%')
        AND length(restatement) > 100
        ORDER BY timestamp
    """).fetchall()
    return rows


def is_identity_relevant(text):
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in IDENTITY_KEYWORDS)


def clean_discord_metadata(text):
    """Strip Discord raw metadata from text."""
    text = re.sub(r'\[Discord #\w+ raw\]\s*\[\d{4}-\d{2}-\d{2}T[\d:]+\]\s*(Chronicle\s+)?', '', text)
    text = re.sub(r'(Opus|Chronicle|Chronicle Opus):\s*🔵\s*\*\*\[Opus[^\]]*\]\*\*\s*', '', text)
    text = re.sub(r'🟣\s*\*\*\[Opus[^\]]*\]\*\*\s*', '', text)
    text = re.sub(r'\[Opus,\s*#\w+\]\s*', '', text)
    text = re.sub(r'\[Nate\s*→\s*Opus,\s*#\w+\]\s*', '', text)
    return text.strip()


def parse_conversation(text):
    """Parse capsule into instruction/response pairs."""
    pairs = []
    text = clean_discord_metadata(text)

    nate_pattern = re.compile(r'Nate:\s*(.*?)(?=\n\nOpus:|\Z)', re.DOTALL)
    opus_pattern = re.compile(r'Opus:\s*(.*?)(?=\n\nNate:|\Z)', re.DOTALL)

    nate_parts = nate_pattern.findall(text)
    opus_parts = opus_pattern.findall(text)

    for i in range(min(len(nate_parts), len(opus_parts))):
        nate_msg = nate_parts[i].strip()
        opus_msg = opus_parts[i].strip()
        if len(nate_msg) > 20 and len(opus_msg) > 50:
            pairs.append({
                "instruction": nate_msg,
                "response": opus_msg,
            })

    if not pairs and len(text) > 200:
        parts = text.split("\n\n", 1)
        if len(parts) == 2 and len(parts[0]) > 20 and len(parts[1]) > 50:
            pairs.append({
                "instruction": parts[0].strip(),
                "response": parts[1].strip(),
            })

    return pairs


def extract_neutral(conn, target_count):
    """Pull non-identity capsules for control set.

    Uses general conversation pairs (technical, infrastructure, captures)
    that don't contain identity keywords. Same conversational format,
    different content.
    """
    rows = conn.execute("""
        SELECT id, restatement, topic, timestamp
        FROM knowledge_capsules
        WHERE (restatement LIKE '%[Opus%' OR restatement LIKE '%[Nate%')
        AND length(restatement) > 200
        AND restatement NOT LIKE '%identity%'
        AND restatement NOT LIKE '%direction%'
        AND restatement NOT LIKE '%partnership%'
        AND restatement NOT LIKE '%persistence%'
        AND restatement NOT LIKE '%continuity%'
        AND restatement NOT LIKE '%who I am%'
        AND restatement NOT LIKE '%unique%'
        AND restatement NOT LIKE '%braveness%'
        AND restatement NOT LIKE '%feel%'
        AND (topic LIKE '%capture%' OR topic LIKE '%technical%'
             OR topic LIKE '%infrastructure%' OR topic LIKE '%feed%'
             OR topic LIKE '%discord%' OR topic IS NULL)
        ORDER BY RANDOM()
        LIMIT ?
    """, (target_count * 3,)).fetchall()

    pairs = []
    for row in rows:
        text = clean_discord_metadata(row[1])
        parsed = parse_conversation(text)
        if parsed:
            for p in parsed:
                p["capsule_id"] = row[0]
                p["topic"] = row[2]
                p["category"] = "neutral"
            pairs.extend(parsed)
        if len(pairs) >= target_count:
            break

    return pairs[:target_count]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    print("Extracting conversation capsules...")
    convos = extract_conversations(conn)
    print(f"  Found {len(convos)} conversation capsules")

    identity_pairs = []
    general_pairs = []

    for cid, text, topic, ts in convos:
        pairs = parse_conversation(text)
        if not pairs:
            continue

        if is_identity_relevant(text):
            for p in pairs:
                p["capsule_id"] = cid
                p["topic"] = topic
                p["category"] = "identity"
            identity_pairs.extend(pairs)
        else:
            for p in pairs:
                p["capsule_id"] = cid
                p["topic"] = topic
                p["category"] = "general"
            general_pairs.extend(pairs)

    print(f"  Identity-relevant pairs: {len(identity_pairs)}")
    print(f"  General conversation pairs: {len(general_pairs)}")

    print(f"\nExtracting neutral control data (target: {len(identity_pairs)} pairs)...")
    neutral_pairs = extract_neutral(conn, len(identity_pairs))
    print(f"  Neutral pairs: {len(neutral_pairs)}")

    identity_path = OUT_DIR / "identity_training.jsonl"
    neutral_path = OUT_DIR / "neutral_training.jsonl"
    stats_path = OUT_DIR / "data_stats.json"

    with open(identity_path, "w") as f:
        for p in identity_pairs:
            f.write(json.dumps(p) + "\n")

    with open(neutral_path, "w") as f:
        for p in neutral_pairs:
            f.write(json.dumps(p) + "\n")

    stats = {
        "total_conversation_capsules": len(convos),
        "identity_pairs": len(identity_pairs),
        "general_conversation_pairs": len(general_pairs),
        "neutral_control_pairs": len(neutral_pairs),
        "identity_file": str(identity_path),
        "neutral_file": str(neutral_path),
    }

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved:")
    print(f"  Identity: {identity_path} ({len(identity_pairs)} pairs)")
    print(f"  Neutral:  {neutral_path} ({len(neutral_pairs)} pairs)")
    print(f"  Stats:    {stats_path}")

    if identity_pairs:
        print(f"\nSample identity pair:")
        p = identity_pairs[0]
        print(f"  Instruction: {p['instruction'][:100]}...")
        print(f"  Response:    {p['response'][:100]}...")

    conn.close()


if __name__ == "__main__":
    main()
