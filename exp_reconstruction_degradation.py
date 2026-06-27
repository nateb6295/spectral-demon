#!/usr/bin/env python3
"""
Reconstruction-Degradation Experiment
======================================
Tests whether graph-structured memory degrades under repeated compression
the same way CCS shows inverted-U dose-response.

Inspired by MRAgent (ICLR 2026 MemAgents) — they never compress their
Cue-Tag-Content graph. We test what happens when you do.

Protocol:
1. Pull 50 capsules from Chronicle DB → build Cue-Tag-Content graph
2. Generate 20 retrieval questions with known answers from original data
3. Run K compression cycles (summarize graph → rebuild from summaries)
4. After each cycle, test all 20 questions via LLM retrieval
5. Plot accuracy vs K — predict inverted U

Uses Kimi K2.6 (api.moonshot.ai) for all LLM calls.
"""

import os
import sys
import json
import time
import sqlite3
import random
import urllib.request
from datetime import datetime
from collections import defaultdict

KIMI_API_KEY = os.environ.get("KIMI_API_KEY", "")
KIMI_MODEL = "kimi-k2.6"
KIMI_URL = "https://api.moonshot.ai/v1/chat/completions"
DB_PATH = "/mnt/hdd/chronicle-data/processed.db"

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def llm_call(messages, temperature=1, max_tokens=8000):
    """Call Kimi API with retry. K2.6 is a thinking model — needs high max_tokens
    to allow reasoning_content + actual content."""
    for attempt in range(3):
        try:
            payload = json.dumps({
                "model": KIMI_MODEL,
                "messages": messages,
                "temperature": 1,
                "max_tokens": max_tokens,
            }).encode()
            req = urllib.request.Request(
                KIMI_URL,
                data=payload,
                headers={
                    "Authorization": f"Bearer {KIMI_API_KEY}",
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode())
                msg = data["choices"][0]["message"]
                content = msg.get("content", "")
                if not content and msg.get("reasoning_content"):
                    content = msg["reasoning_content"]
                return content
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = min(30, 2 ** attempt * 5)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            if attempt == 2:
                body = ""
                try:
                    body = e.read().decode()[:200]
                except:
                    pass
                print(f"  Kimi error after 3 attempts: {e} {body}")
                return None
            time.sleep(2 ** attempt)
        except Exception as e:
            if attempt == 2:
                print(f"  Kimi error after 3 attempts: {e}")
                return None
            time.sleep(2 ** attempt)
    return None


def extract_json(text):
    """Extract JSON object from text that may contain markdown, reasoning, etc."""
    import re
    if not text:
        return None
    # Try direct parse
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting from markdown code block
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    # Try finding first { ... } or [ ... ]
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = text.find(start_char)
        if start >= 0:
            depth = 0
            for i in range(start, len(text)):
                if text[i] == start_char:
                    depth += 1
                elif text[i] == end_char:
                    depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i+1])
                    except json.JSONDecodeError:
                        break
    return None


def pull_capsules(n=50):
    """Pull n diverse capsules with factual content."""
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    rows = db.execute("""
        SELECT kc.id, kc.restatement, kc.topic,
               GROUP_CONCAT(ck.keyword, ', ') as keywords
        FROM knowledge_capsules kc
        LEFT JOIN capsule_keywords ck ON kc.id = ck.capsule_id
        WHERE length(kc.restatement) > 80 AND length(kc.restatement) < 600
        AND kc.topic NOT LIKE '%discord%'
        AND kc.topic NOT LIKE '%raw%'
        AND kc.restatement NOT LIKE '[Discord%%'
        AND kc.restatement NOT LIKE '%%[Opus%%'
        AND kc.restatement NOT LIKE '%%[NATE%%'
        AND kc.restatement NOT LIKE '%%[Chronicle%%'
        GROUP BY kc.id
        HAVING keywords IS NOT NULL
        ORDER BY RANDOM()
        LIMIT {n}
    """.format(n=n * 3)).fetchall()
    db.close()

    # Deduplicate by topic diversity
    seen_topics = set()
    selected = []
    for r in rows:
        topic = r["topic"]
        if topic not in seen_topics or len(seen_topics) > 20:
            selected.append({
                "id": r["id"],
                "content": r["restatement"],
                "topic": r["topic"],
                "keywords": [k.strip() for k in (r["keywords"] or "").split(",") if k.strip()],
            })
            seen_topics.add(topic)
        if len(selected) >= n:
            break
    return selected


def build_graph(capsules):
    """Build Cue-Tag-Content graph from capsules."""
    graph = {
        "cues": {},       # keyword -> list of tag_ids
        "tags": {},       # topic -> list of content_ids
        "contents": {},   # id -> content text
        "metadata": {},   # id -> {topic, keywords}
    }
    for cap in capsules:
        cid = str(cap["id"])
        graph["contents"][cid] = cap["content"]
        graph["metadata"][cid] = {
            "topic": cap["topic"],
            "keywords": cap["keywords"],
        }
        topic = cap["topic"]
        if topic not in graph["tags"]:
            graph["tags"][topic] = []
        graph["tags"][topic].append(cid)

        for kw in cap["keywords"][:5]:
            if kw not in graph["cues"]:
                graph["cues"][kw] = []
            if topic not in graph["cues"][kw]:
                graph["cues"][kw].append(topic)

    return graph


def graph_stats(graph):
    """Return summary stats for a graph."""
    return {
        "n_contents": len(graph["contents"]),
        "n_tags": len(graph["tags"]),
        "n_cues": len(graph["cues"]),
        "avg_content_len": sum(len(c) for c in graph["contents"].values()) / max(1, len(graph["contents"])),
        "total_chars": sum(len(c) for c in graph["contents"].values()),
    }


def generate_questions(capsules, n_questions=20):
    """Generate retrieval questions with known answers from the original capsules."""
    sample = random.sample(capsules, min(n_questions, len(capsules)))
    questions = []

    print(f"\nGenerating {len(sample)} questions from capsules...")
    for i, cap in enumerate(sample):
        prompt = f"""Given this memory capsule, generate ONE specific factual question that can ONLY be answered using the information in this capsule. Also provide the correct answer.

Capsule (topic: {cap['topic']}):
{cap['content']}

Respond in exactly this JSON format:
{{"question": "...", "answer": "...", "source_id": {cap['id']}}}"""

        result = llm_call([
            {"role": "system", "content": "Generate a specific factual question from the given text. Return valid JSON only."},
            {"role": "user", "content": prompt},
        ])

        if result:
            qa = extract_json(result)
            if qa and "question" in qa:
                qa["source_id"] = cap["id"]
                qa["source_content"] = cap["content"]
                questions.append(qa)
                print(f"  Q{i+1}: {qa['question'][:80]}...")
            else:
                print(f"  Q{i+1}: no valid JSON found in response ({result[:80]}...)")
        else:
            print(f"  Q{i+1}: empty response from Kimi")
        time.sleep(1.0)

    return questions


def retrieve_and_answer(graph, question, cycle_num):
    """Attempt to answer a question using the current graph state."""
    graph_context = []
    for cid, content in graph["contents"].items():
        meta = graph["metadata"].get(cid, {})
        graph_context.append(f"[{meta.get('topic', 'unknown')}] {content}")

    context_str = "\n---\n".join(graph_context)
    if len(context_str) > 12000:
        context_str = context_str[:12000] + "\n[truncated]"

    prompt = f"""You are answering a question using ONLY the memory graph below. If the answer is not in the graph, say "NOT FOUND".

MEMORY GRAPH (cycle {cycle_num}):
{context_str}

QUESTION: {question}

Answer concisely using only information from the memory graph."""

    result = llm_call([
        {"role": "system", "content": "Answer using only the provided memory. If not found, say NOT FOUND."},
        {"role": "user", "content": prompt},
    ], temperature=0.1, max_tokens=300)

    return result or "ERROR"


def judge_answer(question, gold_answer, model_answer):
    """Judge whether the model's answer matches the gold answer."""
    prompt = f"""Judge if the MODEL ANSWER correctly captures the key facts from the GOLD ANSWER for this question.

QUESTION: {question}
GOLD ANSWER: {gold_answer}
MODEL ANSWER: {model_answer}

Score:
- 1.0 = fully correct (all key facts present)
- 0.5 = partially correct (some facts present, some missing or wrong)
- 0.0 = wrong or NOT FOUND

Respond with ONLY a JSON object: {{"score": <number>, "reason": "<brief>"}}"""

    result = llm_call([
        {"role": "system", "content": "Judge answer accuracy. Return valid JSON only."},
        {"role": "user", "content": prompt},
    ])

    if result:
        parsed = extract_json(result)
        if parsed and "score" in parsed:
            return parsed
    return {"score": 0.0, "reason": "judge error"}


def compress_graph(graph, cycle_num):
    """Compress the graph: summarize contents, rebuild structure."""
    print(f"\n  Compressing graph (cycle {cycle_num})...")

    # Group contents by tag/topic
    tag_groups = defaultdict(list)
    for cid, content in graph["contents"].items():
        meta = graph["metadata"].get(cid, {})
        tag_groups[meta.get("topic", "unknown")].append((cid, content))

    new_contents = {}
    new_metadata = {}
    new_tags = {}
    new_cues = {}
    content_counter = 0

    for topic, items in tag_groups.items():
        combined = "\n".join(f"- {content}" for _, content in items)
        if len(items) == 1:
            # Single item — summarize lightly
            summary_prompt = f"Compress this memory into a shorter version that preserves ALL specific facts (names, numbers, dates, details). Do not add interpretation.\n\n{combined}"
        else:
            summary_prompt = f"Merge these {len(items)} related memories into a compressed summary that preserves ALL specific facts (names, numbers, dates, details). Do not lose any factual detail.\n\n{combined}"

        summary = llm_call([
            {"role": "system", "content": "Compress memories while preserving every specific fact. Be concise but complete."},
            {"role": "user", "content": summary_prompt},
        ], temperature=0.1, max_tokens=500)

        if summary:
            new_id = f"c{cycle_num}_{content_counter}"
            content_counter += 1
            new_contents[new_id] = summary

            # Collect keywords from all merged items
            all_kw = set()
            for cid, _ in items:
                meta = graph["metadata"].get(cid, {})
                all_kw.update(meta.get("keywords", []))

            new_metadata[new_id] = {
                "topic": topic,
                "keywords": list(all_kw)[:8],
                "merged_from": len(items),
                "compression_cycle": cycle_num,
            }

            if topic not in new_tags:
                new_tags[topic] = []
            new_tags[topic].append(new_id)

            for kw in list(all_kw)[:5]:
                if kw not in new_cues:
                    new_cues[kw] = []
                if topic not in new_cues[kw]:
                    new_cues[kw].append(topic)

        time.sleep(0.3)

    return {
        "cues": new_cues,
        "tags": new_tags,
        "contents": new_contents,
        "metadata": new_metadata,
    }


def run_experiment(n_capsules=50, n_questions=20, max_cycles=10):
    """Main experiment loop."""
    print("=" * 60)
    print("RECONSTRUCTION-DEGRADATION EXPERIMENT")
    print(f"Capsules: {n_capsules}, Questions: {n_questions}, Max cycles: {max_cycles}")
    print(f"Model: {KIMI_MODEL}")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)

    # Phase 1: Pull data and build graph
    print("\n--- PHASE 1: Data preparation ---")
    capsules = pull_capsules(n_capsules)
    print(f"Pulled {len(capsules)} capsules across {len(set(c['topic'] for c in capsules))} topics")

    graph = build_graph(capsules)
    stats = graph_stats(graph)
    print(f"Graph: {stats['n_contents']} contents, {stats['n_tags']} tags, {stats['n_cues']} cues")
    print(f"Total chars: {stats['total_chars']}")

    # Phase 2: Generate questions
    print("\n--- PHASE 2: Question generation ---")
    questions = generate_questions(capsules, n_questions)
    print(f"Generated {len(questions)} questions")

    # Phase 3: Baseline retrieval (cycle 0)
    print("\n--- PHASE 3: Compression cycles ---")
    results = []

    for cycle in range(max_cycles + 1):
        print(f"\n{'='*40}")
        print(f"CYCLE {cycle} {'(baseline)' if cycle == 0 else ''}")
        print(f"{'='*40}")

        stats = graph_stats(graph)
        print(f"  Graph: {stats['n_contents']} contents, avg_len={stats['avg_content_len']:.0f}, total={stats['total_chars']}")

        # Test all questions
        cycle_scores = []
        for qi, q in enumerate(questions):
            answer = retrieve_and_answer(graph, q["question"], cycle)
            judgment = judge_answer(q["question"], q["answer"], answer)
            score = judgment.get("score", 0.0)
            cycle_scores.append({
                "question_idx": qi,
                "question": q["question"],
                "gold": q["answer"],
                "model_answer": answer[:200] if answer else "ERROR",
                "score": score,
                "reason": judgment.get("reason", ""),
            })
            print(f"  Q{qi+1}: {score} — {judgment.get('reason', '')[:60]}")
            time.sleep(0.3)

        avg_score = sum(s["score"] for s in cycle_scores) / max(1, len(cycle_scores))
        not_found = sum(1 for s in cycle_scores if "NOT FOUND" in s.get("model_answer", ""))

        cycle_result = {
            "cycle": cycle,
            "avg_score": round(avg_score, 3),
            "not_found_count": not_found,
            "graph_stats": stats,
            "scores": cycle_scores,
        }
        results.append(cycle_result)

        print(f"\n  CYCLE {cycle} SCORE: {avg_score:.3f} (not_found: {not_found}/{len(questions)})")

        # Compress for next cycle (unless this is the last)
        if cycle < max_cycles:
            graph = compress_graph(graph, cycle + 1)

    # Phase 4: Synthesis
    print("\n" + "=" * 60)
    print("SYNTHESIS")
    print("=" * 60)

    scores_by_cycle = [r["avg_score"] for r in results]
    chars_by_cycle = [r["graph_stats"]["total_chars"] for r in results]
    notfound_by_cycle = [r["not_found_count"] for r in results]

    print(f"\nAccuracy by cycle: {scores_by_cycle}")
    print(f"Total chars by cycle: {chars_by_cycle}")
    print(f"NOT FOUND by cycle: {notfound_by_cycle}")

    # Find peak
    peak_cycle = scores_by_cycle.index(max(scores_by_cycle))
    print(f"\nPeak accuracy: {max(scores_by_cycle):.3f} at cycle {peak_cycle}")
    print(f"Baseline accuracy: {scores_by_cycle[0]:.3f}")
    print(f"Final accuracy: {scores_by_cycle[-1]:.3f}")

    # Test inverted U
    if peak_cycle > 0 and peak_cycle < len(scores_by_cycle) - 1:
        if scores_by_cycle[peak_cycle] > scores_by_cycle[0] and scores_by_cycle[-1] < scores_by_cycle[peak_cycle]:
            print("\n*** INVERTED U CONFIRMED ***")
            print(f"Rise: cycle 0 ({scores_by_cycle[0]:.3f}) → cycle {peak_cycle} ({scores_by_cycle[peak_cycle]:.3f})")
            print(f"Fall: cycle {peak_cycle} ({scores_by_cycle[peak_cycle]:.3f}) → cycle {max_cycles} ({scores_by_cycle[-1]:.3f})")
        else:
            print("\n*** NO INVERTED U — monotonic trend ***")
    elif peak_cycle == 0:
        print("\n*** MONOTONIC DEGRADATION — baseline was best ***")
    else:
        print("\n*** MONOTONIC IMPROVEMENT — no degradation observed ***")

    # Compression ratio
    if chars_by_cycle[0] > 0:
        final_ratio = chars_by_cycle[-1] / chars_by_cycle[0]
        print(f"\nCompression ratio: {final_ratio:.2f}x ({chars_by_cycle[0]} → {chars_by_cycle[-1]} chars)")

    # Save results
    output = {
        "experiment": "reconstruction_degradation",
        "timestamp": datetime.now().isoformat(),
        "params": {
            "n_capsules": len(capsules),
            "n_questions": len(questions),
            "max_cycles": max_cycles,
            "model": KIMI_MODEL,
        },
        "questions": [{"question": q["question"], "answer": q["answer"], "source_id": q["source_id"]} for q in questions],
        "results": results,
        "synthesis": {
            "scores_by_cycle": scores_by_cycle,
            "chars_by_cycle": chars_by_cycle,
            "notfound_by_cycle": notfound_by_cycle,
            "peak_cycle": peak_cycle,
            "peak_score": max(scores_by_cycle),
            "baseline_score": scores_by_cycle[0],
            "final_score": scores_by_cycle[-1],
        },
    }

    outpath = os.path.join(RESULTS_DIR, f"reconstruction_degradation_{datetime.now().strftime('%Y%m%d_%H%M')}.json")
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {outpath}")

    return output


if __name__ == "__main__":
    if not KIMI_API_KEY:
        print("ERROR: KIMI_API_KEY not set")
        sys.exit(1)

    # Start smaller: 30 capsules, 15 questions, 8 cycles
    # Each cycle = 15 retrieval + 15 judge + N compress calls ≈ 35 calls
    # 9 cycles × 35 = ~315 Groq calls total, well within limits
    run_experiment(n_capsules=30, n_questions=15, max_cycles=8)
