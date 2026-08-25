#!/usr/bin/env python3
"""Map finding cross-references across all working documents.

Usage:
  finding_graph.py                # show most-connected findings
  finding_graph.py F122           # show what references F122 and what F122 references
  finding_graph.py --load-bearing # findings referenced by 3+ other findings
"""
import re, sys, glob
from pathlib import Path
from collections import defaultdict

SEARCH_PATHS = [
    Path.home() / "chronicle" / ".claude" / "projects" / "-home-nate-agx-chronicle" / "memory" / "*.md",
    Path.home() / "chronicle" / "spectral-demon" / "*.md",
    Path.home() / "chronicle" / "cycle-context.md",
]

FINDING_RE = re.compile(r'\bF(\d{2,3})\b')


def scan_files():
    files = []
    for pattern in SEARCH_PATHS:
        files.extend(glob.glob(str(pattern)))
    return files


def extract_co_occurrences():
    """Find which findings appear in the same paragraph/section."""
    co_ref = defaultdict(lambda: defaultdict(int))
    mention_count = defaultdict(int)
    file_mentions = defaultdict(set)

    for fpath in scan_files():
        fname = Path(fpath).name
        with open(fpath) as f:
            text = f.read()

        for para in re.split(r'\n\n+', text):
            findings = set(f"F{m}" for m in FINDING_RE.findall(para))
            for f1 in findings:
                mention_count[f1] += 1
                file_mentions[f1].add(fname)
                for f2 in findings:
                    if f1 != f2:
                        co_ref[f1][f2] += 1

    return co_ref, mention_count, file_mentions


def cmd_overview(co_ref, mention_count, file_mentions):
    ranked = sorted(mention_count.items(), key=lambda x: -x[1])
    print(f"{'Finding':>8} {'Mentions':>9} {'Co-refs':>8} {'Files':>6}")
    print("-" * 38)
    for finding, count in ranked[:25]:
        n_co = len(co_ref.get(finding, {}))
        n_files = len(file_mentions.get(finding, set()))
        print(f"{finding:>8} {count:>9} {n_co:>8} {n_files:>6}")


def cmd_detail(finding, co_ref, mention_count, file_mentions):
    if finding not in mention_count:
        print(f"{finding} not found in any document.")
        return

    print(f"\n{finding}: {mention_count[finding]} mentions across {len(file_mentions[finding])} files")
    print(f"  Files: {', '.join(sorted(file_mentions[finding]))}")

    refs = co_ref.get(finding, {})
    if refs:
        print(f"\n  Co-occurs with ({len(refs)} findings):")
        for other, count in sorted(refs.items(), key=lambda x: -x[1]):
            print(f"    {other}: {count}x")
    else:
        print("\n  No co-occurrences with other findings.")


def cmd_load_bearing(co_ref, mention_count, file_mentions, threshold=3):
    print(f"Load-bearing findings (co-referenced by {threshold}+ others):\n")
    candidates = []
    for finding in mention_count:
        n_co = len(co_ref.get(finding, {}))
        if n_co >= threshold:
            candidates.append((finding, n_co, mention_count[finding]))

    candidates.sort(key=lambda x: -x[1])
    for finding, n_co, n_mentions in candidates:
        neighbors = sorted(co_ref[finding].keys(), key=lambda x: -co_ref[finding][x])[:5]
        neighbor_str = ", ".join(neighbors)
        print(f"  {finding}: {n_co} connections, {n_mentions} mentions — [{neighbor_str}]")


def main():
    co_ref, mention_count, file_mentions = extract_co_occurrences()

    if len(sys.argv) < 2:
        cmd_overview(co_ref, mention_count, file_mentions)
    elif sys.argv[1] == "--load-bearing":
        threshold = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        cmd_load_bearing(co_ref, mention_count, file_mentions, threshold)
    elif sys.argv[1].startswith("F"):
        cmd_detail(sys.argv[1], co_ref, mention_count, file_mentions)
    else:
        print(f"Usage: finding_graph.py [F<num>] [--load-bearing [N]]")


if __name__ == "__main__":
    main()
