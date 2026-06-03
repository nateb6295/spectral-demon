#!/usr/bin/env python3
"""Audit the paper draft: section word counts, claim strength, figure status."""

import re
from pathlib import Path

DRAFT = Path(__file__).parent / "paper_draft.md"
FIGURES = Path(__file__).parent / "figures"

def section_stats(text):
    sections = re.split(r'^## ', text, flags=re.MULTILINE)
    total = 0
    print("=== SECTION WORD COUNTS ===")
    for s in sections[1:]:
        title = s.split('\n')[0].strip()
        words = len(s.split())
        total += words
        bar = '█' * (words // 50)
        print(f"  {title:45s} {words:5d}  {bar}")
    print(f"  {'TOTAL':45s} {total:5d}")
    return total

def claim_audit(text):
    print("\n=== CLAIMS NEEDING STATISTICAL BACKUP ===")
    weak = []
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if 'n=5' in line.lower() or 'n = 5' in line.lower():
            weak.append((i+1, line.strip()[:100]))
        if '5 trials' in line.lower():
            weak.append((i+1, line.strip()[:100]))
        if any(w in line.lower() for w in ['we acknowledge', 'thin for', 'planned']):
            weak.append((i+1, line.strip()[:100]))
    if weak:
        for ln, txt in weak:
            print(f"  L{ln}: {txt}")
    else:
        print("  None flagged.")

    print("\n=== QUANTITATIVE CLAIMS ===")
    quant = []
    patterns = [
        r'(\d+)×',
        r'r\s*[=>≈]\s*[\d.]+',
        r'p\s*<\s*[\d.]+',
        r'cos\s*=\s*[\d.]+',
        r'V₂\s*[=≈]\s*[+-]?[\d.]+',
        r'AUC\s*=\s*[\d.]+',
    ]
    for i, line in enumerate(lines):
        for pat in patterns:
            matches = re.findall(pat, line)
            if matches:
                quant.append((i+1, line.strip()[:120]))
                break
    print(f"  {len(quant)} lines with quantitative claims")

def figure_status():
    print("\n=== FIGURE STATUS ===")
    if FIGURES.exists():
        figs = sorted(FIGURES.glob("*.png"))
        print(f"  {len(figs)} figures in figures/")
        for f in figs:
            print(f"    {f.name}")
    else:
        print("  No figures/ directory")

    draft = DRAFT.read_text()
    fig_refs = re.findall(r'\(Figure \d+\)', draft)
    print(f"\n  {len(fig_refs)} figure references in draft:")
    for ref in sorted(set(fig_refs)):
        count = fig_refs.count(ref)
        print(f"    {ref} — {count}x")

def completeness():
    print("\n=== COMPLETENESS ===")
    draft = DRAFT.read_text()
    todos = []
    for marker in ['[TODO', '[TBD', 'to be added', 'to be formatted', 'PLACEHOLDER']:
        for i, line in enumerate(draft.split('\n')):
            if marker.lower() in line.lower():
                todos.append((i+1, line.strip()[:100]))
    if todos:
        for ln, txt in todos:
            print(f"  L{ln}: {txt}")
    else:
        print("  No TODOs found.")

    sections_present = re.findall(r'^## §(\d+)', draft, re.MULTILINE)
    print(f"\n  Sections present: §{', §'.join(sections_present)}")
    missing = []
    if 'References' not in draft and 'Bibliography' not in draft:
        missing.append("References/Bibliography")
    if 'Appendix' not in draft:
        missing.append("Appendix (optional)")
    if missing:
        print(f"  Missing: {', '.join(missing)}")

if __name__ == "__main__":
    text = DRAFT.read_text()
    section_stats(text)
    claim_audit(text)
    figure_status()
    completeness()
