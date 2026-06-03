#!/usr/bin/env python3
"""Replace Unicode characters with LaTeX equivalents."""

import re, sys

with open(sys.argv[1]) as f:
    text = f.read()

replacements = [
    ('α', '$\\alpha$'),
    ('β', '$\\beta$'),
    ('γ', '$\\gamma$'),
    ('δ', '$\\delta$'),
    ('σ', '$\\sigma$'),
    ('Δ', '$\\Delta$'),
    ('π', '$\\pi$'),
    ('Σ', '$\\Sigma$'),
    ('₁', '$_1$'),
    ('₂', '$_2$'),
    ('₃', '$_3$'),
    ('₄', '$_4$'),
    ('ᵢ', '$_i$'),
    ('ⱼ', '$_j$'),
    ('ₖ', '$_k$'),
    ('≥', '$\\geq$'),
    ('≤', '$\\leq$'),
    ('≈', '$\\approx$'),
    ('×', '$\\times$'),
    ('→', '$\\to$'),
    ('−', '$-$'),
    ('±', '$\\pm$'),
    ('∝', '$\\propto$'),
    ('∞', '$\\infty$'),
    ('√', '$\\sqrt{}$'),
    ('°', '$^\\circ$'),
    ('—', '---'),
    ('–', '--'),
    ('“', '``'),
    ('”', "''"),
    ('‘', '`'),
    ('’', "'"),
    ('≡', '$\\equiv$'),
]

for old, new in replacements:
    text = text.replace(old, new)

# Fix adjacent math: $\sigma$$_1$ -> $\sigma_1$
text = re.sub(r'\$\$', '', text)

# But we may have killed legitimate $$ ... let's check: we shouldn't have any display math
# Fix ~4500 etc -- tilde in running text
text = text.replace('\\textasciitilde{}4500', '{\\textasciitilde}4500')

with open(sys.argv[1], 'w') as f:
    f.write(text)

print("Done")
