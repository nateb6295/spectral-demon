#!/usr/bin/env python3
"""Fix remaining Unicode characters."""

import sys

with open(sys.argv[1]) as f:
    text = f.read()

replacements = [
    ('š', '\\v{s}'),
    ('ᵀ', '$^T$'),
    ('⁴', '$^4$'),
    ('⁻', '$^-$'),
    ('₀', '$_0$'),
    ('∘', '$\\circ$'),
    ('≠', '$\\neq$'),
    ('θ', '$\\theta$'),
    ('κ', '$\\kappa$'),
    ('§', '\\S{}'),
    ('²', '$^2$'),
    ('·', '$\\cdot$'),
    ('¹', '$^1$'),
]

for old, new in replacements:
    text = text.replace(old, new)

# Now fix doubled math delimiters created by adjacent replacements
# e.g., $\sigma$$_1$ -> should be $\sigma_1$
import re
# Pattern: end of math immediately followed by start of math
text = re.sub(r'\$\$', '', text)

with open(sys.argv[1], 'w') as f:
    f.write(text)

print("Done")
