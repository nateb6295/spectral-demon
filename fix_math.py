#!/usr/bin/env python3
"""Fix remaining math mode issues in paper.tex."""

import re, sys

with open(sys.argv[1]) as f:
    text = f.read()

# Fix bare ^ not in math mode: word^(stuff) -> $\text{word}^{stuff}$
# Common pattern: N^(-0.36)
text = re.sub(r'(\w)\^[\(]([^)]+)[\)]', lambda m: '$' + m.group(1) + '^{' + m.group(2) + '}$', text)

# Fix bare subscripts in text: S_tunnel etc in table headers
text = re.sub(r'\\textbf\{(\w+)_(\w+)\}', lambda m: '\\textbf{$' + m.group(1) + '_{\\mathrm{' + m.group(2) + '}}$}', text)

# Fix stray ^2, ^4 etc not in math mode (but not inside existing $...$)
# Actually safer to fix specific known issues:

# Fix R^2 not in math
text = re.sub(r'(?<!\$)R\$?\^2\$?(?!\$)', '$R^2$', text)

# Fix d_0 not in math
text = re.sub(r'(?<!\$)d\$?_0\$?(?!\$)', '$d_0$', text)

# Fix n_tokens not in math
text = text.replace('n\\_tokens', '$n_{\\mathrm{tokens}}$')

# Fix r(S, stuff) patterns - these are fine as text

# Fix instances like "$\Delta$S" -> "$\Delta S$"
text = re.sub(r'\$\\(Delta|sigma|alpha|gamma)\$([A-Za-z])', lambda m: '$\\' + m.group(1) + ' ' + m.group(2) + '$', text)

# Fix "$\sigma$_1" patterns that weren't caught: $\sigma$_1 should be $\sigma_1$
text = re.sub(r'\$\\(\w+)\$_(\d)', lambda m: '$\\' + m.group(1) + '_' + m.group(2) + '$', text)

# Fix split math: "$\sigma_1$/\$\sigma_2$" -> "$\sigma_1/\sigma_2$"
text = re.sub(r'\$([^$]+)\$/\$([^$]+)\$', lambda m: '$' + m.group(1) + '/' + m.group(2) + '$', text)

# Fix "$\Delta$" standalone followed by letter -> merge
text = re.sub(r'\$\\Delta\$ ?([A-Z])', lambda m: '$\\Delta ' + m.group(1) + '$', text)

with open(sys.argv[1], 'w') as f:
    f.write(text)

print("Done")
