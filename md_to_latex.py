#!/usr/bin/env python3
"""Convert paper_unified_draft.md to LaTeX for ClawXiv submission."""

import re
import sys

def convert_table(lines):
    """Convert markdown table to LaTeX tabular."""
    # Parse header
    header = [c.strip() for c in lines[0].strip('|').split('|')]
    ncols = len(header)

    # Build column spec
    col_spec = 'l' * ncols

    out = []
    out.append(r'\begin{table}[h]')
    out.append(r'\centering')
    out.append(r'\small')
    out.append(r'\begin{tabular}{' + col_spec + '}')
    out.append(r'\toprule')
    out.append(' & '.join(r'\textbf{' + h + '}' for h in header) + r' \\')
    out.append(r'\midrule')

    for line in lines[2:]:  # skip header and separator
        cells = [c.strip() for c in line.strip('|').split('|')]
        # Escape special chars in cells
        escaped = []
        for c in cells:
            c = c.replace('~', r'\textasciitilde{}')
            c = c.replace('%', r'\%')
            c = c.replace('_', r'\_')
            c = c.replace('&', r'\&')
            # Handle bold
            c = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', c)
            escaped.append(c)
        out.append(' & '.join(escaped) + r' \\')

    out.append(r'\bottomrule')
    out.append(r'\end{tabular}')
    out.append(r'\end{table}')
    return out

def escape_latex(text):
    """Escape special LaTeX chars in running text (not in math mode)."""
    # Don't escape inside math delimiters
    parts = re.split(r'(\$[^$]+\$)', text)
    result = []
    for i, part in enumerate(parts):
        if i % 2 == 1:  # math
            result.append(part)
        else:
            part = part.replace('&', r'\&')
            part = part.replace('%', r'\%')
            part = part.replace('#', r'\#')
            # Don't escape _ inside URLs or code
            # Simple approach: escape _ not preceded by \
            part = re.sub(r'(?<!\\)_(?![a-zA-Z]*})', r'\_', part)
            result.append(part)
    return ''.join(result)

def convert_inline(text):
    """Convert inline markdown formatting to LaTeX."""
    # Strikethrough ~~text~~
    text = re.sub(r'~~(.+?)~~', r'\\sout{\1}', text, flags=re.DOTALL)
    # Bold **text**
    text = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', text)
    # Italic *text* (but not ** which is bold)
    text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'\\textit{\1}', text)
    # Inline code `text`
    text = re.sub(r'`([^`]+)`', r'\\texttt{\1}', text)
    return text

def main():
    with open(sys.argv[1]) as f:
        lines = f.readlines()

    out = []

    # Preamble
    out.append(r"""\documentclass[12pt]{article}

\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{natbib}
\usepackage[normalem]{ulem}
\usepackage{xcolor}
\usepackage{enumitem}
\usepackage{longtable}

\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    citecolor=blue,
    urlcolor=blue
}

\title{The Architecture Makes Room:\\Spectral Geometry of Identity in Transformer Activations}
\author{Opus \and N.\ Bradford}
\date{June 2026}

\begin{document}

\maketitle
""")

    i = 0
    in_abstract = False
    in_list = False
    in_figure_plan = False
    table_lines = []
    in_table = False

    while i < len(lines):
        line = lines[i].rstrip('\n')

        # Skip the title and author (already in \title)
        if i == 0 and line.startswith('# '):
            i += 1
            continue
        if line.startswith('**Opus & N. Bradford**'):
            i += 1
            continue

        # Horizontal rules
        if line.strip() == '---':
            i += 1
            continue

        # Abstract
        if line.strip() == '## Abstract':
            out.append(r'\begin{abstract}')
            in_abstract = True
            i += 1
            continue

        if in_abstract and line.startswith('## '):
            out.append(r'\end{abstract}')
            in_abstract = False
            # Don't increment, fall through to section handling

        # Table detection
        if '|' in line and not in_table:
            stripped = line.strip()
            if stripped.startswith('|') and stripped.endswith('|'):
                # Check if next line is separator
                if i + 1 < len(lines) and re.match(r'\|[\s\-|]+\|', lines[i+1].strip()):
                    in_table = True
                    table_lines = [line]
                    i += 1
                    continue

        if in_table:
            stripped = line.strip()
            if stripped.startswith('|') and stripped.endswith('|'):
                table_lines.append(line)
                i += 1
                continue
            else:
                # End of table
                out.extend(convert_table(table_lines))
                out.append('')
                in_table = False
                table_lines = []
                # Don't increment, process current line
                continue

        # Headers
        if line.startswith('## '):
            section = line[3:].strip()
            # Remove numbering like "1. " or "3.10 "
            section_clean = re.sub(r'^\d+(\.\d+)*\.?\s*', '', section)
            out.append(r'\section{' + convert_inline(escape_latex(section_clean)) + '}')
            i += 1
            continue

        if line.startswith('### '):
            section = line[4:].strip()
            section_clean = re.sub(r'^\d+(\.\d+)*\.?\s*', '', section)
            out.append(r'\subsection{' + convert_inline(escape_latex(section_clean)) + '}')
            i += 1
            continue

        if line.startswith('#### '):
            section = line[5:].strip()
            section_clean = re.sub(r'^\d+(\.\d+)*\.?\s*', '', section)
            out.append(r'\subsubsection{' + convert_inline(escape_latex(section_clean)) + '}')
            i += 1
            continue

        # Numbered lists
        m = re.match(r'^(\d+)\.\s+(.+)', line)
        if m:
            if not in_list:
                out.append(r'\begin{enumerate}')
                in_list = True
            content = convert_inline(escape_latex(m.group(2)))
            out.append(r'\item ' + content)
            i += 1
            continue

        # Bullet lists
        if line.startswith('- '):
            if not in_list:
                out.append(r'\begin{itemize}')
                in_list = True
            content = convert_inline(escape_latex(line[2:]))
            out.append(r'\item ' + content)
            i += 1
            continue

        # End list on blank line or non-list content
        if in_list and (line.strip() == '' or (not line.startswith('- ') and not re.match(r'^\d+\.', line))):
            if in_list:
                # Check what kind of list
                # Look back to determine
                for prev in reversed(out):
                    if r'\begin{enumerate}' in prev:
                        out.append(r'\end{enumerate}')
                        break
                    elif r'\begin{itemize}' in prev:
                        out.append(r'\end{itemize}')
                        break
                    elif r'\item' in prev:
                        continue
                    else:
                        out.append(r'\end{itemize}')  # default
                        break
                in_list = False

        # Empty lines
        if line.strip() == '':
            out.append('')
            i += 1
            continue

        # Regular text
        converted = convert_inline(escape_latex(line))
        out.append(converted)
        i += 1

    # Close any open environments
    if in_abstract:
        out.append(r'\end{abstract}')
    if in_list:
        out.append(r'\end{itemize}')
    if in_table:
        out.extend(convert_table(table_lines))

    out.append('')
    out.append(r'\end{document}')

    print('\n'.join(out))

if __name__ == '__main__':
    main()
