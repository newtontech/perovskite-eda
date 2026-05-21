#!/usr/bin/env python3
"""Reorder references by appearance order and generate PDF.

Key: convert [^N] footnote syntax to plain [N] inline citations,
so pandoc does NOT treat them as footnotes (which go to page bottom).
References are collected as a numbered list at the document end.
"""

import re
import subprocess
import sys
from pathlib import Path

ARTICLE_DIR = Path(__file__).parent
MD_FILE = ARTICLE_DIR / "manuscript_v2.md"
OUT_MD = ARTICLE_DIR / "manuscript_v2_ordered.md"
OUT_PDF = ARTICLE_DIR / "manuscript_v2.pdf"
HEADER = ARTICLE_DIR / "header.tex"


def find_citation_order(text):
    """Find the order of first appearance of each citation in the body."""
    parts = text.split("## References")
    body = parts[0]

    seen = set()
    order = []
    for m in re.finditer(r'\[\^(\d+)\]', body):
        num = int(m.group(1))
        if num not in seen:
            seen.add(num)
            order.append(num)
    return order


def extract_ref_defs(text):
    """Extract reference definitions from the ## References section."""
    ref_match = re.search(r'^## References\s*$', text, re.MULTILINE)
    if not ref_match:
        print("ERROR: Could not find '## References' section")
        sys.exit(1)

    after_ref_header = text[ref_match.end():]

    ref_defs = {}
    current_num = None
    current_lines = []
    for line in after_ref_header.split('\n'):
        ref_start = re.match(r'^\[\^(\d+)\]:\s*', line)
        if ref_start:
            if current_num is not None:
                ref_defs[current_num] = '\n'.join(current_lines)
            current_num = int(ref_start.group(1))
            # Strip the [^N]: prefix, keep only the reference text
            current_lines = [re.sub(r'^\[\^\d+\]:\s*', '', line)]
        elif current_num is not None:
            stripped = line.strip()
            if stripped and not stripped.startswith('#') and not stripped.startswith('---'):
                current_lines.append(line)
            else:
                if current_num is not None:
                    ref_defs[current_num] = '\n'.join(current_lines)
                    current_num = None

    if current_num is not None:
        ref_defs[current_num] = '\n'.join(current_lines)

    return ref_defs, text[:ref_match.start()]


def reorder_and_rebuild(text):
    """Reorder references by appearance, convert to inline citations, place at end."""
    citation_order = find_citation_order(text)
    ref_defs, body = extract_ref_defs(text)

    # Build old->new mapping
    old_to_new = {}
    for new_num, old_num in enumerate(citation_order, start=1):
        old_to_new[old_num] = new_num

    # Step 1: Replace [^N] in body with [N] (plain text, NOT footnote syntax)
    # Use placeholders to avoid number collision
    for old_num in sorted(old_to_new.keys(), reverse=True):
        placeholder = f'__REFPH{old_num}__'
        body = body.replace(f'[^{old_num}]', placeholder)

    for old_num, new_num in old_to_new.items():
        body = body.replace(f'__REFPH{old_num}__', f'[{new_num}]')

    # Step 2: Clean trailing whitespace/footer from body
    body = body.rstrip()
    footer_pattern = re.compile(
        r'\n\*Submitted to.*?\n\n\*Corresponding Author:.*?\n\*Email:.*?\n\n\*Keywords:.*?$',
        re.DOTALL
    )
    body = footer_pattern.sub('', body)

    # Step 3: Build References section as a numbered list (not footnotes)
    refs_section = "\n\n## References\n\n"
    for new_num in range(1, len(citation_order) + 1):
        old_num = citation_order[new_num - 1]
        if old_num in ref_defs:
            ref_text = ref_defs[old_num].strip()
            refs_section += f"{new_num}. {ref_text}\n\n"

    # Footer at very end
    footer = "\n---\n\n*Submitted to example journal*\n\n*Corresponding Author: [Author Name]*  \n*Email: [email@institution.edu]*\n"

    result = body + refs_section + footer
    result = re.sub(r'\n{3,}', '\n\n', result)

    return result


def main():
    print(f"Reading {MD_FILE}...")
    text = MD_FILE.read_text(encoding='utf-8')

    print("Reordering references by appearance (inline [N] format)...")
    result = reorder_and_rebuild(text)

    print(f"Writing ordered markdown to {OUT_MD}...")
    OUT_MD.write_text(result, encoding='utf-8')

    print(f"Generating PDF with pandoc + xelatex...")
    cmd = [
        'pandoc',
        str(OUT_MD),
        '-o', str(OUT_PDF),
        '--pdf-engine=xelatex',
        '-H', str(HEADER),
        '--resource-path', str(ARTICLE_DIR),
        '-V', 'mainfont=DejaVu Serif',
        '-V', 'sansfont=DejaVu Sans',
        '-V', 'monofont=DejaVu Sans Mono',
        '-V', 'fontsize=11pt',
        '-V', 'documentclass=article',
        '-V', 'colorlinks=true',
        '-V', 'linkcolor=blue',
        '-V', 'urlcolor=blue',
        '--toc',
        '--toc-depth=3',
        '--number-sections',
        '-f', 'markdown+pipe_tables-fenced_code_blocks',
        '-s',
        '--wrap=none',
    ]

    print(f"Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ARTICLE_DIR))

    if proc.returncode != 0:
        print(f"STDERR:\n{proc.stderr}")
        print(f"STDOUT:\n{proc.stdout}")
        print(f"Pandoc failed with exit code {proc.returncode}")
        sys.exit(1)

    print(f"\nPDF generated successfully: {OUT_PDF}")
    print(f"File size: {OUT_PDF.stat().st_size / 1024:.1f} KB")


if __name__ == '__main__':
    main()
