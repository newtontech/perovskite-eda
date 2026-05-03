#!/usr/bin/env python3
"""Generate academic-style conceptual figures using Azure OpenAI GPT Image API."""

import base64
import os
from pathlib import Path
from openai import OpenAI

ENDPOINT = "https://szty2-Gpt5.openai.azure.com/openai/v1/"
DEPLOYMENT = "gpt-image-1.5"  # Keep using 1.5 if 2 is not available
API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

OUTPUT_DIR = Path(__file__).parent / "figures_v2"
OUTPUT_DIR.mkdir(exist_ok=True)

# Academic scientific illustration style - NO cartoon, NO playful characters
ACADEMIC_STYLE = (
    "A professional scientific illustration for a high-impact chemistry/materials science journal. "
    "Clean white background. Precise technical drawing style with clear labels. "
    "Color palette: muted scientific tones (navy blue, forest green, burnt orange, slate gray). "
    "No cartoon characters, no playful elements, no emoji faces. "
    "Serious academic aesthetic suitable for Journal of Physical Chemistry Letters or Nature Energy. "
    "All text labels in English, professional sans-serif font."
)

client = OpenAI(base_url=ENDPOINT, api_key=API_KEY)


def generate(prompt: str, output_path: Path, size: str = "1536x1024", quality: str = "high"):
    print(f"Generating: {output_path.name}")
    resp = client.images.generate(
        model=DEPLOYMENT,
        prompt=prompt,
        n=1,
        size=size,
        quality=quality,
    )
    data = base64.b64decode(resp.data[0].b64_json)
    output_path.write_bytes(data)
    print(f"  Saved: {output_path} ({len(data)} bytes)")


# Conceptual Figure S1: Multi-Agent Architecture (academic style)
prompt_s1 = f"""{ACADEMIC_STYLE}

Title: "Multi-Agent Cross-Layer Exploration Architecture for Perovskite Solar Cell Additive Design"

A clean, professional 5-layer horizontal pipeline diagram:

Top row - Five rounded rectangular boxes connected by thick arrows, left to right:
1. "L1: Data Sources" (navy blue box) - icon shows database cylinders, document icons, and a funnel filtering 91,357 to 4,934 samples
2. "L2: Feature Engineering" (forest green box) - icon shows molecular structures, fingerprint bitstrings, and descriptor tables
3. "L3: Model Selection" (burnt orange box) - icon shows decision trees, gradient boost diagrams, and neural network nodes
4. "L4: Evaluation" (slate gray box) - icon shows cross-validation folds, ROC curves, and SHAP beeswarm plots
5. "L5: Virtual Screening" (purple box) - icon shows molecular library, ranking bars, and target molecules

Below the pipeline, a large gray box labeled "Multi-Agent Orchestrator" with sub-labels:
- "Weighted Random Sampling"
- "Multiprocessing Spawn (Deadlock-Free)"
- "Checkpointing & Resume"
- "Leaderboard Ranking"

Thin dashed lines connect the orchestrator to each layer box.

Style: Flat design, minimal shadows, clean vector-like appearance. No 3D effects. No cartoon characters. Professional technical diagram aesthetic. White background."""

# Conceptual Figure S2: SAM Molecular Design (academic style)
prompt_s2 = f"""{ACADEMIC_STYLE}

Title: "Structure-Performance Relationship in SAM Hole Transport Materials"

A professional 2×2 grid of scientific diagrams:

Top-left (a): "SAM Molecular Architecture"
- Cross-section diagram of an inverted perovskite solar cell
- ITO substrate at bottom (gray rectangle)
- SAM monolayer above ITO (thin ordered molecular layer with phosphonic acid anchors pointing down)
- Perovskite absorber layer above SAM (orange-yellow gradient crystal lattice pattern)
- Three SAM molecules drawn in 2D chemical structure style:
  * Left: phosphonic acid anchor (PO3H2) connected to benzene ring spacer, connected to morpholine head group with SCN side chain
  * Center: same anchor + spacer, piperidine head group with SCN side chain  
  * Right: same anchor + spacer, piperidine head group with thioamide side chain
- Labels clearly mark: "Anchor Group", "Spacer/Linker", "Head Group", "Side Chain"
- Small electron (e-) and hole (h+) arrows showing charge transport direction

Top-right (b): "Key SHAP Descriptors"
- Four horizontal bar charts showing descriptor importance
- Bars labeled: EState_VSA5, fr_benzene, SlogP_VSA1, Chi0v
- Color gradient from dark blue (high) to light gray (low)
- Small molecular fragment icons next to each bar showing what the descriptor represents

Bottom-left (c): "Design Rules from ML Model"
- Three checkmarked rules with simple molecular diagrams:
  1. "Include aromatic rings" - shows benzene rings in ordered linear arrangement
  2. "Optimize electrostatic surface" - shows dipole moment vector on molecule
  3. "Balance lipophilicity (LogP ≈ 2)" - shows hydrophobic/hydrophilic balance scale

Bottom-right (d): "Predicted vs Experimental Validation"
- Scatter plot showing predicted PCE on x-axis vs experimental PCE on y-axis
- Points cluster near y=x diagonal line
- Three new SAM molecules marked with red stars at high PCE values (~26-27%)
- Inset text box: "Mean absolute error = 2.2%"

Style: Clean technical drawing. Molecule structures drawn in standard chemical notation (Kekulé style for aromatic rings). No cartoon characters. White background. Navy blue and forest green as primary accent colors."""

if __name__ == '__main__':
    generate(prompt_s1, OUTPUT_DIR / "figS1_architecture_academic.png")
    generate(prompt_s2, OUTPUT_DIR / "figS2_sam_design_academic.png")
    print("\nAll academic conceptual figures generated.")
