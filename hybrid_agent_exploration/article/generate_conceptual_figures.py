#!/usr/bin/env python3
"""Generate conceptual figures using Azure OpenAI GPT Image 1.5."""

import base64
import os
from pathlib import Path
from openai import OpenAI

ENDPOINT = "https://szty2-Gpt5.openai.azure.com/openai/v1/"
DEPLOYMENT = "gpt-image-1.5"
API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

OUTPUT_DIR = Path(__file__).parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

STYLE_TEMPLATE = (
    "A vibrant and playful scientific illustration for academic review, "
    "featuring iconic cartoon-style symbols with clear English text labels. "
    "Clean white background with colorful flat design elements. "
    "Professional yet accessible storytelling layout that effectively "
    "communicates research insights to academic peers."
)

client = OpenAI(base_url=ENDPOINT, api_key=API_KEY)


def generate_image(prompt: str, output_path: Path, size: str = "1536x1024", quality: str = "high"):
    print(f"Generating: {output_path.name}")
    resp = client.images.generate(
        model=DEPLOYMENT,
        prompt=prompt,
        n=1,
        size=size,
        quality=quality,
    )
    image_data = base64.b64decode(resp.data[0].b64_json)
    output_path.write_bytes(image_data)
    print(f"  Saved: {output_path} ({len(image_data)} bytes)")


# Figure S1: Multi-Agent Architecture Concept
prompt_s1 = f"""{STYLE_TEMPLATE}

Title: "Multi-Agent Cross-Layer Exploration Architecture for Perovskite Solar Cell Additive Design"

This single-panel workflow diagram uses a top-down storytelling layout with five colorful horizontal layers connected by vertical data flow arrows:

Layer 1 - Data Collection (top, blue theme): Friendly robot librarian character (blue metallic body with glasses, holding a magnifying glass) standing in front of a massive bookshelf made of solar panel icons. Books have smiling faces with chemical structure spines. Speech bubble says "91,357 Papers → 4,934 Clean Samples". Small document icons with perovskite crystal faces floating around. Large blue downward arrow labeled "Cleaned Dataset".

Layer 2 - Feature Engineering (green theme): Green robot chemist character with lab coat and molecular orbital eyes, holding a large colorful molecule model. Surrounding the robot are five feature icons: "RDKit Descriptors" (blue beaker with numbers), "MACCS Keys" (green keyring with 166 tiny keys), "ECFP Fingerprints" (orange circular maze pattern), "Atom Pair" (purple connected dots), "Topological Torsion" (yellow twisted ribbon). Green downward arrow labeled "Feature Matrix (N × D)".

Layer 3 - Multi-Agent Model Training (center, orange theme): Four worker robot characters in orange hard hats working in parallel at separate workstations. Each workstation has a different ML model mascot: "Random Forest" (blue tree character with leaf crown), "XGBoost" (green rocket character), "LightGBM" (yellow lightbulb character), "SVR" (purple support vector hyperplane character). Above them, a large golden orchestrator robot with a conductor's baton coordinates the workers. Speech bubble from orchestrator says "Explore → Execute → Rank". Orange downward arrow labeled "Trained Models + Metrics".

Layer 4 - Evaluation & Interpretability (purple theme): Purple owl scientist character with graduation cap and SHAP logo on chest, standing in front of three magical crystal balls. Left ball shows "R² = 0.296" with green glow. Center ball shows SHAP beeswarm plot with colorful dots. Right ball shows "RMSE = 2.24%" with blue glow. Small magnifying glass icons examining each ball. Purple downward arrow labeled "Insights + Design Rules".

Layer 5 - Virtual Screening & New Molecules (bottom, yellow theme): Yellow explorer robot with a treasure map standing on a cliff overlooking a vast molecular landscape. In the landscape below: three shining golden molecules on pedestals labeled "SAM-1" (26.8%), "SAM-2" (25.4%), "SAM-3" (24.9%). Behind them, a factory building labeled "Self-Driving Lab". Rainbow arc above with "Discovery!" text. Small confetti and star icons celebrating.

All layers connected by thick colored arrows (blue → green → orange → purple → yellow). Small decorative icons scattered: gear icons, lightbulb icons, crystal icons, solar panel icons. Background is clean white. All text in English with playful rounded font. Color palette: blue, green, orange, purple, yellow gradients."""

# Figure S2: SAM Molecular Design Concept
prompt_s2 = f"""{STYLE_TEMPLATE}

Title: "Structure-Performance Relationship in SAM Hole Transport Materials for Perovskite Solar Cells"

This composite figure uses a 2×2 grid layout with uniform spacing:

(a) SAM Molecular Architecture (top-left): Large central illustration of a generic SAM molecule shown as a friendly character with three labeled body parts. "Anchor Group" (left foot, blue phosphonic acid icon with PO3H2 label, strong binding to ITO substrate). "Spacer/Linker" (body, green flexible chain character with benzene ring segments, controlling molecular tilt and packing density). "Head Group" (right hand, red amine/piperidine character with N label, responsible for hole extraction). Background shows perovskite crystal layer (orange cube pattern) and ITO substrate (gray grid). Small electron and hole characters (blue minus and red plus) moving through the SAM layer. Label "p-i-n Device Structure" at top.

(b) Key Descriptors from SHAP Analysis (top-right): Four descriptor characters standing on importance podiums. "EState_VSA5" (gold trophy, 1st place, electrostatic surface area character with lightning bolts). "fr_benzene" (silver trophy, 2nd place, benzene ring stack character with aromatic cloud). "SlogP_VSA1" (bronze trophy, 3rd place, oil drop character with water/oil separation). "Chi0v" (4th place, branching tree character with connectivity branches). Podiums labeled with SHAP importance scores. Background has molecular formula floating around.

(c) Design Rules from ML Model (bottom-left): Three design rule icons in vertical arrangement. Rule 1: "Include Aromatic Rings" — illustrated as benzene ring characters holding hands in ordered line. Rule 2: "Optimize Electrostatic Surface" — illustrated as dipole character (magnet with plus/minus) aligning with perovskite surface. Rule 3: "Balance Lipophilicity" — illustrated as scale balance with hydrophobic oil drop on one side and hydrophilic water drop on other, centered at "LogP ≈ 2". Each rule has a green checkmark icon.

(d) Predicted vs Experimental Validation (bottom-right): Two friendly character scientists shaking hands. Left scientist (blue, labeled "ML Prediction") holds a crystal ball showing "Predicted PCE = 26.8%". Right scientist (green, labeled "Experiment") holds a solar cell device showing "Measured PCE = 26.2%". Between them, a small gauge icon shows "Error = 2.2%". Speech bubble says "Excellent Agreement!". Background has laboratory equipment icons (beakers, solar simulators, measurement probes).

All subfigures use consistent vibrant color palette (blue, green, orange, purple, yellow, red), friendly cartoon-style character mascots with expressive faces, clean white background, and clear English labels in playful rounded font."""

if __name__ == '__main__':
    generate_image(prompt_s1, OUTPUT_DIR / "figS1_architecture_concept.png")
    generate_image(prompt_s2, OUTPUT_DIR / "figS2_sam_design_concept.png")
    print("\nAll conceptual figures generated.")
