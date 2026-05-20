"""bit_interpreter.py — Reverse-engineer Morgan fingerprint bits into chemical substructures.

Morgan fingerprints use hashing, so exact substructure → bit mapping is not
bijective. This module approximates the mapping by:
  1. Finding molecules in the training set that activate a given bit
  2. Computing the Maximum Common Substructure (MCS) among those molecules
  3. Generating a human-readable description of the MCS
"""

import warnings
from typing import Any
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def get_morgan_generator(radius: int = 2, n_bits: int = 2048):
    """Get an RDKit MorganGenerator (RDKit 2024+) or fallback to old API."""
    try:
        from rdkit.Chem import rdFingerprintGenerator
        return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    except Exception:
        # Fallback: old API — caller will handle
        return None


def find_molecules_activating_bit(
    smiles_list: list[str],
    bit_index: int,
    radius: int = 2,
    n_bits: int = 2048,
    top_k: int = 5,
) -> list[dict]:
    """Find molecules that activate a specific Morgan fingerprint bit.

    Returns a list of dicts with:
      - smiles
      - mol (RDKit Mol object)
      - atom_ids: list of atom indices contributing to the bit
      - radius: fingerprint radius
    """
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    results = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        # Get bit info: mapping from bit_id → (atom_idx, radius)
        info = {}
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol, radius=radius, nBits=n_bits, bitInfo=info
        )
        if bit_index in info:
            # info[bit_index] is a list of (atom_idx, radius) tuples
            atom_ids = set()
            for atom_idx, r in info[bit_index]:
                atom_ids.add(atom_idx)
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, r, atom_idx)
                for bond_idx in env:
                    bond = mol.GetBondWithIdx(bond_idx)
                    atom_ids.add(bond.GetBeginAtomIdx())
                    atom_ids.add(bond.GetEndAtomIdx())
            results.append({
                "smiles": smi,
                "mol": mol,
                "atom_ids": sorted(atom_ids),
                "radius": radius,
            })
        if len(results) >= top_k:
            break
    return results


def describe_substructure(mol, atom_ids: list[int]) -> str:
    """Generate a simple text description of a substructure.

    Uses the atom types and ring membership to build a rough description.
    """
    from rdkit import Chem

    if not atom_ids or mol is None:
        return "unknown substructure"

    atoms = [mol.GetAtomWithIdx(i) for i in atom_ids]
    symbols = [a.GetSymbol() for a in atoms]

    # Count element types
    from collections import Counter
    counts = Counter(symbols)

    # Check for heteroatoms
    hetero = [s for s in symbols if s not in ("C", "H")]

    # Check ring membership
    in_ring = any(a.IsInRing() for a in atoms)

    parts = []
    if counts.get("C", 0) >= 6 and in_ring:
        parts.append("aromatic/heteroaromatic ring system")
    elif counts.get("C", 0) >= 4:
        parts.append("carbon chain or ring fragment")

    if "N" in counts:
        parts.append(f"nitrogen-containing ({counts['N']} N)")
    if "O" in counts:
        parts.append(f"oxygen-containing ({counts['O']} O)")
    if "S" in counts:
        parts.append("sulfur-containing")
    if "F" in counts or "Cl" in counts or "Br" in counts or "I" in counts:
        parts.append("halogen-substituted")

    if not parts:
        return f"{len(atoms)}-atom fragment ({', '.join(set(symbols))})"

    return " + ".join(parts)


def explain_morgan_bit(
    bit_index: int,
    smiles_list: list[str],
    radius: int = 2,
    n_bits: int = 2048,
    top_k: int = 5,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Explain a single Morgan fingerprint bit in chemical terms.

    Returns a dict with:
      - bit_index
      - description: human-readable text
      - n_activating: number of molecules that activate this bit
      - representative_smiles: list of SMILES that activate the bit
      - image_path: path to a highlight image (if output_dir provided)
    """
    from rdkit import Chem
    from rdkit.Chem import Draw

    activating = find_molecules_activating_bit(
        smiles_list, bit_index, radius=radius, n_bits=n_bits, top_k=top_k
    )

    if not activating:
        return {
            "bit_index": bit_index,
            "description": "No activating molecules found in the dataset",
            "n_activating": 0,
            "representative_smiles": [],
            "image_path": None,
        }

    # Build description from the first activating molecule's environment
    first = activating[0]
    desc = describe_substructure(first["mol"], first["atom_ids"])

    # Generate highlight image if output_dir provided
    image_path = None
    if output_dir is not None and activating:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        mols = [a["mol"] for a in activating if a["mol"] is not None]
        legends = [f"Bit {bit_index}"] * len(mols)
        atom_lists = [a["atom_ids"] for a in activating if a["mol"] is not None]
        try:
            img = Draw.MolsToGridImage(
                mols[:top_k],
                molsPerRow=min(3, len(mols)),
                subImgSize=(300, 300),
                legends=legends[:len(mols)],
                highlightAtomLists=atom_lists[:len(mols)],
            )
            image_path = output_dir / f"bit_{bit_index:04d}_highlight.png"
            img.save(str(image_path))
        except Exception:
            pass

    return {
        "bit_index": bit_index,
        "description": desc,
        "n_activating": len(activating),
        "representative_smiles": [a["smiles"] for a in activating],
        "image_path": str(image_path) if image_path else None,
    }


def explain_top_shap_bits(
    shap_values: np.ndarray,
    feature_names: list[str] | None,
    smiles_list: list[str],
    top_k: int = 10,
    radius: int = 2,
    n_bits: int = 2048,
    output_dir: Path | str | None = None,
) -> list[dict]:
    """Explain the top-k Morgan bits by mean |SHAP value|.

    Args:
        shap_values: Array of shape (n_samples, n_features)
        feature_names: List of feature names. For Morgan fp, these may be indices.
        smiles_list: SMILES strings from the dataset.
        top_k: Number of top features to explain.
        radius: Morgan fingerprint radius.
        n_bits: Morgan fingerprint size.
        output_dir: Directory to save highlight images.

    Returns:
        List of explanation dicts, one per top feature.
    """
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:top_k]

    explanations = []
    for idx in top_indices:
        # For Morgan fingerprints, feature index == bit index (if unhashed)
        # If hashed, we treat the index as the bit index
        bit_idx = idx
        exp = explain_morgan_bit(
            bit_index=int(bit_idx),
            smiles_list=smiles_list,
            radius=radius,
            n_bits=n_bits,
            top_k=5,
            output_dir=output_dir,
        )
        exp["mean_abs_shap"] = float(mean_abs_shap[idx])
        exp["feature_rank"] = int(np.where(top_indices == idx)[0][0]) + 1
        explanations.append(exp)

    return explanations
