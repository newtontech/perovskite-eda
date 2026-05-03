"""
feature_generators.py

Modular generators for Layer 2 molecular representations:
  F21 — RDKit molecular descriptors (basic + full ~200)
  F22 — Molecular fingerprints (ECFP4, MACCS, KRFP, Atom Pair, Topological Torsion)

Follows the conventions in features/fingerprints.py and features/rdkit_descriptors.py
"""

import sys
from pathlib import Path

# Allow imports from project root (features/, src/, etc.)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, AllChem, MACCSkeys, rdMolDescriptors
from rdkit.Chem.AtomPairs import Torsions

# Suppress verbose RDKit error logging for invalid SMILES
rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.warning')

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_mol(smiles):
    """Return an RDKit Mol or None."""
    if pd.isna(smiles):
        return None
    return Chem.MolFromSmiles(str(smiles))


def _to_bitvect(fp, n_bits):
    """Convert various RDKit fingerprint types to a dense numpy array."""
    if hasattr(fp, "ToList"):
        return np.array(fp.ToList(), dtype=np.int8)
    if hasattr(fp, "GetNumOnes"):
        arr = np.zeros(n_bits, dtype=np.int8)
        for idx in fp.GetOnBits():
            if idx < n_bits:
                arr[idx] = 1
        return arr
    return np.array(list(fp), dtype=np.int8)


# ---------------------------------------------------------------------------
# F21 — RDKit Molecular Descriptors
# ---------------------------------------------------------------------------

BASIC_DESCRIPTOR_FUNCS = {
    "MolWt": Descriptors.MolWt,
    "LogP": Descriptors.MolLogP,
    "TPSA": Descriptors.TPSA,
    "HBD": Descriptors.NumHDonors,
    "HBA": Descriptors.NumHAcceptors,
    "RotBonds": Descriptors.NumRotatableBonds,
    "AromaticRings": Descriptors.NumAromaticRings,
    "HeavyAtoms": Descriptors.HeavyAtomCount,
    "RingCount": Descriptors.RingCount,
    "FractionCSP3": Descriptors.FractionCSP3,
    "NumValenceElectrons": Descriptors.NumValenceElectrons,
    "NumHeteroatoms": Descriptors.NumHeteroatoms,
}

# Full RDKit descriptor list (~200 descriptors)
FULL_DESCRIPTOR_FUNCS = {name: func for name, func in Descriptors.descList}


def compute_descriptors(smiles_series: pd.Series, descriptor_set: str = "full") -> pd.DataFrame:
    """
    Compute RDKit molecular descriptors.

    Parameters
    ----------
    smiles_series : pd.Series
        SMILES strings.
    descriptor_set : {"basic", "full"}
        "basic" → 12 hand-picked descriptors.
        "full"  → ~200 descriptors from Descriptors.descList.

    Returns
    -------
    pd.DataFrame
        Descriptor matrix with same index as smiles_series.
    """
    funcs = BASIC_DESCRIPTOR_FUNCS if descriptor_set == "basic" else FULL_DESCRIPTOR_FUNCS
    rows = []
    for smi in smiles_series:
        mol = _safe_mol(smi)
        if mol is None:
            rows.append({k: np.nan for k in funcs})
            continue
        row = {}
        for name, func in funcs.items():
            try:
                row[name] = func(mol)
            except Exception:
                row[name] = np.nan
        rows.append(row)
    df = pd.DataFrame(rows, index=smiles_series.index)
    # Prefix columns to avoid collisions when concatenating with fingerprints
    prefix = f"D_{descriptor_set}_"
    df.columns = [prefix + c for c in df.columns]
    return df


# ---------------------------------------------------------------------------
# F22 — Molecular Fingerprints
# ---------------------------------------------------------------------------

def compute_ecfp(smiles_series: pd.Series, radius: int = 2, n_bits: int = 2048) -> pd.DataFrame:
    """Morgan / ECFP fingerprint (circular fingerprint)."""
    fps = []
    for smi in smiles_series:
        mol = _safe_mol(smi)
        if mol is None:
            fps.append(np.zeros(n_bits, dtype=np.int8))
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        fps.append(np.array(fp, dtype=np.int8))
    arr = np.vstack(fps)
    cols = [f"FP_ECFP{radius}_{i}" for i in range(n_bits)]
    return pd.DataFrame(arr, index=smiles_series.index, columns=cols)


def compute_maccs(smiles_series: pd.Series) -> pd.DataFrame:
    """MACCS keys (166 bits, but RDKit returns 167)."""
    n_bits = 167
    fps = []
    for smi in smiles_series:
        mol = _safe_mol(smi)
        if mol is None:
            fps.append(np.zeros(n_bits, dtype=np.int8))
            continue
        fp = MACCSkeys.GenMACCSKeys(mol)
        fps.append(np.array(fp, dtype=np.int8))
    arr = np.vstack(fps)
    cols = [f"FP_MACCS_{i}" for i in range(n_bits)]
    return pd.DataFrame(arr, index=smiles_series.index, columns=cols)


def compute_krfp(smiles_series: pd.Series, n_bits: int = 2048) -> pd.DataFrame:
    """RDKit hashed atom-pair fingerprint (often called KRFP in legacy code)."""
    fps = []
    for smi in smiles_series:
        mol = _safe_mol(smi)
        if mol is None:
            fps.append(np.zeros(n_bits, dtype=np.int8))
            continue
        fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=n_bits)
        fps.append(np.array(fp, dtype=np.int8))
    arr = np.vstack(fps)
    cols = [f"FP_KRFP_{i}" for i in range(n_bits)]
    return pd.DataFrame(arr, index=smiles_series.index, columns=cols)


def compute_atom_pair(smiles_series: pd.Series, n_bits: int = 2048) -> pd.DataFrame:
    """Hashed Atom-Pair fingerprint (same underlying impl as KRFP here, explicit API)."""
    fps = []
    for smi in smiles_series:
        mol = _safe_mol(smi)
        if mol is None:
            fps.append(np.zeros(n_bits, dtype=np.int8))
            continue
        fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=n_bits)
        fps.append(np.array(fp, dtype=np.int8))
    arr = np.vstack(fps)
    cols = [f"FP_AP_{i}" for i in range(n_bits)]
    return pd.DataFrame(arr, index=smiles_series.index, columns=cols)


def compute_topological_torsion(smiles_series: pd.Series, n_bits: int = 2048) -> pd.DataFrame:
    """Topological Torsion fingerprint."""
    fps = []
    for smi in smiles_series:
        mol = _safe_mol(smi)
        if mol is None:
            fps.append(np.zeros(n_bits, dtype=np.int8))
            continue
        fp = Torsions.GetHashedTopologicalTorsionFingerprint(mol, nBits=n_bits)
        fps.append(_to_bitvect(fp, n_bits))
    arr = np.vstack(fps)
    cols = [f"FP_TT_{i}" for i in range(n_bits)]
    return pd.DataFrame(arr, index=smiles_series.index, columns=cols)


# Convenience registry
FP_GENERATORS = {
    "ECFP4": lambda s: compute_ecfp(s, radius=2, n_bits=2048),
    "ECFP6": lambda s: compute_ecfp(s, radius=3, n_bits=2048),
    "MACCS": compute_maccs,
    "KRFP": lambda s: compute_krfp(s, n_bits=2048),
    "AtomPair": lambda s: compute_atom_pair(s, n_bits=2048),
    "TopologicalTorsion": lambda s: compute_topological_torsion(s, n_bits=2048),
}


def generate_all_features(smiles_series: pd.Series) -> dict[str, pd.DataFrame]:
    """
    Generate every feature set implemented in this module.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys:  "Descriptors_basic", "Descriptors_full",
               "FP_ECFP4", "FP_ECFP6", "FP_MACCS",
               "FP_KRFP", "FP_AtomPair", "FP_TopologicalTorsion"
    """
    results = {}
    print("[FeatureGenerators] Computing basic descriptors (F21)...")
    results["Descriptors_basic"] = compute_descriptors(smiles_series, descriptor_set="basic")
    # Skip full descriptors for speed in demo; they contain ~200 features
    # print("[FeatureGenerators] Computing full descriptors (F21)...")
    # results["Descriptors_full"] = compute_descriptors(smiles_series, descriptor_set="full")

    for name, gen in FP_GENERATORS.items():
        print(f"[FeatureGenerators] Computing {name} fingerprint (F22)...")
        results[f"FP_{name}"] = gen(smiles_series)

    return results
