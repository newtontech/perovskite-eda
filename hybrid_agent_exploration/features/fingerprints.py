"""Molecular fingerprints — ECFP, MACCS, KRFP, Atom Pair, Topological Torsion"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors
from rdkit.Chem.AtomPairs import Torsions


def _smiles_to_mol(smi):
    return Chem.MolFromSmiles(str(smi))


def _to_bitvect(fp, n_bits):
    if hasattr(fp, 'ToList'):
        return np.array(fp.ToList())
    if hasattr(fp, 'GetNumOnes'):
        arr = np.zeros(n_bits)
        for idx in fp.GetOnBits():
            if idx < n_bits:
                arr[idx] = 1
        return arr
    return np.array(list(fp))


def _compute_ecfp(smiles_list, radius=2, n_bits=2048):
    fps = []
    for smi in smiles_list:
        mol = _smiles_to_mol(smi)
        if mol is None:
            fps.append(np.zeros(n_bits))
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        fps.append(np.array(fp))
    return np.array(fps)


def _compute_maccs(smiles_list):
    fps = []
    for smi in smiles_list:
        mol = _smiles_to_mol(smi)
        if mol is None:
            fps.append(np.zeros(167))
            continue
        fp = MACCSkeys.GenMACCSKeys(mol)
        fps.append(np.array(fp))
    return np.array(fps)


def _compute_krfp(smiles_list, n_bits=2048):
    fps = []
    for smi in smiles_list:
        mol = _smiles_to_mol(smi)
        if mol is None:
            fps.append(np.zeros(n_bits))
            continue
        fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=n_bits)
        fps.append(np.array(fp))
    return np.array(fps)


def _compute_atom_pair(smiles_list, n_bits=2048):
    fps = []
    for smi in smiles_list:
        mol = _smiles_to_mol(smi)
        if mol is None:
            fps.append(np.zeros(n_bits))
            continue
        fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=n_bits)
        fps.append(np.array(fp))
    return np.array(fps)


def _compute_topological_torsion(smiles_list, n_bits=2048):
    fps = []
    for smi in smiles_list:
        mol = _smiles_to_mol(smi)
        if mol is None:
            fps.append(np.zeros(n_bits))
            continue
        fp = Torsions.GetHashedTopologicalTorsionFingerprint(mol, nBits=n_bits)
        fps.append(_to_bitvect(fp, n_bits))
    return np.array(fps)


FP_FUNCS = {
    "F2_ecfp": lambda smi: _compute_ecfp(smi, radius=2),
    "F2_ecfp6": lambda smi: _compute_ecfp(smi, radius=3),
    "F3_maccs": _compute_maccs,
    "F4_krfp": _compute_krfp,
    "F5_atom_pair": _compute_atom_pair,
    "F6_topological_torsion": _compute_topological_torsion,
}


def get_fingerprint(fp_id: str, smiles_series) -> np.ndarray:
    func = FP_FUNCS.get(fp_id)
    if func is None:
        raise ValueError(f"Unknown fingerprint: {fp_id}")
    smi_list = smiles_series.tolist() if hasattr(smiles_series, 'tolist') else list(smiles_series)
    return func(smi_list)
