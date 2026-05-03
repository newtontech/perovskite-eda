"""RDKit molecular descriptors"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors


DESCRIPTOR_FUNCS = {
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


def compute_basic_descriptors(smiles_series: pd.Series) -> pd.DataFrame:
    rows = []
    for smi in smiles_series:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            rows.append({k: np.nan for k in DESCRIPTOR_FUNCS})
            continue
        rows.append({k: f(mol) for k, f in DESCRIPTOR_FUNCS.items()})
    return pd.DataFrame(rows, index=smiles_series.index)
