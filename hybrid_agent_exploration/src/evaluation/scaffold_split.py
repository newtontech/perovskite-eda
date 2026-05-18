"""scaffold_split.py — Scaffold-based train/test split for external validation."""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupShuffleSplit
from typing import Tuple


def get_scaffold(smiles: str) -> str:
    """Extract Murcko scaffold from SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold) if scaffold else ""
    except Exception:
        return ""


def scaffold_train_test_split(
    df: pd.DataFrame,
    smiles_col: str = "smiles",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split DataFrame by molecular scaffold to simulate external validation.

    Returns (train_df, test_df) where test set contains different scaffolds
    than train set — a more rigorous test of generalization than random split.
    """
    df = df.copy()
    df["_scaffold"] = df[smiles_col].apply(get_scaffold)

    # Remove rows with invalid scaffolds
    valid = df["_scaffold"] != ""
    df_invalid = df[~valid].copy()
    df = df[valid].copy()

    # Group by scaffold
    scaffold_groups = df.groupby("_scaffold").ngroups
    print(f"Scaffold split: {len(df)} valid molecules, {scaffold_groups} unique scaffolds")

    # Use GroupShuffleSplit to keep scaffolds together
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    groups = df.groupby("_scaffold").ngroup().values
    train_idx, test_idx = next(gss.split(df, groups=groups))

    train_df = df.iloc[train_idx].drop(columns=["_scaffold"])
    test_df = df.iloc[test_idx].drop(columns=["_scaffold"])

    # Add invalid rows to train (conservative)
    if len(df_invalid) > 0:
        train_df = pd.concat([train_df, df_invalid.drop(columns=["_scaffold"])], ignore_index=True)

    print(f"Train: {len(train_df)} | Test: {len(test_df)}")
    print(f"Train scaffolds: {train_idx.size} groups | Test scaffolds: {test_idx.size} groups")

    return train_df, test_df
