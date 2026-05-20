"""
splitters.py
============
Data splitting implementations for molecular regression data.

Implements:
- E42: random split, scaffold split (RDKit scaffold clustering), temporal split
- E43: k-fold, repeated k-fold, nested CV split generators
"""

import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    RepeatedKFold,
    train_test_split,
)


# ---------------------------------------------------------------------------
# Scaffold utilities
# ---------------------------------------------------------------------------

def generate_scaffold(smiles: str, include_chirality: bool = False) -> Optional[str]:
    """Compute the Bemis-Murcko scaffold for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=include_chirality
        )
        return scaffold
    except Exception:
        return None


def scaffold_to_smiles(
    smiles_list: List[str],
    use_indices: bool = False,
) -> dict:
    """
    Map each scaffold to the set of SMILES (or indices) that share it.

    Returns
    -------
    dict: {scaffold_str: {smiles_or_index, ...}}
    """
    scaffold_to_items = {}
    for i, smi in enumerate(smiles_list):
        scaffold = generate_scaffold(smi)
        if scaffold is None:
            scaffold = smi  # fallback: treat molecule as its own scaffold
        if scaffold not in scaffold_to_items:
            scaffold_to_items[scaffold] = set()
        scaffold_to_items[scaffold].add(i if use_indices else smi)
    return scaffold_to_items


# ---------------------------------------------------------------------------
# Splitters
# ---------------------------------------------------------------------------

class RandomSplitter:
    """Random train/validation/test split."""

    def __init__(self, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42):
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

    def split(
        self, df: pd.DataFrame, smiles_col: str = "smiles"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns train, val, test indices.
        """
        indices = np.arange(len(df))
        train_val, test = train_test_split(
            indices, test_size=self.test_size, random_state=self.random_state
        )
        if self.val_size > 0:
            val_ratio = self.val_size / (1 - self.test_size)
            train, val = train_test_split(
                train_val, test_size=val_ratio, random_state=self.random_state
            )
        else:
            train, val = train_val, np.array([], dtype=int)
        return train, val, test


class ScaffoldSplitter:
    """
    Scaffold-based split: ensures that molecules sharing the same Bemis-Murcko
    scaffold do not appear in both train and test.
    This is a stricter evaluation of generalisation than random split.
    """

    def __init__(
        self,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        balanced: bool = True,
    ):
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.balanced = balanced

    def split(
        self, df: pd.DataFrame, smiles_col: str = "smiles", y_col: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        smiles_list = df[smiles_col].tolist()
        scaffold_map = scaffold_to_smiles(smiles_list, use_indices=True)
        scaffolds = list(scaffold_map.keys())
        rng = np.random.RandomState(self.random_state)
        rng.shuffle(scaffolds)

        # Sort scaffolds by size to allow greedy balanced assignment
        scaffolds = sorted(scaffolds, key=lambda s: len(scaffold_map[s]), reverse=True)

        train_indices, val_indices, test_indices = [], [], []
        train_size_target = 1.0 - self.test_size - self.val_size
        val_size_target = self.val_size

        for scaffold in scaffolds:
            idxs = list(scaffold_map[scaffold])
            n_train = len(train_indices)
            n_val = len(val_indices)
            n_test = len(test_indices)
            n_total = len(smiles_list)

            if self.balanced:
                # greedy assign to the split that is currently most under-represented
                train_frac = n_train / n_total if n_total > 0 else 0
                val_frac = n_val / n_total if n_total > 0 else 0
                # test_frac is implicit

                if train_frac < train_size_target:
                    train_indices.extend(idxs)
                elif val_frac < val_size_target:
                    val_indices.extend(idxs)
                else:
                    test_indices.extend(idxs)
            else:
                # simple cumulative assignment
                cumsum = len(train_indices) + len(val_indices) + len(test_indices)
                if cumsum < int(train_size_target * n_total):
                    train_indices.extend(idxs)
                elif cumsum < int((train_size_target + val_size_target) * n_total):
                    val_indices.extend(idxs)
                else:
                    test_indices.extend(idxs)

        return (
            np.array(train_indices, dtype=int),
            np.array(val_indices, dtype=int),
            np.array(test_indices, dtype=int),
        )


class TemporalSplitter:
    """
    Temporal split: sorts data by a date/year column and takes the oldest
    fraction as train, newest as test.  Mimics real-world deployment where
    models are trained on past data and evaluated on future data.
    """

    def __init__(
        self,
        test_size: float = 0.2,
        val_size: float = 0.1,
        time_col: str = "year",
    ):
        self.test_size = test_size
        self.val_size = val_size
        self.time_col = time_col

    def split(
        self, df: pd.DataFrame, smiles_col: str = "smiles"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.time_col not in df.columns:
            raise ValueError(f"TemporalSplitter: column '{self.time_col}' not found in DataFrame.")

        sorted_idx = np.argsort(df[self.time_col].values)
        n = len(df)
        n_test = int(n * self.test_size)
        n_val = int(n * self.val_size)
        n_train = n - n_test - n_val

        train = sorted_idx[:n_train]
        val = sorted_idx[n_train : n_train + n_val]
        test = sorted_idx[n_train + n_val :]
        return train, val, test


# ---------------------------------------------------------------------------
# Cross-validation split generators
# ---------------------------------------------------------------------------

class KFoldSplitter:
    """Standard k-fold CV splitter."""

    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        return kf.split(X, y)


class RepeatedKFoldSplitter:
    """Repeated k-fold CV with different random seeds."""

    def __init__(
        self,
        n_splits: int = 5,
        n_repeats: int = 3,
        random_state: int = 42,
    ):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    def split(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        rkf = RepeatedKFold(
            n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=self.random_state
        )
        return rkf.split(X, y)


class NestedCVSplitter:
    """
    Nested CV: outer loop for unbiased performance estimation,
    inner loop for hyper-parameter selection.

    Usage
    -----
    for outer_train_idx, outer_test_idx in nested.outer_split(X, y):
        for inner_train_idx, inner_val_idx in nested.inner_split(X[outer_train_idx]):
            ... train & validate ...
        ... evaluate on outer_test_idx ...
    """

    def __init__(
        self,
        outer_n_splits: int = 5,
        inner_n_splits: int = 3,
        shuffle: bool = True,
        random_state: int = 42,
    ):
        self.outer_n_splits = outer_n_splits
        self.inner_n_splits = inner_n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def outer_split(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        kf = KFold(
            n_splits=self.outer_n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state,
        )
        return kf.split(X, y)

    def inner_split(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        kf = KFold(
            n_splits=self.inner_n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state + 1,  # different seed
        )
        return kf.split(X, y)


class ScaffoldKFoldSplitter:
    """
    K-fold where folds are created by grouping molecules that share the same
    scaffold.  This ensures no scaffold leakage across folds.
    """

    def __init__(
        self,
        n_splits: int = 5,
        smiles_col: str = "smiles",
        random_state: int = 42,
    ):
        self.n_splits = n_splits
        self.smiles_col = smiles_col
        self.random_state = random_state

    def split(self, df: pd.DataFrame, y: Optional[np.ndarray] = None):
        smiles_list = df[self.smiles_col].tolist()
        scaffold_map = scaffold_to_smiles(smiles_list, use_indices=True)
        scaffolds = list(scaffold_map.keys())
        rng = np.random.RandomState(self.random_state)
        rng.shuffle(scaffolds)

        # Distribute scaffolds round-robin into folds
        fold_scaffolds = [[] for _ in range(self.n_splits)]
        for i, scaffold in enumerate(scaffolds):
            fold_scaffolds[i % self.n_splits].append(scaffold)

        # Build index sets
        fold_indices = []
        for fold in fold_scaffolds:
            idxs = []
            for scaf in fold:
                idxs.extend(scaffold_map[scaf])
            fold_indices.append(set(idxs))

        for i in range(self.n_splits):
            test_idx = np.array(list(fold_indices[i]), dtype=int)
            train_idx = np.array(
                list(set(range(len(df))) - fold_indices[i]), dtype=int
            )
            yield train_idx, test_idx


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def split_df(
    df: pd.DataFrame,
    splitter,
    smiles_col: str = "smiles",
    y_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply a splitter and return train/val/test DataFrames."""
    train_idx, val_idx, test_idx = splitter.split(df, smiles_col=smiles_col, y_col=y_col)
    return df.iloc[train_idx].copy(), df.iloc[val_idx].copy(), df.iloc[test_idx].copy()
