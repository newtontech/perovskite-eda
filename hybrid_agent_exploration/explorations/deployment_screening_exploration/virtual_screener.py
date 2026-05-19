"""Virtual Screening Pipeline (D52) for Perovskite Additive Discovery"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, DataStructs, Descriptors
from sklearn.preprocessing import StandardScaler

rdBase.DisableLog("rdApp.error")
rdBase.DisableLog("rdApp.warning")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.rdkit_descriptors import compute_basic_descriptors

warnings.filterwarnings("ignore", category=FutureWarning)


class CandidateLibrary:
    """Generate or load candidate molecule libraries for virtual screening."""

    FRAGMENTS = [
        "C", "CC", "CCC", "CCCC", "CCCCC",
        "c1ccccc1", "c1ccc(C)cc1", "c1cc(C)ccc1O",
        "CCO", "CCCO", "CCCCO", "CCN", "CCCN",
        "COC", "COCC", "OCCO", "NCCN",
        "C=C", "C=CC", "C#C", "C=CC=C",
        "CC(=O)O", "CC(=O)N", "c1ccncc1", "C1CCCCC1",
        "CS", "CCS", "c1ccc(Cl)cc1", "c1ccc(F)cc1",
        "CC(C)C", "C(C)(C)C", "CC(=O)OC", "CN(C)C",
        "c1ccc(O)cc1", "c1ccc(N)cc1", "CC(=O)Oc1ccccc1",
    ]

    @classmethod
    def load_real_data(
        cls,
        path: Path | str = PROJECT_ROOT / "data_cache.csv",
        max_train: int = 500,
        max_candidates: int = 1000,
        target_col: str = "delta_pce",
        smiles_col: str = "smiles",
        random_state: int = 42,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load real data and split into initial training set and candidate pool."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Real data not found at {path}")

        df = pd.read_csv(path)
        df = df.dropna(subset=[smiles_col, target_col]).reset_index(drop=True)
        df[smiles_col] = df[smiles_col].astype(str)
        # Keep only valid SMILES
        valid_mask = df[smiles_col].apply(lambda s: Chem.MolFromSmiles(s) is not None)
        df = df[valid_mask].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError("No valid SMILES found in real data.")

        # Shuffle
        df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

        n_train = min(max_train, len(df) // 2)
        train_df = df.iloc[:n_train].copy()
        candidate_df = df.iloc[n_train:].copy()
        if len(candidate_df) > max_candidates:
            candidate_df = candidate_df.sample(
                n=max_candidates, random_state=random_state
            ).reset_index(drop=True)

        print(
            f"[Library] Real data: {len(train_df)} train | {len(candidate_df)} candidates"
        )
        return train_df, candidate_df

    @classmethod
    def generate_synthetic(
        cls,
        n_train: int = 300,
        n_candidates: int = 700,
        target_col: str = "delta_pce",
        smiles_col: str = "smiles",
        random_state: int = 42,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Generate a fully synthetic dataset with structured target values."""
        rng = np.random.RandomState(random_state)
        smiles_list = []
        for _ in range(n_train + n_candidates):
            n_frag = rng.randint(1, 4)
            smi = "".join(rng.choice(cls.FRAGMENTS, n_frag, replace=False))
            smiles_list.append(smi)

        df = pd.DataFrame({smiles_col: smiles_list})
        df = cls._assign_synthetic_target(df, target_col, smiles_col, rng)
        train_df = df.iloc[:n_train].copy().reset_index(drop=True)
        candidate_df = df.iloc[n_train:].copy().reset_index(drop=True)
        print(
            f"[Library] Synthetic: {len(train_df)} train | {len(candidate_df)} candidates"
        )
        return train_df, candidate_df

    @classmethod
    def _assign_synthetic_target(
        cls,
        df: pd.DataFrame,
        target_col: str,
        smiles_col: str,
        rng: np.random.RandomState,
    ) -> pd.DataFrame:
        """Heuristic target influenced by molecular properties."""
        desc = compute_basic_descriptors(df[smiles_col]).fillna(0)
        # Structured relationship
        y = (
            0.5 * desc.get("LogP", 0)
            - 0.3 * desc.get("TPSA", 0) / 100.0
            + 0.1 * desc.get("MolWt", 0) / 100.0
            + 1.0 * (desc.get("HBA", 0) > 3)
            + rng.normal(0, 0.5, size=len(df))
        )
        df[target_col] = y.values
        return df


class VirtualScreener:
    """Score and rank candidate molecules using a trained model."""

    def __init__(
        self,
        model,
        feature_fn: Callable[[pd.Series], pd.DataFrame],
        scaler: Optional[StandardScaler] = None,
        feature_name: str = "basic_descriptors",
    ):
        self.model = model
        self.feature_fn = feature_fn
        self.scaler = scaler or StandardScaler()
        self.feature_name = feature_name
        self.is_fitted = False

    def fit(
        self, df: pd.DataFrame, target_col: str, smiles_col: str = "smiles"
    ) -> "VirtualScreener":
        """Fit scaler and train model on provided data."""
        X = self._featurize(df[smiles_col])
        y = df[target_col].values
        self.scaler.fit(X)
        Xs = self.scaler.transform(X)
        self.model.fit(Xs, y)
        self.is_fitted = True
        return self

    def score_candidates(self, smiles_list: list[str]) -> pd.DataFrame:
        """Return scored and ranked candidate DataFrame."""
        if not self.is_fitted:
            raise RuntimeError("Screener must be fitted before scoring.")
        df = pd.DataFrame({"smiles": smiles_list})
        valid_mask = df["smiles"].apply(lambda s: Chem.MolFromSmiles(s) is not None)
        df = df[valid_mask].reset_index(drop=True)
        if len(df) == 0:
            return pd.DataFrame(
                columns=["smiles", "score", "rank", "uncertainty"]
            )

        X = self._featurize(df["smiles"])
        Xs = self.scaler.transform(X)
        scores = self.model.predict(Xs)

        # Uncertainty proxy (tree ensemble std)
        uncertainty = self._estimate_uncertainty(Xs)

        df["score"] = scores
        df["uncertainty"] = uncertainty
        df["rank"] = (
            df["score"].rank(ascending=False, method="min").astype(int)
        )
        return df.sort_values("rank").reset_index(drop=True)

    def rank_by_strategy(
        self,
        scored_df: pd.DataFrame,
        strategy: str = "top_k",
        k: int = 10,
        exploration_lambda: float = 0.5,
        diversity_radius: float = 0.3,
    ) -> pd.DataFrame:
        """Select top-k candidates using different strategies."""
        if scored_df.empty:
            return scored_df.copy()

        if strategy == "top_k":
            return scored_df.nsmallest(k, "rank").copy()

        if strategy == "uncertainty_weighted":
            # Upper confidence bound: score + lambda * uncertainty
            scored_df = scored_df.copy()
            scored_df["ucb"] = (
                scored_df["score"] + exploration_lambda * scored_df["uncertainty"]
            )
            top_idx = scored_df["ucb"].nlargest(k).index
            return (
                scored_df.loc[top_idx]
                .sort_values("ucb", ascending=False)
                .reset_index(drop=True)
            )

        if strategy == "diverse_top_k":
            return self._diverse_pick(
                scored_df, k, radius=diversity_radius
            )

        raise ValueError(f"Unknown strategy: {strategy}")

    def filter_by_property(
        self,
        smiles_list: list[str],
        molwt_range: Optional[tuple] = None,
        logp_range: Optional[tuple] = None,
        tpsa_max: Optional[float] = None,
    ) -> list[str]:
        """Filter SMILES by simple physicochemical constraints."""
        filtered = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            if molwt_range is not None:
                mw = Descriptors.MolWt(mol)
                if not (molwt_range[0] <= mw <= molwt_range[1]):
                    continue
            if logp_range is not None:
                logp = Descriptors.MolLogP(mol)
                if not (logp_range[0] <= logp <= logp_range[1]):
                    continue
            if tpsa_max is not None:
                tpsa = Descriptors.TPSA(mol)
                if tpsa > tpsa_max:
                    continue
            filtered.append(smi)
        return filtered

    def _featurize(self, smiles_series: pd.Series) -> np.ndarray:
        df = self.feature_fn(smiles_series)
        return df.fillna(0).values

    def _estimate_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """Use tree prediction variance as a cheap uncertainty proxy."""
        if hasattr(self.model, "estimators_"):
            preds = np.array(
                [tree.predict(X) for tree in self.model.estimators_]
            )
            return preds.std(axis=0)
        return np.zeros(len(X))

    def _diverse_pick(
        self, scored_df: pd.DataFrame, k: int, radius: float = 0.3
    ) -> pd.DataFrame:
        """Greedy diversity picker using ECFP4 Tanimoto distance."""
        if len(scored_df) <= k:
            return scored_df.copy()

        smiles = scored_df["smiles"].tolist()
        fps = []
        for smi in smiles:
            mol = Chem.MolFromSmiles(smi)
            fp = (
                AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                if mol
                else None
            )
            fps.append(fp)

        # Sort by score descending
        order = scored_df["score"].argsort()[::-1].values
        picked = [order[0]]
        for idx in order[1:]:
            if len(picked) >= k:
                break
            if fps[idx] is None:
                continue
            # Compute max similarity to already picked
            max_sim = 0.0
            for p in picked:
                if fps[p] is None:
                    continue
                max_sim = max(
                    max_sim,
                    DataStructs.TanimotoSimilarity(fps[idx], fps[p]),
                )
            if max_sim < (1.0 - radius):
                picked.append(idx)

        # If we didn't get enough, fill with next best
        if len(picked) < k:
            for idx in order:
                if idx not in picked:
                    picked.append(idx)
                if len(picked) >= k:
                    break

        return (
            scored_df.iloc[picked]
            .sort_values("score", ascending=False)
            .reset_index(drop=True)
        )
