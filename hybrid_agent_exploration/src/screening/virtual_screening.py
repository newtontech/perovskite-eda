"""virtual_screening.py — Virtual screening pipeline for molecular candidates."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


def load_candidate_pool(
    data_path: Optional[str] = None,
    smiles_col: str = "smiles",
    deduplicate: bool = True,
) -> pd.DataFrame:
    """Load candidate molecules for virtual screening.

    If no external library is provided, uses the existing dataset's
    unique SMILES as the candidate pool (in-silico re-screening).
    """
    if data_path and Path(data_path).exists():
        df = pd.read_csv(data_path) if data_path.endswith(".csv") else pd.read_excel(data_path)
    else:
        # Fallback: use existing merged data
        default_path = "/share/yhm/test/20260216_with_chemical_merge_fast/merged_llm_crossref_data_streaming_with_chemical_data_fast.xlsx"
        df = pd.read_excel(default_path)

    # Ensure SMILES column exists
    if smiles_col not in df.columns:
        # Try common variations
        for col in ["SMILES", "Smiles", "canonical_smiles", "CanonicalSMILES"]:
            if col in df.columns:
                smiles_col = col
                break

    df = df[[smiles_col]].copy()
    df.columns = ["smiles"]
    df = df.dropna(subset=["smiles"])

    if deduplicate:
        before = len(df)
        df = df.drop_duplicates(subset=["smiles"])
        print(f"Deduplicated: {before} → {len(df)} unique SMILES")

    return df.reset_index(drop=True)


def rank_candidates(
    candidate_df: pd.DataFrame,
    model,
    feature_fn,
    top_k: int = 100,
    uncertainty_fn: Optional[callable] = None,
) -> pd.DataFrame:
    """Rank candidate molecules by predicted ΔPCE.

    Args:
        candidate_df: DataFrame with 'smiles' column
        model: Trained sklearn estimator
        feature_fn: Function(smiles_series) → feature_array
        top_k: Number of top candidates to return
        uncertainty_fn: Optional function(model, X) → uncertainty_array

    Returns:
        DataFrame with smiles, predicted_delta_pce, rank, [uncertainty]
    """
    X = feature_fn(candidate_df["smiles"])
    y_pred = model.predict(X)

    result = candidate_df.copy()
    result["predicted_delta_pce"] = y_pred

    if uncertainty_fn:
        result["uncertainty"] = uncertainty_fn(model, X)

    result = result.sort_values("predicted_delta_pce", ascending=False)
    result["rank"] = np.arange(1, len(result) + 1)

    return result.head(top_k).reset_index(drop=True)
