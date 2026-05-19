"""
data_profiler.py
Reusable data profiling module for PSC device data exploration.
Handles missing values, distributions, correlations, and outlier detection.
Uses only matplotlib (no seaborn dependency) for plotting.
"""

import os
import json
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary DataFrame of missing values per column."""
    missing = df.isnull().sum()
    pct = 100 * missing / len(df)
    summary = pd.DataFrame({
        "missing_count": missing,
        "missing_pct": pct.round(4),
        "dtype": df.dtypes
    })
    summary = summary.sort_values("missing_pct", ascending=False)
    return summary


def numeric_summary(df: pd.DataFrame, cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Return extended summary statistics for numeric columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if cols is not None:
        numeric_cols = [c for c in cols if c in numeric_cols]
    if not numeric_cols:
        return pd.DataFrame()
    desc = df[numeric_cols].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    desc["skewness"] = df[numeric_cols].skew()
    desc["kurtosis"] = df[numeric_cols].kurtosis()
    desc["zeros"] = (df[numeric_cols] == 0).sum()
    desc["negatives"] = (df[numeric_cols] < 0).sum()
    return desc.round(4)


def correlation_matrix(df: pd.DataFrame, cols: Optional[List[str]] = None,
                       method: str = "pearson") -> pd.DataFrame:
    """Return correlation matrix for numeric columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if cols is not None:
        numeric_cols = [c for c in cols if c in numeric_cols]
    if len(numeric_cols) < 2:
        return pd.DataFrame()
    return df[numeric_cols].corr(method=method)


def outlier_summary(df: pd.DataFrame, cols: Optional[List[str]] = None,
                    method: str = "iqr") -> pd.DataFrame:
    """
    Detect outliers using IQR or Z-score.
    Returns count and bounds per column.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if cols is not None:
        numeric_cols = [c for c in cols if c in numeric_cols]
    records = []
    for col in numeric_cols:
        s = df[col].dropna()
        if len(s) == 0:
            records.append({"column": col, "outlier_count": 0, "lower_bound": np.nan, "upper_bound": np.nan})
            continue
        if method == "iqr":
            q1, q3 = s.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
        elif method == "zscore":
            mu, sigma = s.mean(), s.std()
            lower = mu - 3 * sigma
            upper = mu + 3 * sigma
        else:
            raise ValueError("method must be 'iqr' or 'zscore'")
        outliers = ((s < lower) | (s > upper)).sum()
        records.append({"column": col, "outlier_count": outliers,
                        "lower_bound": round(lower, 4), "upper_bound": round(upper, 4)})
    return pd.DataFrame(records)


def plot_distributions(df: pd.DataFrame, cols: Optional[List[str]] = None,
                       out_dir: str = ".", max_plots: int = 20,
                       figsize: Tuple[int, int] = (14, 10)) -> List[str]:
    """
    Plot histograms + KDE-like overlays for numeric columns.
    Saves PNG files and returns their paths.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if cols is not None:
        numeric_cols = [c for c in cols if c in numeric_cols]
    if not numeric_cols:
        return []
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    # Batch columns into grids of up to 6 per figure
    batch_size = 6
    for batch_start in range(0, min(len(numeric_cols), max_plots), batch_size):
        subset = numeric_cols[batch_start:batch_start + batch_size]
        n = len(subset)
        rows = (n + 2) // 3
        fig, axes = plt.subplots(rows, 3, figsize=figsize)
        axes = axes.flatten()
        for idx, col in enumerate(subset):
            ax = axes[idx]
            data = df[col].dropna()
            ax.hist(data, bins=min(50, max(10, int(len(data) ** 0.5))), color="steelblue", edgecolor="white", alpha=0.7)
            ax.set_title(col, fontsize=9)
            ax.set_xlabel("")
            ax.set_ylabel("Count")
            # simple rug plot at bottom
            if len(data) > 0:
                y_rug = np.zeros_like(data) - ax.get_ylim()[1] * 0.02
                ax.scatter(data, y_rug, s=5, color="darkred", alpha=0.3)
        for idx in range(n, len(axes)):
            axes[idx].axis("off")
        fig.tight_layout()
        fname = os.path.join(out_dir, f"dist_batch_{batch_start // batch_size + 1}.png")
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        saved.append(fname)
    return saved


def plot_correlation_heatmap(corr: pd.DataFrame, out_path: str,
                             figsize: Tuple[int, int] = (12, 10)) -> str:
    """Plot a correlation heatmap and save to file."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    n = len(corr)
    figsize = (max(6, n * 0.6), max(5, n * 0.5))
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(corr.columns, rotation=90, ha="center", fontsize=7)
    ax.set_yticklabels(corr.columns, fontsize=7)
    # add colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label("Correlation", rotation=270, labelpad=15)
    # annotate values for small matrices
    if n <= 20:
        for i in range(n):
            for j in range(n):
                text = ax.text(j, i, f"{corr.values[i, j]:.2f}",
                               ha="center", va="center", color="black", fontsize=5)
    ax.set_title("Correlation Matrix")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_missing_matrix(df: pd.DataFrame, out_path: str,
                        figsize: Tuple[int, int] = (14, 8)) -> str:
    """
    Plot a binary missing-value matrix (missing = dark, present = light).
    Samples up to 5,000 rows for visibility.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plot_df = df.sample(min(5000, len(df)), random_state=42) if len(df) > 5000 else df
    missing = plot_df.isnull().astype(int)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(missing.values.T, cmap="Greys", aspect="auto", interpolation="nearest")
    ax.set_xlabel("Sample index", fontsize=9)
    ax.set_ylabel("Column", fontsize=9)
    ax.set_yticks(np.arange(len(plot_df.columns)))
    ax.set_yticklabels(plot_df.columns, fontsize=6)
    ax.set_title("Missing Value Matrix (dark = missing)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def profile_dataframe(df: pd.DataFrame, name: str = "dataset",
                      out_dir: str = ".") -> Dict:
    """
    Run full profiling pipeline and save artifacts.
    Returns a dict of file paths and summary tables.
    """
    os.makedirs(out_dir, exist_ok=True)
    artifacts = {}

    # 1. Missing values
    missing = missing_value_summary(df)
    missing_path = os.path.join(out_dir, f"{name}_missing_summary.csv")
    missing.to_csv(missing_path)
    artifacts["missing_summary"] = missing_path
    plot_missing_matrix(df, os.path.join(out_dir, f"{name}_missing_matrix.png"))
    artifacts["missing_matrix_plot"] = os.path.join(out_dir, f"{name}_missing_matrix.png")

    # 2. Numeric summary
    numeric = numeric_summary(df)
    if not numeric.empty:
        numeric_path = os.path.join(out_dir, f"{name}_numeric_summary.csv")
        numeric.to_csv(numeric_path)
        artifacts["numeric_summary"] = numeric_path

    # 3. Correlation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 2:
        corr = correlation_matrix(df)
        corr_path = os.path.join(out_dir, f"{name}_correlation.csv")
        corr.to_csv(corr_path)
        artifacts["correlation_matrix"] = corr_path
        plot_correlation_heatmap(corr, os.path.join(out_dir, f"{name}_correlation_heatmap.png"))
        artifacts["correlation_heatmap"] = os.path.join(out_dir, f"{name}_correlation_heatmap.png")

    # 4. Outliers
    outliers = outlier_summary(df)
    if not outliers.empty:
        outlier_path = os.path.join(out_dir, f"{name}_outlier_summary.csv")
        outliers.to_csv(outlier_path, index=False)
        artifacts["outlier_summary"] = outlier_path

    # 5. Distributions
    dist_plots = plot_distributions(df, out_dir=out_dir, max_plots=24)
    artifacts["distribution_plots"] = dist_plots

    # 6. JSON metadata
    meta = {
        "name": name,
        "n_rows": int(len(df)),
        "n_cols": int(len(df.columns)),
        "columns": df.columns.tolist(),
        "artifacts": artifacts
    }
    meta_path = os.path.join(out_dir, f"{name}_profile_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    artifacts["meta"] = meta_path

    return artifacts
