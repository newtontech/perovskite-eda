"""
explore_screening.py

Layer 5 — Deployment, Virtual Screening & Closed-Loop Exploration
=================================================================
Demonstrates the full Agent loop for autonomous PSC additive discovery:
  D52 — Virtual Screening (load model → score candidates → rank)
  D54 — Closed-Loop Feedback (train → top-k → simulate experiment → retrain)

Outputs saved in the same folder as this script.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
rdBase.DisableLog("rdApp.error")
rdBase.DisableLog("rdApp.warning")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.rdkit_descriptors import compute_basic_descriptors
from virtual_screener import CandidateLibrary, VirtualScreener
from closed_loop import ClosedLoopSimulator

warnings.filterwarnings("ignore", category=FutureWarning)

OUTPUT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data_cache.csv"
TARGET_COL = "delta_pce"
SMILES_COL = "smiles"
RANDOM_STATE = 42
MAX_TRAIN = 400
MAX_CANDIDATES = 1500
N_ITERATIONS = 5
K_PER_ITER = 5


def agent_step(name: str):
    print("\n" + "=" * 70)
    print(f"Agent Step: {name}")
    print("=" * 70)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        train_df, candidate_df = CandidateLibrary.load_real_data(
            path=DATA_PATH,
            max_train=MAX_TRAIN,
            max_candidates=MAX_CANDIDATES,
            target_col=TARGET_COL,
            smiles_col=SMILES_COL,
            random_state=RANDOM_STATE,
        )
    except Exception as e:
        print(f"[Data] Real data unavailable ({e}) — falling back to synthetic.")
        train_df, candidate_df = CandidateLibrary.generate_synthetic(
            n_train=MAX_TRAIN,
            n_candidates=MAX_CANDIDATES,
            target_col=TARGET_COL,
            smiles_col=SMILES_COL,
            random_state=RANDOM_STATE,
        )
    return train_df, candidate_df


def feature_agent(smiles_series: pd.Series) -> pd.DataFrame:
    return compute_basic_descriptors(smiles_series)


def model_agent() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=200, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1
    )


def evaluation_agent(model, X: np.ndarray, y: np.ndarray) -> dict:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    cv_scores = cross_val_score(model, Xs, y, cv=5, scoring="r2")
    X_tr, X_te, y_tr, y_te = train_test_split(
        Xs, y, test_size=0.2, random_state=RANDOM_STATE
    )
    model.fit(X_tr, y_tr)
    test_r2 = model.score(X_te, y_te)
    y_pred = model.predict(X_te)
    test_rmse = float(np.sqrt(np.mean((y_te - y_pred) ** 2)))
    return {
        "cv_r2_mean": float(cv_scores.mean()),
        "cv_r2_std": float(cv_scores.std()),
        "test_r2": float(test_r2),
        "test_rmse": test_rmse,
    }


def main():
    print("=" * 70)
    print("Layer 5 — Deployment, Virtual Screening & Closed-Loop Exploration")
    print("=" * 70)

    agent_step("Planner: Define scientific goal")
    print("Goal: Discover high-ΔPCE molecular additives for PSC via virtual screening")
    print("      and refine predictions through simulated experimental feedback (D54).")

    agent_step("Retriever: Load training data & candidate library")
    train_df, candidate_df = load_data()

    agent_step("Feature Agent: F21 RDKit basic descriptors")
    X_train = feature_agent(train_df[SMILES_COL]).fillna(0).values
    print(f"  → Feature matrix: {X_train.shape}")

    agent_step("Model Agent: M31 RandomForestRegressor")
    model = model_agent()

    agent_step("Evaluation Agent: 5-fold CV + hold-out test")
    metrics = evaluation_agent(model, X_train, train_df[TARGET_COL].values)
    print(f"  → 5-fold CV R²: {metrics['cv_r2_mean']:.4f} (+/- {metrics['cv_r2_std']:.4f})")
    print(f"  → Hold-out  R²: {metrics['test_r2']:.4f}")
    print(f"  → Hold-out RMSE: {metrics['test_rmse']:.4f}")

    agent_step("Virtual Screening (D52): Score candidate library")
    screener = VirtualScreener(
        model=model_agent(),
        feature_fn=feature_agent,
        feature_name="F21_basic_descriptors",
    )
    screener.fit(train_df, target_col=TARGET_COL, smiles_col=SMILES_COL)

    all_candidates = candidate_df[SMILES_COL].tolist()
    print(f"  → Screening {len(all_candidates)} candidates ...")
    scored_df = screener.score_candidates(all_candidates)

    filtered_smiles = screener.filter_by_property(
        all_candidates,
        molwt_range=(100, 600),
        logp_range=(-2, 5),
        tpsa_max=150,
    )
    print(f"  → After property filters: {len(filtered_smiles)} / {len(all_candidates)}")
    scored_df = scored_df[scored_df["smiles"].isin(filtered_smiles)].reset_index(drop=True)

    strategies = ["top_k", "uncertainty_weighted", "diverse_top_k"]
    ranking_results = {}
    for strat in strategies:
        ranked = screener.rank_by_strategy(scored_df, strategy=strat, k=20)
        ranking_results[strat] = ranked
        print(f"  → Strategy '{strat}': top-5 mean predicted ΔPCE = {ranked['score'].head(5).mean():.3f}")

    for strat, ranked in ranking_results.items():
        out_csv = OUTPUT_DIR / f"ranking_{strat}_top20.csv"
        ranked.to_csv(out_csv, index=False)
        print(f"  → Saved {out_csv}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(scored_df["score"], bins=60, color="steelblue", edgecolor="black", alpha=0.7)
    ax.axvline(scored_df["score"].max(), color="crimson", linestyle="--", label="Best predicted")
    ax.set_xlabel("Predicted ΔPCE")
    ax.set_ylabel("Count")
    ax.set_title("Virtual Screening Score Distribution (D52)")
    ax.legend()
    plt.tight_layout()
    score_dist_path = OUTPUT_DIR / "score_distribution.png"
    plt.savefig(score_dist_path, dpi=200)
    plt.close()
    print(f"[Plot] Saved score distribution → {score_dist_path}")

    agent_step("Closed-Loop Simulation (D54): Train → Screen → Validate → Retrain")

    def ground_truth_fn(smi: str) -> float:
        row = candidate_df[candidate_df[SMILES_COL] == smi]
        if not row.empty:
            return float(row[TARGET_COL].iloc[0])
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return 0.0
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        mw = Descriptors.MolWt(mol)
        return 0.5 * logp - 0.3 * tpsa / 100.0 + 0.1 * mw / 100.0

    loop_histories = {}
    for strat in strategies:
        print(f"\n[Main] Running closed-loop with strategy '{strat}' ...")
        sim = ClosedLoopSimulator(
            screener=VirtualScreener(
                model=model_agent(),
                feature_fn=feature_agent,
                feature_name="F21_basic_descriptors",
            ),
            candidate_pool=candidate_df.copy(),
            ground_truth_fn=ground_truth_fn,
            noise_std=0.3,
            random_state=RANDOM_STATE,
        )
        sim.initialize(train_df.copy())
        sim.run(n_iterations=N_ITERATIONS, k=K_PER_ITER, strategy=strat)
        loop_histories[strat] = sim.history
        sim.plot_history(OUTPUT_DIR / f"closed_loop_trajectory_{strat}.png")

    agent_step("Memory: Persist exploration artefacts")

    full_report = {
        "planner_goal": "Maximize ΔPCE via virtual screening + closed-loop feedback",
        "data_source": "data_cache.csv" if DATA_PATH.exists() else "synthetic",
        "n_train_initial": len(train_df),
        "n_candidates": len(candidate_df),
        "initial_model_metrics": metrics,
        "screening": {
            "strategies": {
                s: {
                    "top_5_mean_pred": float(ranking_results[s]["score"].head(5).mean()),
                    "top_5_mean_unc": float(ranking_results[s]["uncertainty"].head(5).mean())
                    if "uncertainty" in ranking_results[s]
                    else None,
                }
                for s in strategies
            },
        },
        "closed_loop": {
            "n_iterations": N_ITERATIONS,
            "k_per_iteration": K_PER_ITER,
            "histories": loop_histories,
        },
    }

    report_path = OUTPUT_DIR / "screening_exploration_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    print(f"  → {report_path}")

    print("\n" + "=" * 70)
    print("Exploration Complete — Summary")
    print("=" * 70)
    print(f"{'Strategy':<20} {'Top-5 Pred ΔPCE':>16} {'Final Val R²':>14}")
    print("-" * 70)
    for strat in strategies:
        top5 = ranking_results[strat]["score"].head(5).mean()
        final_r2 = loop_histories[strat][-1]["val_r2"] if loop_histories[strat] else 0.0
        print(f"{strat:<20} {top5:>16.3f} {final_r2:>14.3f}")
    print("=" * 70)
    print(f"\nAll outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
