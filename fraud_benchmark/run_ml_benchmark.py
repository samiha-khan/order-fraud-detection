"""Trains and evaluates the transaction-level ML layer from ml_model.py on
real PaySim data, with a temporal (not random) train/test split.

Runs two variants on purpose, not one: a "full" feature set and a
"realistic" one with the known leaky features removed (see ml_model.py's
module docstring and REALISTIC_FEATURE_COLUMNS comment for why). Reporting
only the full variant's number would be the same mistake
diabetes-risk-prediction's README already documents fixing for its own
leakage bug: a headline metric that looks great but doesn't mean what it
appears to.

Run from repo root:
    python -m fraud_benchmark.run_ml_benchmark
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from fraud_benchmark.data import load_paysim
from fraud_benchmark.ml_model import (
    ALL_FEATURE_COLUMNS, REALISTIC_FEATURE_COLUMNS, engineer_features, evaluate, temporal_split, train,
)


def permutation_importance(model, test_df: pd.DataFrame, feature_columns: list[str], seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    base_auc = roc_auc_score(test_df.isFraud, model.predict_proba(test_df[feature_columns])[:, 1])
    drops = {}
    for col in feature_columns:
        shuffled = test_df[feature_columns].copy()
        shuffled[col] = rng.permutation(shuffled[col].to_numpy())
        auc = roc_auc_score(test_df.isFraud, model.predict_proba(shuffled)[:, 1])
        drops[col] = round(base_auc - auc, 4)
    return {"base_auc": round(float(base_auc), 4), "auc_drop_when_shuffled": drops}


def main():
    print("loading PaySim...")
    df = load_paysim()
    featured = engineer_features(df)
    print(f"TRANSFER/CASH_OUT transactions: {len(featured)} of {len(df)} total "
          f"({len(featured) / len(df):.1%}); fraud rate within that subset: {featured.isFraud.mean():.4%}")

    train_df, test_df = temporal_split(featured)
    print(f"temporal split: train n={len(train_df)} (step<={train_df.step.max()}), "
          f"test n={len(test_df)} (step>{train_df.step.max()})")

    results = {}

    print("\ntraining FULL feature set (includes errorBalanceOrig, newbalanceOrig)...")
    full_model = train(train_df, ALL_FEATURE_COLUMNS)
    results["full_features"] = evaluate(full_model, test_df, ALL_FEATURE_COLUMNS)
    results["full_features"]["permutation_importance"] = permutation_importance(full_model, test_df, ALL_FEATURE_COLUMNS)

    print("training REALISTIC feature set (excludes post-transaction origin balance)...")
    realistic_model = train(train_df, REALISTIC_FEATURE_COLUMNS)
    results["realistic_features"] = evaluate(realistic_model, test_df, REALISTIC_FEATURE_COLUMNS)

    print("\n=== Full feature set (inflated by a known PaySim artifact, see below) ===")
    for k, v in results["full_features"].items():
        print(f"  {k}: {v}")

    print("\n=== Realistic feature set (pre-transaction + destination info only) ===")
    for k, v in results["realistic_features"].items():
        print(f"  {k}: {v}")

    out_path = Path(__file__).resolve().parent / "ml_benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
