"""A transaction-level fraud classifier, trained on signals that don't
depend on repeat-customer history, since fraud_benchmark/data.py's own
dataset facts (see docs/fraud-benchmark.md) show that kind of history is
essentially absent even in PaySim, let alone in a cold-start real system.

Features used:
  - type: PaySim fraud only ever occurs in TRANSFER and CASH_OUT
    transactions in this dataset (verified in run_benchmark.py's dataset
    summary, not assumed), so the model is trained and evaluated on that
    subset, the same scoping every published PaySim analysis uses.
  - amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest
  - errorBalanceOrig = newbalanceOrig - (oldbalanceOrg - amount): zero if
    the sender's balance moved by exactly the transaction amount; nonzero
    values are a well-documented PaySim fraud signature (Lopez-Rojas et
    al.) because several fraud patterns in the simulation drain or zero
    an account in a way that doesn't reconcile with the stated amount.
  - errorBalanceDest = oldbalanceDest + amount - newbalanceDest: the same
    idea on the receiving side.

Split: temporal, not random. Training on a random split of a time-ordered
log would leak future transaction patterns into the training set; this
splits by `step` (PaySim's simulated hour) so the model is only ever
evaluated on transactions that happened after everything it was trained on,
same as evaluating a real fraud model on tomorrow's traffic.
"""
from __future__ import annotations

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

FRAUD_TYPES = ("TRANSFER", "CASH_OUT")

# errorBalanceOrig and newbalanceOrig are excluded from REALISTIC_FEATURE_COLUMNS
# on purpose: permutation importance (see docs/fraud-benchmark.md) showed
# errorBalanceOrig alone collapses this model's AUC from 0.9999980 to 0.5597
# when shuffled out, meaning the "full" model below is almost entirely reading
# a PaySim simulation-construction artifact in how fraudulent transactions'
# post-transaction origin balance gets set, not a generalizable fraud signal.
# REALISTIC_FEATURE_COLUMNS keeps only pre-transaction and destination-side
# information, closer to what a model would actually have to work with before
# a transaction's outcome is known.
ALL_FEATURE_COLUMNS = [
    "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
    "errorBalanceOrig", "errorBalanceDest", "is_transfer",
]
REALISTIC_FEATURE_COLUMNS = [
    "amount", "oldbalanceOrg", "oldbalanceDest", "newbalanceDest",
    "errorBalanceDest", "is_transfer",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df.type.isin(FRAUD_TYPES)].copy()
    df["errorBalanceOrig"] = df.newbalanceOrig - (df.oldbalanceOrg - df.amount)
    df["errorBalanceDest"] = df.oldbalanceDest + df.amount - df.newbalanceDest
    df["is_transfer"] = (df.type == "TRANSFER").astype(int)
    return df


def temporal_split(df: pd.DataFrame, train_fraction: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff_step = df.step.quantile(train_fraction)
    train = df[df.step <= cutoff_step]
    test = df[df.step > cutoff_step]
    return train, test


def train(train_df: pd.DataFrame, feature_columns: list[str]) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(class_weight="balanced", random_state=42)
    model.fit(train_df[feature_columns], train_df.isFraud)
    return model


def evaluate(model: HistGradientBoostingClassifier, test_df: pd.DataFrame, feature_columns: list[str]) -> dict:
    y_true = test_df.isFraud.to_numpy()
    scores = model.predict_proba(test_df[feature_columns])[:, 1]

    roc_auc = roc_auc_score(y_true, scores)
    pr_auc = average_precision_score(y_true, scores)

    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    # Report the lowest-threshold operating point that still clears 90%
    # precision, a "blocked" tier analogous to the rule engine's own
    # status >= 0.8 cutoff: a deployed system would rather miss some fraud
    # than block 1 in 10 legitimate transactions outright.
    at_90pr = [
        (p, r, t) for p, r, t in zip(precision[:-1], recall[:-1], thresholds)
        if p >= 0.90
    ]
    best_90pr = max(at_90pr, key=lambda x: x[1]) if at_90pr else None

    return {
        "n_test": int(len(test_df)),
        "n_fraud_test": int(y_true.sum()),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "precision_at_90pct_precision_tier": (
            {"precision": round(best_90pr[0], 4), "recall": round(best_90pr[1], 4), "threshold": round(float(best_90pr[2]), 4)}
            if best_90pr else None
        ),
    }
