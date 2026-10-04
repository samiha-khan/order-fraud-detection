"""Replays the real price-outlier rule (fraud_service/main.py's actual
check_price_outlier, through the actual /assess endpoint, not a
reimplementation) against the one real signal PaySim can offer it: the
9,298 accounts (of 6,353,307) that have more than one transaction.

The velocity rule is deliberately NOT evaluated here. PaySim's time
resolution is one simulated hour per `step`; the velocity rule's window is
10 real minutes. Replaying transactions through a live TestClient call also
happens in real wall-clock milliseconds regardless of how far apart the
original steps were, so any velocity result from this setup would measure
"how fast can this script make HTTP calls," not real customer behavior.
Testing it anyway and reporting a number would be worse than not testing it.

Method: for each repeat account, replay its transactions through /assess in
step order using that account's real nameOrig as customer_id and real
amount as price, exactly like a real customer's order history accumulating
over time. The account's LAST transaction is scored against the model's
prior history, same as a real system scoring a new order; its predicted
flag (review/blocked vs. approved) is compared to PaySim's real isFraud
label for that transaction.

Run from repo root:
    python -m fraud_benchmark.evaluate_rules_on_real_data
"""
from __future__ import annotations

import os

os.environ["USE_MOCK_DB"] = "true"

import pandas as pd
from fastapi.testclient import TestClient

from fraud_benchmark.data import load_paysim
from fraud_service.main import app, history_collection

client = TestClient(app)


def evaluate_price_outlier_rule() -> dict:
    df = load_paysim()
    counts = df.nameOrig.value_counts()
    repeat_accounts = counts[counts > 1].index
    subset = df[df.nameOrig.isin(repeat_accounts)].sort_values(["nameOrig", "step"])

    tp = fp = tn = fn = 0
    for i, (name_orig, group) in enumerate(subset.groupby("nameOrig")):
        # Each account's history is independent of every other account's,
        # by construction (nameOrig values never repeat across groups), so
        # nothing is lost by clearing the shared mock collection between
        # accounts. Everything DOES get lost in how long this takes without
        # it: mongomock's find() has no real index, so leaving ~21,500
        # records from prior accounts sitting in the collection makes every
        # later account's query scan the entire thing, turning a run that
        # should take well under a minute into one that doesn't finish in
        # 7 minutes (measured; killed mid-run before this fix).
        history_collection.delete_many({})
        rows = group.to_dict("records")
        last_status = None
        for j, row in enumerate(rows):
            resp = client.post("/assess", json={
                "order_id": f"paysim-{name_orig}-{j}",
                "customer_id": name_orig,
                "item": "transfer",
                "quantity": 1,
                "price": float(row["amount"]),
            })
            last_status = resp.json()["status"]

        predicted_flag = last_status != "approved"
        actual_fraud = bool(rows[-1]["isFraud"])
        if predicted_flag and actual_fraud:
            tp += 1
        elif predicted_flag and not actual_fraud:
            fp += 1
        elif not predicted_flag and actual_fraud:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    return {
        "n_accounts": len(repeat_accounts),
        "n_fraud_in_evaluated_transaction": tp + fn,
        "true_positives": tp, "false_positives": fp,
        "true_negatives": tn, "false_negatives": fn,
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
    }


if __name__ == "__main__":
    result = evaluate_price_outlier_rule()
    print("price-outlier rule, replayed on real PaySim repeat-account histories:")
    for k, v in result.items():
        print(f"  {k}: {v}")
