import pandas as pd

from fraud_benchmark.ml_model import ALL_FEATURE_COLUMNS, REALISTIC_FEATURE_COLUMNS, engineer_features, temporal_split


def _sample_df():
    return pd.DataFrame({
        "step": [1, 1, 2, 2, 3, 3],
        "type": ["TRANSFER", "PAYMENT", "CASH_OUT", "TRANSFER", "CASH_IN", "DEBIT"],
        "amount": [100.0, 50.0, 200.0, 10.0, 30.0, 20.0],
        "oldbalanceOrg": [100.0, 50.0, 200.0, 10.0, 30.0, 20.0],
        "newbalanceOrig": [0.0, 0.0, 0.0, 0.0, 60.0, 0.0],
        "oldbalanceDest": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "newbalanceDest": [100.0, 50.0, 200.0, 10.0, 0.0, 0.0],
        "isFraud": [1, 0, 1, 0, 0, 0],
    })


def test_engineer_features_keeps_only_transfer_and_cash_out():
    df = _sample_df()
    featured = engineer_features(df)
    assert set(featured.type.unique()) == {"TRANSFER", "CASH_OUT"}
    assert len(featured) == 3  # PAYMENT, CASH_IN, DEBIT rows dropped


def test_error_balance_orig_is_zero_when_balance_reconciles():
    df = pd.DataFrame({
        "step": [1], "type": ["TRANSFER"], "amount": [100.0],
        "oldbalanceOrg": [500.0], "newbalanceOrig": [400.0],
        "oldbalanceDest": [0.0], "newbalanceDest": [100.0], "isFraud": [0],
    })
    featured = engineer_features(df)
    assert featured.errorBalanceOrig.iloc[0] == 0.0  # 400 - (500 - 100) == 0


def test_error_balance_orig_is_nonzero_on_the_known_paysim_fraud_pattern():
    # newbalanceOrig doesn't reconcile with oldbalanceOrg - amount: this is
    # the exact arithmetic gap that turned out to dominate the "full
    # feature set" model (see docs/fraud-benchmark.md).
    df = pd.DataFrame({
        "step": [1], "type": ["TRANSFER"], "amount": [100.0],
        "oldbalanceOrg": [500.0], "newbalanceOrig": [0.0],
        "oldbalanceDest": [0.0], "newbalanceDest": [100.0], "isFraud": [1],
    })
    featured = engineer_features(df)
    assert featured.errorBalanceOrig.iloc[0] == -400.0  # 0 - (500 - 100)


def test_realistic_feature_columns_exclude_post_transaction_origin_balance():
    assert "newbalanceOrig" not in REALISTIC_FEATURE_COLUMNS
    assert "errorBalanceOrig" not in REALISTIC_FEATURE_COLUMNS
    assert "newbalanceOrig" in ALL_FEATURE_COLUMNS
    assert "errorBalanceOrig" in ALL_FEATURE_COLUMNS


def test_temporal_split_puts_earlier_steps_in_train_and_later_in_test():
    df = pd.DataFrame({"step": list(range(1, 11)), "value": list(range(10))})
    train, test = temporal_split(df, train_fraction=0.8)
    assert train.step.max() <= test.step.min()
    assert len(train) + len(test) == len(df)


def test_temporal_split_is_not_a_random_shuffle():
    # A seeded random split could coincidentally pass the ordering check
    # above; this confirms train only ever contains the lowest steps.
    df = pd.DataFrame({"step": [5, 1, 3, 2, 4], "value": range(5)})
    train, test = temporal_split(df, train_fraction=0.6)
    assert set(train.step) == {1, 2, 3}
    assert set(test.step) == {4, 5}
