# Does this hold up against real fraud data, and what would make it stronger

The velocity and price-outlier rules were never checked against anything
but hand-written test cases. This closes that gap, on real data, and finds
a genuine limitation in the rules as they stand: they depend on
repeat-customer order history that real-world fraud data essentially never
has, and the fix for that is the kind of transaction-level model a real
fraud team would actually run alongside rules like these, not instead of
investing more in the rules themselves.

## There's no public dataset with real repeat-customer order history

Both `check_velocity` and `check_price_outlier` need a sequence of past
orders from the *same* customer. Two real-world-standard fraud datasets
were checked for that structure before picking one, and neither has it:

- [rzhou1/FraudDetection](https://github.com/rzhou1/FraudDetection)'s
  `Fraud_Data.csv` (151,112 real-labeled e-commerce signups, 9.36% fraud)
  has exactly one transaction per user. 151,112 rows, 151,112 unique
  users.
- [PaySim](https://github.com/EdgarLopezPhD/PaySim) (Lopez-Rojas, Elmir &
  Axelsson, "PaySim: A financial mobile money simulator for fraud
  detection," 28th European Modeling and Simulation Symposium, 2016;
  6,362,620 simulated mobile-money transactions, 8,213 fraud, 0.129%,
  matching the dataset's own documented fraud rate) has 6,353,307 unique
  sender accounts for 6,362,620 transactions. Only 9,298 accounts (0.15%)
  have more than one transaction, and none have more than three.

This isn't a dataset-search failure, it's a real constraint: persistent
per-customer transaction history is exactly the kind of data real
companies keep privately and never publish, for the same privacy reasons
this project's own rules exist to protect. Public fraud benchmarks
anonymize it away. A production system gets to keep its own customer
history (which is the whole premise behind `check_velocity` and
`check_price_outlier`), but anyone validating *rules like these* against
public data runs into this every time.

## The price-outlier rule, replayed on the real history that does exist

PaySim's 9,298 repeat accounts are a small but real test of the
price-outlier rule specifically. `fraud_benchmark/evaluate_rules_on_real_data.py`
replays each account's real transactions through the actual `/assess`
endpoint (the production code path, not a reimplementation) in order,
using the account's real `nameOrig` as `customer_id` and real `amount` as
`price`, and checks whether the account's last transaction was flagged,
against PaySim's real `isFraud` label for it.

```
n_accounts:                       9,298
n_fraud_in_evaluated_transaction: 18
true_positives:  0      false_positives: 0
true_negatives:  9,280  false_negatives: 18
precision: undefined (rule never fired)
recall:    0.0
```

The rule never fired once, on fraud or non-fraud alike. The reason is
visible in `fraud_service/main.py`:
`MIN_HISTORY_FOR_ZSCORE = 3`, meaning the rule needs 3 *prior* orders
before it will compute a z-score at all. Every account in this real subset
has 3 or fewer transactions *total*, so the last transaction (the one
being evaluated) never has 3 prior orders behind it. `check_velocity` has
the same problem from the other direction: it needs `VELOCITY_THRESHOLD`
= 3 to be exceeded (4+ orders), which no account here reaches, so it's
excluded from this replay entirely rather than reported as a false "0
flags" result that would look like a measurement, not an artifact of
PaySim's hour-level time resolution (`step`) being far coarser than the
rule's real 10-minute window.

This is a real, if small-sample, demonstration of the thing the dataset
facts above already implied: a rule that needs 3+ prior orders is
correctly *cautious* on a cold-start customer, but that caution means it
contributes nothing on exactly the population (new or rarely-returning
customers) where public fraud data, and likely a lot of real traffic, is
concentrated.

## A transaction-level model that doesn't need customer history

`fraud_benchmark/ml_model.py` trains a `HistGradientBoostingClassifier` on
PaySim's full 6.36M transactions (filtered to `TRANSFER`/`CASH_OUT`, the
only two transaction types PaySim ever labels as fraud, confirmed from the
raw data, not assumed), split *temporally* by `step` (80% earliest steps
for training, the rest held out), not randomly, so the model is only ever
evaluated on transactions that happened after everything it trained on.

First run used every available feature, including the transaction's
resulting balances, and scored **ROC-AUC 1.0, PR-AUC 0.9999**. That
number was not trusted and not reported as a result until it was
diagnosed: permutation importance (`fraud_benchmark/run_ml_benchmark.py`)
showed the model's AUC collapses from 0.9999980 to 0.5597, barely above
chance, when a single feature (`errorBalanceOrig`, the arithmetic gap
between an account's stated transaction amount and its actual balance
change) is shuffled out. That feature is a documented PaySim
construction artifact: fraudulent transactions in the simulation get their
origin balance set in a way that leaves a specific, exploitable
arithmetic inconsistency, not something a real fraud signal would
reliably produce. Reporting the 1.0 number without this check would have
been the same mistake this project's sibling repo, diabetes-risk-prediction,
already documents catching and fixing for its own leakage bug.

The honest result drops that feature (and `newbalanceOrig`, which leaks
the same information) and uses only information available *before* a
transaction completes, plus destination-side signals:

```
=== Realistic feature set (pre-transaction + destination info only) ===
  n_test:    552,504
  roc_auc:   0.9985
  pr_auc:    0.9109
  precision/recall at the 90%-precision operating point: 0.9009 / 0.7964
```

Permutation importance on this version shows `amount` and `oldbalanceOrg`
driving almost all of it (dropping either collapses AUC by ~0.25-0.27),
which is a sensible, interpretable signal (a transaction close to or
exceeding the account's available balance is a real risk pattern), not an
artifact. 90% precision at 80% recall, with no customer history required
at all, is a meaningfully stronger result than the existing rules can
offer on cold-start traffic, which this project's own real-data check
above shows is most of it.

## Why this isn't wired into `/assess`

This benchmark is kept separate from the live fraud service on purpose,
not left unfinished. PaySim's schema (`type`, `amount`, account balances)
describes mobile-money transfers; `fraud_service`'s actual `AssessRequest`
(`item`, `quantity`, `price`) describes e-commerce orders. There's no
honest way to point a model trained on one at live traffic shaped like the
other without inventing a mapping between them that this project doesn't
have evidence for. Actually deploying a model like this would need either
real order-level training data (which, per the dataset search above,
essentially doesn't exist publicly) or building the equivalent of
`errorBalanceDest`-style consistency features from whatever payment or
account data this system's own domain actually has, and validating *that*
honestly before trusting it, the same way this document just did for
PaySim's version.

## Running it

```bash
pip install -r requirements.txt -r fraud_benchmark/requirements.txt
python -m fraud_benchmark.evaluate_rules_on_real_data
python -m fraud_benchmark.run_ml_benchmark
```

Both download PaySim (Hugging Face, ~330MB, no auth) on first run and
cache it in `.cache/paysim/`.
