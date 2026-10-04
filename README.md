# Order Fraud Detection System

Two microservices, each with its own MongoDB database: an orders service,
and a fraud detection service that checks every order in real time before
it's approved.

## How the fraud checks work

Two signals, both computed against a customer's own order history (not
some global average):

- **Velocity**: more than 3 orders from the same customer in a 10-minute
  window gets flagged.
- **Price outlier**: an order priced more than 2.5 standard deviations
  from that customer's historical average gets flagged. Needs at least 3
  past orders to compute a meaningful standard deviation, otherwise it's
  skipped.

Each flag adds to a risk score. Above 0.8 the order is blocked, above 0.4
it's sent for review, otherwise it's approved automatically.

## Architecture

```
Client -> Orders Service
              |
              v (synchronous call, before the order is finalized)
        Fraud Service -> checks velocity + price outlier -> risk score
              |
              v
        Order saved with its risk assessment attached
```

If the fraud service is down, orders still go through (marked approved by
default) rather than blocking checkout entirely on a dependency outage.

## Services

**Orders Service** (`:8000`)
```
GET  /health
POST /orders
GET  /orders/{order_id}
GET  /orders?customer_id=...
```

**Fraud Service** (`:8001`)
```
GET  /health
POST /assess
GET  /assessments/{order_id}
```

## Run with Docker Compose

```bash
docker-compose up --build
```

## Run locally without Docker

```bash
pip install -r requirements.txt
export USE_MOCK_DB=true
uvicorn fraud_service.main:app --port 8001 &
uvicorn orders_service.main:app --port 8000 &
```

Try it: a normal order, then a suspiciously large one from the same
customer:

```bash
curl -X POST http://localhost:8000/orders -H "Content-Type: application/json" \
  -d '{"customer_id":"cust-1","item":"Widget","quantity":1,"price":10.00}'

curl -X POST http://localhost:8000/orders -H "Content-Type: application/json" \
  -d '{"customer_id":"cust-1","item":"Widget","quantity":1,"price":5000.00}'
```

The second one comes back `"status": "blocked"` with the actual reason
attached.

## Validated against real fraud data, and a real limitation found

The rules above were only ever checked against hand-written test cases
until now. `docs/fraud-benchmark.md` checks them against real data and
finds a genuine gap: both rules depend on repeat-customer order history,
and two real-world-standard fraud datasets (a real-labeled e-commerce
signup dataset, and PaySim, the standard mobile-money fraud simulator)
were checked and neither has meaningful repeat-customer structure (PaySim:
9,298 of 6,353,307 accounts have more than one transaction; the other
dataset has exactly one transaction per user, period). Replaying the
price-outlier rule on PaySim's real-money repeat accounts (through the
actual `/assess` endpoint, not a reimplementation) confirms the result
that implies: 0 of 18 real fraud cases caught, because the rule requires
3 prior orders and no account in that real subset has that many.

A transaction-level model trained on real PaySim data, using signals that
don't require any customer history, reaches 99.85% ROC-AUC / 91.1% PR-AUC
(90% precision at 80% recall) on a temporally held-out test set, after
catching and discarding a feature that was inflating an earlier run to a
suspicious 1.0 AUC (diagnosed as a known PaySim simulation artifact, not
a real signal). It's a transaction-level benchmark, not an "Amazon-level
fraud model" claim, and it's deliberately not wired into the live
`/assess` endpoint, since PaySim's schema (money transfers) and this
service's own schema (e-commerce orders) are genuinely different domains.
Full methodology, the artifact diagnosis, and what real deployment would
actually require: [`docs/fraud-benchmark.md`](docs/fraud-benchmark.md).

```bash
pip install -r requirements.txt -r fraud_benchmark/requirements.txt
python -m fraud_benchmark.evaluate_rules_on_real_data
python -m fraud_benchmark.run_ml_benchmark
```

## Tests

```bash
pytest tests/ -v
```

18 tests: health checks, order creation and retrieval, 404 handling,
input validation, the fraud logic itself, and the real-data benchmark's
feature engineering and temporal split. First-time customers aren't
flagged with no history, a real price outlier gets flagged, and a burst
of orders in a short window trips the velocity check.
