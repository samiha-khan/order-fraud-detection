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

## Tests

```bash
pytest tests/ -v
```

12 tests: health checks, order creation and retrieval, 404 handling,
input validation, and the fraud logic itself. First-time customers
aren't flagged with no history, a real price outlier gets flagged, and a
burst of orders in a short window trips the velocity check.
