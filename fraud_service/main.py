import os
import statistics
from datetime import datetime, timezone, timedelta
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Fraud Detection Service", version="1.0.0")

USE_MOCK_DB = os.environ.get("USE_MOCK_DB", "false").lower() == "true"

if USE_MOCK_DB:
    import mongomock
    client = mongomock.MongoClient()
else:
    from pymongo import MongoClient
    MONGO_URL = os.environ.get("MONGO_URL", "mongodb://localhost:27017")
    client = MongoClient(MONGO_URL)

db = client["fraud_db"]
history_collection = db["order_history"]
assessments_collection = db["assessments"]

VELOCITY_WINDOW_MINUTES = 10
VELOCITY_THRESHOLD = 3       # more than this many orders in the window -> flag
Z_SCORE_THRESHOLD = 2.5      # order value this many std devs from the mean -> flag
MIN_HISTORY_FOR_ZSCORE = 3


class AssessRequest(BaseModel):
    order_id: str
    customer_id: str
    item: str
    quantity: int
    price: float


class AssessResponse(BaseModel):
    order_id: str
    risk_score: float
    status: str
    reasons: list[str]


def check_velocity(customer_id: str, now: datetime) -> tuple[int, bool]:
    window_start = now - timedelta(minutes=VELOCITY_WINDOW_MINUTES)
    recent = list(history_collection.find({
        "customer_id": customer_id,
        "timestamp": {"$gte": window_start.isoformat()},
    }))
    return len(recent), len(recent) > VELOCITY_THRESHOLD


def check_price_outlier(customer_id: str, price: float) -> tuple[float | None, bool]:
    past = list(history_collection.find({"customer_id": customer_id}))
    past_prices = [p["price"] for p in past]
    if len(past_prices) < MIN_HISTORY_FOR_ZSCORE:
        return None, False
    mean = statistics.mean(past_prices)
    stdev = statistics.stdev(past_prices)
    if stdev == 0:
        return 0.0, False
    z = (price - mean) / stdev
    return z, abs(z) > Z_SCORE_THRESHOLD


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/assess", response_model=AssessResponse)
def assess(order: AssessRequest):
    now = datetime.now(timezone.utc)
    reasons = []
    risk_score = 0.0

    order_count, velocity_flag = check_velocity(order.customer_id, now)
    if velocity_flag:
        reasons.append(f"{order_count} orders from this customer in the last {VELOCITY_WINDOW_MINUTES} minutes")
        risk_score += 0.5

    z_score, price_flag = check_price_outlier(order.customer_id, order.price)
    if price_flag:
        reasons.append(f"order price is {z_score:.1f} standard deviations from this customer's usual spend")
        risk_score += 0.5

    if risk_score >= 0.8:
        status = "blocked"
    elif risk_score >= 0.4:
        status = "review"
    else:
        status = "approved"

    # Record this order for future velocity/outlier checks, then store the
    # assessment so it can be looked up later by order_id.
    history_collection.insert_one({
        "order_id": order.order_id,
        "customer_id": order.customer_id,
        "price": order.price,
        "timestamp": now.isoformat(),
    })
    result = {
        "order_id": order.order_id,
        "risk_score": round(risk_score, 2),
        "status": status,
        "reasons": reasons,
    }
    assessments_collection.insert_one(dict(result))
    return AssessResponse(**result)


@app.get("/assessments/{order_id}", response_model=AssessResponse)
def get_assessment(order_id: str):
    record = assessments_collection.find_one({"order_id": order_id})
    if not record:
        return AssessResponse(order_id=order_id, risk_score=0.0, status="unknown", reasons=["no assessment found"])
    return AssessResponse(**{k: v for k, v in record.items() if k != "_id"})
