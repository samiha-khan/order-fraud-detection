import os
import uuid
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime, timezone

app = FastAPI(title="Orders Service", version="1.0.0")

MONGO_URL = os.environ.get("MONGO_URL", "mongodb://localhost:27017")
FRAUD_SERVICE_URL = os.environ.get("FRAUD_SERVICE_URL", "http://localhost:8001")
USE_MOCK_DB = os.environ.get("USE_MOCK_DB", "false").lower() == "true"

if USE_MOCK_DB:
    import mongomock
    client = mongomock.MongoClient()
else:
    from pymongo import MongoClient
    client = MongoClient(MONGO_URL)

db = client["orders_db"]
orders_collection = db["orders"]


class OrderCreate(BaseModel):
    customer_id: str
    item: str
    quantity: int = Field(..., gt=0)
    price: float = Field(..., gt=0)


class Order(BaseModel):
    order_id: str
    customer_id: str
    item: str
    quantity: int
    price: float
    status: str
    risk_score: float
    risk_reasons: list[str]
    created_at: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/orders", response_model=Order)
def create_order(order: OrderCreate):
    order_id = str(uuid.uuid4())

    # Ask the fraud service to assess this order before we commit to it.
    # If the fraud service is unreachable, we still let the order through
    # rather than blocking checkout on a downstream outage -- but with no
    # risk data attached.
    risk_score, status, reasons = 0.0, "approved", []
    try:
        resp = httpx.post(f"{FRAUD_SERVICE_URL}/assess", json={
            "order_id": order_id,
            "customer_id": order.customer_id,
            "item": order.item,
            "quantity": order.quantity,
            "price": order.price,
        }, timeout=2.0)
        if resp.status_code == 200:
            data = resp.json()
            risk_score, status, reasons = data["risk_score"], data["status"], data["reasons"]
    except httpx.HTTPError:
        reasons = ["fraud service unavailable, order approved by default"]

    record = {
        "order_id": order_id,
        "customer_id": order.customer_id,
        "item": order.item,
        "quantity": order.quantity,
        "price": order.price,
        "status": status,
        "risk_score": risk_score,
        "risk_reasons": reasons,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    orders_collection.insert_one(dict(record))
    return Order(**record)


@app.get("/orders/{order_id}", response_model=Order)
def get_order(order_id: str):
    record = orders_collection.find_one({"order_id": order_id})
    if not record:
        raise HTTPException(status_code=404, detail="Order not found")
    return Order(**{k: v for k, v in record.items() if k != "_id"})


@app.get("/orders")
def list_orders(customer_id: str | None = None):
    query = {"customer_id": customer_id} if customer_id else {}
    records = list(orders_collection.find(query))
    return [{k: v for k, v in r.items() if k != "_id"} for r in records]

