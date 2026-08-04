import os
os.environ["USE_MOCK_DB"] = "true"

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from fraud_service.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200


def test_first_order_has_no_history_to_flag():
    resp = client.post("/assess", json={
        "order_id": "order-1", "customer_id": "fresh-customer",
        "item": "Widget", "quantity": 1, "price": 25.0,
    })
    body = resp.json()
    assert resp.status_code == 200
    assert body["status"] == "approved"


def test_price_far_outside_customer_history_is_flagged():
    customer = "outlier-customer"
    for price in [10.0, 11.0, 9.5, 10.5]:
        client.post("/assess", json={
            "order_id": f"order-{price}", "customer_id": customer,
            "item": "Widget", "quantity": 1, "price": price,
        })
    resp = client.post("/assess", json={
        "order_id": "order-huge", "customer_id": customer,
        "item": "Widget", "quantity": 1, "price": 5000.0,
    })
    body = resp.json()
    assert body["status"] in ("review", "blocked")
    assert any("standard deviations" in r for r in body["reasons"])


def test_many_orders_in_short_window_triggers_velocity_flag():
    customer = "velocity-customer"
    for i in range(5):
        resp = client.post("/assess", json={
            "order_id": f"order-v{i}", "customer_id": customer,
            "item": "Widget", "quantity": 1, "price": 10.0,
        })
    body = resp.json()
    assert any("orders from this customer" in r for r in body["reasons"])


def test_get_assessment_by_order_id():
    client.post("/assess", json={
        "order_id": "lookup-me", "customer_id": "cust-x",
        "item": "Widget", "quantity": 1, "price": 10.0,
    })
    resp = client.get("/assessments/lookup-me")
    assert resp.status_code == 200
    assert resp.json()["order_id"] == "lookup-me"


def test_unknown_order_id_returns_unknown_status():
    resp = client.get("/assessments/never-existed")
    assert resp.json()["status"] == "unknown"
