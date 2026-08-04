import os
os.environ["USE_MOCK_DB"] = "true"

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from orders_service.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200


def test_create_order():
    resp = client.post("/orders", json={
        "customer_id": "cust-1", "item": "Widget", "quantity": 2, "price": 5.0,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in ("approved", "review", "blocked")
    assert "order_id" in body


def test_get_order_by_id():
    created = client.post("/orders", json={
        "customer_id": "cust-2", "item": "Gadget", "quantity": 1, "price": 20.0,
    }).json()
    resp = client.get(f"/orders/{created['order_id']}")
    assert resp.status_code == 200
    assert resp.json()["item"] == "Gadget"


def test_get_nonexistent_order_returns_404():
    resp = client.get("/orders/does-not-exist")
    assert resp.status_code == 404


def test_list_orders_filtered_by_customer():
    client.post("/orders", json={
        "customer_id": "cust-filter-test", "item": "Thing", "quantity": 1, "price": 1.0,
    })
    resp = client.get("/orders", params={"customer_id": "cust-filter-test"})
    assert resp.status_code == 200
    assert all(o["customer_id"] == "cust-filter-test" for o in resp.json())


def test_invalid_quantity_returns_422():
    resp = client.post("/orders", json={
        "customer_id": "cust-1", "item": "Widget", "quantity": 0, "price": 5.0,
    })
    assert resp.status_code == 422
