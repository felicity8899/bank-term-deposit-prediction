"""Smoke-test client for the prediction service.

Sends one realistic client to POST /predict and prints the response.
Start the service first, e.g.:

    uvicorn predict:app --host 0.0.0.0 --port 9696
    python predict_test.py

The base URL can be overridden with the SERVICE_URL environment variable
(useful once the service is deployed to the cloud).
"""

import json
import os

import requests

SERVICE_URL = os.environ.get("SERVICE_URL", "http://localhost:9696")

# A realistic client: 42-year-old married manager with a tertiary degree,
# a healthy balance, a housing loan, reachable by cellular phone, never
# contacted in a previous campaign (pdays = -1, poutcome = "unknown").
client = {
    "age": 42,
    "job": "management",
    "marital": "married",
    "education": "tertiary",
    "default": "no",
    "balance": 1500,
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "day": 15,
    "month": "may",
    "campaign": 2,
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown",
}

if __name__ == "__main__":
    health = requests.get(f"{SERVICE_URL}/health", timeout=10)
    print("GET /health ->", health.status_code, health.json())

    response = requests.post(f"{SERVICE_URL}/predict", json=client, timeout=10)
    print("POST /predict")
    print("client:", json.dumps(client))
    print("response:", response.status_code, response.json())
