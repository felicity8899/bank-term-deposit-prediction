"""Bank Term Deposit Prediction — web service.

Loads the model artifact produced by train.py (model.bin) and serves
predictions over HTTP with FastAPI:

    POST /predict   client JSON  -> {"subscribe_probability": ..., "subscribe": ...}
    GET  /health                 -> {"status": "ok"}

The request schema contains the 15 deployable features of the UCI Bank
Marketing dataset — everything the bank knows about a client BEFORE the
campaign call. The 16th dataset column, `duration` (length of the final
call), is deliberately absent: it is only known after the call ends and
leaks the target (see notebook.ipynb, Section 3.6).

Run locally:
    uvicorn predict:app --host 0.0.0.0 --port 9696
"""

import pickle
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parent / "model.bin"

# ---------------------------------------------------------------------------
# Load the model artifact once at startup.
# model.bin is a pickled dict: {"dv", "model", "features", "threshold"}.
# ---------------------------------------------------------------------------
with open(MODEL_PATH, "rb") as f:
    artifact = pickle.load(f)

dv = artifact["dv"]                # fitted DictVectorizer (one-hot encoder)
model = artifact["model"]          # trained classifier
FEATURES = artifact["features"]    # feature names, in the order dv expects
THRESHOLD = artifact["threshold"]  # decision threshold chosen on validation


class Client(BaseModel):
    """One bank client, described by the pre-call features only."""

    age: int = Field(..., examples=[42])
    job: str = Field(..., examples=["management"])
    marital: str = Field(..., examples=["married"])
    education: str = Field(..., examples=["tertiary"])
    default: str = Field(..., examples=["no"])
    balance: float = Field(..., examples=[1500])
    housing: str = Field(..., examples=["yes"])
    loan: str = Field(..., examples=["no"])
    contact: str = Field(..., examples=["cellular"])
    day: int = Field(..., examples=[15])
    month: str = Field(..., examples=["may"])
    campaign: int = Field(..., examples=[2])
    pdays: int = Field(..., examples=[-1])  # -1 = never contacted before
    previous: int = Field(..., examples=[0])
    poutcome: str = Field(..., examples=["unknown"])


app = FastAPI(title="Bank Term Deposit Prediction")

LANDING_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Bank Term Deposit Prediction — live service</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 44rem; margin: 3rem auto; padding: 0 1rem; line-height: 1.55; color: #1a1a1a; }
  code, pre { background: #f2f2f2; border-radius: 6px; }
  code { padding: 0.1rem 0.35rem; }
  pre { padding: 0.9rem; overflow-x: auto; }
  a { color: #0b5cad; }
</style>
</head>
<body>
<h1>Bank Term Deposit Prediction</h1>
<p>This service is <strong>live</strong>. It serves an XGBoost model (test ROC AUC
0.806) that predicts whether a bank client will subscribe to a term deposit,
trained on the UCI Bank Marketing dataset. Midterm project for the
DataTalksClub Machine Learning Zoomcamp.</p>

<h2>Try it</h2>
<ul>
  <li><a href="/docs"><strong>Interactive API docs (Swagger UI)</strong></a> —
      open <code>POST /predict</code>, click <em>Try it out</em>, and send the
      example client.</li>
  <li><a href="/health">Health check</a> — returns <code>{"status": "ok"}</code>.</li>
</ul>

<h2>Example request</h2>
<pre>curl -X POST https://bank-term-deposit-prediction-s1y8.onrender.com/predict \\
  -H 'Content-Type: application/json' \\
  -d '{"age": 58, "job": "retired", "marital": "married",
       "education": "secondary", "default": "no", "balance": 2143,
       "housing": "no", "loan": "no", "contact": "cellular",
       "day": 5, "month": "mar", "campaign": 1, "pdays": 180,
       "previous": 2, "poutcome": "success"}'</pre>
<p>Response: <code>{"subscribe_probability": 0.81, "subscribe": true}</code></p>

<p>Note: the first request after a period of inactivity can take up to a
minute — the free hosting tier sleeps when idle. Code, notebook and full
documentation:
<a href="https://github.com/felicity8899/bank-term-deposit-prediction">github.com/felicity8899/bank-term-deposit-prediction</a></p>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return LANDING_PAGE


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(client: Client):
    # Same dict shape the model was trained on (DictVectorizer input).
    record = client.model_dump()
    X = dv.transform([record])
    proba = float(model.predict_proba(X)[0, 1])
    return {
        "subscribe_probability": round(proba, 4),
        "subscribe": bool(proba >= THRESHOLD),
    }
