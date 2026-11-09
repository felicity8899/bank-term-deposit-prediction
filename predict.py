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
