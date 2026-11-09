"""Train the final Bank Term Deposit model and save it as model.bin.

This script reproduces, outside the notebook, the exact recipe that
notebook.ipynb (Sections 4, 7 and 8) selected:

  1. Split off the test set (never touched here) and work on the 80%
     "full-train" block, split 75/25 into train/validation.
  2. One-hot encode the 15 deployable features with DictVectorizer.
     `duration` is excluded on purpose: it is the length of the final
     campaign call, unknowable before the call, and leaks the target
     (notebook Section 3.6 — it inflates validation AUC from 0.798 to
     0.937 without adding any deployable skill).
  3. Fit the tuned winner (XGBoost, see PARAMS) on the train part and pick
     the decision threshold that maximises F1 on the validation part.
  4. Retrain the winner on the full-train block (train + validation).
  5. Save {"dv", "model", "features", "threshold"} to model.bin.

Usage:
    python train.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "bank-full.csv"
MODEL_PATH = BASE_DIR / "model.bin"

# The 15 features known BEFORE the campaign call (no 'duration' — it leaks).
FEATURES = [
    "age", "job", "marital", "education", "default", "balance",
    "housing", "loan", "contact", "day", "month",
    "campaign", "pdays", "previous", "poutcome",
]

# Winning hyperparameters from the notebook's validation tuning (Section 7).
PARAMS = {
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.1,
    "min_child_weight": 5,
}

SEED = 1  # same seed as the notebook: identical splits, identical model


def build_model() -> XGBClassifier:
    return XGBClassifier(**PARAMS, tree_method="hist", random_state=SEED, n_jobs=-1)


def main() -> None:
    df = pd.read_csv(DATA_PATH, sep=";")
    df["y"] = (df["y"] == "yes").astype(int)

    # Same two-step stratified split as the notebook (60/20/20 overall).
    df_full_train, _df_test = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["y"]
    )
    df_train, df_val = train_test_split(
        df_full_train, test_size=0.25, random_state=SEED, stratify=df_full_train["y"]
    )

    # --- Step 1: fit on the train part, choose the threshold on validation ---
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(df_train[FEATURES].to_dict(orient="records"))
    X_val = dv.transform(df_val[FEATURES].to_dict(orient="records"))
    y_train, y_val = df_train["y"].values, df_val["y"].values

    val_model = build_model()
    val_model.fit(X_train, y_train)
    val_proba = val_model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_proba)

    thresholds = np.arange(0.05, 0.9, 0.01)
    f1s = [f1_score(y_val, (val_proba >= t).astype(int)) for t in thresholds]
    threshold = float(thresholds[int(np.argmax(f1s))])
    print(f"Validation ROC AUC (train-only fit): {val_auc:.4f}")
    print(f"Decision threshold (max F1 on validation): {threshold:.2f}")

    # --- Step 2: retrain on the full-train block (train + validation) ---
    dv_final = DictVectorizer(sparse=False)
    X_full = dv_final.fit_transform(df_full_train[FEATURES].to_dict(orient="records"))
    y_full = df_full_train["y"].values

    final_model = build_model()
    final_model.fit(X_full, y_full)

    artifact = {
        "dv": dv_final,
        "model": final_model,
        "features": FEATURES,
        "threshold": threshold,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f)

    print(f"Saved {MODEL_PATH} "
          f"(model: XGBoost {PARAMS}, features: {len(FEATURES)}, "
          f"trained on {len(df_full_train):,} rows)")


if __name__ == "__main__":
    main()
