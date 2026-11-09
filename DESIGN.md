# Design — ML Zoomcamp Midterm Project

Owner: Xinli Zhang. Course: DataTalksClub `machine-learning-zoomcamp`, midterm project.
Grading: 9 criteria, 16 points max (peer-reviewed; must also review 3 peers or fail).

## 1. What the course demands (official projects page)

Deliverables: README (problem + how to run) · data (committed or clear download
instructions) · `notebook.ipynb` (data prep/cleaning, EDA + feature importance,
model selection + tuning) · `train.py` (trains final model, saves it) ·
`predict.py` (loads model, serves it as a web service) · dependency files ·
`Dockerfile` · deployment (URL, or video/image of the deployed service).

Criteria (max): problem description 2 · EDA 2 · model training (multiple models
+ tuning) 3 · notebook→script 1 · reproducibility 1 · model deployment 1 ·
dependency & environment management 2 · containerization 2 · cloud deployment 2.
**Total 16.**

Banned datasets (used in course modules): car prices, telco churn, the module-6
credit-risk dataset, clothing classification. Plagiarism rules: no reusing own
projects from other courses, no reusing the midterm in a capstone.

## 2. The project: **Bank Term Deposit Prediction**

**Problem.** A Portuguese bank ran phone marketing campaigns for term deposits.
Calling everyone is expensive and annoys clients. Predict, from client +
campaign features, whether a client will subscribe (`y`), so the bank can rank
and target the call list. Business framing fits Xinli's banking background
(HSBC/Barclays) without touching any banned dataset.

**Dataset.** UCI Bank Marketing (`bank-full.csv`, 45,211 rows, 16 features +
target). Direct download from UCI (zip), no account/auth needed. Committed to
the repo under `data/` (~4.6 MB CSV) AND a `data/download_data.py` script +
README instructions (reproducibility criterion, belt and braces).
Class balance ~11.7% positive — ROC AUC is the headline metric, with a
threshold/business analysis (precision/recall at operating points).

## 3. Modeling plan (notebook.ipynb, executed with outputs)

1. Load, inspect, clean: `unknown` values treated as their own category,
   no missing-value black holes; type fixes.
2. EDA (extensive, for the 2 points): target distribution, numeric ranges and
   distributions, categorical breakdowns vs target rate, month/duration
   effects, correlation heatmap, feature importance (mutual information +
   a quick random-forest importance).
3. Split: train/validation/test 60/20/20, stratified, fixed seed.
4. Models: Logistic Regression (DictVectorizer one-hot), Decision Tree,
   Random Forest, XGBoost. 
5. Tuning (for the 3rd point): parameter search over tree depth / estimators /
   learning rate / min_child_weight etc. on validation AUC, documented in a
   small results table; final model re-trained on train+val, single test eval.
6. Business readout: AUC on test, ROC curve, precision/recall at 2–3
   thresholds, "call top-X% capture Y% of subscribers" lift table.

Note on `duration`: it is known to leak the target (call length is only known
after the call). The notebook discusses it explicitly and reports the headline
model **without** duration as the deployable model — reviewers reward this
kind of honesty; a with-duration comparison shows the leak's size.

## 4. Serving & deployment

- `train.py`: trains the exact final pipeline (DictVectorizer + XGBoost or the
  notebook's winner), saves `model.bin` (pickle of (vectorizer, model,
  threshold metadata)).
- `predict.py`: **FastAPI** service, `POST /predict` with a client JSON →
  `{subscribe_probability, subscribe}`; `GET /health`. Runs locally with
  uvicorn; test client `predict_test.py` included.
- `Dockerfile`: python:3.12-slim, installs requirements, copies model +
  predict.py, uvicorn on port 9696 (course convention) — actually use 8080
  and document it; simpler: keep 9696, Render maps its own PORT env → the
  CMD must respect `$PORT` (Render sets it). Use shell-form CMD with $PORT
  defaulting to 9696.
- Dependencies: `requirements.txt` + README section creating/activating a
  venv (dependency criterion wants both file + documented env workflow).
- **Cloud**: second free service on her existing Render account, from the new
  GitHub repo, root Dockerfile. Live URL into README.

## 5. Repo

New public repo `felicity8899/bank-term-deposit-prediction`:

```
bank-term-deposit-prediction/
  README.md
  notebook.ipynb        (executed, outputs saved)
  train.py  predict.py  predict_test.py
  data/bank-full.csv  data/download_data.py
  model.bin             (produced by train.py; committed so the service runs anywhere)
  requirements.txt  Dockerfile  .gitignore
```

## 6. Execution plan

1. Download data from UCI, sanity-check shape/target.
2. Build notebook with jupyter/nbformat programmatically, execute it
   end-to-end in the project venv (outputs must be real), verify final metrics.
3. train.py → model.bin; predict.py → local uvicorn + predict_test.py passes.
4. Dockerfile written to serve on $PORT; local (non-Docker) verification of
   the same command path (no Docker in this VM — Render build is the proof).
5. README (problem, EDA summary, results table, how to run: venv, notebook,
   train, serve, docker, cloud URL).
6. Main chat: create GitHub repo, push, create Render service, verify URL,
   final commit SHA + criteria audit for Xinli.

## 7. Honesty constraints

- All numbers in README must come from the executed notebook/train run.
- No claim of Docker verification locally (none possible in this VM);
  Render's build is the container proof.
- Dataset is UCI's, cited; not one of the banned module datasets.
