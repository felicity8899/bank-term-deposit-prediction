# Bank Term Deposit Prediction

**Midterm project — DataTalksClub Machine Learning Zoomcamp**

Predicting which bank clients will subscribe to a term deposit, so a phone
marketing campaign can call the right people first instead of everyone.

**Live demo:** LIVE_URL_PLACEHOLDER

---

## 1. Problem description

A Portuguese bank sells term deposits (fixed-term savings products) through
phone marketing campaigns. Agents call clients one by one and try to persuade
them to subscribe. Calling everyone is expensive — agent time is the scarce
resource — and repeatedly calling people who will never buy damages the
client relationship.

This project builds a classifier that, from what the bank knows about a
client **before** the campaign call (client profile, account and credit
status, and the history of previous campaigns), predicts the probability
that the client will subscribe.

**How the predictions are used.** The model does not replace the campaign;
it *ranks* it. Clients are scored, the call list is sorted by predicted
probability, and agents work from the top down. The evaluation therefore
focuses on ranking quality (ROC AUC) and on a lift analysis ("if we call the
top 20% of the list, what share of all subscribers do we reach?") rather
than raw accuracy — with only 11.7% of clients subscribing, a model that
predicts "nobody subscribes" is already 88.3% accurate and completely
useless.

## 2. Dataset

[UCI Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)
(Moro et al., 2014): **45,211 clients**, 16 input features + the target
`y` (`yes`/`no` — did the client subscribe?). Positive rate: **11.7%**
(5,289 subscribers).

The data is committed to this repo at `data/bank-full.csv`. To re-download
it from UCI (the zip contains a nested `bank.zip`; the script handles that):

```bash
python data/download_data.py
```

## 3. EDA highlights (details in `notebook.ipynb`)

- **Class imbalance** — 11.7% positive; this drives the choice of ROC AUC
  over accuracy everywhere in the project.
- **Strongest honest signal:** `poutcome = success` — clients whose previous
  campaign succeeded subscribe **64.7%** of the time (vs the 11.7% base rate).
- **Who subscribes:** students (28.7%) and retired clients (22.8%) lead;
  singles (14.9%) ahead of married (10.1%); tertiary education (15.0%) ahead
  of primary (8.6%). Subscribers also carry higher balances (mean ≈1,804 vs
  ≈1,304 EUR).
- **Debt hurts:** a housing loan cuts the rate to 7.7% (vs 16.7% without);
  a personal loan roughly halves it too.
- **Seasonality is dramatic:** March converts at 52.0% and December at
  46.7%, while May — the largest campaign month — converts at only 6.7%.
- **Reachability matters:** `contact = unknown` clients subscribe at just
  4.1%, vs 14.9% for cellular contact.
- **`pdays = -1` is a code, not a number** (never contacted before, ~82% of
  clients) — kept as-is; `'unknown'` categories are kept as their own
  category rather than imputed or dropped.

## 4. Models and results

Four model families were trained on one-hot encoded features
(`DictVectorizer`), compared on a stratified validation set (train /
validation / test = 60 / 20 / 20, seed fixed), then tuned. All figures are
**validation ROC AUC** on the 15 deployable features (no `duration` — see
the leakage note below).

| Model | Baseline | After tuning | Tuned parameters |
|---|---|---|---|
| Logistic Regression | 0.7680 | 0.7680 | C = 1.0 |
| Decision Tree | 0.6057 | 0.7646 | max_depth = None, min_samples_leaf = 50 |
| Random Forest | 0.7845 | 0.8000 | n_estimators = 200, max_depth = 16, min_samples_leaf = 5 |
| **XGBoost** | 0.7984 | **0.8046** | n_estimators = 300, max_depth = 3, learning_rate = 0.1, min_child_weight = 5 |

**Final model:** XGBoost with the tuned parameters, retrained on
train + validation. **Test ROC AUC = 0.8060** (single evaluation on the
held-out test set). Decision threshold 0.18 (chosen by maximising F1 on the
validation set); at that threshold on the test set: precision 0.459,
recall 0.546.

**What the ranking buys the bank** (test set, clients ranked by predicted
probability):

| Call top | Clients called | Share of subscribers captured | Precision in group | Lift vs random |
|---|---|---|---|---|
| 10% | 904 | 44.0% | 51.4% | 4.40× |
| 20% | 1,808 | 62.5% | 36.6% | 3.12× |
| 30% | 2,712 | 72.6% | 28.3% | 2.42× |
| 50% | 4,521 | 83.9% | 19.6% | 1.68× |

### The `duration` leakage finding

`duration` (length of the final campaign call) is the dataset's trap: it is
only known **after** the call — the very call in which the client decides.
The UCI paper itself warns that if `duration = 0` then `y = no`. Measured
in this project with identical XGBoost configurations:

- With `duration`: validation ROC AUC **0.9370**
- Without `duration` (deployable): validation ROC AUC **0.7984**

The +0.139 gap is pure hindsight, not skill. The deployed model **excludes
`duration`**; the notebook (Sections 3.6 and 6.2) documents the effect in
full. Subscription rate by call-length bucket tells the same story: 0.2%
for calls under a minute, rising to 48.4% for calls over 10 minutes —
because interested clients keep talking, not the other way round.

## 5. How to run everything

### 5.1 Environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 5.2 Explore the analysis

```bash
jupyter notebook notebook.ipynb   # fully executed; all outputs embedded
```

### 5.3 Train the final model

```bash
python train.py                   # writes model.bin (vectorizer + model + threshold)
```

### 5.4 Serve predictions locally

```bash
uvicorn predict:app --host 0.0.0.0 --port 9696
python predict_test.py            # sends one sample client, prints the answer
```

`POST /predict` accepts a client JSON with the 15 pre-call features
(everything except `duration`) and returns
`{"subscribe_probability": <float>, "subscribe": <bool>}`;
`GET /health` returns `{"status": "ok"}`.

### 5.5 Docker

```bash
docker build -t bank-term-deposit .
docker run -p 9696:9696 bank-term-deposit
# the image honours $PORT (cloud platforms inject their own):
#   CMD uvicorn predict:app --host 0.0.0.0 --port ${PORT:-9696}
```

### 5.6 Cloud deployment

The same Docker image runs on Render (free web service, Docker runtime from
this repo). **Live URL:** LIVE_URL_PLACEHOLDER

## 6. Project structure

```
bank-term-deposit-prediction/
├── README.md               <- this file
├── DESIGN.md               <- project design notes
├── notebook.ipynb          <- full analysis, executed with outputs
├── train.py                <- trains the final model, saves model.bin
├── predict.py              <- FastAPI prediction service
├── predict_test.py         <- smoke-test client for the service
├── model.bin               <- trained artifact (produced by train.py)
├── requirements.txt        <- dependencies (with version floors)
├── Dockerfile              <- service image, $PORT-aware
└── data/
    ├── bank-full.csv       <- the dataset (45,211 rows)
    └── download_data.py    <- re-downloads the dataset from UCI
```

## 7. How it works — study notes

*Pipeline decisions, in the order the notebook makes them.*

1. **Metric first.** With an 11.7% positive rate, accuracy rewards doing
   nothing (88.3% for "always no"). ROC AUC measures ranking quality —
   "is a random subscriber scored above a random non-subscriber?" — which
   is exactly the bank's task (rank the call list). Threshold-dependent
   numbers (precision/recall) are reported at one deliberate operating
   point, not optimised away.
2. **Missingness is data.** `'unknown'` categories and the `pdays = -1`
   sentinel are kept, not imputed or dropped: they identify real segments
   (e.g. never-contacted clients), and dropping `poutcome = unknown` alone
   would delete ~82% of the dataset.
3. **One-hot encoding via `DictVectorizer`.** Categories have no natural
   order, so each becomes its own 0/1 column (50 columns total); numeric
   features pass through. The same dict shape is what the web service
   accepts as JSON, which keeps training and serving perfectly aligned.
4. **Split discipline.** Stratified 60/20/20 with a fixed seed. Validation
   is spent freely (model selection + tuning); the test set is touched
   exactly once, at the end — the small val→test gap (0.805 → 0.806) is
   the evidence this discipline worked.
5. **Leakage before modelling.** The `duration` decision (Section 4) is
   the project's most important one: recognising that a feature is
   *unknowable at decision time* matters more than any hyperparameter.
   A model may only use information available when the decision is made.
6. **Tuning lessons.** The unconstrained decision tree was the worst model
   (0.606) and was rescued purely by regularisation (min_samples_leaf = 50
   → 0.765): a clean demonstration that "bad model" often means
   "unregularised model". The ensembles gained only a few thousandths from
   tuning (XGBoost prefers *shallower* trees than its default: depth 3) —
   defaults of mature libraries are already good; tuning polishes, it does
   not transform. Logistic regression was flat in C: it is limited by its
   linear form, not its regularisation.
7. **Threshold is a business setting, not a model property.** The stored
   threshold (0.18) maximises F1 on validation — a balanced default. The
   lift table is the tool for choosing a different one: it converts any
   calling budget directly into expected subscriber capture.
8. **Serving mirrors training.** `train.py` re-runs the notebook's exact
   recipe (same splits, same seed, same hyperparameters) so `model.bin`
   is reproducible from `data/bank-full.csv` with one command;
   `predict.py` loads the pickled {vectorizer, model, features,
   threshold} and applies the identical transformation path — no
   training/serving skew.

*File-by-file:* `notebook.ipynb` is the analysis and the evidence;
`train.py` is the notebook's conclusion turned into a reproducible script;
`predict.py` + `predict_test.py` are the service and its smoke test;
`data/download_data.py` makes the dataset provenance reproducible;
`Dockerfile` packages the service honouring `$PORT` so the same image runs
locally (9696) and on the cloud platform unchanged.
