# SENSEI — Session Intelligence

> **S**ession **I**ntelligence for E-commerce — turning raw clickstream data into actionable predictions.

SENSEI is a modular analytics framework built on the [RetailRocket e-commerce dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset).  
It transforms a raw event log (views, add-to-carts, transactions) into a session-level feature store and uses that store to power a growing suite of intelligence modules.

---

## Vision

Most e-commerce analytics stops at aggregated reports. SENSEI goes deeper:  
**every browsing session tells a story** — who the visitor is, what they intend, and what they will do next.

The goal is to build a reusable session intelligence layer that can answer questions like:

| Question | Module |
|---|---|
| Will this session end in a purchase? | **Module 1 — Purchase Prediction** ✅ |
| How engaged is this session? | Module 2 — Session Quality Score *(planned)* |
| What will the visitor do next? | Module 3 — Next-Action Prediction *(planned)* |
| Is this visitor about to leave? | Module 4 — Bounce / Exit Prediction *(planned)* |
| Which sessions are anomalous? | Module 5 — Anomaly Detection *(planned)* |

---

## Module 1 — Purchase Prediction

Predict whether a browsing session ends in a purchase (binary classification).  
This is the first and most business-critical question: **is this visitor about to buy?**

**Challenge:** Only 0.81 % of sessions result in a purchase — severe class imbalance.

### Notebooks

Run in order:

| Notebook | Purpose |
|---|---|
| [`01_eda.ipynb`](notebooks/01_eda.ipynb) | Explore the raw event log — distributions, funnel, time patterns |
| [`02_sessionize.ipynb`](notebooks/02_sessionize.ipynb) | Build the SENSEI session feature store (one row per session) |
| [`03_modeling.ipynb`](notebooks/03_modeling.ipynb) | Train, evaluate and tune purchase prediction classifiers |

### Session features (feature store)

| Feature | Description |
|---|---|
| `n_views` | Views in session |
| `n_addtocart` | Add-to-cart events |
| `n_items` | Unique items interacted with |
| `n_revisited_items` | Items viewed more than once |
| `duration_sec` | Session length in seconds |
| `hour_of_day` | Hour of session start (UTC) |
| `day_of_week` | 0 = Monday … 6 = Sunday |
| `view_to_cart_ratio` | n_addtocart / n_views |
| `is_first_session` | 1 if visitor's first ever session |
| **`purchased`** | **Target — 1 if session contains a transaction** |

### Key results

| Metric | Value |
|---|---|
| Sessions in feature store | ~1,761,675 |
| Features | 9 |
| Purchase rate | 0.81 % (imbalance ratio 1:122) |
| Evaluation metrics | PR-AUC, F1, ROC-AUC (not accuracy) |

---

## Project structure

```
sensei/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_sessionize.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── __init__.py
│   └── session_utils.py        ← SENSEI session engine
├── data/                       ← not tracked by git
│   └── .gitkeep
├── requirements.txt
└── .gitignore
```

## Data

Download from Kaggle and place in `data/`:

```bash
kaggle datasets download -d retailrocket/ecommerce-dataset -p data/ --unzip
```

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

RetailRocket (2015). *E-Commerce Dataset*. Kaggle. License: CC0 Public Domain.
