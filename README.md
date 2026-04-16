# SENSEI — Session Intelligence

A session-based analytics framework built on the [RetailRocket e-commerce dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset).

The core idea is to transform a raw event log (views, add-to-carts, transactions) into a
**session feature store** — one row per session — and use that store as the foundation for
multiple intelligence modules. The framework is designed to be extensible: each module
is a separate analysis built on the same feature store.

---

## Module 1 — Purchase Prediction

Predict whether a browsing session ends in a purchase (binary classification).

**Target:** `purchased` (1 = session contains a transaction event)  
**Challenge:** 0.81 % positive class — a 1:122 imbalance.

### Notebooks

| Notebook | Description |
|---|---|
| [`01_eda.ipynb`](notebooks/01_eda.ipynb) | Explore the raw event log: distributions, funnel analysis, time patterns, bot detection |
| [`02_sessionize.ipynb`](notebooks/02_sessionize.ipynb) | Build the session feature store (one row per session) and export to parquet |
| [`03_modeling.ipynb`](notebooks/03_modeling.ipynb) | Train and evaluate classifiers; includes cross-validation, permutation importance, threshold tuning |

### Session features

| Feature | Description |
|---|---|
| `n_views` | Number of view events in the session |
| `n_addtocart` | Number of add-to-cart events |
| `n_items` | Unique items interacted with |
| `n_revisited_items` | Items viewed more than once |
| `duration_sec` | Session duration in seconds |
| `hour_of_day` | Hour of session start (UTC) |
| `day_of_week` | 0 = Monday … 6 = Sunday |
| `view_to_cart_ratio` | n_addtocart / n_views |
| `is_first_session` | 1 if this is the visitor's first recorded session |
| **`purchased`** | **Target** |

### Key numbers

| | |
|---|---|
| Sessions in feature store | ~1,761,675 |
| Features | 9 |
| Purchase rate | 0.81 % |
| Evaluation | PR-AUC, F1, ROC-AUC |

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
│   └── session_utils.py       ← load, sessionise, featurise
├── data/                      ← not tracked by git
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
