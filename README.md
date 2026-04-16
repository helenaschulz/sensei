# SENSEI — Session Intelligence

A session-based analytics framework built on the [RetailRocket e-commerce dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset).

Transforms a raw event log (views, add-to-carts, transactions) into a **session feature store** — one row per session — and uses that store as the foundation for intelligence modules.

---

## Module 1 — Sessionisation & Feature Engineering

Builds the session feature store from the raw event log.

### Notebooks

| Notebook | Description |
|---|---|
| [`01_eda.ipynb`](notebooks/01_eda.ipynb) | Explore the raw event log: distributions, funnel analysis, time patterns, bot detection |
| [`02_sessionize.ipynb`](notebooks/02_sessionize.ipynb) | Build the session feature store (one row per session) and export to parquet |

### Key numbers
| | |
|---|---|
| Sessions in feature store | ~1,761,675 |
| Features | 9 |
| Purchase rate | 0.81 % |
| Evaluation | PR-AUC, F1, ROC-AUC |


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

---

## Module 2 — Purchase Prediction

Binary classification: does a session end in a purchase?

**Target:** `purchased` (1 = session contains a transaction event)
**Challenge:** 0.81 % positive class — a 1:122 imbalance.
