# RetailRocket — Session-Based Purchase Prediction

Predicting whether an e-commerce browsing session ends in a purchase, using the [RetailRocket dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset).

## Problem

Given a visitor's session (a sequence of `view`, `addtocart`, and `transaction` events), predict whether the session results in a purchase.

The target is heavily imbalanced: only **0.81 %** of sessions end in a purchase.

## Project structure

```
sessie/
├── notebooks/
│   ├── 01_eda.ipynb            ← Exploratory Data Analysis
│   ├── 02_sessionize.ipynb     ← Sessionisation & Feature Engineering
│   └── 03_modeling.ipynb       ← Classification models & evaluation
├── src/
│   └── session_utils.py        ← Reusable helper functions
├── data/                       ← Not tracked by git (add files manually)
│   └── .gitkeep
├── requirements.txt
└── .gitignore
```

## Data

Download from Kaggle and place in `data/`:

```bash
kaggle datasets download -d retailrocket/ecommerce-dataset -p data/ --unzip
```

Required files:
- `data/events.csv`
- `data/category_tree.csv`
- `data/item_properties_part1.csv`
- `data/item_properties_part2.csv`

## Setup

```bash
pip install -r requirements.txt
```

## Run the notebooks in order

1. `01_eda.ipynb` — understand the raw data
2. `02_sessionize.ipynb` — build the session feature table (saves `data/sessions_features.parquet`)
3. `03_modeling.ipynb` — train and evaluate classification models

## Key results

| Metric | Value |
|--------|-------|
| Total sessions | 1,761,675 |
| Features | 6 |
| Target | `purchased` (binary) |
| Purchase rate | 0.81 % |

## Dataset

Seyfi, I. et al. (2015). *RetailRocket E-Commerce Dataset*. Kaggle.  
License: CC0 Public Domain.
