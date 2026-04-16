"""
Utility functions for RetailRocket session analysis.
"""

import pandas as pd
import numpy as np


SESSION_GAP = pd.Timedelta(minutes=30)


def load_events(path: str) -> pd.DataFrame:
    """Load and preprocess the events CSV."""
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.sort_values(["visitorid", "timestamp"]).reset_index(drop=True)
    return df


def assign_session_ids(df: pd.DataFrame, gap: pd.Timedelta = SESSION_GAP) -> pd.DataFrame:
    """
    Assign a session_id to each row using a time-based gap rule.
    A new session starts when the gap between two consecutive events
    from the same visitor exceeds `gap` (default: 30 minutes).
    """
    df = df.copy()
    prev_ts = df.groupby("visitorid")["timestamp"].shift(1)
    time_diff = df["timestamp"] - prev_ts
    new_session = time_diff.isna() | (time_diff > gap)
    df["session_id"] = new_session.cumsum()
    return df


def build_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate event-level rows into one row per session.

    Features:
        n_events       - total number of events in the session
        n_views        - number of view events
        n_addtocart    - number of addtocart events
        n_items        - number of unique items interacted with
        duration_sec   - session duration in seconds
        has_addtocart  - binary: did visitor add anything to cart?
        purchased      - TARGET: did session end in a transaction?
    """
    agg = df.groupby("session_id").agg(
        visitorid=("visitorid", "first"),
        n_events=("event", "count"),
        n_views=("event", lambda x: (x == "view").sum()),
        n_addtocart=("event", lambda x: (x == "addtocart").sum()),
        n_items=("itemid", "nunique"),
        duration_sec=("timestamp", lambda x: (x.max() - x.min()).total_seconds()),
        purchased=("event", lambda x: int((x == "transaction").any())),
    )
    agg["has_addtocart"] = (agg["n_addtocart"] > 0).astype(int)
    return agg.reset_index()
