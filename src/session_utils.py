"""
SENSEI — Session Intelligence
==============================
Core session engine: load, sessionise, and featurise the RetailRocket event log.

This module is the foundation of the SENSEI feature store.
Every intelligence module (purchase prediction, session quality,
next-action prediction, etc.) builds on the session-level features
produced here.

Functions
---------
load_events          : Load and preprocess the raw events CSV.
assign_session_ids   : Split the event stream into sessions (30-min gap rule).
build_session_features : Aggregate events into one row per session.
"""

import pandas as pd
import numpy as np


SESSION_GAP = pd.Timedelta(minutes=30)


def load_events(path: str) -> pd.DataFrame:
    """
    Load and preprocess the RetailRocket events CSV.

    Steps:
    - Parse the Unix millisecond timestamp to datetime.
    - Sort by visitor and time (required for correct session assignment).
    """
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.sort_values(["visitorid", "timestamp"]).reset_index(drop=True)
    return df


def assign_session_ids(df: pd.DataFrame, gap: pd.Timedelta = SESSION_GAP) -> pd.DataFrame:
    """
    Assign a session_id to each event row using a time-based inactivity gap rule.

    A new session starts when a visitor is inactive for longer than `gap`
    (default: 30 minutes — the Google Analytics standard).

    The dataframe is sorted internally, so this function is safe to call
    on unsorted input.
    """
    df = df.sort_values(["visitorid", "timestamp"]).copy()
    prev_ts = df.groupby("visitorid")["timestamp"].shift(1)
    time_diff = df["timestamp"] - prev_ts
    new_session = time_diff.isna() | (time_diff > gap)
    df["session_id"] = new_session.cumsum()
    return df


def build_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the event-level data into the SENSEI session feature store.
    One row per session. Used as input for all SENSEI intelligence modules.

    Features produced
    -----------------
    n_views             : number of view events
    n_addtocart         : number of addtocart events
    n_items             : number of unique items interacted with
    n_revisited_items   : items viewed more than once (re-engagement / intent signal)
    duration_sec        : session duration in seconds
    hour_of_day         : hour of session start (0–23, UTC)
    day_of_week         : day of session start (0=Monday, 6=Sunday)
    view_to_cart_ratio  : n_addtocart / n_views — visitor decisiveness
    is_first_session    : 1 if this is the visitor's first recorded session

    Target (Module 1 — Purchase Prediction)
    ----------------------------------------
    purchased           : 1 if the session contains at least one transaction event
    """
    # --- core aggregation ---
    agg = df.groupby("session_id").agg(
        visitorid=("visitorid", "first"),
        startdate=("timestamp", "min"),
        n_views=("event", lambda x: (x == "view").sum()),
        n_addtocart=("event", lambda x: (x == "addtocart").sum()),
        n_items=("itemid", "nunique"),
        duration_sec=("timestamp", lambda x: (x.max() - x.min()).total_seconds()),
        purchased=("event", lambda x: int((x == "transaction").any())),
    ).reset_index()

    # --- repeated item views (re-engagement signal) ---
    revisited = (
        df[df["event"] == "view"]
        .groupby(["session_id", "itemid"])
        .size()
        .reset_index(name="view_count")
    )
    revisited = (
        revisited[revisited["view_count"] > 1]
        .groupby("session_id")
        .size()
        .rename("n_revisited_items")
        .reset_index()
    )
    agg = agg.merge(revisited, on="session_id", how="left")
    agg["n_revisited_items"] = agg["n_revisited_items"].fillna(0).astype(int)

    # --- temporal features ---
    agg["hour_of_day"] = agg["startdate"].dt.hour
    agg["day_of_week"] = agg["startdate"].dt.dayofweek

    # --- decisiveness ratio ---
    agg["view_to_cart_ratio"] = np.where(
        agg["n_views"] > 0, agg["n_addtocart"] / agg["n_views"], 0.0
    )

    # --- returning visitor signal ---
    agg = agg.sort_values(["visitorid", "startdate"]).reset_index(drop=True)
    agg["is_first_session"] = (~agg.duplicated(subset="visitorid", keep="first")).astype(int)

    return agg.drop(columns=["startdate"])
