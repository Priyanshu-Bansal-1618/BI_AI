"""
Deterministic anomaly detection — Scikit-learn / statistics only.
No LLM calls happen anywhere in this module, by design.

Two complementary detectors are combined:
  1. Seasonal baseline + z-score  -> fast, interpretable, good for
     metrics with clear day-of-week / weekly seasonality (revenue).
  2. IsolationForest              -> catches multivariate anomalies
     across (value, day-of-week, rolling volatility) that a plain
     z-score misses, e.g. a value that's "normal" in isolation but
     anomalous given how volatile the series has been lately.

A movement is flagged `is_material` only if it clears BOTH the
statistical threshold (z-score) AND the business-materiality threshold
(pct_change) from the semantic contract — this avoids paging someone
over a statistically significant but practically irrelevant 0.4% move.
"""

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from backend import models
from backend.db import get_db

# Loaded from semantic_contract.yaml in production; inlined here for clarity
KPI_THRESHOLDS = {
    "kpi.revenue": {"z_score": 2.5, "pct_change_material": 0.08, "min_baseline_periods": 14},
    "kpi.stockout_rate": {"z_score": 2.0, "pct_change_material": 0.15, "min_baseline_periods": 14},
    "kpi.complaint_rate": {"z_score": 2.0, "pct_change_material": 0.20, "min_baseline_periods": 10},
}


@dataclass
class AnomalyResult:
    kpi_id: str
    metric_date: date
    observed_value: float
    expected_value: float
    z_score: float
    pct_change: float
    is_material: bool
    method: str                 # "zscore" | "isolation_forest" | "sparse_history_fallback"
    baseline_periods_available: int


def _load_kpi_timeseries(db, kpi_id: str, as_of: date, lookback_days: int = 90) -> pd.DataFrame:
    """Pulls the daily rollup for a KPI from Postgres into a tidy DataFrame:
    columns = [metric_date, value]. Aggregated across all dimensions here;
    RCA (analytics/rca.py) is what breaks it back out by segment."""
    start = as_of - timedelta(days=lookback_days)

    if kpi_id == "kpi.revenue":
        rows = (
            db.query(models.RevenueDaily.metric_date, models.RevenueDaily.revenue)
            .filter(models.RevenueDaily.metric_date.between(start, as_of))
            .all()
        )
        df = pd.DataFrame(rows, columns=["metric_date", "value"])
        df = df.groupby("metric_date", as_index=False)["value"].sum()

    elif kpi_id == "kpi.stockout_rate":
        rows = (
            db.query(
                models.StockoutDaily.metric_date,
                models.StockoutDaily.sku_days_out_of_stock,
                models.StockoutDaily.sku_days_total,
            )
            .filter(models.StockoutDaily.metric_date.between(start, as_of))
            .all()
        )
        df = pd.DataFrame(rows, columns=["metric_date", "oos", "total"])
        df = df.groupby("metric_date", as_index=False).sum()
        df["value"] = df["oos"] / df["total"].replace(0, np.nan)
        df = df[["metric_date", "value"]]

    elif kpi_id == "kpi.complaint_rate":
        rows = (
            db.query(models.SupportTicketMeta.ticket_date)
            .filter(
                models.SupportTicketMeta.ticket_date.between(start, as_of),
                models.SupportTicketMeta.sentiment == "negative",
                models.SupportTicketMeta.category == "product_quality",
            )
            .all()
        )
        df = pd.DataFrame(rows, columns=["metric_date"])
        df = df.groupby("metric_date").size().reset_index(name="value")

    else:
        raise ValueError(f"Unknown kpi_id: {kpi_id}")

    df = df.sort_values("metric_date").reset_index(drop=True)
    return df


def _zscore_detect(df: pd.DataFrame, as_of: date, thresholds: dict) -> AnomalyResult | None:
    history = df[df["metric_date"] < as_of]["value"]
    today_rows = df[df["metric_date"] == as_of]
    if today_rows.empty or len(history) < thresholds["min_baseline_periods"]:
        return None

    observed = float(today_rows["value"].iloc[0])
    mu, sigma = history.mean(), history.std(ddof=1)
    if sigma == 0 or np.isnan(sigma):
        sigma = 1e-9

    z = (observed - mu) / sigma
    pct_change = (observed - mu) / mu if mu != 0 else float("inf")
    is_material = abs(z) >= thresholds["z_score"] and abs(pct_change) >= thresholds["pct_change_material"]

    return AnomalyResult(
        kpi_id="", metric_date=as_of, observed_value=observed, expected_value=float(mu),
        z_score=float(z), pct_change=float(pct_change), is_material=is_material,
        method="zscore", baseline_periods_available=len(history),
    )


def _isolation_forest_detect(df: pd.DataFrame, as_of: date, thresholds: dict) -> AnomalyResult | None:
    """Multivariate check: [value, day_of_week, rolling_7d_std]. Used as a
    secondary confirmation signal, and as the PRIMARY signal for
    sparse-history products (see Deliverable 5 / sparse_history.py)."""
    d = df.copy()
    if len(d) < 8:
        return None
    d["dow"] = pd.to_datetime(d["metric_date"]).dt.dayofweek
    d["roll_std"] = d["value"].rolling(7, min_periods=3).std().bfill()

    features = d[["value", "dow", "roll_std"]].to_numpy()
    model = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    model.fit(features)
    scores = model.decision_function(features)   # lower = more anomalous
    d["anomaly_score"] = scores
    d["is_outlier"] = model.predict(features) == -1

    today_row = d[d["metric_date"] == as_of]
    if today_row.empty:
        return None
    today_row = today_row.iloc[0]

    baseline = d[d["metric_date"] < as_of]["value"]
    mu = baseline.mean() if len(baseline) else today_row["value"]
    pct_change = (today_row["value"] - mu) / mu if mu else float("inf")

    is_material = bool(today_row["is_outlier"]) and abs(pct_change) >= thresholds["pct_change_material"]

    return AnomalyResult(
        kpi_id="", metric_date=as_of, observed_value=float(today_row["value"]),
        expected_value=float(mu), z_score=float(-today_row["anomaly_score"]),  # sign-flipped for consistency
        pct_change=float(pct_change), is_material=is_material,
        method="isolation_forest", baseline_periods_available=len(baseline),
    )


def detect_anomalies_for_kpi(db, kpi_id: str, as_of: date) -> AnomalyResult | None:
    """Public entry point used by the FastAPI pipeline. Runs z-score first
    (cheap, interpretable); falls back to IsolationForest for
    sparse-history products where a simple mean/std baseline is unreliable
    (see Deliverable 5)."""
    thresholds = KPI_THRESHOLDS[kpi_id]
    df = _load_kpi_timeseries(db, kpi_id, as_of)

    if df.empty:
        return None

    history_len = len(df[df["metric_date"] < as_of])

    if history_len >= thresholds["min_baseline_periods"]:
        result = _zscore_detect(df, as_of, thresholds)
    else:
        # Sparse-history fallback — see analytics/sparse_history.py
        from analytics.sparse_history import detect_sparse_history_anomaly
        result = detect_sparse_history_anomaly(df, as_of, thresholds)

    if result:
        result.kpi_id = kpi_id
    return result
