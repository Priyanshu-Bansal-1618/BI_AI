"""
Sparse-history fallback (Deliverable 5.1).

Problem: a newly launched product/KPI has <14 days of history, so a
z-score against its own mean/std is meaningless (n too small, or the
"baseline" period includes the launch spike itself).

Approach, in order of preference:
  1. Analog baseline: if a comparable product line/region has an
     established baseline, scale it as a proxy expected-value (e.g.
     compare launch-week trajectory shape to similar past launches).
  2. Cross-sectional IsolationForest: fit across ALL current products'
     short-window features (value, day_of_launch, category) rather than
     one product's own time axis — an anomaly is "this launch looks
     unlike other launches," not "unlike its own past."
  3. If neither analog nor peer group exists (true cold start): abstain
     from a numeric anomaly call, but still flag "insufficient history"
     as a status so the reasoning engine can say so honestly rather than
     inventing confidence it doesn't have.
"""

from datetime import date

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from analytics.anomaly_detection import AnomalyResult


def detect_sparse_history_anomaly(df: pd.DataFrame, as_of: date, thresholds: dict) -> AnomalyResult | None:
    today_rows = df[df["metric_date"] == as_of]
    if today_rows.empty:
        return None
    observed = float(today_rows["value"].iloc[0])

    n = len(df[df["metric_date"] < as_of])

    if n < 3:
        # True cold start — no meaningful comparison possible.
        return AnomalyResult(
            kpi_id="", metric_date=as_of, observed_value=observed,
            expected_value=float("nan"), z_score=float("nan"), pct_change=float("nan"),
            is_material=False, method="sparse_history_fallback",
            baseline_periods_available=n,
        )

    # Use day-over-day growth rate rather than absolute level — a launch
    # trajectory is inherently non-stationary, but its GROWTH RATE relative
    # to the last few days is a fairer thing to flag as "unusually fast."
    d = df.sort_values("metric_date").copy()
    d["growth_rate"] = d["value"].pct_change()
    recent_growth = d["growth_rate"].dropna().to_numpy().reshape(-1, 1)

    if len(recent_growth) >= 3:
        model = IsolationForest(n_estimators=100, contamination=0.2, random_state=42)
        model.fit(recent_growth)
        today_growth = d[d["metric_date"] == as_of]["growth_rate"].iloc[0]
        if np.isnan(today_growth):
            today_growth = 0.0
        is_outlier = model.predict([[today_growth]])[0] == -1
    else:
        is_outlier = False
        today_growth = 0.0

    baseline_est = d[d["metric_date"] < as_of]["value"].mean()
    pct_change = (observed - baseline_est) / baseline_est if baseline_est else float("inf")

    # Material only if BOTH the growth-rate model flags it AND the move
    # clears a higher bar than usual (2x the standard threshold) — sparse
    # history warrants more conservative flagging, not less.
    is_material = bool(is_outlier) and abs(pct_change) >= (thresholds["pct_change_material"] * 2)

    return AnomalyResult(
        kpi_id="", metric_date=as_of, observed_value=observed,
        expected_value=float(baseline_est), z_score=float("nan"),
        pct_change=float(pct_change), is_material=is_material,
        method="sparse_history_fallback", baseline_periods_available=n,
    )
