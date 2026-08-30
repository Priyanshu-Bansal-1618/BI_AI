"""
Deterministic Root-Cause Analysis (RCA).

Given a flagged anomaly on an aggregate KPI, decomposes the movement into
the quantitative contribution of each sub-segment (region, product line,
warehouse), so the LLM downstream is only ever narrating numbers computed
here — never inventing or estimating them.

Method: additive contribution decomposition (a simplified, fast Shapley-
style approach appropriate for a small number of segments). For each
segment s, contribution_pct(s) = (segment's delta from its own baseline)
/ (total aggregate delta). Contributions sum to ~100% of the explained
movement; any unexplained residual is reported explicitly rather than
silently absorbed, so the LLM can be honest about ambiguity.
"""

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from backend import models
from analytics.anomaly_detection import AnomalyResult


@dataclass
class SegmentContribution:
    dimension: str        # "product_line" | "region" | "warehouse"
    segment_value: str    # e.g. "PROD-B"
    baseline_value: float
    observed_value: float
    delta: float
    contribution_pct: float   # share of total aggregate delta explained by this segment


def _segment_query(db, kpi_id: str, dimension: str, as_of: date, lookback_days: int = 90):
    start = as_of - timedelta(days=lookback_days)

    if kpi_id == "kpi.revenue" and dimension == "product_line":
        rows = (
            db.query(
                models.ProductLine.code,
                models.RevenueDaily.metric_date,
                models.RevenueDaily.revenue,
            )
            .join(models.ProductLine, models.RevenueDaily.product_line_id == models.ProductLine.id)
            .filter(models.RevenueDaily.metric_date.between(start, as_of))
            .all()
        )
        df = pd.DataFrame(rows, columns=["segment", "metric_date", "value"])

    elif kpi_id == "kpi.revenue" and dimension == "region":
        rows = (
            db.query(
                models.Region.code,
                models.RevenueDaily.metric_date,
                models.RevenueDaily.revenue,
            )
            .join(models.Region, models.RevenueDaily.region_id == models.Region.id)
            .filter(models.RevenueDaily.metric_date.between(start, as_of))
            .all()
        )
        df = pd.DataFrame(rows, columns=["segment", "metric_date", "value"])

    elif kpi_id == "kpi.stockout_rate" and dimension == "product_line":
        rows = (
            db.query(
                models.ProductLine.code,
                models.StockoutDaily.metric_date,
                models.StockoutDaily.sku_days_out_of_stock,
                models.StockoutDaily.sku_days_total,
            )
            .join(models.ProductLine, models.StockoutDaily.product_line_id == models.ProductLine.id)
            .filter(models.StockoutDaily.metric_date.between(start, as_of))
            .all()
        )
        df = pd.DataFrame(rows, columns=["segment", "metric_date", "oos", "total"])
        df["value"] = df["oos"] / df["total"].replace(0, 1)
        df = df[["segment", "metric_date", "value"]]

    else:
        return pd.DataFrame(columns=["segment", "metric_date", "value"])

    return df


def _decompose_dimension(db, kpi_id: str, dimension: str, anomaly: AnomalyResult) -> list[SegmentContribution]:
    df = _segment_query(db, kpi_id, dimension, anomaly.metric_date)
    if df.empty:
        return []

    contributions = []
    total_delta = anomaly.observed_value - anomaly.expected_value
    if total_delta == 0:
        return []

    for segment, seg_df in df.groupby("segment"):
        seg_df = seg_df.sort_values("metric_date")
        history = seg_df[seg_df["metric_date"] < anomaly.metric_date]["value"]
        today = seg_df[seg_df["metric_date"] == anomaly.metric_date]["value"]
        if today.empty or len(history) < 3:
            continue

        baseline = history.mean()
        observed = float(today.iloc[0])
        delta = observed - baseline

        contributions.append(SegmentContribution(
            dimension=dimension, segment_value=segment,
            baseline_value=float(baseline), observed_value=observed,
            delta=float(delta), contribution_pct=float(delta / total_delta),
        ))

    # Sort by absolute contribution, largest driver first
    contributions.sort(key=lambda c: abs(c.contribution_pct), reverse=True)
    return contributions


def compute_rca(db, kpi_id: str, anomaly: AnomalyResult) -> dict:
    """Returns a JSON-serializable breakdown across all relevant dimensions
    for this KPI, plus an explicit `unexplained_pct` residual."""
    dimensions_by_kpi = {
        "kpi.revenue": ["product_line", "region"],
        "kpi.stockout_rate": ["product_line"],
        "kpi.complaint_rate": ["product_line"],
    }

    breakdown = {}
    for dim in dimensions_by_kpi.get(kpi_id, []):
        contributions = _decompose_dimension(db, kpi_id, dim, anomaly)
        explained = sum(c.contribution_pct for c in contributions)
        breakdown[dim] = {
            "segments": [
                {
                    "segment": c.segment_value,
                    "baseline_value": round(c.baseline_value, 4),
                    "observed_value": round(c.observed_value, 4),
                    "delta": round(c.delta, 4),
                    "contribution_pct": round(c.contribution_pct * 100, 1),
                }
                for c in contributions
            ],
            "unexplained_pct": round((1 - explained) * 100, 1),
        }

    return breakdown
