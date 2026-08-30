"""
Human-in-the-loop feedback (Deliverable 5.3).

Two things happen when feedback arrives via POST /feedback (see backend/main.py):

1. Structured logging: every rating is persisted to `feedback` (models.py)
   for offline analysis — e.g. "which KPI/role combos get thumbs-down most,"
   which becomes a prompt-tuning or threshold-tuning backlog.

2. Live pipeline update: a thumbs-down WITH a correction is immediately
   embedded into the vector store as an `expert_correction` record scoped
   to that kpi_story_id's anomaly. Future retrieval for similar anomalies
   (same KPI + overlapping driver segment) will surface that correction
   alongside raw customer evidence, so the LLM sees "an analyst previously
   flagged this exact reasoning as wrong" as retrieved context.

This module defines the schema/contract; the FastAPI route itself lives in
backend/main.py (`submit_feedback`) and the vector write in
backend/vector_db.py (`upsert_feedback_text`).
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field


class FeedbackRequest(BaseModel):
    kpi_story_id: int
    user_role: str
    rating: Literal["up", "down"]
    correction_text: Optional[str] = Field(
        default=None,
        description="Required in practice when rating='down' and the user wants "
                    "the correction to influence future retrieval; optional otherwise."
    )


class FeedbackResponse(BaseModel):
    status: Literal["recorded"]
    feedback_id: int


# --- Business-rule update path -----------------------------------------
# Beyond the per-story vector correction above, aggregated feedback should
# periodically (e.g. nightly batch job) update BUSINESS RULES, not just
# retrieval context:
#
#   - If a given KPI's stories are down-voted > X% over a rolling window,
#     auto-raise its `min_confidence_to_auto_send` threshold in the
#     semantic contract until narrative quality improves.
#   - If corrections repeatedly point at the same RCA dimension being
#     mis-weighted (e.g. "it's actually pricing, not stockouts"), flag
#     that KPI's RCA dimension list (analytics/rca.py) for a data-eng
#     review — the fix belongs in the deterministic engine, not a prompt
#     patch, since the contract mandates the LLM never overrides RCA math.
#
# Sketch of the nightly job:

def compute_confidence_threshold_adjustments(db, lookback_days: int = 30, downvote_rate_trigger: float = 0.3):
    """Returns {kpi_id: suggested_new_min_confidence} for any KPI whose
    down-vote rate over the window exceeds the trigger. This is advisory —
    a human reviews and applies it to semantic_contract.yaml rather than
    the job silently rewriting the contract."""
    from datetime import datetime, timedelta
    from backend import models

    cutoff = datetime.utcnow() - timedelta(days=lookback_days)
    rows = (
        db.query(models.Feedback, models.KPIStory, models.AnomalyEvent)
        .join(models.KPIStory, models.Feedback.kpi_story_id == models.KPIStory.id)
        .join(models.AnomalyEvent, models.KPIStory.anomaly_event_id == models.AnomalyEvent.id)
        .filter(models.Feedback.created_at >= cutoff)
        .all()
    )

    tally: dict[str, dict[str, int]] = {}
    for fb, story, event in rows:
        tally.setdefault(event.kpi_id, {"up": 0, "down": 0})
        tally[event.kpi_id][fb.rating] += 1

    adjustments = {}
    for kpi_id, counts in tally.items():
        total = counts["up"] + counts["down"]
        if total == 0:
            continue
        downvote_rate = counts["down"] / total
        if downvote_rate > downvote_rate_trigger:
            adjustments[kpi_id] = {
                "downvote_rate": round(downvote_rate, 2),
                "suggested_min_confidence": min(0.9, 0.65 + downvote_rate * 0.3),
            }
    return adjustments
