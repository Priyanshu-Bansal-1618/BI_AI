"""
FastAPI ingestion layer.

Routes:
  - structured metrics (orders, inventory, ticket metadata) -> PostgreSQL
  - unstructured text (reviews, CRM notes, ticket bodies)    -> Vector DB

This file also exposes the read endpoints the React dashboard calls, and
the feedback endpoint used by the human-in-the-loop loop (Deliverable 5).
"""

from datetime import date
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.db import get_db, engine
from backend import models
from backend.vector_db import upsert_feedback_text, VectorRecord
from analytics.anomaly_detection import detect_anomalies_for_kpi
from analytics.rca import compute_rca
from rag.retrieval import retrieve_evidence
from rag.llm_reasoning import generate_kpi_story
from telemetry.telemetry import track_insight_pipeline

# Display metadata for formatting API output into what KPIStoryCard.jsx
# expects (kpiName, metricChangeLabel, direction, drivers). Mirrors
# backend/semantic_contract.yaml's `name`/`unit` fields — in production,
# load this from the parsed YAML instead of duplicating it here.
KPI_DISPLAY = {
    "kpi.revenue": {"name": "Revenue", "unit": "$", "is_percent": False},
    "kpi.stockout_rate": {"name": "Stockout Rate", "unit": "%", "is_percent": True},
    "kpi.complaint_rate": {"name": "Complaint Rate", "unit": "%", "is_percent": True},
}


def _format_change_label(name: str, pct_change: float) -> tuple[str, str]:
    direction = "down" if pct_change < 0 else "up"
    arrow = "↓" if direction == "down" else "↑"
    return f"{name} {arrow} {abs(pct_change) * 100:.0f}%", direction


def _format_drivers(rca_breakdown: dict) -> list[dict]:
    drivers = []
    for dim, data in rca_breakdown.items():
        for seg in data["segments"][:3]:   # top 3 per dimension, avoid an overlong card
            pct = seg["contribution_pct"]
            direction = "down" if seg["delta"] < 0 else "up"
            arrow = "↓" if direction == "down" else "↑"
            drivers.append({
                "label": f"{seg['segment']} ({dim.replace('_', ' ')})",
                "changeLabel": f"{arrow} {abs(pct):.0f}%",
                "direction": direction,
            })
    return drivers


app = FastAPI(title="KPI Intelligence-to-Action Engine")

# Dev-friendly CORS so a locally-run React app (e.g. Vite on :5173,
# create-react-app on :3000) can call this API directly. Restrict
# allow_origins to your real frontend origin(s) before deploying.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

models.Base.metadata.create_all(bind=engine)


# ---------------------------------------------------------------------
# 1. INGESTION — structured
# ---------------------------------------------------------------------
class StructuredMetricIn(BaseModel):
    kpi_id: str
    metric_date: date
    dimensions: dict            # e.g. {"region_id": 3, "product_line_id": 7}
    value_fields: dict          # e.g. {"revenue": 10432.5} or {"sku_days_total": 30, "sku_days_out_of_stock": 4}


@app.post("/ingest/structured")
def ingest_structured(payload: StructuredMetricIn, db: Session = Depends(get_db)):
    """Routes a structured metric row into the correct Postgres fact table
    based on kpi_id, per the semantic contract's `source_tables`."""
    if payload.kpi_id == "kpi.revenue":
        row = models.RevenueDaily(
            metric_date=payload.metric_date,
            region_id=payload.dimensions["region_id"],
            product_line_id=payload.dimensions["product_line_id"],
            revenue=payload.value_fields["revenue"],
        )
    elif payload.kpi_id == "kpi.stockout_rate":
        row = models.StockoutDaily(
            metric_date=payload.metric_date,
            warehouse_id=payload.dimensions["warehouse_id"],
            product_line_id=payload.dimensions["product_line_id"],
            sku_days_total=payload.value_fields["sku_days_total"],
            sku_days_out_of_stock=payload.value_fields["sku_days_out_of_stock"],
        )
    else:
        raise HTTPException(400, f"Unknown or non-structured kpi_id: {payload.kpi_id}")

    db.add(row)
    db.commit()
    return {"status": "ingested", "kpi_id": payload.kpi_id}


# ---------------------------------------------------------------------
# 2. INGESTION — unstructured (reviews, CRM notes, ticket bodies)
# ---------------------------------------------------------------------
class UnstructuredTextIn(BaseModel):
    source_type: Literal["review", "crm_note", "support_ticket"]
    text: str
    product_line_code: str
    region_code: Optional[str] = None
    sentiment: Optional[Literal["negative", "neutral", "positive"]] = None
    category: Optional[str] = None
    event_date: date


@app.post("/ingest/unstructured")
def ingest_unstructured(payload: UnstructuredTextIn, db: Session = Depends(get_db)):
    """Embeds free text into the Vector DB and, if it's ticket data that
    feeds a KPI (complaint_rate), writes the structured *metadata* row
    into Postgres so the anomaly detector can aggregate it numerically —
    the LLM never sees or counts raw tickets to produce that number."""
    record = VectorRecord(
        text=payload.text,
        metadata={
            "source_type": payload.source_type,
            "product_line_code": payload.product_line_code,
            "region_code": payload.region_code,
            "sentiment": payload.sentiment,
            "category": payload.category,
            "event_date": payload.event_date.isoformat(),
        },
    )
    source_ref = upsert_feedback_text(record)

    if payload.source_type == "support_ticket" and payload.sentiment and payload.category:
        product_line = db.query(models.ProductLine).filter_by(code=payload.product_line_code).first()
        if not product_line:
            raise HTTPException(400, "Unknown product_line_code")
        meta_row = models.SupportTicketMeta(
            ticket_date=payload.event_date,
            product_line_id=product_line.id,
            sentiment=payload.sentiment,
            category=payload.category,
            source_ref=source_ref,
        )
        db.add(meta_row)
        db.commit()

    return {"status": "ingested", "vector_ref": source_ref}


# ---------------------------------------------------------------------
# 3. THE FULL PIPELINE — anomaly -> RCA -> RAG -> LLM story
# ---------------------------------------------------------------------
class KPIStoryRequest(BaseModel):
    kpi_id: str
    as_of_date: date
    user_role: str


@app.post("/kpi-story")
@track_insight_pipeline
def generate_story(req: KPIStoryRequest, db: Session = Depends(get_db)):
    # Step 1+2: deterministic quantitative engine (NEVER the LLM)
    anomaly = detect_anomalies_for_kpi(db, req.kpi_id, req.as_of_date)
    if anomaly is None or not anomaly.is_material:
        return {"status": "no_material_anomaly", "kpi_id": req.kpi_id}

    rca_breakdown = compute_rca(db, req.kpi_id, anomaly)

    # Step 3: RAG evidence retrieval, scoped by the anomaly's dimensions
    evidence = retrieve_evidence(anomaly=anomaly, rca_breakdown=rca_breakdown)

    # Step 4: LLM narrates ONLY what it was handed — see rag/llm_reasoning.py
    story = generate_kpi_story(
        anomaly=anomaly,
        rca_breakdown=rca_breakdown,
        evidence=evidence,
        user_role=req.user_role,
    )

    # Persist for audit trail / frontend replay
    event_row = models.AnomalyEvent(
        kpi_id=req.kpi_id, metric_date=req.as_of_date,
        observed_value=anomaly.observed_value, expected_value=anomaly.expected_value,
        z_score=anomaly.z_score, pct_change=anomaly.pct_change,
        is_material=anomaly.is_material, rca_breakdown=rca_breakdown,
        status="narrated" if not story["abstained"] else "new",
    )
    db.add(event_row)
    db.commit()
    db.refresh(event_row)

    story_row = models.KPIStory(
        anomaly_event_id=event_row.id,
        descriptive=story["descriptive"], diagnostic=story["diagnostic"],
        prescriptive=story["prescriptive"], confidence_score=story["confidence_score"],
        evidence_ids=story["evidence_ids"], abstained=story["abstained"],
        llm_model=story["llm_model"], prompt_tokens=story["usage"]["prompt_tokens"],
        completion_tokens=story["usage"]["completion_tokens"],
    )
    db.add(story_row)
    db.commit()

    # Shape the response into what frontend/KPIStoryCard.jsx expects,
    # on top of the raw story fields (descriptive/diagnostic/etc.)
    display = KPI_DISPLAY.get(req.kpi_id, {"name": req.kpi_id, "unit": "", "is_percent": False})
    metric_change_label, direction = _format_change_label(display["name"], anomaly.pct_change)

    return {
        **story,
        "anomaly_event_id": event_row.id,
        "kpi_story_id": story_row.id,
        "kpiName": display["name"],
        "metricChangeLabel": metric_change_label,
        "direction": direction,
        "drivers": _format_drivers(rca_breakdown),
    }


# ---------------------------------------------------------------------
# 4. FEEDBACK — human-in-the-loop (Deliverable 5)
# ---------------------------------------------------------------------
class FeedbackIn(BaseModel):
    kpi_story_id: int
    user_role: str
    rating: Literal["up", "down"]
    correction_text: Optional[str] = None


@app.post("/feedback")
def submit_feedback(payload: FeedbackIn, db: Session = Depends(get_db)):
    fb = models.Feedback(
        kpi_story_id=payload.kpi_story_id,
        user_role=payload.user_role,
        rating=payload.rating,
        correction_text=payload.correction_text,
    )
    db.add(fb)
    db.commit()

    # Down-votes with a correction get embedded back into the vector store
    # as a labeled "expert correction" record so future retrieval for
    # similar anomalies surfaces the correction. See feedback/feedback_schema.py.
    if payload.rating == "down" and payload.correction_text:
        upsert_feedback_text(VectorRecord(
            text=payload.correction_text,
            metadata={"source_type": "expert_correction", "kpi_story_id": payload.kpi_story_id},
        ))

    return {"status": "recorded", "feedback_id": fb.id}
