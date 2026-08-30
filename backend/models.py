"""
SQLAlchemy models — PostgreSQL side of the Semantic Contract.

Three connected KPIs modeled here:
  1. Revenue            (structured: orders / order_lines)
  2. Stockout Rate       (structured: inventory_snapshots)
  3. Complaint Rate      (unstructured_derived: support_tickets_meta,
                          with raw text living in the Vector DB — see
                          backend/vector_db.py)

Design note: structured metric TABLES stay in Postgres. Anything free-text
(review body, CRM note, ticket transcript) is NEVER stored/queried here —
it lives only in the vector store, referenced by `source_ref` ids so the
two systems can be joined at query time.
"""

from datetime import datetime, date
from enum import Enum as PyEnum

from sqlalchemy import (
    Column, Integer, String, Float, Date, DateTime, ForeignKey,
    Enum, JSON, Boolean, UniqueConstraint, Index
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Role(PyEnum):
    REGIONAL_SALES_MANAGER = "Regional Sales Manager"
    SUPPLY_CHAIN_VP = "Supply Chain VP"
    EXECUTIVE = "Executive"


class Region(Base):
    __tablename__ = "regions"
    id = Column(Integer, primary_key=True)
    code = Column(String(10), unique=True, nullable=False)   # e.g. "NA-EAST"
    name = Column(String(100), nullable=False)


class ProductLine(Base):
    __tablename__ = "product_lines"
    id = Column(Integer, primary_key=True)
    code = Column(String(20), unique=True, nullable=False)   # e.g. "PROD-B"
    name = Column(String(100), nullable=False)


# ---------------------------------------------------------------------
# KPI 1: Revenue (structured)
# ---------------------------------------------------------------------
class OrderLine(Base):
    __tablename__ = "order_lines"
    id = Column(Integer, primary_key=True)
    order_date = Column(Date, nullable=False, index=True)
    region_id = Column(Integer, ForeignKey("regions.id"), nullable=False)
    product_line_id = Column(Integer, ForeignKey("product_lines.id"), nullable=False)
    net_amount = Column(Float, nullable=False)
    status = Column(String(20), nullable=False, default="completed")

    region = relationship("Region")
    product_line = relationship("ProductLine")

    __table_args__ = (
        Index("ix_orderline_date_region_product", "order_date", "region_id", "product_line_id"),
    )


class RevenueDaily(Base):
    """Materialized rollup — what the anomaly detector actually reads.
    Populated by a nightly dbt job (fct_revenue_daily) aggregating OrderLine.
    """
    __tablename__ = "fct_revenue_daily"
    id = Column(Integer, primary_key=True)
    metric_date = Column(Date, nullable=False, index=True)
    region_id = Column(Integer, ForeignKey("regions.id"), nullable=False)
    product_line_id = Column(Integer, ForeignKey("product_lines.id"), nullable=False)
    revenue = Column(Float, nullable=False)

    __table_args__ = (
        UniqueConstraint("metric_date", "region_id", "product_line_id", name="uq_revenue_grain"),
    )


# ---------------------------------------------------------------------
# KPI 2: Stockout Rate (structured)
# ---------------------------------------------------------------------
class Warehouse(Base):
    __tablename__ = "warehouses"
    id = Column(Integer, primary_key=True)
    code = Column(String(20), unique=True, nullable=False)
    region_id = Column(Integer, ForeignKey("regions.id"), nullable=False)


class StockoutDaily(Base):
    """Materialized rollup for stockout rate, grain = day/warehouse/product_line."""
    __tablename__ = "fct_stockout_daily"
    id = Column(Integer, primary_key=True)
    metric_date = Column(Date, nullable=False, index=True)
    warehouse_id = Column(Integer, ForeignKey("warehouses.id"), nullable=False)
    product_line_id = Column(Integer, ForeignKey("product_lines.id"), nullable=False)
    sku_days_total = Column(Integer, nullable=False)
    sku_days_out_of_stock = Column(Integer, nullable=False)

    @property
    def stockout_rate(self) -> float:
        return self.sku_days_out_of_stock / self.sku_days_total if self.sku_days_total else 0.0

    __table_args__ = (
        UniqueConstraint("metric_date", "warehouse_id", "product_line_id", name="uq_stockout_grain"),
    )


# ---------------------------------------------------------------------
# KPI 3: Complaint Rate (unstructured_derived)
# Structured METADATA lives here; the raw ticket/review TEXT lives in
# the vector DB and is joined via `source_ref` (see vector_db.py).
# ---------------------------------------------------------------------
class SupportTicketMeta(Base):
    __tablename__ = "support_tickets_meta"
    id = Column(Integer, primary_key=True)
    ticket_date = Column(Date, nullable=False, index=True)
    product_line_id = Column(Integer, ForeignKey("product_lines.id"), nullable=False)
    region_id = Column(Integer, ForeignKey("regions.id"), nullable=True)
    sentiment = Column(String(20), nullable=False)     # 'negative' | 'neutral' | 'positive'
    category = Column(String(50), nullable=False)      # 'product_quality' | 'shipping' | 'billing' ...
    source_ref = Column(String(64), nullable=False, unique=True)  # FK into vector DB metadata.source_ref
    order_count_denominator = Column(Integer, nullable=False, default=0)


# ---------------------------------------------------------------------
# Anomaly + RCA output persistence (so the frontend / audit trail can
# replay a "KPI Story" without recomputing it)
# ---------------------------------------------------------------------
class AnomalyEvent(Base):
    __tablename__ = "anomaly_events"
    id = Column(Integer, primary_key=True)
    kpi_id = Column(String(64), nullable=False)          # matches semantic_contract.yaml kpis[].id
    detected_at = Column(DateTime, default=datetime.utcnow)
    metric_date = Column(Date, nullable=False)
    observed_value = Column(Float, nullable=False)
    expected_value = Column(Float, nullable=False)
    z_score = Column(Float, nullable=False)
    pct_change = Column(Float, nullable=False)
    is_material = Column(Boolean, nullable=False)
    rca_breakdown = Column(JSON, nullable=True)   # output of analytics/rca.py, e.g. per-segment contribution %
    status = Column(String(20), default="new")    # new | narrated | dismissed | acted_on


class KPIStory(Base):
    """Persisted LLM output — always tied 1:1 to the AnomalyEvent that
    triggered it, and always carries the evidence ids it cited."""
    __tablename__ = "kpi_stories"
    id = Column(Integer, primary_key=True)
    anomaly_event_id = Column(Integer, ForeignKey("anomaly_events.id"), nullable=False)
    generated_at = Column(DateTime, default=datetime.utcnow)
    descriptive = Column(String, nullable=False)
    diagnostic = Column(String, nullable=False)
    prescriptive = Column(String, nullable=False)
    confidence_score = Column(Float, nullable=False)
    evidence_ids = Column(JSON, nullable=False)     # list of vector DB ids cited
    abstained = Column(Boolean, default=False)
    llm_model = Column(String(64), nullable=False)
    prompt_tokens = Column(Integer, nullable=True)
    completion_tokens = Column(Integer, nullable=True)


class Feedback(Base):
    """Human-in-the-loop signal — see feedback/feedback_schema.py for the
    FastAPI request/response contract this backs."""
    __tablename__ = "feedback"
    id = Column(Integer, primary_key=True)
    kpi_story_id = Column(Integer, ForeignKey("kpi_stories.id"), nullable=False)
    user_role = Column(Enum(Role), nullable=False)
    rating = Column(String(10), nullable=False)   # 'up' | 'down'
    correction_text = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
