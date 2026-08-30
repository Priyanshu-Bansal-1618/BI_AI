"""
Seed script — run this once after the schema exists (`models.Base.metadata
.create_all` runs automatically when backend/main.py starts, so start the
server first, or call create_all here directly) and before your first
POST /kpi-story call.

Why this is necessary: anomaly_detection.py requires
`min_baseline_periods` (14 days for revenue) of history before it will
even attempt a z-score. Without seed data, every /kpi-story call returns
{"status": "no_material_anomaly"} — not because nothing is wrong with the
code, but because there's no history to compare against.

This script creates:
  - 2 regions, 3 product lines, 2 warehouses
  - 30 days of "normal" revenue + stockout history
  - a DELIBERATE anomaly on day 30 for PROD-B: revenue drops ~15%,
    stockout rate spikes, and matching complaint tickets + review text
    are embedded into the vector store — so your first /kpi-story call
    for kpi.revenue on the seeded "as_of" date has something real to find.

Usage:
    python seed_data.py
"""

import random
from datetime import date, timedelta

from backend.db import SessionLocal, engine
from backend import models
from backend.vector_db import upsert_feedback_text, VectorRecord

random.seed(7)

models.Base.metadata.create_all(bind=engine)
db = SessionLocal()

AS_OF = date.today()
START = AS_OF - timedelta(days=29)   # 30 days of history ending on AS_OF

# --- Dimensions ---------------------------------------------------------
regions = [models.Region(code="NA-EAST", name="North America East"),
           models.Region(code="NA-WEST", name="North America West")]
product_lines = [models.ProductLine(code="PROD-A", name="Product A"),
                  models.ProductLine(code="PROD-B", name="Product B"),
                  models.ProductLine(code="PROD-C", name="Product C")]
db.add_all(regions + product_lines)
db.commit()

warehouses = [models.Warehouse(code="WH-EAST-1", region_id=regions[0].id),
              models.Warehouse(code="WH-WEST-1", region_id=regions[1].id)]
db.add_all(warehouses)
db.commit()

pl_by_code = {p.code: p for p in product_lines}
region_by_code = {r.code: r for r in regions}

# --- 30 days of "normal" revenue + stockout history ---------------------
for i in range(30):
    d = START + timedelta(days=i)
    is_anomaly_day = (d == AS_OF)

    for region in regions:
        for pl in product_lines:
            base = {"PROD-A": 4000, "PROD-B": 6000, "PROD-C": 3000}[pl.code]
            noise = random.uniform(-0.05, 0.05)
            revenue = base * (1 + noise)

            if is_anomaly_day and pl.code == "PROD-B":
                revenue = base * 0.85   # ~15% drop, deliberate anomaly

            db.add(models.RevenueDaily(
                metric_date=d, region_id=region.id, product_line_id=pl.id,
                revenue=round(revenue, 2),
            ))

    for wh in warehouses:
        for pl in product_lines:
            total = 30
            oos = random.randint(0, 1)   # normally near-zero stockout days

            if is_anomaly_day and pl.code == "PROD-B":
                oos = 11   # spike

            db.add(models.StockoutDaily(
                metric_date=d, warehouse_id=wh.id, product_line_id=pl.id,
                sku_days_total=total, sku_days_out_of_stock=oos,
            ))

db.commit()

# --- Support ticket metadata (drives kpi.complaint_rate) ----------------
for i in range(30):
    d = START + timedelta(days=i)
    n_tickets = 1 if d != AS_OF else 6   # spike on anomaly day
    for _ in range(n_tickets):
        ref = str(random.random())
        db.add(models.SupportTicketMeta(
            ticket_date=d, product_line_id=pl_by_code["PROD-B"].id,
            sentiment="negative", category="product_quality",
            source_ref=f"seed-ticket-{d}-{ref}",
        ))
db.commit()
print(f"Seeded dimensions + 30 days of revenue/stockout/ticket history, "
      f"anomaly on {AS_OF} for PROD-B.")

# --- Unstructured evidence (drives RAG retrieval) ------------------------
evidence_texts = [
    ("Customer wrote in saying Product B has been out of stock for two "
     "weeks and they had to cancel their order.", "review"),
    ("CRM note: regional sales rep flagged that Product B inventory has "
     "been unavailable at the East warehouse since last week.", "crm_note"),
    ("Support ticket: customer complained about a defective unit and long "
     "wait times for a replacement of Product B.", "support_ticket"),
    ("Review: 'Product B was on backorder and customer service couldn't "
     "give me a delivery date.'", "review"),
]

for text, source_type in evidence_texts:
    kwargs = dict(
        text=text,
        metadata={
            "source_type": source_type,
            "product_line_code": "PROD-B",
            "region_code": "NA-EAST",
            "sentiment": "negative",
            "category": "product_quality",
            "event_date": AS_OF.isoformat(),
        },
    )
    upsert_feedback_text(VectorRecord(**kwargs))

print("Seeded 4 evidence snippets into the vector store for PROD-B.")
print(f"\nNow call: POST /kpi-story with "
      f'{{"kpi_id": "kpi.revenue", "as_of_date": "{AS_OF}", "user_role": "Executive"}}')

db.close()
