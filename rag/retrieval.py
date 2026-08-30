"""
Evidence retrieval — correlates a flagged quantitative anomaly + its RCA
breakdown with qualitative evidence in the Vector DB (reviews, CRM notes,
support tickets). This is retrieval only: no generation happens here.
"""

from dataclasses import dataclass, asdict
from datetime import timedelta

from backend.vector_db import semantic_search
from analytics.anomaly_detection import AnomalyResult

# Maps a KPI's "shape of movement" to the semantic queries worth running.
# In production this could itself be LLM-assisted query expansion, but the
# prototype keeps it template-based and deterministic for the top driver.
KPI_EVIDENCE_QUERY_TEMPLATES = {
    "kpi.revenue": [
        "customers reporting products out of stock or unavailable",
        "customer complaints about product quality or defects",
        "shipping delays or delivery problems",
    ],
    "kpi.stockout_rate": [
        "supplier delay or delayed shipment from vendor",
        "warehouse capacity or fulfillment issues",
    ],
    "kpi.complaint_rate": [
        "product defect or quality issue reported by customer",
        "customer service response time complaints",
    ],
}


@dataclass
class EvidenceItem:
    id: str
    text: str
    source_type: str
    similarity: float
    product_line_code: str | None
    region_code: str | None
    event_date: str | None


def retrieve_evidence(anomaly: AnomalyResult, rca_breakdown: dict, top_k_per_query: int = 5) -> list[EvidenceItem]:
    """Runs semantic search scoped to the anomaly's date window and the
    top RCA-identified driver segment(s), across each relevant query
    template for this KPI. Returns deduplicated, similarity-ranked results."""
    queries = KPI_EVIDENCE_QUERY_TEMPLATES.get(anomaly.kpi_id, [])

    # Scope filters to the top driver segment from RCA, if one exists,
    # so retrieval isn't polluted by unrelated product lines/regions.
    top_product_line = None
    top_region = None
    for dim, data in rca_breakdown.items():
        if not data["segments"]:
            continue
        top_segment = data["segments"][0]["segment"]
        if dim == "product_line":
            top_product_line = top_segment
        elif dim == "region":
            top_region = top_segment

    date_from = (anomaly.metric_date - timedelta(days=14)).isoformat()
    date_to = anomaly.metric_date.isoformat()

    filters = {"date_from": date_from, "date_to": date_to}
    if top_product_line:
        filters["product_line_code"] = top_product_line
    if top_region:
        filters["region_code"] = top_region

    seen_ids = set()
    results: list[EvidenceItem] = []
    for query in queries:
        hits = semantic_search(query, filters=filters, top_k=top_k_per_query)
        for h in hits:
            if h["id"] in seen_ids:
                continue
            seen_ids.add(h["id"])
            results.append(EvidenceItem(
                id=str(h["id"]), text=h["text"], source_type=h["source_type"],
                similarity=float(h["similarity"]),
                product_line_code=h.get("product_line_code"),
                region_code=h.get("region_code"),
                event_date=str(h.get("event_date")) if h.get("event_date") else None,
            ))

    results.sort(key=lambda e: e.similarity, reverse=True)
    return results[:15]
