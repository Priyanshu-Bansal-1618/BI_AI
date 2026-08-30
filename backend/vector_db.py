"""
Vector DB layer — implemented with pgvector for prototype simplicity
(one less moving part than standing up Pinecone/Chroma separately), but
isolated behind this module so swapping to Pinecone/Chroma later only
touches this file.

Schema (pgvector table, created via raw DDL below — kept separate from
the SQLAlchemy ORM models since it's accessed almost exclusively via
similarity search, not row CRUD):

    CREATE EXTENSION IF NOT EXISTS vector;

    CREATE TABLE customer_feedback (
        id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        source_ref   TEXT UNIQUE NOT NULL,     -- joins to support_tickets_meta.source_ref
        text         TEXT NOT NULL,
        embedding    VECTOR(1536) NOT NULL,     -- text-embedding-3-small dimension
        source_type  TEXT NOT NULL,             -- review | crm_note | support_ticket | expert_correction
        product_line_code TEXT,
        region_code  TEXT,
        sentiment    TEXT,
        category     TEXT,
        event_date   DATE,
        created_at   TIMESTAMPTZ DEFAULT now()
    );

    CREATE INDEX ON customer_feedback USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
"""

import os
import uuid
from dataclasses import dataclass, field
from typing import Optional

import psycopg2
from psycopg2.extras import Json, RealDictCursor

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

VECTOR_DB_DSN = os.environ.get(
    "VECTOR_DB_DSN", "postgresql://postgres:postgres@localhost:5432/kpi_engine"
)


@dataclass
class VectorRecord:
    text: str
    metadata: dict
    id: Optional[str] = field(default_factory=lambda: str(uuid.uuid4()))


def _embed(text: str) -> list[float]:
    """Calls the embedding API. Stubbed here to keep the prototype
    runnable offline — swap for a real OpenAI/Anthropic/Voyage call."""
    import hashlib
    import struct

    h = hashlib.sha256(text.encode()).digest()
    # deterministic pseudo-embedding for local dev/testing only
    vals = [b / 255.0 for b in (h * (EMBEDDING_DIM // len(h) + 1))[:EMBEDDING_DIM]]
    return vals


def _connect():
    return psycopg2.connect(VECTOR_DB_DSN)


def upsert_feedback_text(record: VectorRecord) -> str:
    """Embeds and stores one piece of unstructured text. Returns source_ref."""
    embedding = _embed(record.text)
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO customer_feedback
                    (id, source_ref, text, embedding, source_type,
                     product_line_code, region_code, sentiment, category, event_date)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (source_ref) DO UPDATE SET text = EXCLUDED.text
                """,
                (
                    record.id, record.id, record.text, embedding,
                    record.metadata.get("source_type"),
                    record.metadata.get("product_line_code"),
                    record.metadata.get("region_code"),
                    record.metadata.get("sentiment"),
                    record.metadata.get("category"),
                    record.metadata.get("event_date"),
                ),
            )
        conn.commit()
    return record.id


def semantic_search(query_text: str, filters: dict, top_k: int = 8) -> list[dict]:
    """Cosine-similarity search filtered by metadata (product line, region,
    date window) — used by rag/retrieval.py to correlate a quantitative
    anomaly with qualitative evidence."""
    query_embedding = _embed(query_text)

    # Collect ONLY the filter params here — the two embedding params (one
    # for the similarity SELECT, one for ORDER BY) are added explicitly
    # below in the exact order they appear in the SQL, to avoid subtle
    # param-ordering bugs if filters are added/removed later.
    where_clauses = []
    filter_params = []
    for col in ("product_line_code", "region_code", "source_type"):
        if filters.get(col):
            where_clauses.append(f"{col} = %s")
            filter_params.append(filters[col])
    if filters.get("date_from"):
        where_clauses.append("event_date >= %s")
        filter_params.append(filters["date_from"])
    if filters.get("date_to"):
        where_clauses.append("event_date <= %s")
        filter_params.append(filters["date_to"])

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    sql = f"""
        SELECT id, source_ref, text, source_type, product_line_code,
               region_code, sentiment, category, event_date,
               1 - (embedding <=> %s) AS similarity
        FROM customer_feedback
        {where_sql}
        ORDER BY embedding <=> %s
        LIMIT %s
    """
    params = [query_embedding] + filter_params + [query_embedding, top_k]

    with _connect() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            return cur.fetchall()
