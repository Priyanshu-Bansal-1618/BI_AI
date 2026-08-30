# KPI Intelligence-to-Action Engine — Prototype

Bridges the "Last-Mile Insight Gap": raw KPI anomalies get automatically
turned into causal, evidence-cited, role-specific narratives and actions —
without ever letting an LLM touch the math.

## Architecture rule, enforced structurally (not just by prompt)

```
┌─────────────────────────┐        ┌──────────────────────────┐
│   QUANTITATIVE ENGINE    │        │   REASONING ENGINE        │
│  (deterministic, no LLM) │  JSON  │  (RAG + LLM, no math)     │
│                          │ ─────► │                           │
│  analytics/               │        │  rag/                     │
│   anomaly_detection.py   │        │   retrieval.py            │
│   rca.py                 │        │   prompts.py              │
│   sparse_history.py      │        │   llm_reasoning.py        │
└─────────────────────────┘        └──────────────────────────┘
```

The LLM (`rag/llm_reasoning.py`) is handed pre-computed JSON and is
instructed — and then independently re-validated — to never invent or
recompute a number, and to abstain when evidence is weak or missing.

## Pipeline (matches the mandated 5 steps)

1. **Ingestion** — `backend/main.py` (`/ingest/structured`, `/ingest/unstructured`)
   routes metrics to Postgres (`backend/models.py`) and free text to the
   vector store (`backend/vector_db.py`, pgvector-backed).
2. **Anomaly Detection** — `analytics/anomaly_detection.py`: z-score against
   seasonal baseline, IsolationForest as a secondary/sparse-history signal.
3. **RCA + Evidence Retrieval** — `analytics/rca.py` decomposes the anomaly
   by segment; `rag/retrieval.py` semantically searches the vector store
   scoped to the top driver segment and date window.
4. **LLM Reasoning** — `rag/llm_reasoning.py` + `rag/prompts.py` produce the
   structured KPI Story (descriptive / diagnostic / prescriptive / confidence).
5. **Delivery** — `frontend/KPIStoryCard.jsx` renders the story, with a
   thumbs up/down feedback loop wired to `/feedback`.

## File map

| File | Deliverable | Purpose |
|---|---|---|
| `backend/semantic_contract.yaml` | 1 | KPI definitions, thresholds, lineage, RBAC |
| `backend/models.py` | 1 | SQLAlchemy models (Postgres) |
| `backend/vector_db.py` | 1 | pgvector schema + embed/search client |
| `backend/db.py` | 1 | DB session wiring |
| `backend/main.py` | 1, 5 | FastAPI ingestion + pipeline + feedback routes |
| `analytics/anomaly_detection.py` | 2 | Scikit-learn anomaly detection |
| `analytics/rca.py` | 2 | Deterministic root-cause contribution |
| `analytics/sparse_history.py` | 5.1 | New-product / cold-start handling |
| `rag/retrieval.py` | 3 | Semantic search evidence retrieval |
| `rag/prompts.py` | 3 | Exact system/user prompt templates |
| `rag/llm_reasoning.py` | 3 | LLM call + independent abstention validation |
| `frontend/KPIStoryCard.jsx` | 4 | Executive-ready KPI Story UI |
| `telemetry/telemetry.py` | 5.2 | Latency, tokens, cost-per-insight |
| `feedback/feedback_schema.py` | 5.3 | Feedback contract + rule-update job |

## Three connected KPIs modeled

1. **Revenue** (structured) — orders/order_lines → `fct_revenue_daily`
2. **Stockout Rate** (structured) — inventory snapshots → `fct_stockout_daily`
3. **Complaint Rate** (unstructured-derived) — ticket metadata in Postgres,
   raw ticket/review text in the vector store, joined via `source_ref`

These are the intentionally-linked example: a revenue dip's RCA points at
Product B → retrieval scoped to Product B surfaces stockout/complaint
evidence → the story tells a coherent supply-chain-to-revenue causal chain.

## Running it

See the step-by-step setup and run guide in the conversation this project
was generated from, or follow this summary:

1. `pip install -r requirements.txt`
2. Stand up Postgres 14+ with `CREATE EXTENSION vector;` and run the
   `customer_feedback` table DDL documented in `backend/vector_db.py`'s
   docstring.
3. Set `DATABASE_URL` and `VECTOR_DB_DSN` (same DSN is fine if using one
   Postgres instance) and `ANTHROPIC_API_KEY`.
4. Replace the stub `_embed()` in `backend/vector_db.py` with a real
   embedding call before relying on retrieval quality.
5. `uvicorn backend.main:app --reload` from the `kpi-engine/` directory
   (creates tables automatically on startup).
6. `python seed_data.py` — required before your first `/kpi-story` call,
   since the anomaly detector needs baseline history to compare against.
7. `POST /kpi-story` with the KPI id/date printed by the seed script.

## Design decisions worth flagging

- **RCA residual is reported, not hidden.** `unexplained_pct` in `rca.py`'s
  output is a first-class field the LLM sees and factors into confidence —
  a system that always claims 100% explanation is lying by omission.
- **Two abstention layers.** The prompt asks the model to self-abstain;
  `llm_reasoning.py`'s `_independent_validation` re-checks citations against
  actually-retrieved evidence IDs and overrides the model if it hallucinated
  a citation or the evidence set was empty. The model's self-report is a
  signal, not a source of truth.
- **Materiality is a two-condition AND**, not just statistical significance
  (`z_score` AND `pct_change_material`), so trivial-but-significant wiggles
  don't spam an executive dashboard.
- **Vector DB uses pgvector** rather than a separate Pinecone/Chroma service
  to keep the prototype to one datastore; `vector_db.py` isolates all access
  so swapping providers later is a one-file change.
