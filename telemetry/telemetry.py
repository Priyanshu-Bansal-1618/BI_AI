"""
Telemetry (Deliverable 5.2): end-to-end latency, LLM token usage, and
estimated cost per insight, captured as a decorator around the pipeline
endpoint so every /kpi-story call emits one structured log line /
metrics-store row regardless of which code path (abstained, material,
sparse-history) it took.
"""

import functools
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime

# Per-million-token pricing, illustrative — keep in a config/pricing table
# in production so it can be updated without a code change.
PRICING_USD_PER_MILLION_TOKENS = {
    "claude-sonnet-5": {"input": 3.00, "output": 15.00},
}


@dataclass
class PipelineTelemetry:
    request_id: str
    kpi_id: str
    started_at: str
    total_latency_ms: float
    stage_latency_ms: dict
    llm_model: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    estimated_cost_usd: float | None
    outcome: str   # "no_material_anomaly" | "narrated" | "abstained" | "error"


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    rates = PRICING_USD_PER_MILLION_TOKENS.get(model)
    if not rates or prompt_tokens is None:
        return 0.0
    return (prompt_tokens / 1e6) * rates["input"] + (completion_tokens / 1e6) * rates["output"]


def track_insight_pipeline(fn):
    """Wraps the /kpi-story FastAPI handler. In production this would push
    PipelineTelemetry rows to a metrics store (Prometheus pushgateway,
    a Postgres telemetry table, or Datadog) — stubbed here as structured
    stdout logging so the prototype is dependency-free."""

    @functools.wraps(fn)
    def wrapper(req, *args, **kwargs):
        import uuid
        request_id = str(uuid.uuid4())
        t0 = time.perf_counter()

        result = fn(req, *args, **kwargs)

        total_latency_ms = (time.perf_counter() - t0) * 1000
        outcome = result.get("status", "narrated" if not result.get("abstained") else "abstained")

        usage = result.get("usage", {})
        telemetry = PipelineTelemetry(
            request_id=request_id,
            kpi_id=getattr(req, "kpi_id", "unknown"),
            started_at=datetime.utcnow().isoformat(),
            total_latency_ms=round(total_latency_ms, 1),
            stage_latency_ms={},   # populate via time.perf_counter() checkpoints inside each stage in production
            llm_model=result.get("llm_model"),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            estimated_cost_usd=_estimate_cost(
                result.get("llm_model"), usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
            ) if result.get("llm_model") else None,
            outcome=outcome,
        )

        print(json.dumps({"telemetry": asdict(telemetry)}))  # replace with real sink
        return result

    return wrapper
