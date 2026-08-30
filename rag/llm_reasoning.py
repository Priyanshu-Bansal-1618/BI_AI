"""
Reasoning/Generative Engine — the ONLY module in this codebase allowed to
call the LLM. It hands the model exactly three pre-computed JSON blocks
(anomaly, RCA, evidence) and validates the response before it's ever
persisted or shown to a user.

Abstention logic lives at two levels:
  - Soft: the prompt instructs the model to self-abstain on weak evidence.
  - Hard: this module independently re-checks the model's output against
    the same evidence set and overrides "abstained": false -> true if the
    model's own confidence/citations don't hold up. The LLM's self-report
    is never trusted blindly.
"""

import json
import os
import re
from dataclasses import asdict

from analytics.anomaly_detection import AnomalyResult
from rag.retrieval import EvidenceItem
from rag.prompts import SYSTEM_PROMPT, build_user_prompt

MIN_CONFIDENCE_TO_AUTO_SEND = 0.65
ABSTAIN_BELOW_CONFIDENCE = 0.35

ROLE_ACCESS = {
    "Regional Sales Manager": {"fields": ["revenue", "region_scope"]},
    "Supply Chain VP": {"fields": ["revenue_total", "product_line", "stockout_rate"]},
    "Executive": {"fields": "all"},
}


def _call_llm(system_prompt: str, user_prompt: str) -> dict:
    """Real call to the Anthropic Messages API. Kept as a thin wrapper so
    it's easy to mock in tests."""
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=800,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    text = re.sub(r"^```json\s*|\s*```$", "", text.strip())
    parsed = json.loads(text)
    parsed["_usage"] = {
        "prompt_tokens": response.usage.input_tokens,
        "completion_tokens": response.usage.output_tokens,
    }
    return parsed


def _independent_validation(parsed: dict, evidence: list[EvidenceItem], rca_breakdown: dict) -> dict:
    """Hard abstention check — does not trust the model's self-reported
    abstained/confidence fields without re-verifying against ground truth."""
    cited_ids = set(parsed.get("evidence_ids", []))
    valid_ids = {e.id for e in evidence}

    # 1. Every cited id must actually exist in the evidence we retrieved.
    hallucinated_citations = cited_ids - valid_ids
    if hallucinated_citations:
        parsed["abstained"] = True
        parsed["confidence_score"] = min(parsed.get("confidence_score", 0), ABSTAIN_BELOW_CONFIDENCE)
        parsed["diagnostic"] += " [SYSTEM: response cited unretrieved evidence ids and was overridden to abstain.]"

    # 2. If evidence set is empty, force abstention regardless of what the model said.
    if not evidence:
        parsed["abstained"] = True
        parsed["confidence_score"] = 0.0

    # 3. Penalize confidence for high unexplained RCA residual.
    max_unexplained = max((d.get("unexplained_pct", 0) for d in rca_breakdown.values()), default=0)
    if max_unexplained > 40:
        parsed["confidence_score"] = min(parsed.get("confidence_score", 1.0), 0.5)

    # 4. Clamp.
    parsed["confidence_score"] = max(0.0, min(1.0, float(parsed.get("confidence_score", 0.0))))
    if parsed["confidence_score"] <= ABSTAIN_BELOW_CONFIDENCE:
        parsed["abstained"] = True

    return parsed


def generate_kpi_story(anomaly: AnomalyResult, rca_breakdown: dict,
                        evidence: list[EvidenceItem], user_role: str) -> dict:
    anomaly_json = asdict(anomaly)
    anomaly_json["metric_date"] = str(anomaly_json["metric_date"])
    evidence_json = [asdict(e) for e in evidence]

    user_prompt = build_user_prompt(
        anomaly_json=anomaly_json, rca_json=rca_breakdown,
        evidence_json=evidence_json, user_role=user_role,
        role_access=ROLE_ACCESS.get(user_role, {}),
    )

    try:
        parsed = _call_llm(SYSTEM_PROMPT, user_prompt)
    except (json.JSONDecodeError, Exception) as e:
        # Any parsing/API failure -> hard abstain, never surface a broken story.
        return {
            "descriptive": f"{anomaly.kpi_id} moved from {anomaly.expected_value:.2f} to "
                            f"{anomaly.observed_value:.2f} on {anomaly.metric_date}.",
            "diagnostic": f"Unable to generate a diagnostic narrative due to a system error ({type(e).__name__}). "
                           f"RCA data is available in the dashboard; manual review recommended.",
            "prescriptive": "Escalate to analyst for manual review before acting.",
            "confidence_score": 0.0,
            "evidence_ids": [],
            "abstained": True,
            "llm_model": "claude-sonnet-5",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }

    usage = parsed.pop("_usage", {"prompt_tokens": None, "completion_tokens": None})
    parsed = _independent_validation(parsed, evidence, rca_breakdown)
    parsed["llm_model"] = "claude-sonnet-5"
    parsed["usage"] = usage
    return parsed
