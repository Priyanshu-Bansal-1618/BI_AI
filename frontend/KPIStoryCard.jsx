import React, { useState } from "react";

/**
 * KPIStoryCard
 *
 * Renders one "KPI Story" as returned by POST /kpi-story:
 *   { descriptive, diagnostic, prescriptive, confidence_score,
 *     evidence_ids, abstained, kpi_story_id, anomaly_event_id, ... }
 * plus the raw anomaly/RCA payload for the headline metric + driver bars.
 *
 * Expected `story` shape (see backend/main.py generate_story response):
 * {
 *   kpiName: "Revenue",
 *   metricChangeLabel: "Revenue ↓ 12%",
 *   direction: "down" | "up",
 *   drivers: [{ label: "Product B sales", changeLabel: "↓ 27%", direction: "down" }, ...],
 *   descriptive, diagnostic, prescriptive,
 *   confidence_score, abstained, kpi_story_id
 * }
 */

function ConfidenceBadge({ score, abstained }) {
  if (abstained) {
    return (
      <span className="inline-flex items-center gap-1 rounded-full bg-amber-100 text-amber-800 text-xs font-medium px-2.5 py-1">
        ⚠ Abstained — insufficient evidence
      </span>
    );
  }
  const pct = Math.round(score * 100);
  const tier = score >= 0.65 ? "bg-emerald-100 text-emerald-800" : score >= 0.35 ? "bg-amber-100 text-amber-800" : "bg-rose-100 text-rose-800";
  return (
    <span className={`inline-flex items-center gap-1 rounded-full text-xs font-medium px-2.5 py-1 ${tier}`}>
      Confidence: {pct}%
    </span>
  );
}

function DriverBar({ label, changeLabel, direction }) {
  const isDown = direction === "down";
  return (
    <div className="flex items-center justify-between py-2 border-b border-gray-100 last:border-0">
      <span className="text-sm text-gray-700">{label}</span>
      <span className={`text-sm font-semibold ${isDown ? "text-rose-600" : "text-emerald-600"}`}>
        {changeLabel}
      </span>
    </div>
  );
}

function renderWithCitations(text) {
  // Splits on [ev:<id>] markers so citation chips render distinctly
  // from prose without exposing raw evidence text inline.
  const parts = text.split(/(\[ev:[^\]]+\])/g);
  return parts.map((part, i) => {
    const match = part.match(/\[ev:([^\]]+)\]/);
    if (match) {
      return (
        <sup key={i} className="ml-0.5 text-[10px] font-semibold text-indigo-600 cursor-help" title={`Evidence ${match[1]}`}>
          [{i}]
        </sup>
      );
    }
    return <React.Fragment key={i}>{part}</React.Fragment>;
  });
}

export default function KPIStoryCard({ story, onFeedback }) {
  const [rated, setRated] = useState(null);
  const [showCorrection, setShowCorrection] = useState(false);
  const [correction, setCorrection] = useState("");

  const handleRate = (rating) => {
    setRated(rating);
    if (rating === "down") {
      setShowCorrection(true);
    } else {
      onFeedback?.({ kpi_story_id: story.kpi_story_id, rating, correction_text: null });
    }
  };

  const submitCorrection = () => {
    onFeedback?.({ kpi_story_id: story.kpi_story_id, rating: "down", correction_text: correction });
    setShowCorrection(false);
  };

  return (
    <div className="max-w-xl rounded-2xl border border-gray-200 shadow-sm bg-white p-6 space-y-4">
      {/* Headline */}
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs uppercase tracking-wide text-gray-400 font-medium">{story.kpiName}</p>
          <h2 className={`text-2xl font-bold ${story.direction === "down" ? "text-rose-600" : "text-emerald-600"}`}>
            {story.metricChangeLabel}
          </h2>
        </div>
        <ConfidenceBadge score={story.confidence_score} abstained={story.abstained} />
      </div>

      {/* Descriptive */}
      <p className="text-sm text-gray-700">{story.descriptive}</p>

      {/* Drivers */}
      {story.drivers?.length > 0 && (
        <div>
          <p className="text-xs font-semibold text-gray-500 mb-1">Correlated drivers</p>
          <div>
            {story.drivers.map((d, i) => (
              <DriverBar key={i} {...d} />
            ))}
          </div>
        </div>
      )}

      {/* Diagnostic (why) */}
      <div>
        <p className="text-xs font-semibold text-gray-500 mb-1">Why</p>
        <p className="text-sm text-gray-700 leading-relaxed">{renderWithCitations(story.diagnostic)}</p>
      </div>

      {/* Prescriptive (action) */}
      {!story.abstained && (
        <div className="rounded-xl bg-indigo-50 p-4">
          <p className="text-xs font-semibold text-indigo-600 mb-1">Recommended action</p>
          <p className="text-sm text-indigo-900">{story.prescriptive}</p>
        </div>
      )}

      {/* Feedback */}
      <div className="flex items-center justify-between pt-2 border-t border-gray-100">
        <span className="text-xs text-gray-400">Was this insight useful?</span>
        <div className="flex gap-2">
          <button
            onClick={() => handleRate("up")}
            className={`px-2 py-1 rounded-lg text-sm ${rated === "up" ? "bg-emerald-100" : "hover:bg-gray-100"}`}
            aria-label="Thumbs up"
          >
            👍
          </button>
          <button
            onClick={() => handleRate("down")}
            className={`px-2 py-1 rounded-lg text-sm ${rated === "down" ? "bg-rose-100" : "hover:bg-gray-100"}`}
            aria-label="Thumbs down"
          >
            👎
          </button>
        </div>
      </div>

      {showCorrection && (
        <div className="space-y-2">
          <textarea
            className="w-full text-sm border border-gray-200 rounded-lg p-2"
            rows={3}
            placeholder="What did this get wrong? (goes back into the retrieval pipeline)"
            value={correction}
            onChange={(e) => setCorrection(e.target.value)}
          />
          <button
            onClick={submitCorrection}
            className="text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 rounded-lg px-3 py-1.5"
          >
            Submit correction
          </button>
        </div>
      )}
    </div>
  );
}

// Example usage:
//
// <KPIStoryCard
//   story={{
//     kpiName: "Revenue",
//     metricChangeLabel: "Revenue ↓ 12%",
//     direction: "down",
//     drivers: [
//       { label: "Product B sales", changeLabel: "↓ 27%", direction: "down" },
//       { label: "Stockouts", changeLabel: "↑ 35%", direction: "up" },
//       { label: "Complaints", changeLabel: "↑ 41%", direction: "up" },
//     ],
//     descriptive: "Revenue fell 12% week-over-week, driven primarily by Product B.",
//     diagnostic: "Product B stockouts rose sharply in the same window [ev:a1b2], and customer complaints about availability increased [ev:c3d4].",
//     prescriptive: "Expedite replenishment for Product B in the affected region and notify the regional sales team of expected availability dates.",
//     confidence_score: 0.78,
//     abstained: false,
//     kpi_story_id: 42,
//   }}
//   onFeedback={(payload) => fetch("/feedback", { method: "POST", body: JSON.stringify(payload) })}
// />
