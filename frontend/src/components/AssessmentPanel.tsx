import RiskBadge from "./RiskBadge";
import type { ImageDetectionResponse } from "../types/detection";

interface AssessmentPanelProps {
  result: ImageDetectionResponse;
}

function readableReason(reason: string): string {
  return reason.replaceAll("_", " ");
}

export default function AssessmentPanel({ result }: AssessmentPanelProps) {
  const { assessment, summary } = result;
  const status = result.assessment_status ?? "FULL_EVALUATION";

  return (
    <section className="assessment-card" aria-labelledby="assessment-title">
      <div className="assessment-heading">
        <div>
          <p className="eyebrow">{status === "FULL_EVALUATION" ? "Immediate assessment" : status.replaceAll("_", " ")}</p>
          <h2 id="assessment-title">Scene safety result</h2>
        </div>
        <RiskBadge level={assessment.risk_level} />
      </div>

      <div className="metric-grid">
        <article className="metric">
          <span>People</span>
          <strong>{summary.person_count}</strong>
        </article>
        <article className="metric">
          <span>Hanging loads</span>
          <strong>{summary.load_count}</strong>
        </article>
        <article className="metric">
          <span>Ropes</span>
          <strong>{summary.rope_count}</strong>
        </article>
        <article className="metric">
          <span>Processing</span>
          <strong>{result.processing_time_ms.toFixed(0)}<small> ms</small></strong>
        </article>
      </div>

      <div className="reliability-row">
        <span className={`reliability-icon ${assessment.assessment_reliable ? "is-reliable" : ""}`} aria-hidden="true">
          {assessment.assessment_reliable ? "✓" : "!"}
        </span>
        <div>
          <strong>
            {assessment.assessment_reliable
              ? "Assessment is reliable"
              : "Assessment needs operator review"}
          </strong>
          <p>
            {assessment.assessment_reliable
              ? "Required geometry and detection quality checks passed."
              : "Technical quality limitations prevented a fully reliable result."}
          </p>
        </div>
      </div>

      {assessment.quality_reasons.length > 0 && (
        <div className="quality-block">
          <h3>Quality reasons</h3>
          <ul className="reason-list">
            {assessment.quality_reasons.map((reason) => (
              <li key={reason}>{readableReason(reason)}</li>
            ))}
          </ul>
        </div>
      )}

      {assessment.pairs.length > 0 && (
        <div className="pair-block">
          <div className="section-label-row">
            <h3>Person–load checks</h3>
            <span>{assessment.pairs.length} evaluated</span>
          </div>
          <div className="pair-list">
            {assessment.pairs.map((pair, index) => (
              <article className="pair-row" key={`${pair.person_id}-${pair.load_id}-${index}`}>
                <div>
                  <strong>{pair.person_id}</strong>
                  <span>against {pair.load_id}</span>
                </div>
                <div className="pair-confidence">
                  <span>{Math.round(pair.confidence * 100)}% confidence</span>
                  <RiskBadge level={pair.risk_level} compact />
                </div>
              </article>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}
