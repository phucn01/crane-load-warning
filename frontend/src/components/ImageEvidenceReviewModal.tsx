import { useEffect, useRef } from "react";
import { createPortal } from "react-dom";

import { apiUrl, type ImageEvidenceViews } from "../services/api";

const VIEWS = [
  ["Original camera frame", "original_url"],
  ["Annotated result", "detection_url"],
  ["Pseudo-BEV safety view", "bev_url"],
  ["Combined review", "combined_url"],
] as const;

function readableStatus(status: string, riskLevel: string | null): string {
  if (status === "SAFE_NO_LOAD") return "SAFE · NO LOAD";
  if (status !== "FULL_EVALUATION") return status.replaceAll("_", " ");
  return riskLevel ?? "ASSESSED";
}

function readableReason(reason: string): string {
  return reason.replaceAll("_", " ");
}

export default function ImageEvidenceReviewModal({
  jobName,
  views,
  onClose,
}: {
  jobName: string;
  views: ImageEvidenceViews;
  onClose: () => void;
}) {
  const closeButtonRef = useRef<HTMLButtonElement>(null);
  const level = views.risk_level ?? "SAFE";
  const status = readableStatus(views.assessment_status, views.risk_level);
  const reviewViews = VIEWS.filter(
    ([, key]) => key !== "combined_url" || views.combined_url !== views.detection_url,
  );
  useEffect(() => {
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    closeButtonRef.current?.focus();
    const handleKeyDown = (event: KeyboardEvent) => { if (event.key === "Escape") onClose(); };
    window.addEventListener("keydown", handleKeyDown);
    return () => { document.body.style.overflow = previousOverflow; window.removeEventListener("keydown", handleKeyDown); };
  }, [onClose]);

  return createPortal(
    <div className="frame-evidence-modal" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
      <div className="frame-evidence-dialog" role="dialog" aria-modal="true" aria-labelledby="image-review-title">
        <header className="frame-evidence-header">
          <div><p className="eyebrow">Image evidence</p><h2 id="image-review-title">Review image</h2><p>{jobName}</p></div>
          <div className="frame-evidence-header-actions">
            <span className={`frame-evidence-risk risk-${level.toLowerCase()}`}>{status}</span>
            <button ref={closeButtonRef} type="button" className="frame-evidence-close" aria-label="Close image review" onClick={onClose}>×</button>
          </div>
        </header>
        <div className="frame-evidence-stage snapshot-review-stage">
          <div className="image-review-assessment">
            <div className="reliability-row">
              <span className={`reliability-icon ${views.assessment_reliable ? "is-reliable" : ""}`} aria-hidden="true">
                {views.assessment_reliable ? "✓" : "!"}
              </span>
              <div>
                <strong>
                  {views.assessment_reliable === true
                    ? "Assessment is reliable"
                    : views.assessment_reliable === false
                      ? "Assessment needs operator review"
                      : "Assessment was not completed"}
                </strong>
                <p>
                  {views.assessment_status === "SAFE_NO_LOAD"
                    ? "A person was detected, but no hanging load was found."
                    : views.assessment_reliable === true
                      ? "Required detection and geometry quality checks passed."
                      : "Review the quality reasons and available evidence before interpreting this result."}
                </p>
              </div>
            </div>
            {views.quality_reasons.length > 0 && (
              <div className="quality-block">
                <h3>Quality reasons</h3>
                <ul className="reason-list">
                  {views.quality_reasons.map((reason) => <li key={reason}>{readableReason(reason)}</li>)}
                </ul>
              </div>
            )}
          </div>
          <div className="frame-evidence-views image-review-views">
            {reviewViews.map(([label, key]) => {
              const url = apiUrl(views[key]);
              return <figure key={key}><div>{url ? <img src={url} alt={label} /> : <span className="snapshot-view-missing">Not generated</span>}</div><figcaption><span>{label}</span>{url && <a href={url} target="_blank" rel="noreferrer">Open image ↗</a>}</figcaption></figure>;
            })}
          </div>
        </div>
      </div>
    </div>,
    document.body,
  );
}
