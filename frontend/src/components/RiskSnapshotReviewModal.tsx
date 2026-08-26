import { useEffect, useRef } from "react";
import { createPortal } from "react-dom";

import { apiUrl } from "../services/api";
import type { RiskSnapshotHistory } from "../types/detection";

export default function RiskSnapshotReviewModal({
  snapshot,
  onClose,
}: {
  snapshot: RiskSnapshotHistory;
  onClose: () => void;
}) {
  const closeButtonRef = useRef<HTMLButtonElement>(null);
  const returnFocusRef = useRef<HTMLElement | null>(null);
  const frameNumber = snapshot.frame_index == null ? null : snapshot.frame_index + 1;

  useEffect(() => {
    returnFocusRef.current = document.activeElement instanceof HTMLElement
      ? document.activeElement
      : null;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    closeButtonRef.current?.focus();
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", handleKeyDown);
      window.setTimeout(() => returnFocusRef.current?.focus(), 0);
    };
  }, [onClose]);

  const views = [
    ["Original camera frame", snapshot.evidence_path],
    ["Annotated camera frame", snapshot.rgb_evidence_path],
    ["Pseudo-BEV safety view", snapshot.pseudo_bev_path],
  ] as const;

  return createPortal(
    <div
      className="frame-evidence-modal"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div
        className="frame-evidence-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="snapshot-review-title"
        aria-describedby="snapshot-review-description"
      >
        <header className="frame-evidence-header">
          <div>
            <p className="eyebrow">Sampled risk evidence</p>
            <h2 id="snapshot-review-title">
              {frameNumber == null ? "Image evidence" : `Frame ${frameNumber} evidence`}
            </h2>
            <p>
              {snapshot.timestamp_sec == null ? "Image assessment" : `${snapshot.timestamp_sec.toFixed(2)}s · Video snapshot`}
            </p>
          </div>
          <div className="frame-evidence-header-actions">
            <span className={`frame-evidence-risk risk-${snapshot.risk_level.toLowerCase()}`}>
              {snapshot.risk_level}
            </span>
            <button
              ref={closeButtonRef}
              type="button"
              className="frame-evidence-close"
              aria-label="Close snapshot review"
              onClick={onClose}
            >
              ×
            </button>
          </div>
        </header>

        <div className="frame-evidence-stage snapshot-review-stage">
          <div className="frame-evidence-views">
            {views.map(([label, path]) => {
              const url = apiUrl(path);
              return (
                <figure key={label}>
                  <div>
                    {url ? <img src={url} alt={`${label} ${frameNumber ?? "image"}`} /> : <span className="snapshot-view-missing">Unavailable</span>}
                  </div>
                  <figcaption>
                    <span>{label}</span>
                    {url && <a href={url} target="_blank" rel="noreferrer">Open image ↗</a>}
                  </figcaption>
                </figure>
              );
            })}
          </div>
        </div>

        <footer className="frame-evidence-footer">
          <p id="snapshot-review-description">
            Pseudo-BEV is relative and non-metric. This is sampled frame evidence,
            not a tracked safety event.
          </p>
        </footer>
      </div>
    </div>,
    document.body,
  );
}
