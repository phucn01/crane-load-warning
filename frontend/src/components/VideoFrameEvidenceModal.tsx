import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

import { apiUrl } from "../services/api";
import type { VideoFrameEvidence } from "../types/detection";

interface VideoFrameEvidenceModalProps {
  evidence: VideoFrameEvidence[];
  onClose: () => void;
}

export default function VideoFrameEvidenceModal({
  evidence,
  onClose,
}: VideoFrameEvidenceModalProps) {
  const [activeIndex, setActiveIndex] = useState(0);
  const closeButtonRef = useRef<HTMLButtonElement>(null);
  const returnFocusRef = useRef<HTMLElement | null>(null);
  const active = evidence[activeIndex];

  useEffect(() => {
    returnFocusRef.current =
      document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    closeButtonRef.current?.focus();

    const move = (direction: number) => {
      setActiveIndex((current) =>
        (current + direction + evidence.length) % evidence.length,
      );
    };
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
      if (event.key === "ArrowLeft" && evidence.length > 1) move(-1);
      if (event.key === "ArrowRight" && evidence.length > 1) move(1);
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", handleKeyDown);
      window.setTimeout(() => returnFocusRef.current?.focus(), 0);
    };
  }, [evidence.length, onClose]);

  if (!active) return null;

  const move = (direction: number) => {
    setActiveIndex((current) =>
      (current + direction + evidence.length) % evidence.length,
    );
  };
  const originalUrl = apiUrl(active.original_url);
  const rgbUrl = apiUrl(active.rgb_url);
  const bevUrl = apiUrl(active.pseudo_bev_url);

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
        aria-labelledby="frame-evidence-title"
        aria-describedby="frame-evidence-description"
      >
        <header className="frame-evidence-header">
          <div>
            <p className="eyebrow">
              Evidence {activeIndex + 1} of {evidence.length}
            </p>
            <h2 id="frame-evidence-title">Frame {active.frame_number} evidence</h2>
            <p>
              {active.timestamp_seconds.toFixed(1)}s · Independently assessed frame
            </p>
          </div>
          <div className="frame-evidence-header-actions">
            <span className={`frame-evidence-risk risk-${active.risk_level.toLowerCase()}`}>
              {active.risk_level}
            </span>
            <button
              ref={closeButtonRef}
              type="button"
              className="frame-evidence-close"
              aria-label="Close frame evidence"
              onClick={onClose}
            >
              ×
            </button>
          </div>
        </header>

        <div className="frame-evidence-stage">
          {evidence.length > 1 && (
            <button
              type="button"
              className="frame-evidence-nav frame-evidence-previous"
              aria-label="Previous frame evidence"
              onClick={() => move(-1)}
            >
              ‹
            </button>
          )}
          <div className="frame-evidence-views">
            <figure>
              <div><img src={originalUrl || undefined} alt={`Original frame ${active.frame_number}`} /></div>
              <figcaption>
                <span>Original camera frame</span>
                <a href={originalUrl || undefined} target="_blank" rel="noreferrer">Open original ↗</a>
              </figcaption>
            </figure>
            <figure>
              <div><img src={rgbUrl || undefined} alt={`Annotated frame ${active.frame_number}`} /></div>
              <figcaption>
                <span>Annotated camera frame</span>
                <a href={rgbUrl || undefined} target="_blank" rel="noreferrer">Open image ↗</a>
              </figcaption>
            </figure>
            <figure>
              <div><img src={bevUrl || undefined} alt={`Pseudo-BEV for frame ${active.frame_number}`} /></div>
              <figcaption>
                <span>Pseudo-BEV safety view</span>
                <a href={bevUrl || undefined} target="_blank" rel="noreferrer">Open image ↗</a>
              </figcaption>
            </figure>
          </div>
          {evidence.length > 1 && (
            <button
              type="button"
              className="frame-evidence-nav frame-evidence-next"
              aria-label="Next frame evidence"
              onClick={() => move(1)}
            >
              ›
            </button>
          )}
        </div>

        <footer className="frame-evidence-footer">
          <p id="frame-evidence-description">
            Pseudo-BEV is relative and non-metric. Evidence represents a frame-level
            classification, not a tracked safety event.
          </p>
        </footer>
      </div>
    </div>,
    document.body,
  );
}
