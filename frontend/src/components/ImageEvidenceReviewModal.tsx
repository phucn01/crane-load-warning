import { useEffect, useRef } from "react";
import { createPortal } from "react-dom";

import { apiUrl, type ImageEvidenceViews } from "../services/api";

const VIEWS = [
  ["Original camera frame", "original_url"],
  ["Detected objects", "detection_url"],
  ["Pseudo-BEV safety view", "bev_url"],
  ["Combined review", "combined_url"],
] as const;

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
          <button ref={closeButtonRef} type="button" className="frame-evidence-close" aria-label="Close image review" onClick={onClose}>×</button>
        </header>
        <div className="frame-evidence-stage snapshot-review-stage">
          <div className="frame-evidence-views image-review-views">
            {VIEWS.map(([label, key]) => {
              const url = apiUrl(views[key]);
              return <figure key={key}><div>{url ? <img src={url} alt={label} /> : <span className="snapshot-view-missing">Unavailable</span>}</div><figcaption><span>{label}</span>{url && <a href={url} target="_blank" rel="noreferrer">Open image ↗</a>}</figcaption></figure>;
            })}
          </div>
        </div>
      </div>
    </div>,
    document.body,
  );
}
