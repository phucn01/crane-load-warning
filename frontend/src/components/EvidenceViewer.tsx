import { useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

import { evidenceUrl } from "../services/api";
import type { EvidenceResponse } from "../types/detection";

interface EvidenceViewerProps {
  evidence: EvidenceResponse;
}

interface EvidenceItem {
  key: keyof EvidenceResponse;
  title: string;
  description: string;
  wide?: boolean;
}

interface AvailableEvidence extends EvidenceItem {
  url: string;
}

const ITEMS: EvidenceItem[] = [
  {
    key: "combined_url",
    title: "Combined safety evidence",
    description: "Annotated camera view and shared Pseudo-BEV assessment.",
    wide: true,
  },
  {
    key: "rgb_url",
    title: "Camera evidence",
    description: "Detections and person-level safety status.",
  },
  {
    key: "pseudo_bev_url",
    title: "Pseudo-BEV safety view",
    description: "Relative geometry, load footprint, and safety zones.",
  },
];

export default function EvidenceViewer({ evidence }: EvidenceViewerProps) {
  const available: AvailableEvidence[] = ITEMS.flatMap((item) => {
    const url = evidenceUrl(evidence[item.key]);
    return url ? [{ ...item, url }] : [];
  });
  const [activeIndex, setActiveIndex] = useState<number | null>(null);
  const closeButtonRef = useRef<HTMLButtonElement>(null);
  const returnFocusRef = useRef<HTMLElement | null>(null);
  const isOpen = activeIndex !== null;
  const activeItem = activeIndex === null ? null : available[activeIndex];

  const openLightbox = (index: number) => {
    returnFocusRef.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    setActiveIndex(index);
  };

  const closeLightbox = useCallback(() => {
    setActiveIndex(null);
    window.setTimeout(() => returnFocusRef.current?.focus(), 0);
  }, []);

  const moveLightbox = useCallback(
    (direction: number) => {
      setActiveIndex((current) => {
        if (current === null || available.length < 2) return current;
        return (current + direction + available.length) % available.length;
      });
    },
    [available.length],
  );

  useEffect(() => {
    if (!isOpen) return undefined;

    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    closeButtonRef.current?.focus();

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") closeLightbox();
      if (event.key === "ArrowLeft") moveLightbox(-1);
      if (event.key === "ArrowRight") moveLightbox(1);
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [closeLightbox, isOpen, moveLightbox]);

  return (
    <section className="evidence-section" aria-labelledby="evidence-title">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Visual evidence</p>
          <h2 id="evidence-title">Review what the pipeline saw</h2>
        </div>
        <p>Relative depth and Pseudo-BEV values are non-metric.</p>
      </div>

      <div className="evidence-grid">
        {available.map((item, index) => (
          <article className={`evidence-card ${item.wide ? "evidence-wide" : ""}`} key={item.key}>
            <button
              type="button"
              className="evidence-trigger"
              aria-label={`Preview ${item.title}`}
              onClick={() => openLightbox(index)}
            >
              <img src={item.url} alt={item.title} loading="lazy" />
            </button>
            <div className="evidence-caption">
              <div>
                <h3>{item.title}</h3>
                <p>{item.description}</p>
              </div>
              <button
                type="button"
                className="open-link evidence-view-button"
                aria-label={`View ${item.title}`}
                onClick={() => openLightbox(index)}
              >
                View
              </button>
            </div>
          </article>
        ))}
      </div>

      {activeItem &&
        createPortal(
          <div
            className="evidence-lightbox"
            onMouseDown={(event) => {
              if (event.target === event.currentTarget) closeLightbox();
            }}
          >
            <div
              className="evidence-lightbox-dialog"
              role="dialog"
              aria-modal="true"
              aria-labelledby="evidence-lightbox-title"
              aria-describedby="evidence-lightbox-description"
            >
              <header className="evidence-lightbox-header">
                <div>
                  <p className="eyebrow">
                    Evidence {activeIndex! + 1} of {available.length}
                  </p>
                  <h2 id="evidence-lightbox-title">{activeItem.title}</h2>
                </div>
                <button
                  ref={closeButtonRef}
                  type="button"
                  className="evidence-lightbox-close"
                  aria-label="Close evidence viewer"
                  onClick={closeLightbox}
                >
                  ×
                </button>
              </header>

              <div className="evidence-lightbox-stage">
                {available.length > 1 && (
                  <button
                    type="button"
                    className="evidence-lightbox-nav evidence-lightbox-previous"
                    aria-label="Previous evidence"
                    onClick={() => moveLightbox(-1)}
                  >
                    ‹
                  </button>
                )}
                <img src={activeItem.url} alt={activeItem.title} />
                {available.length > 1 && (
                  <button
                    type="button"
                    className="evidence-lightbox-nav evidence-lightbox-next"
                    aria-label="Next evidence"
                    onClick={() => moveLightbox(1)}
                  >
                    ›
                  </button>
                )}
              </div>

              <footer className="evidence-lightbox-footer">
                <p id="evidence-lightbox-description">{activeItem.description}</p>
                <a href={activeItem.url} target="_blank" rel="noreferrer" className="open-link">
                  Open original ↗
                </a>
              </footer>
            </div>
          </div>,
          document.body,
        )}
    </section>
  );
}
