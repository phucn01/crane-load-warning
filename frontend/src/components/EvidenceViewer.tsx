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
  const available = ITEMS.filter((item) => evidence[item.key]);

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
        {available.map((item) => {
          const url = evidenceUrl(evidence[item.key]);
          if (!url) return null;
          return (
            <article className={`evidence-card ${item.wide ? "evidence-wide" : ""}`} key={item.key}>
              <a href={url} target="_blank" rel="noreferrer" title={`Open ${item.title}`}>
                <img src={url} alt={item.title} loading="lazy" />
              </a>
              <div className="evidence-caption">
                <div>
                  <h3>{item.title}</h3>
                  <p>{item.description}</p>
                </div>
                <a href={url} target="_blank" rel="noreferrer" className="open-link">
                  Open full size ↗
                </a>
              </div>
            </article>
          );
        })}
      </div>

      {!evidence.combined_url && (
        <p className="evidence-note">
          Combined alert evidence is created only for WARNING or DANGER results.
        </p>
      )}
    </section>
  );
}
