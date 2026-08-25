import { useEffect, useState } from "react";

import { apiUrl, getVideoReport } from "../services/api";
import type { VideoReport } from "../types/detection";

export default function VideoReportPage({ jobId }: { jobId: string }) {
  const [report, setReport] = useState<VideoReport | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    getVideoReport(jobId)
      .then((payload) => {
        if (active) setReport(payload);
      })
      .catch((reason: unknown) => {
        if (active) {
          setError(reason instanceof Error ? reason.message : "Could not load report.");
        }
      });
    return () => {
      active = false;
    };
  }, [jobId]);

  if (error) {
    return (
      <main className="report-state">
        <p className="eyebrow">Video safety report</p>
        <h1>Report unavailable</h1>
        <p>{error}</p>
        <a className="button button-secondary" href="/">Return to analysis</a>
      </main>
    );
  }

  if (!report) {
    return (
      <main className="report-state" aria-live="polite">
        <p className="eyebrow">Video safety report</p>
        <h1>Loading report...</h1>
      </main>
    );
  }

  const completedAt = new Date(report.completed_at).toLocaleString();
  return (
    <div className="report-page">
      <header className="report-header">
        <a className="report-brand" href="/">Crane Load Warning</a>
        <div className="report-actions">
          <button className="button button-secondary" type="button" onClick={() => window.print()}>
            Print report
          </button>
          <a className="button button-secondary" href={apiUrl(report.video.download_url) || undefined}>
            Download video
          </a>
        </div>
      </header>

      <main className="report-content">
        <section className="report-hero">
          <div>
            <p className="eyebrow">Completed video assessment</p>
            <h1>Crane load safety report</h1>
            <p>Job {report.job_id} · Completed {completedAt}</p>
          </div>
          <span className={`report-risk risk-${report.summary.max_risk_level?.toLowerCase() || "safe"}`}>
            {report.summary.max_risk_level || "SAFE"}
          </span>
        </section>

        <section className="report-summary" aria-label="Assessment summary">
          <ReportMetric label="Frames assessed" value={report.summary.processed_frames} />
          <ReportMetric label="Safe frames" value={report.summary.safe_frames} tone="safe" />
          <ReportMetric label="Warning frames" value={report.summary.warning_frames} tone="warning" />
          <ReportMetric label="Danger frames" value={report.summary.danger_frames} tone="danger" />
          <ReportMetric label="Risk segments" value={report.summary.risk_segment_count} />
          <ReportMetric label="Processing FPS" value={report.summary.average_processing_fps.toFixed(1)} />
        </section>

        <section className="report-section">
          <div className="report-section-heading">
            <div>
              <p className="eyebrow">Annotated output</p>
              <h2>Processed video</h2>
            </div>
            <span>{report.video.codec.toUpperCase()}</span>
          </div>
          <video controls preload="metadata" src={apiUrl(report.video.url) || undefined} />
          {report.video.playback_warning && <p className="report-warning">{report.video.playback_warning}</p>}
        </section>

        <section className="report-section">
          <div className="report-section-heading">
            <div>
              <p className="eyebrow">Risk review</p>
              <h2>WARNING / DANGER segments</h2>
            </div>
            <span>{report.risk_segments.length} segments</span>
          </div>

          {report.risk_segments.length === 0 ? (
            <p className="report-empty">No WARNING or DANGER segments were detected.</p>
          ) : (
            <div className="report-segments">
              {report.risk_segments.map((segment, index) => (
                <article className="report-segment" key={segment.segment_id}>
                  <header>
                    <div>
                      <p className="eyebrow">Segment {index + 1}</p>
                      <h3>{segment.start_seconds.toFixed(1)}s – {segment.end_seconds.toFixed(1)}s</h3>
                    </div>
                    <span className={`report-risk risk-${segment.max_risk_level.toLowerCase()}`}>
                      {segment.max_risk_level}
                    </span>
                  </header>
                  <div className="report-segment-meta">
                    <span>Frames {segment.start_frame}–{segment.end_frame}</span>
                    <span>{segment.warning_frame_count} warning</span>
                    <span>{segment.danger_frame_count} danger</span>
                  </div>
                  <video controls preload="metadata" src={apiUrl(segment.result_url) || undefined} />
                  <div className="report-evidence-grid">
                    {segment.evidence.map((item) => (
                      <article key={item.frame_number}>
                        <img src={apiUrl(item.rgb_url) || undefined} alt={`Annotated evidence frame ${item.frame_number}`} />
                        <div>
                          <strong>Frame {item.frame_number}</strong>
                          <span>{item.timestamp_seconds.toFixed(1)}s · {item.risk_level}</span>
                          <a href={apiUrl(item.original_url) || undefined} target="_blank" rel="noreferrer">View original</a>
                        </div>
                      </article>
                    ))}
                  </div>
                </article>
              ))}
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

function ReportMetric({
  label,
  value,
  tone,
}: {
  label: string;
  value: number | string;
  tone?: "safe" | "warning" | "danger";
}) {
  return (
    <article className={tone ? `report-metric metric-${tone}` : "report-metric"}>
      <span>{label}</span>
      <strong>{value}</strong>
    </article>
  );
}
