import { useEffect, useState } from "react";

import RiskSnapshotReviewModal from "../components/RiskSnapshotReviewModal";
import {
  apiUrl,
  getProcessingHistory,
  getRiskSnapshotHistory,
} from "../services/api";
import type {
  ProcessingJobHistory,
  RiskSnapshotHistory,
} from "../types/detection";

export default function HistoryPage() {
  const [jobs, setJobs] = useState<ProcessingJobHistory[]>([]);
  const [snapshots, setSnapshots] = useState<RiskSnapshotHistory[]>([]);
  const [mediaType, setMediaType] = useState("");
  const [status, setStatus] = useState("");
  const [riskLevel, setRiskLevel] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [reviewSnapshot, setReviewSnapshot] = useState<RiskSnapshotHistory | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    Promise.all([
      getProcessingHistory({ mediaType, status }, controller.signal),
      getRiskSnapshotHistory({ riskLevel }, controller.signal),
    ])
      .then(([jobPage, snapshotPage]) => {
        setJobs(jobPage.items);
        setSnapshots(snapshotPage.items);
      })
      .catch((caught: unknown) => {
        if (caught instanceof DOMException && caught.name === "AbortError") return;
        setError(caught instanceof Error ? caught.message : "Unable to load history.");
      })
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, [mediaType, status, riskLevel]);

  return (
    <div className="history-page">
      <header className="history-header">
        <a className="brand" href="/" aria-label="Crane Load Warning home">
          <span className="brand-mark" aria-hidden="true">
            <svg viewBox="0 0 32 32" focusable="false">
              <path d="M7 27h8M9 27V7m-3 0h22M9 7l6-4 4 4M9 11h13M22 7v9m-2 0h4m-2 0v4m-3 0h6v6h-6zM9 12l6 15M15 12 9 21" />
            </svg>
          </span>
          <span><strong>Crane Load Warning</strong><small>Processing history</small></span>
        </a>
        <a className="header-link" href="/">New analysis</a>
      </header>

      <main className="history-main">
        <div className="history-title">
          <div>
            <p className="eyebrow">Persisted assessment records</p>
            <h1>History</h1>
            <p>Processing jobs and sampled risk evidence. Snapshots are frame observations, not tracked safety events.</p>
          </div>
        </div>

        {error && <div className="error-banner" role="alert">{error}</div>}
        {loading && <p className="history-empty">Loading processing history…</p>}

        <section className="history-section" aria-labelledby="processing-jobs-title">
          <div className="history-section-heading">
            <div><p className="eyebrow">Image and video</p><h2 id="processing-jobs-title">Processing Jobs</h2></div>
            <div className="history-filters">
              <select aria-label="Filter jobs by media type" value={mediaType} onChange={(event) => setMediaType(event.target.value)}>
                <option value="">All media</option><option value="image">Images</option><option value="video">Videos</option>
              </select>
              <select aria-label="Filter jobs by status" value={status} onChange={(event) => setStatus(event.target.value)}>
                <option value="">All statuses</option><option value="queued">Queued</option><option value="processing">Processing</option><option value="completed">Completed</option><option value="failed">Failed</option>
              </select>
            </div>
          </div>
          {!loading && jobs.length === 0 ? <p className="history-empty">No processing jobs found.</p> : (
            <div className="history-job-list">{jobs.map((job) => <JobRow job={job} key={job.id} />)}</div>
          )}
        </section>

        <section className="history-section" aria-labelledby="risk-snapshots-title">
          <div className="history-section-heading">
            <div><p className="eyebrow">Sampled evidence</p><h2 id="risk-snapshots-title">Risk Snapshots</h2></div>
            <select aria-label="Filter snapshots by risk" value={riskLevel} onChange={(event) => setRiskLevel(event.target.value)}>
              <option value="">All risks</option><option value="WARNING">Warning</option><option value="DANGER">Danger</option>
            </select>
          </div>
          {!loading && snapshots.length === 0 ? <p className="history-empty">No WARNING or DANGER snapshots found.</p> : (
            <div className="snapshot-grid">
              {snapshots.map((snapshot) => (
                <SnapshotCard
                  snapshot={snapshot}
                  onReview={() => setReviewSnapshot(snapshot)}
                  key={snapshot.id}
                />
              ))}
            </div>
          )}
        </section>
      </main>
      {reviewSnapshot && (
        <RiskSnapshotReviewModal
          snapshot={reviewSnapshot}
          onClose={() => setReviewSnapshot(null)}
        />
      )}
    </div>
  );
}

function JobRow({ job }: { job: ProcessingJobHistory }) {
  const reportAvailable = job.media_type === "video" && job.status === "completed";
  return (
    <article className="history-job-row">
      <MediaTypeBadge mediaType={job.media_type} />
      <div><strong>{job.input_name}</strong><small>{new Date(job.created_at).toLocaleString()}</small></div>
      <span className={`history-status status-${job.status}`}>{job.status}</span>
      <span className={`report-risk risk-${(job.max_risk_level || "safe").toLowerCase()}`}>{job.max_risk_level || "—"}</span>
      <div className="history-job-metric history-processing-metric"><strong>{job.processing_time_ms == null ? "—" : `${(job.processing_time_ms / 1000).toFixed(1)}s`}</strong><small>processing</small></div>
      <div className="history-job-metric"><strong>{job.media_type === "video" ? (job.total_frames ?? "—") : "—"}</strong><small>frames</small></div>
      <div className="history-report-action">
        {reportAvailable ? (
          <a
            className="button button-secondary"
            href={`/?report=${encodeURIComponent(job.id)}&from=history`}
          >
            View report
          </a>
        ) : (
          <small>{reportStatusLabel(job)}</small>
        )}
      </div>
    </article>
  );
}

function MediaTypeBadge({ mediaType }: { mediaType: ProcessingJobHistory["media_type"] }) {
  return (
    <span className={`media-type-badge history-media-badge media-${mediaType}`}>
      {mediaType === "image" ? (
        <svg viewBox="0 0 24 24" focusable="false" aria-hidden="true">
          <rect x="3" y="4" width="18" height="16" rx="2" />
          <circle cx="9" cy="10" r="2" />
          <path d="m5 18 5-5 3 3 2-2 4 4" />
        </svg>
      ) : (
        <svg viewBox="0 0 24 24" focusable="false" aria-hidden="true">
          <rect x="3" y="6" width="13" height="12" rx="2" />
          <path d="m16 10 5-3v10l-5-3z" />
        </svg>
      )}
      <span>{mediaType}</span>
    </span>
  );
}

function reportStatusLabel(job: ProcessingJobHistory) {
  if (job.media_type !== "video") return "No video report";
  if (job.status === "queued" || job.status === "processing") return "Report pending";
  return "No report";
}

function SnapshotCard({
  snapshot,
  onReview,
}: {
  snapshot: RiskSnapshotHistory;
  onReview: () => void;
}) {
  const preview = apiUrl(snapshot.rgb_evidence_path || snapshot.evidence_path);
  return (
    <article className="snapshot-card">
      {preview ? (
        <button
          className="snapshot-preview-button"
          type="button"
          aria-label={`Review ${snapshot.risk_level} risk snapshot`}
          onClick={onReview}
        >
          <img src={preview} alt={`${snapshot.risk_level} risk snapshot`} />
        </button>
      ) : <div className="snapshot-placeholder">No preview</div>}
      <div className="snapshot-card-body">
        <span className={`report-risk risk-${snapshot.risk_level.toLowerCase()}`}>{snapshot.risk_level}</span>
        <strong>{snapshot.frame_index == null ? "Image assessment" : `Frame ${snapshot.frame_index + 1}`}</strong>
        <span>{snapshot.timestamp_sec == null ? "No video timestamp" : `${snapshot.timestamp_sec.toFixed(2)}s`} · Job {snapshot.job_id.slice(0, 8)}</span>
        <small>{snapshot.assessment_reliable ? "Reliable assessment" : "Review recommended"}</small>
        <button className="button button-secondary" type="button" onClick={onReview}>
          Review evidence
        </button>
      </div>
    </article>
  );
}
