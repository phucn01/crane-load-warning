import { useEffect, useMemo, useRef, useState } from "react";

import RiskTimeline from "../components/RiskTimeline";
import VideoFrameEvidenceModal from "../components/VideoFrameEvidenceModal";
import {
  apiUrl,
  getAllRiskSnapshotsForJob,
  getAllVideoFrameResults,
  getVideoReport,
} from "../services/api";
import type {
  RiskSnapshotHistory,
  RiskLevel,
  VideoFrameEvidence,
  VideoFrameRiskResult,
  VideoReport,
  VideoReportEvidence,
  VideoReportSegment,
} from "../types/detection";

export default function VideoReportPage({ jobId }: { jobId: string }) {
  const [report, setReport] = useState<VideoReport | null>(null);
  const [frameResults, setFrameResults] = useState<VideoFrameRiskResult[]>([]);
  const [riskSnapshots, setRiskSnapshots] = useState<RiskSnapshotHistory[]>([]);
  const [selectedSnapshotId, setSelectedSnapshotId] = useState<string | null>(null);
  const [selectedSegmentId, setSelectedSegmentId] = useState<string | null>(null);
  const [selectedEvidenceFrame, setSelectedEvidenceFrame] = useState<number | null>(null);
  const [modalEvidence, setModalEvidence] = useState<VideoFrameEvidence[] | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [frameTimelineUnavailable, setFrameTimelineUnavailable] = useState(false);
  const [snapshotsUnavailable, setSnapshotsUnavailable] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    const controller = new AbortController();
    let active = true;
    Promise.all([
      getVideoReport(jobId, controller.signal),
      getAllVideoFrameResults(jobId, controller.signal)
        .then((frames) => ({ frames, unavailable: false }))
        .catch((reason: unknown) => {
          if (reason instanceof DOMException && reason.name === "AbortError") throw reason;
          return { frames: [], unavailable: true };
        }),
      getAllRiskSnapshotsForJob(jobId, controller.signal)
        .then((snapshots) => ({ snapshots, unavailable: false }))
        .catch((reason: unknown) => {
          if (reason instanceof DOMException && reason.name === "AbortError") throw reason;
          return { snapshots: [], unavailable: true };
        }),
    ])
      .then(([reportPayload, timeline, persistedSnapshots]) => {
        if (!active) return;
        setReport(reportPayload);
        setFrameResults(timeline.frames);
        setFrameTimelineUnavailable(timeline.unavailable);
        setRiskSnapshots([...persistedSnapshots.snapshots].sort(
          (left, right) => (left.timestamp_sec ?? 0) - (right.timestamp_sec ?? 0),
        ));
        setSnapshotsUnavailable(persistedSnapshots.unavailable);
        const initialSegment = reportPayload.risk_segments.find(
          (segment) => segment.max_risk_level === "DANGER",
        ) ?? reportPayload.risk_segments[0];
        setSelectedSegmentId(initialSegment?.segment_id ?? null);
        setSelectedEvidenceFrame(initialSegment?.evidence[0]?.frame_number ?? null);
      })
      .catch((reason: unknown) => {
        if (!active || (reason instanceof DOMException && reason.name === "AbortError")) return;
        setError(reason instanceof Error ? reason.message : "Could not load report.");
      });
    return () => {
      active = false;
      controller.abort();
    };
  }, [jobId]);

  const selectedSegment = report?.risk_segments.find(
    (segment) => segment.segment_id === selectedSegmentId,
  ) ?? null;
  const selectedEvidence = selectedSegment?.evidence.find(
    (item) => item.frame_number === selectedEvidenceFrame,
  ) ?? selectedSegment?.evidence[0] ?? null;
  const selectedSnapshot = riskSnapshots.find(
    (snapshot) => snapshot.id === selectedSnapshotId,
  ) ?? null;
  const selectedSnapshotEvidence = selectedSnapshot
    ? snapshotToEvidence(selectedSnapshot)
    : null;
  const allEvidence = useMemo(() => mergeTimelineEvidence(report, riskSnapshots), [report, riskSnapshots]);
  const selectFrame = (frame: VideoFrameRiskResult) => {
    setCurrentTime(frame.timestamp_seconds);
    const video = videoRef.current;
    if (video) {
      video.currentTime = frame.timestamp_seconds;
      video.focus();
    }
  };

  const selectTimestamp = (timestampSeconds: number) => {
    const frame = nearestFrame(frameResults, timestampSeconds);
    if (frame) {
      selectFrame(frame);
      return;
    }
    setCurrentTime(timestampSeconds);
    const video = videoRef.current;
    if (video) {
      video.currentTime = timestampSeconds;
      video.focus();
    }
  };

  const selectSegment = (segment: VideoReportSegment) => {
    setSelectedSnapshotId(null);
    setSelectedSegmentId(segment.segment_id);
    setSelectedEvidenceFrame(segment.evidence[0]?.frame_number ?? null);
  };

  const reviewSnapshot = (snapshot: RiskSnapshotHistory) => {
    setSelectedSnapshotId(snapshot.id);
    selectTimestamp(snapshot.timestamp_sec ?? 0);
  };

  if (error) {
    return (
      <main className="report-state">
        <p className="eyebrow">Video safety report</p>
        <h1>Report unavailable</h1>
        <p>{error}</p>
        <a className="button button-secondary" href="/?history=1">Return to history</a>
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
  const warningRate = percentage(report.summary.warning_frames, report.summary.processed_frames);
  const dangerRate = percentage(report.summary.danger_frames, report.summary.processed_frames);

  return (
    <div className="report-page">
      <header className="report-header">
        <a className="report-brand" href="/">Crane Load Warning</a>
        <div className="report-actions">
          <a className="button button-secondary" href="/?history=1">
            History
          </a>
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
            <p>{report.input_filename} · Completed {completedAt}</p>
            <p className="report-job-id">Job {report.job_id}</p>
          </div>
          <span className={`report-risk risk-${report.summary.max_risk_level?.toLowerCase() || "safe"}`}>
            {report.summary.max_risk_level || "SAFE"}
          </span>
        </section>

        <section className="report-summary" aria-label="Assessment summary">
          <ReportMetric label="Frames assessed" value={report.summary.processed_frames} />
          <ReportMetric label="Safe frames" value={report.summary.safe_frames} tone="safe" />
          <ReportMetric label="Warning frames" value={report.summary.warning_frames} detail={`${warningRate}% of video`} tone="warning" />
          <ReportMetric label="Danger frames" value={report.summary.danger_frames} detail={`${dangerRate}% of video`} tone="danger" />
          <ReportMetric label="Risk segments" value={report.summary.risk_segment_count} />
          <ReportMetric label="Risk snapshots" value={riskSnapshots.length} />
        </section>

        <section className="report-section report-video-section">
          <div className="report-section-heading">
            <div>
              <p className="eyebrow">Annotated output</p>
              <h2>Video and frame-risk timeline</h2>
            </div>
            <span>{report.video.codec.toUpperCase()}</span>
          </div>
          <video
            ref={videoRef}
            controls
            preload="metadata"
            src={apiUrl(report.video.url) || undefined}
            onLoadedMetadata={(event) => setDuration(event.currentTarget.duration)}
            onTimeUpdate={(event) => {
              setCurrentTime(event.currentTarget.currentTime);
            }}
          />
          {report.video.playback_warning && <p className="report-warning">{report.video.playback_warning}</p>}
          {frameTimelineUnavailable && (
            <p className="report-warning">
              The per-frame timeline is no longer in memory. The persisted report,
              video, risk segments, and evidence remain available.
            </p>
          )}
          <RiskTimeline
            results={frameResults}
            evidence={allEvidence}
            totalFrames={report.summary.total_frames || report.summary.processed_frames}
            currentTime={currentTime}
            duration={duration}
            onSeek={selectTimestamp}
          />
        </section>

        <section className="report-section" aria-labelledby="report-snapshots-title">
          <div className="report-section-heading">
            <div>
              <p className="eyebrow">Persisted cooldown sampling</p>
              <h2 id="report-snapshots-title">Risk snapshots</h2>
            </div>
            <span>{riskSnapshots.length} snapshots</span>
          </div>
          <p className="report-section-note">
            These are the same sampled WARNING / DANGER frames shown in History.
            Segment evidence below keeps only first, peak, and last key frames, so
            its count can be smaller when roles overlap.
          </p>
          {snapshotsUnavailable ? (
            <p className="report-warning">Persisted risk snapshots could not be loaded.</p>
          ) : riskSnapshots.length === 0 ? (
            <p className="report-empty">No persisted risk snapshots are available for this video.</p>
          ) : (
            <div className="report-snapshot-grid">
              {riskSnapshots.map((snapshot) => (
                <ReportSnapshotCard
                  snapshot={snapshot}
                  selected={snapshot.id === selectedSnapshotId}
                  onReview={() => reviewSnapshot(snapshot)}
                  key={snapshot.id}
                />
              ))}
            </div>
          )}
        </section>

        <section className="report-section">
          <div className="report-section-heading">
            <div>
              <p className="eyebrow">{selectedSnapshot ? "Frame review" : "Risk review"}</p>
              <h2>
                {selectedSnapshot?.frame_index == null
                  ? "WARNING / DANGER segments"
                  : `Snapshot frame ${selectedSnapshot.frame_index + 1}`}
              </h2>
            </div>
            <span>{report.risk_segments.length} segments</span>
          </div>

          {report.risk_segments.length === 0 && !selectedSnapshot ? (
            <p className="report-empty">No WARNING or DANGER segments were detected.</p>
          ) : (
            <div className="report-segment-layout">
              <div className="report-segments" aria-label="Risk segment review queue">
                {report.risk_segments.map((segment, index) => (
                  <ReportSegmentCard
                    segment={segment}
                    index={index}
                    selected={selectedSnapshot == null && segment.segment_id === selectedSegmentId}
                    onSelect={() => selectSegment(segment)}
                    onSeek={() => selectTimestamp(timestampForFrame(frameResults, segment.risk_start_frame, segment.start_seconds))}
                    key={segment.segment_id}
                  />
                ))}
              </div>
              {selectedSnapshot ? (
                <SnapshotEvidencePanel
                  snapshot={selectedSnapshot}
                  evidence={selectedSnapshotEvidence}
                  onOpen={(item) => setModalEvidence([item])}
                />
              ) : (
                <EvidencePanel
                  segment={selectedSegment}
                  evidence={selectedEvidence}
                  onSelectEvidence={setSelectedEvidenceFrame}
                  onOpen={(item) => setModalEvidence([item])}
                />
              )}
            </div>
          )}
        </section>

        <section className="report-methodology">
          <div>
            <p className="eyebrow">Interpretation notes</p>
            <h2>How to read this report</h2>
          </div>
          <ul>
            <li>SAFE, WARNING, and DANGER totals count independently assessed frames, not tracked events.</li>
            <li>Risk snapshots use cooldown sampling; segment evidence contains first, peak, and last key frames.</li>
            <li>Saved clips include pre-roll and post-roll context outside the true risk range.</li>
            <li>Pseudo-BEV uses relative depth and is not a metric distance measurement.</li>
            <li>Person and load identities are not tracked consistently across video frames.</li>
          </ul>
          <div className="report-processing-meta">
            <span>Average processing speed <strong>{report.summary.average_processing_fps.toFixed(1)} FPS</strong></span>
            <span>Processing elapsed <strong>{report.summary.elapsed_seconds.toFixed(1)}s</strong></span>
            <span>Schema <strong>{report.schema_version}</strong></span>
          </div>
        </section>
      </main>

      {modalEvidence && (
        <VideoFrameEvidenceModal
          evidence={modalEvidence}
          onClose={() => setModalEvidence(null)}
        />
      )}
    </div>
  );
}

function ReportMetric({
  label,
  value,
  detail,
  tone,
}: {
  label: string;
  value: number | string;
  detail?: string;
  tone?: "safe" | "warning" | "danger";
}) {
  return (
    <article className={tone ? `report-metric metric-${tone}` : "report-metric"}>
      <span>{label}</span>
      <strong>{value}</strong>
      {detail && <small>{detail}</small>}
    </article>
  );
}

function ReportSegmentCard({
  segment,
  index,
  selected,
  onSelect,
  onSeek,
}: {
  segment: VideoReportSegment;
  index: number;
  selected: boolean;
  onSelect: () => void;
  onSeek: () => void;
}) {
  const clipFrames = Math.max(1, segment.end_frame - segment.start_frame + 1);
  const riskLeft = (segment.risk_start_frame - segment.start_frame) * 100 / clipFrames;
  const riskWidth = (segment.risk_end_frame - segment.risk_start_frame + 1) * 100 / clipFrames;
  return (
    <article className={`report-segment${selected ? " is-selected" : ""}`}>
      <header>
        <div>
          <p className="eyebrow">Segment {String(index + 1).padStart(2, "0")}</p>
          <h3>{segment.start_seconds.toFixed(1)}s – {segment.end_seconds.toFixed(1)}s</h3>
        </div>
        <span className={`report-risk risk-${segment.max_risk_level.toLowerCase()}`}>
          {segment.max_risk_level}
        </span>
      </header>
      <div className="segment-range">
        <span
          className={`segment-risk-core segment-risk-${segment.max_risk_level.toLowerCase()}`}
          style={{ left: `${riskLeft}%`, width: `${riskWidth}%` }}
        />
      </div>
      <div className="segment-range-labels">
        <span>Clip {segment.start_frame}</span>
        <strong>Risk {segment.risk_start_frame}–{segment.risk_end_frame}</strong>
        <span>{segment.end_frame}</span>
      </div>
      <div className="report-segment-meta">
        <span>{segment.warning_frame_count} warning</span>
        <span>{segment.danger_frame_count} danger</span>
        <span>{segment.evidence.length} key frames</span>
      </div>
      <div className="report-segment-actions">
        <button className="button button-primary" type="button" onClick={onSeek}>Jump to risk</button>
        <button className="button button-secondary" type="button" onClick={onSelect}>Inspect evidence</button>
        <a className="button button-secondary" href={apiUrl(segment.result_url) || undefined} target="_blank" rel="noreferrer">Open clip</a>
      </div>
    </article>
  );
}

function ReportSnapshotCard({
  snapshot,
  selected,
  onReview,
}: {
  snapshot: RiskSnapshotHistory;
  selected: boolean;
  onReview: () => void;
}) {
  const frameNumber = snapshot.frame_index == null ? null : snapshot.frame_index + 1;
  const preview = apiUrl(snapshot.rgb_evidence_path || snapshot.evidence_path);
  return (
    <article className={`report-snapshot-card${selected ? " is-selected" : ""}`}>
      {preview ? (
        <img src={preview} alt={`${snapshot.risk_level} snapshot frame ${frameNumber}`} />
      ) : (
        <div className="snapshot-placeholder">No preview</div>
      )}
      <div>
        <span className={`report-risk risk-${snapshot.risk_level.toLowerCase()}`}>
          {snapshot.risk_level}
        </span>
        <strong>{frameNumber == null ? "Image assessment" : `Frame ${frameNumber}`}</strong>
        <small>{snapshot.timestamp_sec == null ? "No timestamp" : `${snapshot.timestamp_sec.toFixed(2)}s`}</small>
        {snapshot.timestamp_sec != null && (
          <button className="button button-secondary" type="button" onClick={onReview}>
            Review frame
          </button>
        )}
      </div>
    </article>
  );
}

function SnapshotEvidencePanel({
  snapshot,
  evidence,
  onOpen,
}: {
  snapshot: RiskSnapshotHistory;
  evidence: VideoFrameEvidence | null;
  onOpen: (item: VideoFrameEvidence) => void;
}) {
  const frameNumber = snapshot.frame_index == null ? null : snapshot.frame_index + 1;
  if (!evidence) {
    return (
      <aside className="report-evidence-panel">
        <p className="report-empty">Evidence files are unavailable for this snapshot.</p>
      </aside>
    );
  }
  return (
    <aside className="report-evidence-panel" aria-labelledby="report-snapshot-review-title">
      <header>
        <div>
          <p className="eyebrow">Selected risk snapshot</p>
          <h3 id="report-snapshot-review-title">Frame {frameNumber}</h3>
          <span>{evidence.timestamp_seconds.toFixed(2)}s · {snapshot.risk_level}</span>
        </div>
        <div className="report-evidence-roles"><span>Cooldown snapshot</span></div>
      </header>
      <div className="report-evidence-triptych">
        <EvidenceView title="Original" src={evidence.original_url} alt={`Original snapshot frame ${frameNumber}`} />
        <EvidenceView title="Annotated" src={evidence.rgb_url} alt={`Annotated snapshot frame ${frameNumber}`} />
        <EvidenceView title="Pseudo-BEV" src={evidence.pseudo_bev_url} alt={`Pseudo-BEV snapshot frame ${frameNumber}`} />
      </div>
      <button className="button button-secondary report-open-evidence" type="button" onClick={() => onOpen(evidence)}>
        Open full evidence viewer
      </button>
    </aside>
  );
}

function EvidencePanel({
  segment,
  evidence,
  onSelectEvidence,
  onOpen,
}: {
  segment: VideoReportSegment | null;
  evidence: VideoReportEvidence | null;
  onSelectEvidence: (frameNumber: number) => void;
  onOpen: (item: VideoFrameEvidence) => void;
}) {
  if (!segment) return null;
  if (!evidence) {
    return <div className="report-evidence-panel"><p className="report-empty">No saved evidence for this segment.</p></div>;
  }
  const roles = evidenceRoles(segment, evidence);
  return (
    <aside className="report-evidence-panel" aria-labelledby="report-evidence-title">
      <header>
        <div>
          <p className="eyebrow">Selected evidence</p>
          <h3 id="report-evidence-title">Frame {evidence.frame_number}</h3>
          <span>{evidence.timestamp_seconds.toFixed(2)}s · {evidence.risk_level}</span>
        </div>
        <div className="report-evidence-roles">
          {roles.map((role) => <span key={role}>{role}</span>)}
        </div>
      </header>
      <div className="report-evidence-selector">
        {segment.evidence.map((item) => (
          <button
            type="button"
            className={item.frame_number === evidence.frame_number ? "is-active" : ""}
            onClick={() => onSelectEvidence(item.frame_number)}
            key={item.frame_number}
          >
            <span>Frame {item.frame_number}</span>
            <strong>{item.risk_level}</strong>
          </button>
        ))}
      </div>
      <div className="report-evidence-triptych">
        <EvidenceView title="Original" src={evidence.original_url} alt={`Original evidence frame ${evidence.frame_number}`} />
        <EvidenceView title="Annotated" src={evidence.rgb_url} alt={`Annotated evidence frame ${evidence.frame_number}`} />
        <EvidenceView title="Pseudo-BEV" src={evidence.pseudo_bev_url} alt={`Pseudo-BEV evidence frame ${evidence.frame_number}`} />
      </div>
      <button className="button button-secondary report-open-evidence" type="button" onClick={() => onOpen(evidence)}>
        Open full evidence viewer
      </button>
    </aside>
  );
}

function EvidenceView({ title, src, alt }: { title: string; src: string; alt: string }) {
  return (
    <figure>
      <img src={apiUrl(src) || undefined} alt={alt} />
      <figcaption>{title}</figcaption>
    </figure>
  );
}

function nearestFrame(results: VideoFrameRiskResult[], timestamp: number) {
  let nearest = results[0] ?? null;
  for (const result of results) {
    if (result.timestamp_seconds > timestamp) break;
    nearest = result;
  }
  return nearest;
}

function timestampForFrame(
  results: VideoFrameRiskResult[],
  frameNumber: number,
  fallback: number,
) {
  return results.find((item) => item.frame_number === frameNumber)?.timestamp_seconds ?? fallback;
}

function percentage(value: number, total: number) {
  return total > 0 ? (value * 100 / total).toFixed(1) : "0.0";
}

function evidenceRoles(segment: VideoReportSegment, evidence: VideoReportEvidence) {
  const roles: string[] = [];
  if (evidence.frame_number === segment.risk_start_frame) roles.push("First risk");
  const peak = segment.evidence.reduce<VideoReportEvidence | null>((current, item) => {
    if (!current || severity(item.risk_level) > severity(current.risk_level)) return item;
    return current;
  }, null);
  if (evidence.frame_number === peak?.frame_number) roles.push("Peak risk");
  if (evidence.frame_number === segment.risk_end_frame) roles.push("Last risk");
  return roles.length > 0 ? roles : ["Representative"];
}

function mergeTimelineEvidence(
  report: VideoReport | null,
  snapshots: RiskSnapshotHistory[],
): VideoReportEvidence[] {
  const byFrame = new Map<number, VideoReportEvidence>();
  for (const item of report?.risk_segments.flatMap((segment) => segment.evidence) ?? []) {
    byFrame.set(item.frame_number, item);
  }
  for (const snapshot of snapshots) {
    if (snapshot.frame_index == null || snapshot.timestamp_sec == null) continue;
    const originalUrl = snapshot.evidence_path;
    const rgbUrl = snapshot.rgb_evidence_path;
    const pseudoBevUrl = snapshot.pseudo_bev_path;
    if (!originalUrl || !rgbUrl || !pseudoBevUrl) continue;
    const frameNumber = snapshot.frame_index + 1;
    byFrame.set(frameNumber, {
      frame_number: frameNumber,
      timestamp_seconds: snapshot.timestamp_sec,
      risk_level: snapshot.risk_level,
      original_url: originalUrl,
      rgb_url: rgbUrl,
      pseudo_bev_url: pseudoBevUrl,
    });
  }
  return [...byFrame.values()].sort((left, right) => left.frame_number - right.frame_number);
}

function snapshotToEvidence(snapshot: RiskSnapshotHistory): VideoFrameEvidence | null {
  if (
    snapshot.frame_index == null
    || snapshot.timestamp_sec == null
    || !snapshot.evidence_path
    || !snapshot.rgb_evidence_path
    || !snapshot.pseudo_bev_path
  ) {
    return null;
  }
  return {
    frame_number: snapshot.frame_index + 1,
    timestamp_seconds: snapshot.timestamp_sec,
    risk_level: snapshot.risk_level,
    original_url: snapshot.evidence_path,
    rgb_url: snapshot.rgb_evidence_path,
    pseudo_bev_url: snapshot.pseudo_bev_path,
  };
}

function severity(level: RiskLevel) {
  return level === "DANGER" ? 2 : level === "WARNING" ? 1 : 0;
}
