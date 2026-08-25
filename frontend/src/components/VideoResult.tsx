import { useState } from "react";

import { apiUrl } from "../services/api";
import type { VideoFrameEvidence, VideoJob } from "../types/detection";
import VideoFrameEvidenceModal from "./VideoFrameEvidenceModal";

export default function VideoResult({ job }: { job: VideoJob }) {
  const [selectedEvidence, setSelectedEvidence] = useState<VideoFrameEvidence[] | null>(null);
  const resultUrl = apiUrl(job.result_url);
  const downloadUrl = apiUrl(job.download_url);
  if (!resultUrl) return null;
  return (
    <section className="video-result" aria-labelledby="video-result-title">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Processing complete</p>
          <h2 id="video-result-title">Annotated result video</h2>
        </div>
        <p>{job.current_frame} frames assessed at {job.processing_fps.toFixed(1)} processing FPS.</p>
      </div>
      {!job.browser_playback_compatible && (
        <div className="playback-warning" role="status">
          <strong>Browser playback compatibility is limited</strong>
          <span>{job.playback_warning || `Output codec ${job.output_codec} may not be supported by this browser.`}</span>
        </div>
      )}
      <video controls preload="metadata" src={resultUrl}>
        Your browser does not support HTML5 video playback.
      </video>
      <div className="video-result-actions">
        {downloadUrl && (
          <a className="button button-secondary" href={downloadUrl}>
            Download video
          </a>
        )}
        {job.report_url && (
          <a
            className="button button-secondary"
            href={`/?report=${encodeURIComponent(job.job_id)}`}
            target="_blank"
            rel="noreferrer"
          >
            View report
          </a>
        )}
      </div>
      <div className="risk-segments-heading">
        <div>
          <p className="eyebrow">Frame-level risk segments</p>
          <h3>Saved WARNING / DANGER clips</h3>
        </div>
        <span>{job.risk_segments.length} clips</span>
      </div>
      {job.risk_segments.length > 0 ? (
        <div className="risk-segment-grid">
          {job.risk_segments.map((segment) => (
            <article className="risk-segment-card" key={segment.segment_id}>
              <div className={`segment-level segment-${segment.max_risk_level.toLowerCase()}`}>
                {segment.max_risk_level}
              </div>
              {!segment.browser_playback_compatible && (
                <span className="segment-codec-warning" title={segment.playback_warning || undefined}>
                  {segment.output_codec} fallback
                </span>
              )}
              <video controls preload="metadata" src={apiUrl(segment.result_url) || undefined}>
                Your browser does not support HTML5 video playback.
              </video>
              <div className="segment-metadata">
                <strong>{segment.start_seconds.toFixed(1)}s - {segment.end_seconds.toFixed(1)}s</strong>
                <span>Frames {segment.start_frame} - {segment.end_frame}</span>
                <span>{segment.warning_frame_count} WARNING frames</span>
                <span>{segment.danger_frame_count} DANGER frames</span>
                <span>{segment.frame_evidence.length} evidence frames</span>
              </div>
              {segment.frame_evidence.length > 0 && (
                <button
                  type="button"
                  className="button button-secondary segment-evidence-button"
                  onClick={() => setSelectedEvidence(segment.frame_evidence)}
                >
                  View frame evidence
                </button>
              )}
            </article>
          ))}
        </div>
      ) : (
        <p className="empty-segments">No WARNING or DANGER frames were classified.</p>
      )}
      <p className="segment-disclaimer">
        Clips represent contiguous frame-level classifications with context padding. They are not tracked safety events.
      </p>
      {selectedEvidence && (
        <VideoFrameEvidenceModal
          evidence={selectedEvidence}
          onClose={() => setSelectedEvidence(null)}
        />
      )}
    </section>
  );
}
