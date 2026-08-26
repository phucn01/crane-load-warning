import { useMemo, useRef, useState } from "react";

import { apiUrl } from "../services/api";
import type {
  VideoFrameEvidence,
  VideoFrameRiskResult,
  VideoJob,
} from "../types/detection";
import RiskTimeline from "./RiskTimeline";
import VideoFrameEvidenceModal from "./VideoFrameEvidenceModal";

interface Props {
  job: VideoJob;
  frameResults: VideoFrameRiskResult[];
}

export default function VideoResult({ job, frameResults }: Props) {
  const [selectedEvidence, setSelectedEvidence] = useState<VideoFrameEvidence[] | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const videoRef = useRef<HTMLVideoElement>(null);
  const resultUrl = apiUrl(job.result_url);
  const downloadUrl = apiUrl(job.download_url);
  const evidence = useMemo(() => uniqueEvidence(job), [job]);

  const seekVideo = (timestampSeconds: number) => {
    const video = videoRef.current;
    if (!video) return;
    video.currentTime = Math.max(0, timestampSeconds);
    setCurrentTime(video.currentTime);
    video.focus();
  };

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
      <video
        ref={videoRef}
        className="video-result-player"
        controls
        preload="metadata"
        src={resultUrl}
        onTimeUpdate={(event) => setCurrentTime(event.currentTarget.currentTime)}
        onLoadedMetadata={(event) => setDuration(event.currentTarget.duration)}
      >
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
          >
            View report
          </a>
        )}
      </div>

      <RiskTimeline
        results={frameResults}
        evidence={evidence}
        totalFrames={job.total_frames || job.current_frame}
        currentTime={currentTime}
        duration={duration}
        onSeek={seekVideo}
      />

      {false && (<>
      <div className="risk-segments-heading">
        <div>
          <p className="eyebrow">Review queue</p>
          <h3>Saved WARNING / DANGER clips</h3>
        </div>
        <span>{job.risk_segments.length} clips</span>
      </div>
      {job.risk_segments.length > 0 ? (
        <div className="risk-segment-grid">
          {job.risk_segments.map((segment, index) => {
            const clipFrames = Math.max(1, segment.end_frame - segment.start_frame + 1);
            const riskLeft = (segment.risk_start_frame - segment.start_frame) * 100 / clipFrames;
            const riskWidth = (segment.risk_end_frame - segment.risk_start_frame + 1) * 100 / clipFrames;
            const riskTimestamp = timestampForFrame(
              frameResults,
              segment.risk_start_frame,
              segment.start_seconds,
            );
            return (
              <article className="risk-segment-card" key={segment.segment_id}>
                <header className="segment-card-heading">
                  <div>
                    <span>Segment {String(index + 1).padStart(2, "0")}</span>
                    <strong>{segment.start_seconds.toFixed(1)}s – {segment.end_seconds.toFixed(1)}s</strong>
                  </div>
                  <span className={`segment-level segment-${segment.max_risk_level.toLowerCase()}`}>
                    {segment.max_risk_level}
                  </span>
                </header>
                <div className="segment-range" aria-label={`Risk frames ${segment.risk_start_frame} to ${segment.risk_end_frame} inside clip frames ${segment.start_frame} to ${segment.end_frame}`}>
                  <span
                    className={`segment-risk-core segment-risk-${segment.max_risk_level.toLowerCase()}`}
                    style={{ left: `${riskLeft}%`, width: `${riskWidth}%` }}
                  />
                </div>
                <div className="segment-range-labels">
                  <span>Context {segment.start_frame}</span>
                  <strong>Risk {segment.risk_start_frame}–{segment.risk_end_frame}</strong>
                  <span>{segment.end_frame}</span>
                </div>
                {!segment.browser_playback_compatible && (
                  <p className="segment-codec-warning" title={segment.playback_warning || undefined}>
                    {segment.output_codec} playback fallback
                  </p>
                )}
                <div className="segment-metadata">
                  <span>{segment.warning_frame_count} WARNING frames</span>
                  <span>{segment.danger_frame_count} DANGER frames</span>
                  <span>{segment.frame_evidence.length} evidence frames</span>
                </div>
                <div className="segment-actions">
                  <button
                    type="button"
                    className="button button-primary"
                    onClick={() => seekVideo(riskTimestamp)}
                  >
                    Jump to risk
                  </button>
                  <a
                    className="button button-secondary"
                    href={apiUrl(segment.result_url) || undefined}
                    target="_blank"
                    rel="noreferrer"
                  >
                    Open clip
                  </a>
                  {segment.frame_evidence.length > 0 && (
                    <button
                      type="button"
                      className="button button-secondary segment-evidence-button"
                      onClick={() => setSelectedEvidence(segment.frame_evidence)}
                    >
                      View frame evidence
                    </button>
                  )}
                </div>
              </article>
            );
          })}
        </div>
      ) : (
        <p className="empty-segments">No WARNING or DANGER frames were classified.</p>
      )}
      <p className="segment-disclaimer">
        Colored timeline ranges are independent frame classifications. Saved clips include dimmed context padding and are not tracked safety events.
      </p>
      </>)}
      {selectedEvidence && (
        <VideoFrameEvidenceModal
          evidence={selectedEvidence}
          onClose={() => setSelectedEvidence(null)}
        />
      )}
    </section>
  );
}

function uniqueEvidence(job: VideoJob): VideoFrameEvidence[] {
  const byFrame = new Map<number, VideoFrameEvidence>();
  for (const segment of job.risk_segments) {
    for (const item of segment.frame_evidence) byFrame.set(item.frame_number, item);
  }
  return [...byFrame.values()].sort((left, right) => left.frame_number - right.frame_number);
}

function timestampForFrame(
  results: VideoFrameRiskResult[],
  frameNumber: number,
  fallback: number,
): number {
  return results.find((item) => item.frame_number === frameNumber)?.timestamp_seconds ?? fallback;
}
