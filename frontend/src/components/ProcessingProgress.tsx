import RiskBadge from "./RiskBadge";
import type { VideoJob } from "../types/detection";

export default function ProcessingProgress({ job }: { job: VideoJob }) {
  return (
    <div className="video-progress">
      <div className="progress-heading">
        <div>
          <p className="eyebrow">{job.status === "queued" ? "Queued" : "Pipeline active"}</p>
          <h2>{job.progress.toFixed(1)}% processed</h2>
        </div>
        {job.current_risk_level && <RiskBadge level={job.current_risk_level} />}
      </div>
      <div
        className="progress-track"
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={job.progress}
      >
        <span style={{ width: `${job.progress}%` }} />
      </div>
      <div className="video-stat-grid">
        <div><span>Frame</span><strong>{job.current_frame} / {job.total_frames || "?"}</strong></div>
        <div><span>Processing FPS</span><strong>{job.processing_fps.toFixed(1)}</strong></div>
        <div><span>SAFE frames</span><strong>{job.safe_frame_count}</strong></div>
        <div><span>WARNING frames</span><strong>{job.warning_frame_count}</strong></div>
        <div><span>DANGER frames</span><strong>{job.danger_frame_count}</strong></div>
      </div>
    </div>
  );
}
