import type {
  RiskLevel,
  VideoFrameEvidence,
  VideoFrameRiskResult,
} from "../types/detection";

export interface RiskRun {
  startFrame: number;
  endFrame: number;
  startSeconds: number;
  endSeconds: number;
  riskLevel: RiskLevel;
}

interface Props {
  results: VideoFrameRiskResult[];
  evidence: VideoFrameEvidence[];
  totalFrames: number;
  currentTime: number;
  duration: number;
  onSeek: (timestampSeconds: number) => void;
  showMarkers?: boolean;
}

const LEVELS: RiskLevel[] = ["SAFE", "WARNING", "DANGER"];

export function buildRiskRuns(results: VideoFrameRiskResult[]): RiskRun[] {
  const runs: RiskRun[] = [];
  for (const result of results.filter((item): item is VideoFrameRiskResult & { risk_level: RiskLevel } => item.risk_level !== null)) {
    const current = runs[runs.length - 1];
    if (
      current
      && current.riskLevel === result.risk_level
      && current.endFrame + 1 === result.frame_number
    ) {
      current.endFrame = result.frame_number;
      current.endSeconds = result.timestamp_seconds;
      continue;
    }
    runs.push({
      startFrame: result.frame_number,
      endFrame: result.frame_number,
      startSeconds: result.timestamp_seconds,
      endSeconds: result.timestamp_seconds,
      riskLevel: result.risk_level,
    });
  }
  return runs;
}

export default function RiskTimeline({
  results,
  evidence,
  totalFrames,
  currentTime,
  duration,
  onSeek,
  showMarkers = true,
}: Props) {
  const runs = buildRiskRuns(results);
  const frameExtent = Math.max(totalFrames, results.at(-1)?.frame_number ?? 0, 1);
  const counts = LEVELS.map((level) => ({
    level,
    count: results.filter((item) => item.risk_level === level).length,
  }));
  const activeFrame = frameAtTime(results, currentTime);
  const skippedCount = results.filter((item) => item.risk_level === null).length;
  const playheadPercent = duration > 0
    ? Math.min(100, Math.max(0, currentTime * 100 / duration))
    : 0;
  const timelineDuration = duration > 0
    ? duration
    : results.at(-1)?.timestamp_seconds ?? 0;

  return (
    <section className="risk-timeline" aria-labelledby="risk-timeline-title">
      <div className="risk-timeline-heading">
        <div>
          <p className="eyebrow">Frame-level assessment</p>
          <h3 id="risk-timeline-title">Risk timeline</h3>
        </div>
        <div className="risk-timeline-current" aria-live="polite">
          <span>Current frame</span>
          <strong>{activeFrame?.frame_number ?? 0}</strong>
          <em className={`timeline-current-${activeFrame?.risk_level?.toLowerCase() ?? "idle"}`}>
            {activeFrame?.risk_level ?? activeFrame?.assessment_status ?? "READY"}
          </em>
        </div>
      </div>

      <div className="risk-timeline-summary" aria-label="Frame risk totals">
        {counts.map(({ level, count }) => (
          <span key={level} className={`timeline-summary-${level.toLowerCase()}`}>
            {showMarkers && <i aria-hidden="true" />} {level} <strong>{count}</strong>
          </span>
        ))}
        <span className="timeline-frame-total">{results.length} / {frameExtent} frames mapped</span>
        {skippedCount > 0 && <span className="timeline-frame-skipped">{skippedCount} skipped</span>}
      </div>

      <div className="risk-timeline-shell">
        <div
          className="risk-timeline-track"
          role="group"
          aria-label="Video risk timeline. Select a colored range to seek the video."
        >
          {runs.map((run) => {
            const left = (run.startFrame - 1) * 100 / frameExtent;
            const width = (run.endFrame - run.startFrame + 1) * 100 / frameExtent;
            const label = `${run.riskLevel}: frames ${run.startFrame} to ${run.endFrame}, ${formatTime(run.startSeconds)} to ${formatTime(run.endSeconds)}`;
            return (
              <button
                type="button"
                className={`risk-run risk-run-${run.riskLevel.toLowerCase()}`}
                style={{ left: `${left}%`, width: `${width}%` }}
                aria-label={label}
                title={label}
                onClick={() => onSeek(run.startSeconds)}
                key={`${run.startFrame}-${run.endFrame}-${run.riskLevel}`}
              />
            );
          })}
          {showMarkers && evidence.map((item) => {
            const left = (item.frame_number - 0.5) * 100 / frameExtent;
            return (
              <button
                type="button"
                className={`risk-evidence-marker evidence-${item.risk_level.toLowerCase()}`}
                style={{ left: `${left}%` }}
                aria-label={`Evidence frame ${item.frame_number}, ${item.risk_level}, ${formatTime(item.timestamp_seconds)}`}
                title={`Evidence frame ${item.frame_number}`}
                onClick={() => onSeek(item.timestamp_seconds)}
                key={`evidence-${item.frame_number}`}
              />
            );
          })}
          <span
            className="risk-timeline-playhead"
            style={{ left: `${playheadPercent}%` }}
            aria-hidden="true"
          />
        </div>
        <div className="risk-timeline-ticks" aria-hidden="true">
          {[0, 0.25, 0.5, 0.75, 1].map((fraction) => (
            <span key={fraction}>{formatTime(timelineDuration * fraction)}</span>
          ))}
        </div>
      </div>

      <p className="risk-timeline-help">
        {showMarkers
          ? "Select a colored range or evidence marker to jump to that moment in the processed video."
          : "Select a colored range to jump to that moment in the processed video."}
      </p>
    </section>
  );
}

function frameAtTime(
  results: VideoFrameRiskResult[],
  currentTime: number,
): VideoFrameRiskResult | null {
  let active: VideoFrameRiskResult | null = results[0] ?? null;
  for (const result of results) {
    if (result.timestamp_seconds > currentTime) break;
    active = result;
  }
  return active;
}

function formatTime(seconds: number): string {
  const safeSeconds = Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
  const minutes = Math.floor(safeSeconds / 60);
  const remainder = safeSeconds - minutes * 60;
  return `${minutes}:${remainder.toFixed(1).padStart(4, "0")}`;
}
