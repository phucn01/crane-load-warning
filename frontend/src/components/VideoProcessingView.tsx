import { useEffect, useState } from "react";

import { ApiError, getVideoJob } from "../services/api";
import type { VideoJob, VideoJobCreated, VideoJobStatus } from "../types/detection";
import LivePreview from "./LivePreview";
import ProcessingProgress from "./ProcessingProgress";
import VideoResult from "./VideoResult";

const POLL_INTERVAL_MS = 750;

interface Props {
  created: VideoJobCreated;
  onStatusChange?: (status: VideoJobStatus) => void;
}

export default function VideoProcessingView({ created, onStatusChange }: Props) {
  const [job, setJob] = useState<VideoJob | null>(null);
  const [pollError, setPollError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    let timeoutId: number | undefined;
    let controller: AbortController | null = null;

    const poll = async () => {
      controller = new AbortController();
      try {
        const next = await getVideoJob(created.job_id, controller.signal);
        if (!active) return;
        setJob(next);
        setPollError(null);
        onStatusChange?.(next.status);
        if (next.status !== "completed" && next.status !== "failed") {
          timeoutId = window.setTimeout(poll, POLL_INTERVAL_MS);
        }
      } catch (error: unknown) {
        if (!active || (error instanceof DOMException && error.name === "AbortError")) return;
        setPollError(error instanceof ApiError ? error.message : "Could not refresh video status.");
        timeoutId = window.setTimeout(poll, POLL_INTERVAL_MS);
      }
    };

    void poll();
    return () => {
      active = false;
      controller?.abort();
      if (timeoutId !== undefined) window.clearTimeout(timeoutId);
    };
  }, [created.job_id, onStatusChange]);

  if (!job) {
    return <section className="processing-panel"><span className="spinner" /> Creating processing job...</section>;
  }
  if (job.status === "failed") {
    return (
      <section className="error-banner video-terminal" role="alert">
        <span aria-hidden="true">!</span>
        <div><strong>Video processing failed</strong><p>{job.error || "The video could not be processed."}</p></div>
      </section>
    );
  }
  if (job.status === "completed") return <VideoResult job={job} />;

  return (
    <section className="video-processing" aria-live="polite">
      <LivePreview streamUrl={created.stream_url} />
      <ProcessingProgress job={job} />
      {pollError && <p className="poll-warning">{pollError}</p>}
    </section>
  );
}
