import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import AssessmentPanel from "../components/AssessmentPanel";
import EvidenceViewer from "../components/EvidenceViewer";
import ImagePreview from "../components/ImagePreview";
import MediaUploader, { type MediaType } from "../components/MediaUploader";
import VideoProcessingView from "../components/VideoProcessingView";
import { analyzeImage, ApiError, uploadVideo } from "../services/api";
import type {
  ImageDetectionResponse,
  VideoJobCreated,
  VideoJobStatus,
} from "../types/detection";

type PageState = "idle" | "selected" | "processing" | "success" | "error";

const STATE_LABELS: Record<PageState, string> = {
  idle: "Waiting for media",
  selected: "Ready to process",
  processing: "Pipeline in progress",
  success: "Processing complete",
  error: "Action required",
};

export default function DetectionPage() {
  const [file, setFile] = useState<File | null>(null);
  const [mediaType, setMediaType] = useState<MediaType | null>(null);
  const [state, setState] = useState<PageState>("idle");
  const [imageResult, setImageResult] = useState<ImageDetectionResponse | null>(null);
  const [videoJob, setVideoJob] = useState<VideoJobCreated | null>(null);
  const [error, setError] = useState<string | null>(null);
  const requestRef = useRef<AbortController | null>(null);
  const previewUrl = useMemo(
    () => (file && mediaType === "image" ? URL.createObjectURL(file) : null),
    [file, mediaType],
  );

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  useEffect(() => () => requestRef.current?.abort(), []);

  const reset = useCallback(() => {
    requestRef.current?.abort();
    requestRef.current = null;
    setFile(null);
    setMediaType(null);
    setImageResult(null);
    setVideoJob(null);
    setError(null);
    setState("idle");
  }, []);

  const selectMedia = useCallback((selected: File, detected: MediaType) => {
    requestRef.current?.abort();
    setFile(selected);
    setMediaType(detected);
    setImageResult(null);
    setVideoJob(null);
    setError(null);
    setState("selected");
  }, []);

  const processMedia = useCallback(async () => {
    if (!file || !mediaType || state === "processing") return;
    const controller = new AbortController();
    requestRef.current = controller;
    setImageResult(null);
    setVideoJob(null);
    setError(null);
    setState("processing");

    try {
      if (mediaType === "image") {
        setImageResult(await analyzeImage(file, controller.signal));
        setState("success");
      } else {
        setVideoJob(await uploadVideo(file, controller.signal));
      }
    } catch (caught: unknown) {
      if (caught instanceof DOMException && caught.name === "AbortError") return;
      setError(
        caught instanceof ApiError
          ? caught.message
          : `The ${mediaType} processing request was interrupted.`,
      );
      setState("error");
    } finally {
      if (requestRef.current === controller) requestRef.current = null;
    }
  }, [file, mediaType, state]);

  const handleVideoStatus = useCallback((status: VideoJobStatus) => {
    if (status === "completed") setState("success");
    if (status === "failed") setState("error");
  }, []);

  const handleInvalid = (message: string) => {
    setError(message);
    setState("error");
  };

  const hasSelection = file !== null && mediaType !== null;
  const actionLabel = mediaType ? mediaActionLabel(mediaType, state) : "";

  return (
    <div className="app-shell">
      <header className="site-header">
        <a className="brand" href="#top" aria-label="Crane Load Warning home">
          <span className="brand-mark" aria-hidden="true">
            <svg viewBox="0 0 32 32" focusable="false">
              <path d="M7 27h8M9 27V7m-3 0h22M9 7l6-4 4 4M9 11h13M22 7v9m-2 0h4m-2 0v4m-3 0h6v6h-6zM9 12l6 15M15 12 9 21" />
            </svg>
          </span>
          <span>
            <strong>Crane Load Warning</strong>
            <small>Smart media safety assessment</small>
          </span>
        </a>
        <div className="system-status">
          <span aria-hidden="true" /> Local analysis mode
        </div>
      </header>

      <main id="top">
        <section className="hero">
          <div className="hero-copy">
            <p className="eyebrow">Crane suspended-load safety</p>
            <h1>One upload.<br />The right safety flow.</h1>
            <p className="hero-intro">
              Select one worksite image or video. The system identifies the media,
              runs the appropriate safety pipeline, and presents a dedicated result.
            </p>
          </div>
          <div className="workflow-strip" aria-label="Analysis workflow">
            <span className={hasSelection ? "is-complete" : "is-active"}>01 Upload</span>
            <i aria-hidden="true" />
            <span className={state === "selected" || state === "processing" ? "is-active" : state === "success" ? "is-complete" : ""}>
              02 Process
            </span>
            <i aria-hidden="true" />
            <span className={state === "success" ? "is-active" : ""}>03 Review</span>
          </div>
        </section>

        <section className="workspace smart-workspace" aria-labelledby="upload-title">
          <div className="workspace-header">
            <div>
              <p className="step-number">01 / Smart media input</p>
              <h2 id="upload-title">
                {hasSelection ? "Detected media" : "Upload an image or video"}
              </h2>
            </div>
            <span className={`state-pill state-${state}`}>
              <span aria-hidden="true" /> {STATE_LABELS[state]}
            </span>
          </div>

          {!hasSelection ? (
            <MediaUploader
              disabled={state === "processing"}
              onSelect={selectMedia}
              onInvalid={handleInvalid}
            />
          ) : (
            <div className="selected-media">
              <div className="selected-media-info">
                <span className={`media-type-badge media-${mediaType}`}>{mediaType}</span>
                <div>
                  <strong title={file.name}>{file.name}</strong>
                  <span>{formatBytes(file.size)} - automatically routed to {mediaType} processing</span>
                </div>
              </div>
              <div className="selected-media-actions">
                <button className="button button-secondary" type="button" onClick={reset}>
                  Choose another file
                </button>
                <button
                  className="button button-primary"
                  type="button"
                  disabled={state === "processing" || state === "success"}
                  onClick={() => void processMedia()}
                >
                  {actionLabel}
                </button>
              </div>
            </div>
          )}

          {mediaType === "image" && file && previewUrl && (
            <div className="smart-image-preview">
              <ImagePreview file={file} previewUrl={previewUrl} />
            </div>
          )}

          {error && (
            <div className="error-banner" role="alert">
              <span aria-hidden="true">!</span>
              <div>
                <strong>Unable to complete this media request</strong>
                <p>{error}</p>
              </div>
            </div>
          )}
        </section>

        {state === "processing" && mediaType === "image" && (
          <section className="processing-panel" aria-live="polite">
            <div className="scan-line" aria-hidden="true" />
            <div>
              <p className="eyebrow">Image pipeline active</p>
              <h2>Building the safety assessment</h2>
              <p>Vision, relative geometry, risk evaluation, and evidence rendering are running.</p>
            </div>
            <div className="processing-stages" aria-hidden="true">
              <span>Vision</span><span>Geometry</span><span>Risk</span><span>Evidence</span>
            </div>
          </section>
        )}

        {state === "processing" && mediaType === "video" && !videoJob && (
          <section className="processing-panel" aria-live="polite">
            <span className="spinner" aria-hidden="true" />
            <div><p className="eyebrow">Uploading video</p><h2>Creating background job</h2></div>
          </section>
        )}

        {imageResult && state === "success" && (
          <div className="results-wrap">
            <AssessmentPanel result={imageResult} />
            <EvidenceViewer evidence={imageResult.evidence} />
          </div>
        )}

        {videoJob && (
          <VideoProcessingView
            created={videoJob}
            onStatusChange={handleVideoStatus}
          />
        )}
      </main>

      <footer>
        <span>Crane Load Warning - Suspended-load decision support</span>
        <span>AI-assisted crane load safety monitoring</span>
      </footer>
    </div>
  );
}

function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function mediaActionLabel(mediaType: MediaType, state: PageState): string {
  if (state === "processing") {
    return mediaType === "image" ? "Analyzing image..." : "Uploading video...";
  }
  if (state === "success") return "Processing complete";
  if (state === "error") {
    return mediaType === "image" ? "Retry image analysis" : "Retry video processing";
  }
  return mediaType === "image" ? "Run image analysis" : "Upload and process video";
}
