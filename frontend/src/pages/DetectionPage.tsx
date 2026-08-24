import { useEffect, useMemo, useRef, useState } from "react";

import AssessmentPanel from "../components/AssessmentPanel";
import EvidenceViewer from "../components/EvidenceViewer";
import FileUploader from "../components/FileUploader";
import ImagePreview from "../components/ImagePreview";
import { analyzeImage, ApiError } from "../services/api";
import type { AnalysisState, ImageDetectionResponse } from "../types/detection";

const STATE_LABELS: Record<AnalysisState, string> = {
  idle: "Waiting for image",
  selected: "Ready to analyze",
  processing: "Analysis in progress",
  success: "Assessment complete",
  error: "Action required",
};

export default function DetectionPage() {
  const [file, setFile] = useState<File | null>(null);
  const [state, setState] = useState<AnalysisState>("idle");
  const [result, setResult] = useState<ImageDetectionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const previewUrl = useMemo(() => (file ? URL.createObjectURL(file) : null), [file]);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  useEffect(() => {
    return () => abortRef.current?.abort();
  }, []);

  const handleSelect = (selected: File) => {
    setFile(selected);
    setResult(null);
    setError(null);
    setState("selected");
  };

  const handleInvalid = (message: string) => {
    setError(message);
    setState("error");
  };

  const runAnalysis = async () => {
    if (!file || state === "processing") return;
    const controller = new AbortController();
    abortRef.current = controller;
    setState("processing");
    setError(null);
    setResult(null);

    try {
      const response = await analyzeImage(file, controller.signal);
      setResult(response);
      setState("success");
    } catch (caught: unknown) {
      if (caught instanceof DOMException && caught.name === "AbortError") return;
      setError(
        caught instanceof ApiError
          ? caught.message
          : "An unexpected error interrupted the assessment.",
      );
      setState("error");
    } finally {
      abortRef.current = null;
    }
  };

  const isProcessing = state === "processing";

  return (
    <div className="app-shell">
      <header className="site-header">
        <a className="brand" href="#top" aria-label="Crane Load Warning home">
          <span className="brand-mark" aria-hidden="true">🏗</span>
          <span>
            <strong>Crane Load Warning</strong>
            <small>Image safety assessment</small>
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
            <h1>Keep people clear<br />of suspended loads.</h1>
            <p className="hero-intro">
              Analyze one worksite image to identify people, hanging loads, and
              ropes—then review their relative position against the load safety zone.
            </p>
          </div>
          <div className="workflow-strip" aria-label="Analysis workflow">
            <span className={state !== "idle" ? "is-complete" : "is-active"}>01 Select</span>
            <i aria-hidden="true" />
            <span className={isProcessing ? "is-active" : result ? "is-complete" : ""}>02 Analyze</span>
            <i aria-hidden="true" />
            <span className={result ? "is-active" : ""}>03 Review</span>
          </div>
        </section>

        <section className="workspace" aria-labelledby="upload-title">
          <div className="workspace-header">
            <div>
              <p className="step-number">01 / Image input</p>
              <h2 id="upload-title">Upload a crane worksite image</h2>
            </div>
            <span className={`state-pill state-${state}`}>
              <span aria-hidden="true" /> {STATE_LABELS[state]}
            </span>
          </div>

          <div className="input-grid">
            <FileUploader
              disabled={isProcessing}
              selectedFile={file}
              onSelect={handleSelect}
              onInvalid={handleInvalid}
            />
            {file && previewUrl ? (
              <ImagePreview file={file} previewUrl={previewUrl} />
            ) : (
              <div className="preview-placeholder">
                <span aria-hidden="true">◎</span>
                <p>Your local image preview will appear here.</p>
              </div>
            )}
          </div>

          {error && (
            <div className="error-banner" role="alert">
              <span aria-hidden="true">!</span>
              <div>
                <strong>Unable to complete this step</strong>
                <p>{error}</p>
              </div>
            </div>
          )}

          <div className="action-row">
            <p>Decision support only—always follow the site safety procedure.</p>
            <button
              type="button"
              className="button button-primary"
              disabled={!file || isProcessing}
              onClick={runAnalysis}
            >
              {isProcessing ? (
                <><span className="spinner" aria-hidden="true" /> Processing image…</>
              ) : (
                <>Run analysis <span aria-hidden="true">→</span></>
              )}
            </button>
          </div>
        </section>

        {isProcessing && (
          <section className="processing-panel" aria-live="polite">
            <div className="scan-line" aria-hidden="true" />
            <div>
              <p className="eyebrow">Pipeline active</p>
              <h2>Building the safety assessment</h2>
              <p>Vision, relative geometry, risk evaluation, and evidence rendering are running.</p>
            </div>
            <div className="processing-stages" aria-hidden="true">
              <span>Vision</span><span>Geometry</span><span>Risk</span><span>Evidence</span>
            </div>
          </section>
        )}

        {result && state === "success" && (
          <div className="results-wrap">
            <AssessmentPanel result={result} />
            <EvidenceViewer evidence={result.evidence} />
          </div>
        )}
      </main>

      <footer>
        <span>Crane Load Warning · Suspended-load decision support</span>
        <span>No tracking · No crane control · Relative geometry is non-metric</span>
      </footer>
    </div>
  );
}
