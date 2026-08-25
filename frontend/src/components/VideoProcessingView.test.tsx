import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { VideoJob, VideoJobCreated } from "../types/detection";
import VideoProcessingView from "./VideoProcessingView";

const created: VideoJobCreated = {
  job_id: "job-1",
  status: "queued",
  status_url: "/api/v1/jobs/job-1",
  stream_url: "/api/v1/jobs/job-1/stream",
  result_url: "/api/v1/jobs/job-1/result",
};

function job(status: VideoJob["status"]): VideoJob {
  return {
    job_id: "job-1",
    status,
    input_path: "input.mp4",
    output_path: "output.mp4",
    current_frame: status === "queued" ? 0 : 47,
    total_frames: 100,
    progress: status === "completed" ? 100 : 47,
    processing_fps: 8.6,
    elapsed_seconds: 5.5,
    current_risk_level: "WARNING",
    max_risk_level: "WARNING",
    safe_frame_count: 30,
    warning_frame_count: 17,
    danger_frame_count: 0,
    error: status === "failed" ? "fixture failure" : null,
    created_at: "2026-08-25T00:00:00Z",
    started_at: "2026-08-25T00:00:01Z",
    completed_at: status === "completed" || status === "failed" ? "2026-08-25T00:00:06Z" : null,
    stream_url: created.stream_url,
    result_url: status === "completed" ? created.result_url : null,
    download_url: status === "completed" ? "/api/v1/jobs/job-1/download" : null,
    report_url: status === "completed" ? "/api/v1/jobs/job-1/report" : null,
    summary: null,
    risk_segments: status === "completed" ? [
      {
        segment_id: "segment-1",
        start_frame: 20,
        end_frame: 40,
        risk_start_frame: 24,
        risk_end_frame: 36,
        start_seconds: 2,
        end_seconds: 4,
        max_risk_level: "DANGER",
        warning_frame_count: 5,
        danger_frame_count: 8,
        frame_evidence: [
          {
            frame_number: 24,
            timestamp_seconds: 2.4,
            risk_level: "WARNING",
            original_url: "/api/v1/jobs/job-1/segments/segment-1/evidence/24/original",
            rgb_url: "/api/v1/jobs/job-1/segments/segment-1/evidence/24/rgb",
            pseudo_bev_url: "/api/v1/jobs/job-1/segments/segment-1/evidence/24/bev",
          },
          {
            frame_number: 30,
            timestamp_seconds: 3,
            risk_level: "DANGER",
            original_url: "/api/v1/jobs/job-1/segments/segment-1/evidence/30/original",
            rgb_url: "/api/v1/jobs/job-1/segments/segment-1/evidence/30/rgb",
            pseudo_bev_url: "/api/v1/jobs/job-1/segments/segment-1/evidence/30/bev",
          },
        ],
        result_url: "/api/v1/jobs/job-1/segments/segment-1",
        output_codec: "h264",
        browser_playback_compatible: true,
        playback_warning: null,
      },
    ] : [],
    output_codec: "h264",
    browser_playback_compatible: true,
    playback_warning: null,
  };
}

function jsonResponse(payload: VideoJob): Response {
  return new Response(JSON.stringify(payload), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

async function flushPromises() {
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
  });
}

describe("VideoProcessingView", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("shows processing progress and transitions to completed playback", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse(job("processing")))
      .mockResolvedValueOnce(jsonResponse(job("completed")));
    vi.stubGlobal("fetch", fetchMock);
    render(<VideoProcessingView created={created} />);

    await flushPromises();
    expect(screen.getByText("47.0% processed")).toBeInTheDocument();
    expect(screen.getByText("WARNING frames")).toBeInTheDocument();
    expect(screen.queryByText(/Danger Events/i)).not.toBeInTheDocument();

    await act(async () => vi.advanceTimersByTimeAsync(750));
    expect(screen.getByText("Annotated result video")).toBeInTheDocument();
    expect(document.querySelector("video")).toHaveAttribute(
      "src",
      "http://localhost:8000/api/v1/jobs/job-1/result",
    );
    expect(screen.getByRole("link", { name: "View report" })).toHaveAttribute(
      "href",
      "/?report=job-1",
    );
    expect(screen.getByRole("link", { name: "Download video" })).toHaveAttribute(
      "href",
      "http://localhost:8000/api/v1/jobs/job-1/download",
    );
    expect(screen.getByText("Saved WARNING / DANGER clips")).toBeInTheDocument();
    expect(screen.getByText("8 DANGER frames")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "View frame evidence" })).toBeInTheDocument();

    await act(async () => vi.advanceTimersByTimeAsync(3000));
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("opens original, annotated, and Pseudo-BEV evidence", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(job("completed")));
    vi.stubGlobal("fetch", fetchMock);
    render(<VideoProcessingView created={created} />);

    await flushPromises();
    fireEvent.click(screen.getByRole("button", { name: "View frame evidence" }));

    let dialog = screen.getByRole("dialog", { name: "Frame 24 evidence" });
    expect(dialog).toHaveTextContent("2.4s · Independently assessed frame");
    expect(screen.getByRole("img", { name: "Original frame 24" })).toHaveAttribute(
      "src",
      "http://localhost:8000/api/v1/jobs/job-1/segments/segment-1/evidence/24/original",
    );
    expect(screen.getByRole("img", { name: "Annotated frame 24" })).toHaveAttribute(
      "src",
      "http://localhost:8000/api/v1/jobs/job-1/segments/segment-1/evidence/24/rgb",
    );
    expect(screen.getByRole("img", { name: "Pseudo-BEV for frame 24" })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Next frame evidence" }));
    dialog = screen.getByRole("dialog", { name: "Frame 30 evidence" });
    expect(dialog).toHaveTextContent("DANGER");

    fireEvent.click(screen.getByRole("button", { name: "Close frame evidence" }));
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("renders failed jobs and stops polling", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(job("failed")));
    vi.stubGlobal("fetch", fetchMock);
    render(<VideoProcessingView created={created} />);

    await flushPromises();
    expect(screen.getByText("fixture failure")).toBeInTheDocument();
    await act(async () => vi.advanceTimersByTimeAsync(2000));
    expect(fetchMock).toHaveBeenCalledOnce();
  });

  it("shows a clear warning for an mp4v fallback result", async () => {
    const fallback = job("completed");
    fallback.output_codec = "mp4v";
    fallback.browser_playback_compatible = false;
    fallback.playback_warning = "FFmpeg is unavailable; browser playback may fail";
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(fallback));
    vi.stubGlobal("fetch", fetchMock);
    render(<VideoProcessingView created={created} />);

    await flushPromises();
    expect(screen.getByText("Browser playback compatibility is limited")).toBeInTheDocument();
    expect(screen.getByText(fallback.playback_warning)).toBeInTheDocument();
  });

  it("cleans up its polling timer when unmounted", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(job("processing")));
    vi.stubGlobal("fetch", fetchMock);
    const view = render(<VideoProcessingView created={created} />);
    await flushPromises();
    expect(screen.getByText("47.0% processed")).toBeInTheDocument();

    view.unmount();
    await act(async () => vi.advanceTimersByTimeAsync(2000));
    expect(fetchMock).toHaveBeenCalledOnce();
  });
});
