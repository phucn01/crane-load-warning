import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { VideoReport } from "../types/detection";
import VideoReportPage from "./VideoReportPage";

const report: VideoReport = {
  schema_version: "1.0",
  job_id: "job-1",
  status: "completed",
  created_at: "2026-08-25T01:00:00+00:00",
  started_at: "2026-08-25T01:00:01+00:00",
  completed_at: "2026-08-25T01:01:00+00:00",
  input_filename: "clip.mp4",
  summary: {
    processed_frames: 100,
    total_frames: 100,
    safe_frames: 70,
    warning_frames: 20,
    danger_frames: 10,
    max_risk_level: "DANGER",
    average_processing_fps: 8.5,
    elapsed_seconds: 59,
    risk_segment_count: 1,
  },
  video: {
    filename: "result.mp4",
    url: "/api/v1/jobs/job-1/result",
    download_url: "/api/v1/jobs/job-1/download",
    codec: "h264",
    browser_playback_compatible: true,
    playback_warning: null,
  },
  risk_segments: [
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
      result_url: "/api/v1/jobs/job-1/segments/segment-1",
      codec: "h264",
      browser_playback_compatible: true,
      playback_warning: null,
      evidence: [
        {
          frame_number: 30,
          timestamp_seconds: 3,
          risk_level: "DANGER",
          original_url: "/api/v1/jobs/job-1/segments/segment-1/evidence/30/original",
          rgb_url: "/api/v1/jobs/job-1/segments/segment-1/evidence/30/rgb",
          pseudo_bev_url: "/api/v1/jobs/job-1/segments/segment-1/evidence/30/bev",
        },
      ],
    },
  ],
};

describe("VideoReportPage", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("renders the persisted report summary and evidence", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify(report), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
      ),
    );

    render(<VideoReportPage jobId="job-1" />);

    expect(await screen.findByRole("heading", { name: "Crane load safety report" })).toBeInTheDocument();
    expect(screen.getByText("100")).toBeInTheDocument();
    expect(screen.getByText("70")).toBeInTheDocument();
    expect(screen.getByText("8 danger")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Annotated evidence frame 30" })).toHaveAttribute(
      "src",
      "http://localhost:8000/api/v1/jobs/job-1/segments/segment-1/evidence/30/rgb",
    );
    expect(screen.getByRole("link", { name: "Download video" })).toHaveAttribute(
      "href",
      "http://localhost:8000/api/v1/jobs/job-1/download",
    );
  });
});
