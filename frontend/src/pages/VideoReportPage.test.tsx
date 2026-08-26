import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { VideoReport } from "../types/detection";
import VideoReportPage from "./VideoReportPage";

const snapshots = [0, 48, 96, 144].map((frameIndex, index) => ({
  id: `snapshot-${index + 1}`,
  job_id: "job-1",
  frame_index: frameIndex,
  timestamp_sec: frameIndex / 24,
  risk_level: index < 3 ? "DANGER" as const : "WARNING" as const,
  confidence: 0.9,
  assessment_reliable: true,
  quality_reasons: [],
  evidence_path: `/evidence/snapshot-${index + 1}/original.png`,
  rgb_evidence_path: `/evidence/snapshot-${index + 1}/rgb.png`,
  pseudo_bev_path: `/evidence/snapshot-${index + 1}/bev.png`,
  created_at: "2026-08-25T01:00:00+00:00",
}));

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
    const frameResults = [
      { frame_number: 1, timestamp_seconds: 0, risk_level: "SAFE" },
      { frame_number: 24, timestamp_seconds: 2.3, risk_level: "WARNING" },
      { frame_number: 30, timestamp_seconds: 2.9, risk_level: "DANGER" },
      { frame_number: 36, timestamp_seconds: 3.5, risk_level: "WARNING" },
      { frame_number: 100, timestamp_seconds: 9.9, risk_level: "SAFE" },
    ];
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation((input: string) => {
        const url = String(input);
        const payload = url.includes("/frames?")
          ? {
              job_id: "job-1",
              job_status: "completed",
              items: frameResults,
              next_after_frame: 100,
              has_more: false,
            }
          : url.includes("/risk-snapshots?")
            ? { items: snapshots, total: snapshots.length, limit: 200, offset: 0 }
          : report;
        return Promise.resolve(new Response(JSON.stringify(payload), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }));
      }),
    );

    render(<VideoReportPage jobId="job-1" />);

    expect(await screen.findByRole("heading", { name: "Crane load safety report" })).toBeInTheDocument();
    expect(screen.getByText("100")).toBeInTheDocument();
    expect(screen.getByText("70")).toBeInTheDocument();
    expect(screen.getByText("8 danger")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Risk timeline" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Risk snapshots" })).toBeInTheDocument();
    expect(screen.getByText("4 snapshots")).toBeInTheDocument();
    expect(screen.getByText("Frame 145")).toBeInTheDocument();
    expect(screen.getByText("1 key frames")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Annotated evidence frame 30" })).toHaveAttribute(
      "src",
      "http://localhost:8000/api/v1/jobs/job-1/segments/segment-1/evidence/30/rgb",
    );
    expect(screen.getByRole("link", { name: "Download video" })).toHaveAttribute(
      "href",
      "http://localhost:8000/api/v1/jobs/job-1/download",
    );
    expect(screen.getByRole("link", { name: "History" })).toHaveAttribute(
      "href",
      "/?history=1",
    );

    fireEvent.click(screen.getAllByRole("button", { name: "Review frame" })[3]);
    expect(screen.getByRole("heading", { name: "Snapshot frame 145" })).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Annotated snapshot frame 145" })).toHaveAttribute(
      "src",
      "http://localhost:8000/evidence/snapshot-4/rgb.png",
    );

    fireEvent.click(screen.getByRole("button", { name: "Inspect evidence" }));
    expect(screen.getByRole("heading", { name: "WARNING / DANGER segments" })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /DANGER: frames 30 to 30/i }));
    expect(document.querySelector("video")?.currentTime).toBe(2.9);
    fireEvent.click(screen.getByRole("button", { name: "Open full evidence viewer" }));
    expect(screen.getByRole("dialog", { name: "Frame 30 evidence" })).toBeInTheDocument();
  });

  it("keeps the persisted report usable when the in-memory timeline is gone", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation((input: string) => {
        if (String(input).includes("/frames?")) {
          return Promise.resolve(new Response(JSON.stringify({ detail: "not found" }), {
            status: 404,
            headers: { "Content-Type": "application/json" },
          }));
        }
        if (String(input).includes("/risk-snapshots?")) {
          return Promise.resolve(new Response(JSON.stringify({
            items: snapshots,
            total: snapshots.length,
            limit: 200,
            offset: 0,
          }), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          }));
        }
        return Promise.resolve(new Response(JSON.stringify(report), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }));
      }),
    );

    render(<VideoReportPage jobId="job-1" />);

    expect(await screen.findByRole("heading", { name: "Crane load safety report" })).toBeInTheDocument();
    expect(screen.getByText(/timeline is no longer in memory/i)).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "WARNING / DANGER segments" })).toBeInTheDocument();
  });
});
