import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { getProcessingHistory, getRiskSnapshotHistory } from "../services/api";
import HistoryPage from "./HistoryPage";

vi.mock("../services/api", () => ({
  apiUrl: (path: string | null) => path,
  getProcessingHistory: vi.fn(),
  getRiskSnapshotHistory: vi.fn(),
}));

describe("HistoryPage", () => {
  beforeEach(() => {
    vi.mocked(getProcessingHistory).mockResolvedValue({
      items: [{
        id: "job-12345678",
        media_type: "video",
        input_name: "crane.mp4",
        input_path: "storage/crane.mp4",
        output_path: "storage/result.mp4",
        status: "completed",
        total_frames: 120,
        processed_frames: 120,
        safe_frame_count: 100,
        warning_frame_count: 15,
        danger_frame_count: 5,
        max_risk_level: "DANGER",
        processing_time_ms: 4200,
        average_processing_fps: 28.5,
        error_message: null,
        created_at: "2026-08-25T12:00:00Z",
        started_at: "2026-08-25T12:00:01Z",
        completed_at: "2026-08-25T12:00:05Z",
      }],
      total: 1,
      limit: 50,
      offset: 0,
    });
    vi.mocked(getRiskSnapshotHistory).mockResolvedValue({
      items: [{
        id: "snapshot-1",
        job_id: "job-12345678",
        frame_index: 23,
        timestamp_sec: 0.8,
        risk_level: "DANGER",
        confidence: 0.91,
        assessment_reliable: true,
        quality_reasons: [],
        evidence_path: "/evidence/original.png",
        rgb_evidence_path: "/evidence/rgb.png",
        pseudo_bev_path: "/evidence/bev.png",
        created_at: "2026-08-25T12:00:02Z",
      }],
      total: 1,
      limit: 50,
      offset: 0,
    });
  });

  it("renders processing jobs and risk snapshots without event semantics", async () => {
    const { container } = render(<HistoryPage />);

    expect(await screen.findByText("crane.mp4")).toBeInTheDocument();
    expect(screen.getByText("Frame 24")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "View report" })).toHaveAttribute(
      "href",
      "/?report=job-12345678&from=history",
    );
    expect(screen.getByText(/not tracked safety events/i)).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Safety Events" })).not.toBeInTheDocument();
    expect(container.querySelector(".history-header .brand-mark svg")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Review evidence" }));
    expect(screen.getByRole("dialog", { name: "Frame 24 evidence" })).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Original camera frame 24" })).toHaveAttribute(
      "src",
      "/evidence/original.png",
    );
    expect(screen.getByRole("img", { name: "Annotated camera frame 24" })).toHaveAttribute(
      "src",
      "/evidence/rgb.png",
    );
    expect(screen.getByRole("img", { name: "Pseudo-BEV safety view 24" })).toHaveAttribute(
      "src",
      "/evidence/bev.png",
    );
  });
});
