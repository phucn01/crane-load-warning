import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { getImageEvidence, getProcessingHistory } from "../services/api";
import HistoryPage from "./HistoryPage";

vi.mock("../services/api", () => ({
  apiUrl: (path: string | null) => path,
  getImageEvidence: vi.fn(),
  getProcessingHistory: vi.fn(),
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
  });

  it("renders processing jobs without snapshot history", async () => {
    const { container } = render(<HistoryPage />);

    expect(await screen.findByText("crane.mp4")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "View report" })).toHaveAttribute(
      "href",
      "/?report=job-12345678&from=history",
    );
    expect(screen.getByText(/not tracked safety events/i)).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Safety Events" })).not.toBeInTheDocument();
    expect(container.querySelector(".history-header .brand-mark svg")).toBeInTheDocument();

    expect(screen.queryByText("Sampled evidence")).not.toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Risk Snapshots" })).not.toBeInTheDocument();
  });

  it("reviews a persisted SAFE_NO_LOAD image without requiring Pseudo-BEV", async () => {
    vi.mocked(getProcessingHistory).mockResolvedValue({
      items: [{
        id: "image-12345678",
        media_type: "image",
        input_name: "crane.png",
        input_path: "storage/crane.png",
        output_path: "/evidence/image/rgb.png",
        status: "completed",
        total_frames: null,
        processed_frames: null,
        safe_frame_count: 1,
        warning_frame_count: 0,
        danger_frame_count: 0,
        max_risk_level: "SAFE",
        processing_time_ms: 250,
        average_processing_fps: null,
        error_message: null,
        created_at: "2026-08-25T12:00:00Z",
        started_at: "2026-08-25T12:00:01Z",
        completed_at: "2026-08-25T12:00:02Z",
      }],
      total: 1,
      limit: 50,
      offset: 0,
    });
    vi.mocked(getImageEvidence).mockResolvedValue({
      original_url: "/uploads/images/crane.png",
      detection_url: "/evidence/image/rgb.png",
      bev_url: null,
      combined_url: "/evidence/image/rgb.png",
      risk_level: "SAFE",
      assessment_status: "SAFE_NO_LOAD",
      assessment_reliable: true,
      quality_reasons: ["safe_no_load"],
    });

    render(<HistoryPage />);
    fireEvent.click(await screen.findByRole("button", { name: /Review image/i }));

    expect(await screen.findByText("SAFE · NO LOAD")).toBeInTheDocument();
    expect(screen.getByText("A person was detected, but no hanging load was found.")).toBeInTheDocument();
    expect(screen.getByText("safe no load")).toBeInTheDocument();
    expect(screen.getByText("Not generated")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Annotated result" })).toHaveAttribute(
      "src",
      "/evidence/image/rgb.png",
    );
  });
});
