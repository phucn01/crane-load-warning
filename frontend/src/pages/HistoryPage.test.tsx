import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { getProcessingHistory } from "../services/api";
import HistoryPage from "./HistoryPage";

vi.mock("../services/api", () => ({
  apiUrl: (path: string | null) => path,
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
});
