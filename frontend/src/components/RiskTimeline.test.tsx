import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { VideoFrameRiskResult } from "../types/detection";
import RiskTimeline, { buildRiskRuns } from "./RiskTimeline";

const results: VideoFrameRiskResult[] = [
  { frame_number: 1, timestamp_seconds: 0, risk_level: "SAFE" },
  { frame_number: 2, timestamp_seconds: 0.1, risk_level: "SAFE" },
  { frame_number: 3, timestamp_seconds: 0.2, risk_level: "WARNING" },
  { frame_number: 4, timestamp_seconds: 0.3, risk_level: "DANGER" },
  { frame_number: 5, timestamp_seconds: 0.4, risk_level: "DANGER" },
];

describe("RiskTimeline", () => {
  it("compresses consecutive frame classifications into seekable runs", () => {
    expect(buildRiskRuns(results)).toEqual([
      {
        startFrame: 1,
        endFrame: 2,
        startSeconds: 0,
        endSeconds: 0.1,
        riskLevel: "SAFE",
      },
      {
        startFrame: 3,
        endFrame: 3,
        startSeconds: 0.2,
        endSeconds: 0.2,
        riskLevel: "WARNING",
      },
      {
        startFrame: 4,
        endFrame: 5,
        startSeconds: 0.3,
        endSeconds: 0.4,
        riskLevel: "DANGER",
      },
    ]);
  });

  it("shows totals and seeks from ranges and evidence markers", () => {
    const onSeek = vi.fn();
    render(
      <RiskTimeline
        results={results}
        evidence={[
          {
            frame_number: 4,
            timestamp_seconds: 0.3,
            risk_level: "DANGER",
            original_url: "/original",
            rgb_url: "/rgb",
            pseudo_bev_url: "/bev",
          },
        ]}
        totalFrames={5}
        currentTime={0.3}
        duration={0.5}
        onSeek={onSeek}
      />,
    );

    expect(screen.getByText("5 / 5 frames mapped")).toBeInTheDocument();
    expect(screen.getByText("4")).toBeInTheDocument();
    expect(screen.getByText("DANGER", { selector: "em" })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /WARNING: frames 3 to 3/i }));
    fireEvent.click(screen.getByRole("button", { name: /Evidence frame 4/i }));
    expect(onSeek).toHaveBeenNthCalledWith(1, 0.2);
    expect(onSeek).toHaveBeenNthCalledWith(2, 0.3);
  });
});
