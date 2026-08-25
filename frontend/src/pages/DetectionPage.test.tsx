import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import DetectionPage from "./DetectionPage";

describe("DetectionPage smart upload routing", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it.each([
    ["frame.png", "image/png", "/api/v1/detection/image", "Run image analysis"],
    ["worksite.mp4", "video/mp4", "/api/v1/detection/video", "Upload and process video"],
  ])("routes %s to the matching API after confirmation", async (name, type, endpoint, action) => {
    Object.defineProperty(URL, "createObjectURL", {
      configurable: true,
      value: vi.fn(() => "blob:preview"),
    });
    Object.defineProperty(URL, "revokeObjectURL", {
      configurable: true,
      value: vi.fn(),
    });
    const fetchMock = vi.fn(
      (_input: RequestInfo | URL, _init?: RequestInit) =>
        new Promise<Response>(() => undefined),
    );
    vi.stubGlobal("fetch", fetchMock);
    const view = render(<DetectionPage />);
    const file = new File(["fixture"], name, { type });

    fireEvent.change(screen.getByLabelText("Choose an image or video"), {
      target: { files: [file] },
    });

    await waitFor(() => expect(screen.getByRole("button", { name: action })).toBeInTheDocument());
    expect(fetchMock).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: action }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledOnce());
    expect(fetchMock.mock.calls[0]?.[0]).toBe(`http://localhost:8000${endpoint}`);
    expect(screen.getByText(type.startsWith("image") ? "image" : "video")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Choose another file" })).toBeInTheDocument();
    view.unmount();
  });

  it("aborts the active upload when choosing another file", async () => {
    const fetchMock = vi.fn(
      (_input: RequestInfo | URL, _init?: RequestInit) =>
        new Promise<Response>(() => undefined),
    );
    vi.stubGlobal("fetch", fetchMock);
    render(<DetectionPage />);

    fireEvent.change(screen.getByLabelText("Choose an image or video"), {
      target: { files: [new File(["video"], "lift.mp4", { type: "video/mp4" })] },
    });
    await waitFor(() => expect(screen.getByRole("button", { name: "Upload and process video" })).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "Upload and process video" }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledOnce());
    const signal = fetchMock.mock.calls[0]?.[1]?.signal;
    expect(signal?.aborted).toBe(false);

    fireEvent.click(screen.getByRole("button", { name: "Choose another file" }));

    expect(signal?.aborted).toBe(true);
    expect(screen.getByText("Upload an image or video")).toBeInTheDocument();
  });

  it("offers a retry action after a video upload failure", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ detail: "fixture upload failure" }), {
        status: 500,
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);
    render(<DetectionPage />);

    fireEvent.change(screen.getByLabelText("Choose an image or video"), {
      target: { files: [new File(["video"], "lift.mp4", { type: "video/mp4" })] },
    });
    const processButton = await screen.findByRole("button", {
      name: "Upload and process video",
    });
    fireEvent.click(processButton);

    expect(await screen.findByText("fixture upload failure")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Retry video processing" })).toBeEnabled();
  });
});
