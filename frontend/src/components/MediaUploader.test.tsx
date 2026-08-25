import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import MediaUploader, { detectMediaType } from "./MediaUploader";

describe("MediaUploader", () => {
  it("detects a selected video and submits it once", async () => {
    const onSelect = vi.fn();
    render(
      <MediaUploader disabled={false} onSelect={onSelect} onInvalid={vi.fn()} />,
    );
    const file = new File(["video"], "lift.mp4", { type: "video/mp4" });

    fireEvent.change(screen.getByLabelText("Choose an image or video"), {
      target: { files: [file] },
    });

    await waitFor(() => expect(onSelect).toHaveBeenCalledWith(file, "video"));
    expect(onSelect).toHaveBeenCalledOnce();
  });

  it("recognizes image and video file signatures", async () => {
    const png = new File(
      [new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])],
      "unknown.bin",
    );
    const mp4 = new File(
      [new Uint8Array([0, 0, 0, 16, 0x66, 0x74, 0x79, 0x70])],
      "unknown.bin",
    );

    expect(await detectMediaType(png)).toBe("image");
    expect(await detectMediaType(mp4)).toBe("video");
  });

  it("rejects unsupported media", async () => {
    const onInvalid = vi.fn();
    render(
      <MediaUploader disabled={false} onSelect={vi.fn()} onInvalid={onInvalid} />,
    );

    fireEvent.change(screen.getByLabelText("Choose an image or video"), {
      target: { files: [new File(["text"], "notes.txt", { type: "text/plain" })] },
    });

    await waitFor(() => expect(onInvalid).toHaveBeenCalledOnce());
  });
});
