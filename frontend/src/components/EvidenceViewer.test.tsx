import { fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import EvidenceViewer from "./EvidenceViewer";

const evidence = {
  combined_url: "/api/v1/evidence/combined.jpg",
  rgb_url: "/api/v1/evidence/rgb.jpg",
  pseudo_bev_url: "/api/v1/evidence/bev.jpg",
};

describe("EvidenceViewer", () => {
  it("opens evidence in a modal and restores focus when closed", async () => {
    const user = userEvent.setup();
    render(<EvidenceViewer evidence={evidence} />);

    const trigger = screen.getByRole("button", { name: "Preview Combined safety evidence" });
    await user.click(trigger);

    const dialog = screen.getByRole("dialog", { name: "Combined safety evidence" });
    expect(within(dialog).getByRole("img", { name: "Combined safety evidence" })).toBeInTheDocument();
    expect(document.body).toHaveStyle({ overflow: "hidden" });
    expect(screen.getByRole("button", { name: "Close evidence viewer" })).toHaveFocus();

    await user.click(screen.getByRole("button", { name: "Close evidence viewer" }));

    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(document.body.style.overflow).toBe("");
    expect(trigger).toHaveFocus();
  });

  it("navigates between evidence with controls and arrow keys", async () => {
    const user = userEvent.setup();
    render(<EvidenceViewer evidence={evidence} />);

    await user.click(screen.getByRole("button", { name: "View Camera evidence" }));
    let dialog = screen.getByRole("dialog", { name: "Camera evidence" });
    expect(within(dialog).getByText("Evidence 2 of 3")).toBeInTheDocument();

    await user.click(within(dialog).getByRole("button", { name: "Next evidence" }));
    dialog = screen.getByRole("dialog", { name: "Pseudo-BEV safety view" });
    expect(within(dialog).getByText("Evidence 3 of 3")).toBeInTheDocument();

    fireEvent.keyDown(window, { key: "ArrowRight" });
    expect(screen.getByRole("dialog", { name: "Combined safety evidence" })).toBeInTheDocument();

    fireEvent.keyDown(window, { key: "ArrowLeft" });
    expect(screen.getByRole("dialog", { name: "Pseudo-BEV safety view" })).toBeInTheDocument();
  });

  it("closes with Escape and exposes the original image link", async () => {
    const user = userEvent.setup();
    render(<EvidenceViewer evidence={evidence} />);

    await user.click(screen.getByRole("button", { name: "Preview Camera evidence" }));
    const originalLink = screen.getByRole("link", { name: "Open original ↗" });
    expect(originalLink).toHaveAttribute("href", expect.stringContaining("/api/v1/evidence/rgb.jpg"));
    expect(originalLink).toHaveAttribute("target", "_blank");

    fireEvent.keyDown(window, { key: "Escape" });
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });
});
