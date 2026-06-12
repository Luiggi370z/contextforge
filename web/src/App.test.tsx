import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";

describe("App", () => {
  beforeEach(() => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => [],
      }),
    );
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("renders ContextForge heading", () => {
    render(<App />);
    expect(screen.getByText("ContextForge")).toBeInTheDocument();
  });

  it("renders demo prompt cards with title and prompt text", () => {
    render(<App />);
    const directCard = screen.getByRole("button", { name: /^Direct/ });
    expect(directCard).toBeInTheDocument();
    expect(directCard).toHaveTextContent("hi");
    const ragCard = screen.getByRole("button", { name: /^RAG/ });
    expect(ragCard).toHaveTextContent("How many PTO days do employees get?");
  });

  it("renders conversation sidebar navigation", () => {
    render(<App />);
    expect(
      screen.getByRole("navigation", { name: "Conversations" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "New conversation" }),
    ).toBeInTheDocument();
  });

  it("opens the retrieval eval modal from the header button", async () => {
    render(<App />);
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Retrieval eval" }));
    expect(
      screen.getByRole("dialog", { name: "Retrieval eval" }),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Close" }));
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });
});
