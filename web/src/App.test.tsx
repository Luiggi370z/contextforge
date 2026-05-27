import { render, screen } from "@testing-library/react";
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

  it("renders demo route chips", () => {
    render(<App />);
    expect(screen.getByRole("button", { name: "Direct" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "RAG" })).toBeInTheDocument();
  });
});
