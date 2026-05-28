import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ThinkingIndicator } from "./ThinkingIndicator";

describe("ThinkingIndicator", () => {
  it("renders thinking label with shimmer class", () => {
    render(<ThinkingIndicator />);
    const label = screen.getByText("Thinking");
    expect(label).toHaveClass("thinking-shimmer");
  });

  it("shows graph stage when provided", () => {
    render(<ThinkingIndicator stage="retrieve" />);
    expect(screen.getByText(/Graph: retrieve/)).toBeInTheDocument();
  });
});
