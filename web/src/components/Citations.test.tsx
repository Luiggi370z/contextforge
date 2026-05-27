import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Citations } from "./Citations";

describe("Citations", () => {
  it("renders nothing when empty", () => {
    const { container } = render(<Citations citations={[]} />);
    expect(container.firstChild).toBeNull();
  });

  it("renders source snippets", () => {
    render(
      <Citations
        citations={[
          { snippet: "PTO policy excerpt", score: 0.88, chunkId: "abc" },
        ]}
      />,
    );
    expect(screen.getByText("Sources")).toBeInTheDocument();
    expect(screen.getByText("PTO policy excerpt")).toBeInTheDocument();
  });
});
