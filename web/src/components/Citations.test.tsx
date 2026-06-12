import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Citations } from "./Citations";

describe("Citations", () => {
  it("renders nothing when empty", () => {
    const { container } = render(<Citations citations={[]} />);
    expect(container.firstChild).toBeNull();
  });

  it("hides snippets until the sources toggle is clicked", () => {
    render(
      <Citations
        citations={[
          { snippet: "PTO policy excerpt", score: 0.88, chunkId: "abc" },
        ]}
      />,
    );
    const toggle = screen.getByRole("button", { name: /Sources \(1\)/ });
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByText("PTO policy excerpt")).not.toBeInTheDocument();

    fireEvent.click(toggle);
    expect(toggle).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByText("PTO policy excerpt")).toBeInTheDocument();

    fireEvent.click(toggle);
    expect(screen.queryByText("PTO policy excerpt")).not.toBeInTheDocument();
  });
});
