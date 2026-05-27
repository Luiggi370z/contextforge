import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { DebugPanel } from "./DebugPanel";

describe("DebugPanel", () => {
  it("renders route metadata", () => {
    render(
      <DebugPanel
        metadata={{
          route: "single_hop_rag",
          abstained: false,
          nodesVisited: ["route", "retrieve", "generate"],
          retrievalScores: [0.42, 0.31],
          graphCheckpointEnabled: true,
        }}
      />,
    );
    expect(screen.getByText(/single_hop_rag/)).toBeInTheDocument();
    expect(screen.getByText(/retrieve/)).toBeInTheDocument();
    expect(screen.getByText(/true/)).toBeInTheDocument();
  });
});
