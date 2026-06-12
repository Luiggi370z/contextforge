import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { PipelineEvent } from "../types";
import { PipelineStepper, PipelineSummary } from "./PipelineStepper";

const start = (
  stage: string,
  detail?: Record<string, unknown>,
): PipelineEvent => ({
  stage,
  phase: "start",
  detail,
});
const end = (stage: string): PipelineEvent => ({ stage, phase: "end" });

function stepEl(stage: string): HTMLElement | null {
  return document.querySelector(`[data-stage="${stage}"]`);
}

describe("PipelineStepper", () => {
  it("marks the streaming stage active and earlier stages done", () => {
    render(
      <PipelineStepper
        events={[
          start("rewrite"),
          start("route"),
          end("route"),
          start("retrieve.search"),
        ]}
      />,
    );
    expect(stepEl("rewrite")?.dataset.state).toBe("done");
    expect(stepEl("retrieve.search")?.dataset.state).toBe("active");
    expect(stepEl("generate.llm")?.dataset.state).toBe("pending");
  });

  it("hides the conditional judge step unless it ran", () => {
    render(<PipelineStepper events={[start("grade")]} />);
    expect(stepEl("grade.judge")).toBeNull();
  });

  it("shows the judge step when consulted", () => {
    render(
      <PipelineStepper
        events={[start("grade"), start("grade.judge", { provider: "ollama" })]}
      />,
    );
    expect(stepEl("grade.judge")?.dataset.state).toBe("active");
  });

  it("renders the provider badge on generate", () => {
    render(
      <PipelineStepper
        events={[start("generate.llm", { provider: "ollama" })]}
      />,
    );
    expect(screen.getByText("ollama")).toBeInTheDocument();
  });
});

describe("PipelineSummary", () => {
  it("collapses to a one-line traversed path", () => {
    render(
      <PipelineSummary
        events={[
          start("rewrite"),
          start("route"),
          end("route"),
          start("generate.llm"),
          end("generate"),
        ]}
      />,
    );
    expect(
      screen.getByText(/Pipeline: Rewrite → Route → Generate/),
    ).toBeInTheDocument();
  });

  it("renders nothing without events", () => {
    const { container } = render(<PipelineSummary events={[]} />);
    expect(container).toBeEmptyDOMElement();
  });
});
