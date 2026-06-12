import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { PipelineEvent } from "../types";
import { PipelineDiagram } from "./PipelineDiagram";

const start = (
  stage: string,
  detail?: Record<string, unknown>,
): PipelineEvent => ({
  stage,
  phase: "start",
  detail,
});
const end = (stage: string): PipelineEvent => ({ stage, phase: "end" });

function nodeEl(id: string): SVGGElement | null {
  return document.querySelector(`[data-node="${id}"]`);
}

describe("PipelineDiagram", () => {
  it("highlights the live active node while streaming", () => {
    render(
      <PipelineDiagram
        events={[
          start("rewrite"),
          start("route"),
          end("route"),
          start("retrieve.search"),
        ]}
      />,
    );
    expect(nodeEl("retrieve.search")?.dataset.state).toBe("active");
    expect(nodeEl("route")?.dataset.state).toBe("done");
    expect(nodeEl("validate")?.dataset.state).toBe("pending");
    expect(nodeEl("done")?.dataset.state).toBe("pending");
  });

  it("shows the traversed path and skipped branches after completion", () => {
    render(
      <PipelineDiagram
        finished
        events={[
          start("rewrite"),
          start("route"),
          end("route"),
          start("generate.llm", { provider: "ollama" }),
          end("generate"),
          end("validate_answer"),
        ]}
      />,
    );
    expect(nodeEl("generate.llm")?.dataset.state).toBe("done");
    expect(nodeEl("retrieve.search")?.dataset.state).toBe("skipped");
    expect(nodeEl("grade.judge")?.dataset.state).toBe("skipped");
    expect(nodeEl("done")?.dataset.state).toBe("done");
  });

  it("lights the judge node when the run consulted it", () => {
    render(
      <PipelineDiagram
        events={[start("grade"), start("grade.judge", { provider: "ollama" })]}
      />,
    );
    expect(nodeEl("grade.judge")?.dataset.state).toBe("active");
  });

  it("shows step detail badges under the nodes", () => {
    render(
      <PipelineDiagram
        events={[
          start("retrieve.rerank", { candidates: 12 }),
          start("generate.llm", { provider: "ollama" }),
        ]}
      />,
    );
    expect(
      document.querySelector('[data-badge="retrieve.rerank"]')?.textContent,
    ).toBe("12 cand.");
    expect(
      document.querySelector('[data-badge="generate.llm"]')?.textContent,
    ).toBe("ollama");
  });
});
