import { describe, expect, it } from "vitest";
import type { PipelineEvent } from "../types";
import { derivePipeline, pipelinePathSummary } from "./pipeline";

const start = (
  stage: string,
  detail?: Record<string, unknown>,
): PipelineEvent => ({
  stage,
  phase: "start",
  detail,
});
const end = (stage: string): PipelineEvent => ({ stage, phase: "end" });

function statesById(steps: ReturnType<typeof derivePipeline>) {
  return Object.fromEntries(steps.map((step) => [step.id, step.state]));
}

describe("derivePipeline", () => {
  it("returns all steps pending with no events", () => {
    const steps = derivePipeline([]);
    expect(steps.every((step) => step.state === "pending")).toBe(true);
  });

  it("marks the latest started stage active and earlier ones done", () => {
    const steps = derivePipeline([
      start("rewrite"),
      start("route"),
      end("route"),
      start("retrieve.search"),
    ]);
    const states = statesById(steps);
    expect(states.rewrite).toBe("done");
    expect(states.route).toBe("done");
    expect(states["retrieve.search"]).toBe("active");
    expect(states["generate.llm"]).toBe("pending");
  });

  it("node-end events complete the covered sub-stages", () => {
    const steps = derivePipeline([
      start("retrieve.search"),
      start("retrieve.rerank"),
      end("retrieve"),
    ]);
    const states = statesById(steps);
    expect(states["retrieve.search"]).toBe("done");
    expect(states["retrieve.rerank"]).toBe("done");
  });

  it("happy path: full RAG run finishes with retrieval steps done and judge skipped", () => {
    const steps = derivePipeline(
      [
        start("rewrite"),
        start("route"),
        end("route"),
        start("retrieve.search"),
        start("retrieve.rerank", { candidates: 12 }),
        end("retrieve"),
        start("grade"),
        end("grade_context"),
        start("generate.llm", { provider: "ollama" }),
        end("generate"),
        start("validate"),
        end("validate_answer"),
      ],
      { finished: true },
    );
    const states = statesById(steps);
    expect(states.rewrite).toBe("done");
    expect(states["retrieve.rerank"]).toBe("done");
    expect(states["grade.judge"]).toBe("skipped");
    expect(states["generate.llm"]).toBe("done");
    expect(states.validate).toBe("done");
    const rerank = steps.find((step) => step.id === "retrieve.rerank");
    expect(rerank?.detail).toEqual({ candidates: 12 });
  });

  it("direct route: retrieval steps end up skipped when finished", () => {
    const steps = derivePipeline(
      [
        start("rewrite"),
        start("route"),
        end("route"),
        start("generate.llm", { provider: "heuristic" }),
        end("generate"),
        end("validate_answer"),
      ],
      { finished: true },
    );
    const states = statesById(steps);
    expect(states["retrieve.search"]).toBe("skipped");
    expect(states["retrieve.rerank"]).toBe("skipped");
    expect(states.grade).toBe("skipped");
    expect(states["generate.llm"]).toBe("done");
  });

  it("abstain: generate and validate stay skipped when finished", () => {
    const steps = derivePipeline(
      [
        start("rewrite"),
        start("route"),
        end("route"),
        start("retrieve.search"),
        end("retrieve"),
        start("grade"),
        end("grade_context"),
        end("generate"),
        end("validate_answer"),
      ],
      { finished: true },
    );
    const states = statesById(steps);
    expect(states.grade).toBe("done");
    expect(states["generate.llm"]).toBe("skipped");
    expect(states.validate).toBe("skipped");
  });

  it("judge path: grade.judge runs and completes with the grade node", () => {
    const steps = derivePipeline([
      start("grade"),
      start("grade.judge", { provider: "ollama" }),
      end("grade_context"),
    ]);
    const states = statesById(steps);
    expect(states.grade).toBe("done");
    expect(states["grade.judge"]).toBe("done");
  });

  it("finished promotes still-active steps to done", () => {
    const steps = derivePipeline([start("validate")], { finished: true });
    expect(statesById(steps).validate).toBe("done");
  });

  it("ignores unknown stages", () => {
    const steps = derivePipeline([start("started"), start("mystery.stage")]);
    expect(steps.every((step) => step.state === "pending")).toBe(true);
  });
});

describe("pipelinePathSummary", () => {
  it("lists only traversed steps in order", () => {
    const steps = derivePipeline(
      [
        start("rewrite"),
        start("route"),
        end("route"),
        start("generate.llm"),
        end("generate"),
      ],
      { finished: true },
    );
    expect(pipelinePathSummary(steps)).toBe("Rewrite → Route → Generate");
  });
});
