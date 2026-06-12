import type { PipelineEvent } from "../types";

/**
 * Single source of truth for the RAG pipeline visualization.
 *
 * The backend emits two kinds of status events over SSE:
 *  - phase "start" with fine-grained stage ids (`rewrite`, `route`,
 *    `retrieve.search`, `retrieve.rerank`, `grade`, `grade.judge`,
 *    `generate.llm`, `validate`) the moment work begins, and
 *  - phase "end" with graph-node names (`route`, `retrieve`, `grade_context`,
 *    `generate`, `validate_answer`) when a node completes.
 *
 * `derivePipeline` folds that event list into per-step states for the
 * stepper and the debug diagram.
 */

export type StepState = "pending" | "active" | "done" | "skipped";

export interface StageMeta {
  id: string;
  label: string;
  /** Conditional steps are hidden by the stepper unless they actually ran. */
  conditional?: boolean;
}

export const PIPELINE_STAGES: StageMeta[] = [
  { id: "rewrite", label: "Rewrite" },
  { id: "route", label: "Route" },
  { id: "retrieve.search", label: "Search" },
  { id: "retrieve.rerank", label: "Rerank" },
  { id: "grade", label: "Grade" },
  { id: "grade.judge", label: "LLM judge", conditional: true },
  { id: "generate.llm", label: "Generate" },
  { id: "validate", label: "Validate" },
];

/** Node-complete events mark these fine-grained stages as done. */
const NODE_END_COVERS: Record<string, string[]> = {
  route: ["route"],
  retrieve: ["retrieve.search", "retrieve.rerank"],
  grade_context: ["grade", "grade.judge"],
  generate: ["generate.llm"],
  validate_answer: ["validate"],
};

export interface PipelineStep extends StageMeta {
  state: StepState;
  detail?: Record<string, unknown>;
}

/**
 * Fold stage events into per-step states.
 *
 * While streaming (`finished: false`) untouched steps stay `pending`; once the
 * run completes they become `skipped` (e.g. search/rerank/grade on a direct
 * route, generate/validate on abstain, LLM judge when scores were confident).
 */
export function derivePipeline(
  events: PipelineEvent[],
  options: { finished?: boolean } = {},
): PipelineStep[] {
  const finished = options.finished ?? false;
  const order = PIPELINE_STAGES.map((stage) => stage.id);
  const states = new Map<string, StepState>();
  const details = new Map<string, Record<string, unknown>>();

  const markEarlierActiveDone = (beforeIndex: number) => {
    for (const [id, state] of states) {
      if (state === "active" && order.indexOf(id) < beforeIndex) {
        states.set(id, "done");
      }
    }
  };

  for (const event of events) {
    if (event.phase === "start") {
      const index = order.indexOf(event.stage);
      if (index === -1) continue;
      markEarlierActiveDone(index);
      states.set(event.stage, "active");
      if (event.detail) {
        details.set(event.stage, {
          ...details.get(event.stage),
          ...event.detail,
        });
      }
      continue;
    }
    for (const covered of NODE_END_COVERS[event.stage] ?? []) {
      if (states.get(covered) === "active") {
        states.set(covered, "done");
      }
    }
  }

  if (finished) {
    for (const [id, state] of states) {
      if (state === "active") states.set(id, "done");
    }
  }

  return PIPELINE_STAGES.map((stage) => ({
    ...stage,
    state: states.get(stage.id) ?? (finished ? "skipped" : "pending"),
    detail: details.get(stage.id),
  }));
}

/** Compact human-readable badge for a step's detail (provider, candidate count). */
export function stageDetailBadge(
  detail: Record<string, unknown> | undefined,
): string | null {
  if (!detail) return null;
  if (typeof detail.provider === "string") return detail.provider;
  if (typeof detail.candidates === "number")
    return `${detail.candidates} cand.`;
  return null;
}

/** Short human-readable summary of the path taken, e.g. for collapsed view. */
export function pipelinePathSummary(steps: PipelineStep[]): string {
  return steps
    .filter((step) => step.state === "done" || step.state === "active")
    .map((step) => step.label)
    .join(" → ");
}
