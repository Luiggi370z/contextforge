import {
  derivePipeline,
  pipelinePathSummary,
  stageDetailBadge,
} from "../lib/pipeline";
import type { PipelineStep } from "../lib/pipeline";
import type { PipelineEvent } from "../types";

interface StepperProps {
  events: PipelineEvent[];
  finished?: boolean;
}

function StepDot({ step }: { step: PipelineStep }) {
  if (step.state === "done") {
    return (
      <span className="flex h-4 w-4 items-center justify-center rounded-full bg-emerald-600 text-[9px] text-white">
        ✓
      </span>
    );
  }
  if (step.state === "active") {
    return (
      <span className="h-4 w-4 animate-pulse rounded-full border-2 border-sky-400 bg-sky-500/40" />
    );
  }
  return (
    <span
      className={`h-4 w-4 rounded-full border ${step.state === "skipped" ? "border-slate-700" : "border-slate-600"}`}
    />
  );
}

/**
 * Compact horizontal stepper showing live RAG pipeline progress.
 * Rendered inline in the chat while a query streams.
 */
export function PipelineStepper({ events, finished = false }: StepperProps) {
  const steps = derivePipeline(events, { finished });
  // Hide conditional steps (LLM judge) unless they actually ran.
  const visible = steps.filter(
    (step) =>
      !step.conditional || step.state === "active" || step.state === "done",
  );

  return (
    <div
      className="flex flex-wrap items-start gap-x-1 gap-y-2"
      aria-live="polite"
      aria-busy={!finished}
      data-testid="pipeline-stepper"
    >
      {visible.map((step, index) => {
        const badge = stageDetailBadge(step.detail);
        const dimmed = step.state === "pending" || step.state === "skipped";
        return (
          <div key={step.id} className="flex items-start">
            {index > 0 && (
              <span className="mx-1 mt-1.5 h-px w-4 bg-slate-600" aria-hidden />
            )}
            <div
              className={`flex flex-col items-center gap-1 ${dimmed ? "opacity-40" : ""}`}
              data-stage={step.id}
              data-state={step.state}
            >
              <StepDot step={step} />
              <span
                className={`text-[10px] leading-tight ${step.state === "active" ? "font-semibold text-sky-300" : "text-slate-400"}`}
              >
                {step.label}
              </span>
              {badge && (
                <span className="rounded bg-slate-700 px-1 text-[9px] text-slate-300">
                  {badge}
                </span>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

interface SummaryProps {
  events: PipelineEvent[];
}

/**
 * Collapsed one-line path summary attached to a completed assistant message.
 * Expands to the full (finished) stepper on click.
 */
export function PipelineSummary({ events }: SummaryProps) {
  if (events.length === 0) return null;
  const steps = derivePipeline(events, { finished: true });
  return (
    <details className="mt-2 text-xs text-slate-400">
      <summary className="cursor-pointer select-none font-mono text-[11px] text-slate-500 hover:text-slate-300">
        Pipeline: {pipelinePathSummary(steps)}
      </summary>
      <div className="mt-2">
        <PipelineStepper events={events} finished />
      </div>
    </details>
  );
}
