import { derivePipeline, stageDetailBadge } from "../lib/pipeline";
import type { StepState } from "../lib/pipeline";
import type { PipelineEvent } from "../types";

interface Props {
  events: PipelineEvent[];
  finished?: boolean;
}

interface NodeBox {
  id: string;
  label: string;
  x: number;
  y: number;
}

const NODE_WIDTH = 78;
const NODE_HEIGHT = 28;
const MAIN_Y = 64;

/** Mirrors the stage taxonomy: main flow on one row, judge below, branches arced. */
const NODES: NodeBox[] = [
  { id: "rewrite", label: "Rewrite", x: 8, y: MAIN_Y },
  { id: "route", label: "Route", x: 102, y: MAIN_Y },
  { id: "retrieve.search", label: "Search", x: 196, y: MAIN_Y },
  { id: "retrieve.rerank", label: "Rerank", x: 290, y: MAIN_Y },
  { id: "grade", label: "Grade", x: 384, y: MAIN_Y },
  { id: "grade.judge", label: "LLM judge", x: 384, y: 128 },
  { id: "generate.llm", label: "Generate", x: 478, y: MAIN_Y },
  { id: "validate", label: "Validate", x: 572, y: MAIN_Y },
  { id: "done", label: "Done", x: 666, y: MAIN_Y },
];

const MAIN_EDGES: Array<[string, string]> = [
  ["rewrite", "route"],
  ["route", "retrieve.search"],
  ["retrieve.search", "retrieve.rerank"],
  ["retrieve.rerank", "grade"],
  ["grade", "generate.llm"],
  ["generate.llm", "validate"],
  ["validate", "done"],
];

function nodeById(id: string): NodeBox {
  const node = NODES.find((candidate) => candidate.id === id);
  if (!node) throw new Error(`Unknown pipeline node: ${id}`);
  return node;
}

function center(node: NodeBox): { cx: number; cy: number } {
  return { cx: node.x + NODE_WIDTH / 2, cy: node.y + NODE_HEIGHT / 2 };
}

function nodeClasses(state: StepState): { rect: string; text: string } {
  switch (state) {
    case "active":
      return {
        rect: "animate-pulse fill-sky-500/30 stroke-sky-400",
        text: "fill-sky-200 font-semibold",
      };
    case "done":
      return {
        rect: "fill-emerald-500/15 stroke-emerald-500",
        text: "fill-emerald-200",
      };
    case "skipped":
      return {
        rect: "fill-transparent stroke-slate-700",
        text: "fill-slate-600",
      };
    default:
      return {
        rect: "fill-slate-800 stroke-slate-600",
        text: "fill-slate-400",
      };
  }
}

/**
 * Full RAG workflow diagram for the debug panel. Highlights the live active
 * stage while a query streams and the traversed path once it completes.
 */
export function PipelineDiagram({ events, finished = false }: Props) {
  const steps = derivePipeline(events, { finished });
  const stateById = new Map<string, StepState>(
    steps.map((step) => [step.id, step.state]),
  );
  const badgeById = new Map<string, string | null>(
    steps.map((step) => [step.id, stageDetailBadge(step.detail)]),
  );
  // Terminal node lights up when the stream has finished.
  stateById.set("done", finished ? "done" : "pending");

  const judgeRan =
    stateById.get("grade.judge") === "active" ||
    stateById.get("grade.judge") === "done";

  const route = nodeById("route");
  const grade = nodeById("grade");
  const judge = nodeById("grade.judge");
  const generate = nodeById("generate.llm");
  const doneNode = nodeById("done");

  return (
    <div data-testid="pipeline-diagram">
      <svg
        viewBox="0 0 752 170"
        className="w-full"
        role="img"
        aria-label="RAG pipeline diagram"
      >
        <title>RAG pipeline diagram</title>
        <defs>
          <marker
            id="pipeline-arrow"
            viewBox="0 0 8 8"
            refX="7"
            refY="4"
            markerWidth="6"
            markerHeight="6"
            orient="auto-start-reverse"
          >
            <path d="M 0 0 L 8 4 L 0 8 z" className="fill-slate-500" />
          </marker>
        </defs>

        {MAIN_EDGES.map(([fromId, toId]) => {
          const from = nodeById(fromId);
          const to = nodeById(toId);
          return (
            <line
              key={`${fromId}-${toId}`}
              x1={from.x + NODE_WIDTH}
              y1={center(from).cy}
              x2={to.x - 2}
              y2={center(to).cy}
              className="stroke-slate-600"
              strokeWidth="1.5"
              markerEnd="url(#pipeline-arrow)"
            />
          );
        })}

        {/* Direct route: skips retrieval and jumps straight to generate. */}
        <path
          d={`M ${center(route).cx} ${route.y} C ${center(route).cx} 14, ${center(generate).cx} 14, ${center(generate).cx} ${generate.y - 2}`}
          fill="none"
          className="stroke-slate-600"
          strokeWidth="1.5"
          strokeDasharray="4 3"
          markerEnd="url(#pipeline-arrow)"
        />
        <text
          x={(center(route).cx + center(generate).cx) / 2}
          y={12}
          textAnchor="middle"
          className="fill-slate-500 text-[9px]"
        >
          direct
        </text>

        {/* Low-score path: grade consults the LLM judge before generating. */}
        <line
          x1={center(grade).cx}
          y1={grade.y + NODE_HEIGHT}
          x2={center(judge).cx}
          y2={judge.y - 2}
          className="stroke-slate-600"
          strokeWidth="1.5"
          strokeDasharray="4 3"
          markerEnd="url(#pipeline-arrow)"
        />
        <line
          x1={judge.x + NODE_WIDTH}
          y1={center(judge).cy}
          x2={center(generate).cx}
          y2={generate.y + NODE_HEIGHT + 2}
          className="stroke-slate-600"
          strokeWidth="1.5"
          strokeDasharray="4 3"
          markerEnd="url(#pipeline-arrow)"
        />

        {/* Abstain: grade ends the run without generation. */}
        <path
          d={`M ${center(grade).cx + 30} ${grade.y + NODE_HEIGHT} C ${center(grade).cx + 60} 164, ${center(doneNode).cx} 164, ${center(doneNode).cx} ${doneNode.y + NODE_HEIGHT + 2}`}
          fill="none"
          className="stroke-slate-600"
          strokeWidth="1.5"
          strokeDasharray="4 3"
          markerEnd="url(#pipeline-arrow)"
        />
        <text
          x={(center(grade).cx + center(doneNode).cx) / 2}
          y={166}
          textAnchor="middle"
          className="fill-slate-500 text-[9px]"
        >
          abstain
        </text>

        {NODES.map((node) => {
          if (node.id === "grade.judge" && !judgeRan) {
            // Judge stays ghosted unless the run actually consulted it.
            const classes = nodeClasses("skipped");
            return (
              <g key={node.id} data-node={node.id} data-state="skipped">
                <rect
                  x={node.x}
                  y={node.y}
                  width={NODE_WIDTH}
                  height={NODE_HEIGHT}
                  rx="8"
                  className={classes.rect}
                  strokeWidth="1.5"
                />
                <text
                  x={center(node).cx}
                  y={center(node).cy + 3.5}
                  textAnchor="middle"
                  className={`${classes.text} text-[10px]`}
                >
                  {node.label}
                </text>
              </g>
            );
          }
          const state = stateById.get(node.id) ?? "pending";
          const classes = nodeClasses(state);
          const badge = badgeById.get(node.id) ?? null;
          return (
            <g key={node.id} data-node={node.id} data-state={state}>
              <rect
                x={node.x}
                y={node.y}
                width={NODE_WIDTH}
                height={NODE_HEIGHT}
                rx="8"
                className={classes.rect}
                strokeWidth="1.5"
              />
              <text
                x={center(node).cx}
                y={center(node).cy + 3.5}
                textAnchor="middle"
                className={`${classes.text} text-[10px]`}
              >
                {node.label}
              </text>
              {badge && (
                <text
                  x={center(node).cx}
                  y={node.y + NODE_HEIGHT + 11}
                  textAnchor="middle"
                  data-badge={node.id}
                  className="fill-slate-400 text-[9px]"
                >
                  {badge}
                </text>
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
}
