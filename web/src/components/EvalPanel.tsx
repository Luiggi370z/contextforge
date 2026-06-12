import { useCallback, useEffect, useState } from "react";
import type { EvalRun, EvalRunList } from "../types";

const METRIC_KEYS = ["recall@1", "recall@3", "recall@5", "mrr", "ndcg@5"];

// The short question each metric answers, rendered as a subtitle under the
// column header so the table reads without prior IR knowledge.
const METRIC_QUESTIONS: Record<string, string> = {
  "recall@1": "Is a correct doc ranked #1?",
  "recall@3": "Is a correct doc in the top 3?",
  "recall@5": "Is a correct doc in the top 5?",
  mrr: "How high is the first correct doc? (avg 1/rank)",
  "ndcg@5": "Are correct docs ranked near the top?",
};

// Per-metric "good retrieval" bar (scores are averages in [0,1]). A cell at or
// above its bar earns a green yes, below it a red no. These mirror the floors
// the `pytest -m eval` IR gate asserts, so the panel and the gate agree.
const METRIC_THRESHOLDS: Record<string, number> = {
  "recall@1": 0.8,
  "recall@3": 0.9,
  "recall@5": 0.9,
  mrr: 0.7,
  "ndcg@5": 0.7,
};

// The API returns full run history (newest first); each "run eval" appends one
// row per config. Show only the most recent run per config so the table is a
// current scoreboard, not an ever-growing append log of identical results.
function latestPerConfig(items: EvalRun[] | undefined): EvalRun[] {
  const seen = new Set<string>();
  const latest: EvalRun[] = [];
  for (const run of items ?? []) {
    if (!seen.has(run.label)) {
      seen.add(run.label);
      latest.push(run);
    }
  }
  return latest;
}

function MetricCell({
  metricKey,
  value,
}: {
  metricKey: string;
  value: number | undefined;
}) {
  if (typeof value !== "number") {
    return <td className="px-2 text-right">—</td>;
  }
  const bar = METRIC_THRESHOLDS[metricKey] ?? 0.8;
  const good = value >= bar;
  return (
    <td className="whitespace-nowrap px-2 text-right">
      <span className="tabular-nums">{value.toFixed(3)}</span>{" "}
      <span
        className={good ? "text-emerald-400" : "text-rose-400"}
        title={good ? `yes — at or above ${bar}` : `no — below ${bar}`}
      >
        {good ? "✓ yes" : "✗ no"}
      </span>
    </td>
  );
}

export function EvalPanel() {
  const [runs, setRuns] = useState<EvalRun[]>([]);
  const [running, setRunning] = useState(false);

  const load = useCallback(async () => {
    const response = await fetch("/v1/eval/runs");
    if (response.ok) {
      const data = (await response.json()) as EvalRunList;
      setRuns(latestPerConfig(data.items));
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const runEval = useCallback(async () => {
    setRunning(true);
    try {
      await fetch("/v1/eval/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      await load();
    } finally {
      setRunning(false);
    }
  }, [load]);

  return (
    <div className="mt-2 rounded-lg border border-sky-800/50 bg-sky-950/30 p-3 text-xs text-sky-100">
      <div className="flex items-center justify-between">
        <span className="font-mono text-sky-400">retrieval eval</span>
        <button
          type="button"
          onClick={runEval}
          disabled={running}
          className="rounded bg-sky-700 px-2 py-1 disabled:opacity-50"
        >
          {running ? "running…" : "run eval"}
        </button>
      </div>
      <div className="mt-2 overflow-x-auto">
        <table className="w-full min-w-[40rem] font-mono">
          <thead>
            <tr className="align-bottom text-sky-400">
              <th className="px-2 text-left">config</th>
              {METRIC_KEYS.map((key) => (
                <th key={key} className="px-2 text-right">
                  <div className="text-sky-300">{key}</div>
                  <div className="ml-auto mt-0.5 max-w-[9rem] text-[10px] font-normal text-sky-400/70">
                    {METRIC_QUESTIONS[key]}
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {runs.map((run) => (
              <tr key={run.id}>
                <td className="whitespace-nowrap px-2 text-left">
                  {run.label}
                </td>
                {METRIC_KEYS.map((key) => (
                  <MetricCell
                    key={key}
                    metricKey={key}
                    value={run.metrics[key]}
                  />
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-3 border-t border-sky-800/40 pt-2 text-[11px] text-sky-300/70">
        <span className="text-emerald-400">✓ yes</span> /{" "}
        <span className="text-rose-400">✗ no</span> = the metric is at or above
        its quality bar (hover a marker for that metric's threshold).
      </p>
    </div>
  );
}
