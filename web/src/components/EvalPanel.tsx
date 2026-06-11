import { useCallback, useEffect, useState } from "react";
import type { EvalRun, EvalRunList } from "../types";

const METRIC_KEYS = ["recall@1", "recall@3", "recall@5", "mrr", "ndcg@5"];

export function EvalPanel() {
  const [runs, setRuns] = useState<EvalRun[]>([]);
  const [running, setRunning] = useState(false);

  const load = useCallback(async () => {
    const response = await fetch("/v1/eval/runs");
    if (response.ok) {
      const data = (await response.json()) as EvalRunList;
      setRuns(data.items);
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
      <table className="mt-2 w-full font-mono">
        <thead>
          <tr className="text-sky-400">
            <th className="text-left">config</th>
            {METRIC_KEYS.map((key) => (
              <th key={key} className="text-right">
                {key}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {runs.map((run) => (
            <tr key={run.id}>
              <td className="text-left">{run.label}</td>
              {METRIC_KEYS.map((key) => (
                <td key={key} className="text-right">
                  {typeof run.metrics[key] === "number"
                    ? run.metrics[key].toFixed(3)
                    : "—"}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
