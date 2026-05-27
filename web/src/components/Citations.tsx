import type { Citation } from "../types";

interface Props {
  citations: Citation[];
}

export function Citations({ citations }: Props) {
  if (!citations.length) return null;
  return (
    <div className="mt-2 space-y-2">
      <p className="text-xs font-medium uppercase tracking-wide text-slate-400">Sources</p>
      {citations.map((c, i) => (
        <details
          key={`${c.chunk_id ?? i}`}
          className="rounded-lg border border-slate-700 bg-slate-900/60 p-2 text-sm"
        >
          <summary className="cursor-pointer text-slate-300">
            [{i + 1}] score {c.score?.toFixed(3) ?? "—"}
          </summary>
          <p className="mt-1 whitespace-pre-wrap text-slate-400">{c.snippet}</p>
        </details>
      ))}
    </div>
  );
}
