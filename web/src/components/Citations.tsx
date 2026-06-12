import { useState } from "react";
import type { Citation } from "../types";

interface Props {
  citations: Citation[];
}

export function Citations({ citations }: Props) {
  const [open, setOpen] = useState(false);
  if (!citations.length) return null;
  return (
    <div className="mt-2">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
        className="rounded border border-slate-700 px-2 py-0.5 text-xs font-medium uppercase tracking-wide text-slate-400 hover:bg-slate-800 hover:text-slate-200"
      >
        Sources ({citations.length}) {open ? "▾" : "▸"}
      </button>
      {open && (
        <div className="mt-2 space-y-2">
          {citations.map((c, i) => (
            <details
              key={`${c.chunkId ?? i}`}
              className="rounded-lg border border-slate-700 bg-slate-900/60 p-2 text-sm"
            >
              <summary className="cursor-pointer text-slate-300">
                [{i + 1}] score {c.score?.toFixed(3) ?? "—"}
              </summary>
              <p className="mt-1 whitespace-pre-wrap text-slate-400">
                {c.snippet}
              </p>
            </details>
          ))}
        </div>
      )}
    </div>
  );
}
