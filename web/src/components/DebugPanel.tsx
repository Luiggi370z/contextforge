// TODO(retrieval-backend-ui): Add a settings control here (or a sibling Settings drawer) to let
// the user pick retrieval_backend: "qdrant" | "postgres", persist in localStorage, and send
// with POST /v1/query and document ingest once the API supports per-request overrides.

import type { QueryMetadata } from "../types";

interface Props {
  metadata?: QueryMetadata;
}

export function DebugPanel({ metadata }: Props) {
  if (!metadata) return null;
  return (
    <div className="mt-2 rounded-lg border border-amber-800/50 bg-amber-950/30 p-3 font-mono text-xs text-amber-100">
      <p>
        <span className="text-amber-400">route</span> {metadata.route}
      </p>
      <p>
        <span className="text-amber-400">abstained</span> {String(metadata.abstained)}
      </p>
      <p>
        <span className="text-amber-400">nodes</span> {metadata.nodes_visited.join(" → ")}
      </p>
      <p>
        <span className="text-amber-400">scores</span>{" "}
        {metadata.retrieval_scores.map((s) => s.toFixed(3)).join(", ") || "—"}
      </p>
    </div>
  );
}
