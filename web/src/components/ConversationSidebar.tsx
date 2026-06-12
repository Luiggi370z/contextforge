import { useState } from "react";
import type { Thread } from "../types";

interface Props {
  threads: Thread[];
  activeThreadId: string | null;
  loading: boolean;
  deletingThreadId: string | null;
  onNewThread: () => void;
  onSelectThread: (threadId: string) => void;
  onDeleteThread: (threadId: string) => void;
}

function formatThreadLabel(thread: Thread): string {
  if (thread.title?.trim()) {
    return thread.title.trim();
  }
  return `Chat ${thread.id.slice(0, 8)}`;
}

function formatThreadDate(createdAt: string): string {
  const date = new Date(createdAt);
  if (Number.isNaN(date.getTime())) {
    return "";
  }
  return date.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
  });
}

export function ConversationSidebar({
  threads,
  activeThreadId,
  loading,
  deletingThreadId,
  onNewThread,
  onSelectThread,
  onDeleteThread,
}: Props) {
  const [collapsed, setCollapsed] = useState(false);

  if (collapsed) {
    return (
      <aside className="flex w-12 shrink-0 flex-col items-center gap-2 border-r border-slate-700 bg-slate-950/80 py-3">
        <button
          type="button"
          aria-label="Expand conversations"
          title="Expand conversations"
          className="rounded-lg border border-slate-600 px-2 py-1.5 text-sm text-slate-300 hover:bg-slate-800"
          onClick={() => setCollapsed(false)}
        >
          »
        </button>
        <button
          type="button"
          aria-label="New conversation"
          title="New conversation"
          className="rounded-lg bg-sky-600 px-2 py-1.5 text-sm font-medium hover:bg-sky-500"
          onClick={onNewThread}
        >
          +
        </button>
      </aside>
    );
  }

  return (
    <aside className="flex w-64 shrink-0 flex-col border-r border-slate-700 bg-slate-950/80">
      <div className="flex items-center gap-2 border-b border-slate-700 p-3">
        <button
          type="button"
          className="flex-1 rounded-lg bg-sky-600 px-3 py-2 text-sm font-medium hover:bg-sky-500"
          onClick={onNewThread}
        >
          New conversation
        </button>
        <button
          type="button"
          aria-label="Collapse conversations"
          title="Collapse conversations"
          className="rounded-lg border border-slate-600 px-2 py-2 text-sm text-slate-300 hover:bg-slate-800"
          onClick={() => setCollapsed(true)}
        >
          «
        </button>
      </div>

      <nav className="flex-1 overflow-y-auto p-2" aria-label="Conversations">
        {threads.length === 0 ? (
          <p className="px-2 py-3 text-sm text-slate-500">
            No conversations yet
          </p>
        ) : (
          <ul className="space-y-1">
            {threads.map((thread) => {
              const isActive = activeThreadId === thread.id;
              const isDeleting = deletingThreadId === thread.id;
              return (
                <li key={thread.id}>
                  <div
                    className={`group flex items-start gap-1 rounded-lg ${
                      isActive
                        ? "bg-sky-600/20 ring-1 ring-sky-600/50"
                        : "hover:bg-slate-800/80"
                    }`}
                  >
                    <button
                      type="button"
                      disabled={loading || isDeleting}
                      className="min-w-0 flex-1 px-3 py-2 text-left disabled:opacity-50"
                      onClick={() => onSelectThread(thread.id)}
                    >
                      <span className="block truncate text-sm font-medium text-slate-100">
                        {formatThreadLabel(thread)}
                      </span>
                      <span className="block text-xs text-slate-500">
                        {formatThreadDate(thread.createdAt)}
                      </span>
                    </button>
                    <button
                      type="button"
                      aria-label={`Delete ${formatThreadLabel(thread)}`}
                      disabled={loading || isDeleting}
                      className="mt-1 mr-1 shrink-0 rounded px-2 py-1 text-xs text-slate-400 opacity-0 transition hover:bg-red-950 hover:text-red-300 group-hover:opacity-100 disabled:opacity-50"
                      onClick={(event) => {
                        event.stopPropagation();
                        onDeleteThread(thread.id);
                      }}
                    >
                      {isDeleting ? "…" : "Delete"}
                    </button>
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </nav>
    </aside>
  );
}
