import type { Thread } from "../types";

interface Props {
  threads: Thread[];
  activeThreadId: string | null;
  loading: boolean;
  onNewThread: () => void;
  onSelectThread: (threadId: string) => void;
}

export function ThreadBar({
  threads,
  activeThreadId,
  loading,
  onNewThread,
  onSelectThread,
}: Props) {
  return (
    <div className="flex gap-2 overflow-x-auto text-sm">
      <button
        type="button"
        className={`shrink-0 rounded-full px-3 py-1 ${activeThreadId === null ? "bg-sky-600" : "bg-slate-800"}`}
        onClick={onNewThread}
      >
        New thread
      </button>
      {threads.map((thread) => (
        <button
          key={thread.id}
          type="button"
          disabled={loading}
          className={`shrink-0 rounded-full px-3 py-1 ${activeThreadId === thread.id ? "bg-sky-600" : "bg-slate-800"} disabled:opacity-50`}
          onClick={() => onSelectThread(thread.id)}
        >
          {thread.title ?? thread.id.slice(0, 8)}
        </button>
      ))}
    </div>
  );
}
