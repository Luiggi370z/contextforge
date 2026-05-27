import { useCallback, useEffect, useRef, useState } from "react";
import { streamQuery } from "../hooks/useSSE";
import type { ChatMessage, Thread } from "../types";
import { Citations } from "./Citations";
import { DebugPanel } from "./DebugPanel";

export function Chat() {
  const [threads, setThreads] = useState<Thread[]>([]);
  const [threadId, setThreadId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [showDebug, setShowDebug] = useState(true);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);
  const [streamStage, setStreamStage] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  const loadThreads = useCallback(async () => {
    const res = await fetch("/v1/threads");
    if (res.ok) setThreads(await res.json());
  }, []);

  useEffect(() => {
    loadThreads();
  }, [loadThreads]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const send = async () => {
    const text = input.trim();
    if (!text || loading) return;
    setInput("");
    setLoading(true);
    setStreamStage(null);
    setMessages((m) => [...m, { role: "user", content: text }]);
    let assistant = "";
    setMessages((m) => [...m, { role: "assistant", content: "" }]);
    try {
      const result = await streamQuery(
        text,
        threadId,
        (token) => {
          assistant += token;
          setMessages((m) => {
            const copy = [...m];
            copy[copy.length - 1] = { role: "assistant", content: assistant };
            return copy;
          });
        },
        (stage) => setStreamStage(stage),
      );
      if (result.thread_id) setThreadId(result.thread_id);
      setMessages((m) => {
        const copy = [...m];
        copy[copy.length - 1] = {
          role: "assistant",
          content: result.answer,
          metadata: result.metadata,
          citations: result.citations,
        };
        return copy;
      });
      await loadThreads();
    } catch (err) {
      setMessages((m) => {
        const copy = [...m];
        copy[copy.length - 1] = {
          role: "assistant",
          content: `Error: ${err instanceof Error ? err.message : String(err)}`,
        };
        return copy;
      });
    } finally {
      setLoading(false);
      setStreamStage(null);
    }
  };

  const onUpload = async (file: File) => {
    setUploadStatus(`Uploading ${file.name}...`);
    const form = new FormData();
    form.append("file", file);
    const res = await fetch("/v1/documents/upload", { method: "POST", body: form });
    if (!res.ok) {
      setUploadStatus(`Upload failed: ${res.status}`);
      return;
    }
    const doc = await res.json();
    setUploadStatus(`Ingested ${doc.filename} (${doc.status})`);
  };

  return (
    <div className="mx-auto flex h-screen max-w-4xl flex-col gap-3 p-4">
      <header className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-700 pb-3">
        <div>
          <h1 className="text-xl font-semibold text-white">ContextForge</h1>
          <p className="text-sm text-slate-400">Agentic RAG chat — test routes & citations</p>
        </div>
        <div className="flex items-center gap-2">
          <label className="flex cursor-pointer items-center gap-2 rounded-lg border border-slate-600 px-3 py-1.5 text-sm hover:bg-slate-800">
            Upload
            <input
              type="file"
              accept=".md,.txt,.markdown"
              className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) onUpload(f);
              }}
            />
          </label>
          <button
            type="button"
            onClick={() => setShowDebug((v) => !v)}
            className="rounded-lg border border-slate-600 px-3 py-1.5 text-sm hover:bg-slate-800"
          >
            Debug {showDebug ? "on" : "off"}
          </button>
        </div>
      </header>

      {uploadStatus && <p className="text-sm text-emerald-400">{uploadStatus}</p>}

      <div className="flex gap-2 overflow-x-auto text-sm">
        <button
          type="button"
          className={`shrink-0 rounded-full px-3 py-1 ${threadId === null ? "bg-sky-600" : "bg-slate-800"}`}
          onClick={() => {
            setThreadId(null);
            setMessages([]);
          }}
        >
          New thread
        </button>
        {threads.map((t) => (
          <button
            key={t.id}
            type="button"
            className={`shrink-0 rounded-full px-3 py-1 ${threadId === t.id ? "bg-sky-600" : "bg-slate-800"}`}
            onClick={() => setThreadId(t.id)}
          >
            {t.title ?? t.id.slice(0, 8)}
          </button>
        ))}
      </div>

      {loading && streamStage && (
        <p className="text-xs font-mono text-slate-400">Graph: {streamStage}</p>
      )}

      <div className="flex-1 space-y-4 overflow-y-auto rounded-xl border border-slate-700 bg-slate-900/40 p-4">
        {messages.length === 0 && (
          <p className="text-slate-500">
            Try: &quot;hi&quot; (direct), &quot;How many PTO days?&quot; (RAG), &quot;Compare PTO then
            remote work steps&quot; (multi-hop)
          </p>
        )}
        {messages.map((m, i) => (
          <div
            key={`${m.role}-${i}`}
            className={`max-w-[90%] rounded-xl px-4 py-2 ${m.role === "user" ? "ml-auto bg-sky-700" : "bg-slate-800"}`}
          >
            <p className="whitespace-pre-wrap">{m.content}</p>
            {m.role === "assistant" && m.citations && <Citations citations={m.citations} />}
            {m.role === "assistant" && showDebug && <DebugPanel metadata={m.metadata} />}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      <form
        className="flex gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          send();
        }}
      >
        <input
          className="flex-1 rounded-xl border border-slate-600 bg-slate-900 px-4 py-2 outline-none focus:border-sky-500"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about your documents..."
          disabled={loading}
        />
        <button
          type="submit"
          disabled={loading}
          className="rounded-xl bg-sky-600 px-5 py-2 font-medium disabled:opacity-50"
        >
          Send
        </button>
      </form>
    </div>
  );
}
