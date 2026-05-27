import { useEffect, useRef } from "react";
import type { ChatMessage } from "../types";
import { Citations } from "./Citations";
import { DebugPanel } from "./DebugPanel";

interface Props {
  messages: ChatMessage[];
  showDebug: boolean;
  loading: boolean;
  threadLoading: boolean;
}

export function MessageList({
  messages,
  showDebug,
  loading,
  threadLoading,
}: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);

  // Scroll when the list grows or streaming updates the last assistant message.
  // biome-ignore lint/correctness/useExhaustiveDependencies: intentional scroll trigger
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length, loading]);

  return (
    <div className="flex-1 space-y-4 overflow-y-auto rounded-xl border border-slate-700 bg-slate-900/40 p-4">
      {threadLoading && (
        <p className="text-sm text-slate-400">Loading thread…</p>
      )}
      {!threadLoading && messages.length === 0 && (
        <p className="text-slate-500">
          Use the chips above to test each route, or ask about your uploaded
          documents.
        </p>
      )}
      {messages.map((message, index) => (
        <div
          key={`${message.role}-${index}-${message.content.slice(0, 24)}`}
          className={`max-w-[90%] rounded-xl px-4 py-2 ${message.role === "user" ? "ml-auto bg-sky-700" : "bg-slate-800"}`}
        >
          <p className="whitespace-pre-wrap">{message.content}</p>
          {message.role === "assistant" && message.citations && (
            <Citations citations={message.citations} />
          )}
          {message.role === "assistant" && showDebug && (
            <DebugPanel metadata={message.metadata} />
          )}
        </div>
      ))}
      <div ref={bottomRef} />
    </div>
  );
}
