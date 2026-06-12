import { useEffect, useRef } from "react";
import type { ChatMessage, PipelineEvent } from "../types";
import { Citations } from "./Citations";
import { DebugPanel } from "./DebugPanel";
import { PipelineStepper, PipelineSummary } from "./PipelineStepper";

interface Props {
  messages: ChatMessage[];
  loading: boolean;
  threadLoading: boolean;
  pipelineEvents?: PipelineEvent[];
}

export function MessageList({
  messages,
  loading,
  threadLoading,
  pipelineEvents = [],
}: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const lastMessage = messages.at(-1);
  const showThinking =
    loading && lastMessage?.role === "assistant" && !lastMessage.content.trim();

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
      {messages.map((message, index) => {
        const isEmptyStreamingAssistant =
          loading &&
          message.role === "assistant" &&
          index === messages.length - 1 &&
          !message.content.trim();
        if (isEmptyStreamingAssistant) {
          return null;
        }
        return (
          <div
            key={`${message.role}-${index}-${message.content.slice(0, 24)}`}
            className={`max-w-[90%] rounded-xl px-4 py-2 ${message.role === "user" ? "ml-auto bg-sky-700" : "bg-slate-800"}`}
          >
            <p className="whitespace-pre-wrap">{message.content}</p>
            {message.role === "assistant" && message.citations && (
              <Citations citations={message.citations} />
            )}
            {message.role === "assistant" &&
              message.pipelineEvents &&
              message.pipelineEvents.length > 0 && (
                <PipelineSummary events={message.pipelineEvents} />
              )}
            {message.role === "assistant" && message.metadata && (
              <details className="mt-2 text-xs">
                <summary className="cursor-pointer select-none font-mono text-[11px] text-slate-500 hover:text-slate-300">
                  Debug
                </summary>
                <DebugPanel metadata={message.metadata} />
              </details>
            )}
          </div>
        );
      })}
      {showThinking && (
        <div className="max-w-[90%] rounded-xl bg-slate-800 px-4 py-3">
          <p className="thinking-shimmer mb-2 text-sm font-medium tracking-wide">
            Thinking
          </p>
          <PipelineStepper events={pipelineEvents} />
        </div>
      )}
      <div ref={bottomRef} />
    </div>
  );
}
