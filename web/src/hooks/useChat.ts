import { useCallback, useEffect, useReducer, useState } from "react";
import { flushSync } from "react-dom";
import type { ThreadDetail } from "../types";
import { chatReducer, initialChatState } from "./chatReducer";
import { streamQuery } from "./useSSE";

/**
 * Chat screen state: one useReducer for related data, one useState for isolated UI prefs.
 * See docs/CODING_STANDARDS.md — prefer reducer over many useState hooks in one feature.
 */
export function useChat() {
  const [state, dispatch] = useReducer(chatReducer, initialChatState);
  const [showDebug, setShowDebug] = useState(true);
  const [deletingThreadId, setDeletingThreadId] = useState<string | null>(null);

  const loadThreads = useCallback(async () => {
    const response = await fetch("/v1/threads");
    if (response.ok) {
      dispatch({ type: "threads_loaded", threads: await response.json() });
    }
  }, []);

  const loadThread = useCallback(async (selectedThreadId: string) => {
    dispatch({ type: "thread_load_start", threadId: selectedThreadId });
    try {
      const response = await fetch(`/v1/threads/${selectedThreadId}`);
      if (!response.ok) return;
      const detail = (await response.json()) as ThreadDetail;
      dispatch({
        type: "thread_loaded",
        messages: detail.messages.map((message) => ({
          role: message.role === "assistant" ? "assistant" : "user",
          content: message.content,
          metadata: message.metadata,
          citations: message.citations,
        })),
      });
    } finally {
      dispatch({ type: "thread_load_end" });
    }
  }, []);

  const startNewThread = useCallback(() => {
    dispatch({ type: "new_thread" });
  }, []);

  const deleteThread = useCallback(async (threadId: string) => {
    setDeletingThreadId(threadId);
    try {
      const response = await fetch(`/v1/threads/${threadId}`, {
        method: "DELETE",
      });
      if (!response.ok) {
        return;
      }
      dispatch({ type: "thread_deleted", threadId });
    } finally {
      setDeletingThreadId(null);
    }
  }, []);

  useEffect(() => {
    void loadThreads();
  }, [loadThreads]);

  const sendMessage = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || state.loading) return;

      dispatch({ type: "send_start", userMessage: trimmed });
      let assistant = "";

      try {
        const result = await streamQuery(
          trimmed,
          state.threadId,
          async (token) => {
            assistant += token;
            flushSync(() => {
              dispatch({ type: "stream_token", assistantContent: assistant });
            });
            await new Promise<void>((resolve) => {
              requestAnimationFrame(() => resolve());
            });
          },
          async (stage) => {
            flushSync(() => {
              dispatch({ type: "stream_stage", stage });
            });
          },
        );
        dispatch({
          type: "send_success",
          threadId: result.threadId,
          answer: result.answer,
          metadata: result.metadata,
          citations: result.citations,
        });
        await loadThreads();
      } catch (error) {
        dispatch({
          type: "send_error",
          message: `Error: ${error instanceof Error ? error.message : String(error)}`,
        });
      } finally {
        dispatch({ type: "send_end" });
      }
    },
    [state.loading, state.threadId, loadThreads],
  );

  return {
    threads: state.threads,
    threadId: state.threadId,
    messages: state.messages,
    input: state.input,
    setInput: (input: string) => dispatch({ type: "input_changed", input }),
    loading: state.loading,
    showDebug,
    setShowDebug,
    streamStage: state.streamStage,
    threadLoading: state.threadLoading,
    deletingThreadId,
    loadThreads,
    loadThread,
    startNewThread,
    deleteThread,
    sendMessage,
  };
}
