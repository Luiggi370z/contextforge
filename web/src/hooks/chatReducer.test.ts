import { describe, expect, it } from "vitest";
import { chatReducer, initialChatState } from "./chatReducer";

describe("chatReducer", () => {
  it("send_start appends user and empty assistant messages", () => {
    const next = chatReducer(initialChatState, {
      type: "send_start",
      userMessage: "hello",
    });
    expect(next.loading).toBe(true);
    expect(next.input).toBe("");
    expect(next.messages).toHaveLength(2);
    expect(next.messages[0]?.role).toBe("user");
    expect(next.messages[1]?.content).toBe("");
  });

  it("thread_deleted clears active conversation when removed", () => {
    const withThread = chatReducer(
      {
        ...initialChatState,
        threadId: "thread-1",
        messages: [{ role: "user", content: "hi" }],
        threads: [
          {
            id: "thread-1",
            title: "Test",
            createdAt: "2026-05-28T00:00:00.000Z",
          },
        ],
      },
      { type: "thread_deleted", threadId: "thread-1" },
    );
    expect(withThread.threadId).toBeNull();
    expect(withThread.messages).toEqual([]);
    expect(withThread.threads).toEqual([]);
  });

  it("stream_token updates the last assistant message", () => {
    const started = chatReducer(initialChatState, {
      type: "send_start",
      userMessage: "hi",
    });
    const streaming = chatReducer(started, {
      type: "stream_token",
      assistantContent: "partial",
    });
    expect(streaming.messages[1]?.content).toBe("partial");
  });

  it("stream_stage accumulates pipeline events and send_success attaches them", () => {
    const started = chatReducer(initialChatState, {
      type: "send_start",
      userMessage: "hi",
    });
    const withStage = chatReducer(started, {
      type: "stream_stage",
      event: { stage: "route", phase: "start" },
    });
    const withMore = chatReducer(withStage, {
      type: "stream_stage",
      event: {
        stage: "generate.llm",
        phase: "start",
        detail: { provider: "ollama" },
      },
    });
    expect(withMore.pipelineEvents).toHaveLength(2);

    const success = chatReducer(withMore, {
      type: "send_success",
      answer: "done",
      metadata: {
        route: "single_hop_rag",
        abstained: false,
        nodesVisited: [],
        retrievalScores: [],
      },
      citations: [],
    });
    expect(success.messages[1]?.pipelineEvents).toHaveLength(2);

    const ended = chatReducer(success, { type: "send_end" });
    expect(ended.pipelineEvents).toEqual([]);
    expect(ended.messages[1]?.pipelineEvents).toHaveLength(2);
  });

  it("send_start resets pipeline events", () => {
    const withEvents = chatReducer(
      {
        ...initialChatState,
        pipelineEvents: [{ stage: "route", phase: "start" }],
      },
      { type: "send_start", userMessage: "again" },
    );
    expect(withEvents.pipelineEvents).toEqual([]);
  });

  it("new_thread clears thread id and messages", () => {
    const withData = chatReducer(
      {
        ...initialChatState,
        threadId: "abc",
        messages: [{ role: "user", content: "x" }],
      },
      { type: "new_thread" },
    );
    expect(withData.threadId).toBeNull();
    expect(withData.messages).toEqual([]);
  });
});
