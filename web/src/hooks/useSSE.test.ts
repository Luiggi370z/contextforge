import { afterEach, describe, expect, it, vi } from "vitest";
import type { QueryResponse } from "../types";
import { streamQuery } from "./useSSE";

function buildSseResponse(events: object[]): Response {
  const body = events
    .map((event) => `data: ${JSON.stringify(event)}\n\n`)
    .join("");
  return new Response(
    new ReadableStream({
      start(controller) {
        controller.enqueue(new TextEncoder().encode(body));
        controller.close();
      },
    }),
    { status: 200 },
  );
}

describe("streamQuery", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("parses status, token, and done events", async () => {
    const final: QueryResponse = {
      answer: "full answer",
      threadId: "thread-1",
      citations: [{ snippet: "policy text", score: 0.9 }],
      metadata: {
        route: "single_hop_rag",
        abstained: false,
        nodesVisited: ["route", "retrieve"],
        retrievalScores: [0.9],
      },
    };

    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        buildSseResponse([
          { type: "status", stage: "started" },
          { type: "token", content: "full " },
          { type: "token", content: "answer" },
          { type: "done", result: final },
        ]),
      ),
    );

    const tokens: string[] = [];
    const stages: string[] = [];
    const result = await streamQuery(
      "PTO?",
      null,
      (chunk) => tokens.push(chunk),
      (stage) => stages.push(stage),
    );

    expect(tokens.join("")).toBe("full answer");
    expect(stages).toContain("started");
    expect(result.threadId).toBe("thread-1");
    expect(result.citations[0]?.snippet).toBe("policy text");
  });

  it("throws on error events", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(
          buildSseResponse([{ type: "error", detail: "Something went wrong" }]),
        ),
    );

    await expect(streamQuery("x", null, () => {})).rejects.toThrow(
      "Something went wrong",
    );
  });
});
