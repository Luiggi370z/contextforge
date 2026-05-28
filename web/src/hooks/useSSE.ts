import type { QueryResponse } from "../types";

export type StreamStatusHandler = (stage: string) => void | Promise<void>;

export type StreamTokenHandler = (chunk: string) => void | Promise<void>;

interface StreamPayload {
  type: string;
  content?: string;
  stage?: string;
  detail?: string;
  correlationId?: string;
  result?: QueryResponse;
}

export async function streamQuery(
  message: string,
  threadId: string | null,
  onToken: StreamTokenHandler,
  onStatus?: StreamStatusHandler,
): Promise<QueryResponse> {
  const res = await fetch("/v1/query/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, threadId }),
  });
  if (!res.ok) throw new Error(`Query failed: ${res.status}`);
  if (!res.body) throw new Error("No response body");

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalResult: QueryResponse | null = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() ?? "";
    for (const part of parts) {
      for (const line of part.split("\n")) {
        if (!line.startsWith("data: ")) continue;
        const payload = JSON.parse(line.slice(6)) as StreamPayload;
        if (payload.type === "status" && payload.stage) {
          await onStatus?.(payload.stage);
        }
        if (payload.type === "token" && payload.content) {
          await onToken(payload.content);
        }
        if (payload.type === "done" && payload.result) {
          finalResult = payload.result;
        }
        if (payload.type === "error") {
          throw new Error(payload.detail ?? "Stream error");
        }
      }
    }
  }
  if (!finalResult) throw new Error("Stream ended without result");
  return finalResult;
}
