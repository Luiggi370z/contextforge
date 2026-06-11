import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { EvalPanel } from "./EvalPanel";

beforeEach(() => {
  global.fetch = vi.fn(
    async () =>
      new Response(
        JSON.stringify({
          items: [
            {
              id: "1",
              label: "hybrid_rerank",
              config: {},
              metrics: { "recall@5": 0.85, mrr: 0.7 },
              rowCount: 17,
              createdAt: "2026-01-01",
            },
          ],
          total: 1,
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
  ) as unknown as typeof fetch;
});

afterEach(() => vi.restoreAllMocks());

test("renders eval run metrics", async () => {
  render(<EvalPanel />);
  await waitFor(() =>
    expect(screen.getByText("hybrid_rerank")).toBeInTheDocument(),
  );
  expect(screen.getByText("0.850")).toBeInTheDocument();
});
