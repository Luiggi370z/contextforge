import { render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { EvalPanel } from "./EvalPanel";

beforeEach(() => {
  globalThis.fetch = vi.fn(
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

test("explains the question each metric answers", async () => {
  render(<EvalPanel />);
  await waitFor(() =>
    expect(screen.getByText("hybrid_rerank")).toBeInTheDocument(),
  );
  // Each column header carries the plain-language question as a subtitle.
  expect(screen.getByText("Is a correct doc ranked #1?")).toBeInTheDocument();
  expect(
    screen.getByText("How high is the first correct doc? (avg 1/rank)"),
  ).toBeInTheDocument();
});

test("flags pass/fail per metric against that metric's own bar", async () => {
  render(<EvalPanel />);
  await waitFor(() =>
    expect(screen.getByText("hybrid_rerank")).toBeInTheDocument(),
  );
  // Thresholds are per-metric: recall@5 bar is 0.9, so 0.85 fails (✗ no), while
  // mrr's bar is 0.7, so 0.7 passes (✓ yes). The verdict lives in the same cell
  // as the score, so scope each assertion to that number's cell.
  const recallCell = screen.getByText("0.850").closest("td");
  const mrrCell = screen.getByText("0.700").closest("td");
  expect(recallCell).not.toBeNull();
  expect(mrrCell).not.toBeNull();
  expect(
    within(recallCell as HTMLElement).getByText("✗ no"),
  ).toBeInTheDocument();
  expect(within(mrrCell as HTMLElement).getByText("✓ yes")).toBeInTheDocument();
});

test("collapses run history to the latest run per config", async () => {
  // The API returns full history (newest first), so two runs of the same config
  // arrive. The panel must show only the most recent, not append duplicates.
  globalThis.fetch = vi.fn(
    async () =>
      new Response(
        JSON.stringify({
          items: [
            {
              id: "2",
              label: "hybrid_rerank",
              config: {},
              metrics: { "recall@1": 0.9 },
              rowCount: 25,
              createdAt: "2026-01-02",
            },
            {
              id: "1",
              label: "hybrid_rerank",
              config: {},
              metrics: { "recall@1": 0.44 },
              rowCount: 25,
              createdAt: "2026-01-01",
            },
          ],
          total: 2,
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
  ) as unknown as typeof fetch;

  render(<EvalPanel />);
  await waitFor(() =>
    expect(screen.getByText("hybrid_rerank")).toBeInTheDocument(),
  );
  // Only one row for the config, and it is the newest (0.900, not the stale 0.440).
  expect(screen.getAllByText("hybrid_rerank")).toHaveLength(1);
  expect(screen.getByText("0.900")).toBeInTheDocument();
  expect(screen.queryByText("0.440")).not.toBeInTheDocument();
});
