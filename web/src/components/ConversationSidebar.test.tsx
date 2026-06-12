import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ConversationSidebar } from "./ConversationSidebar";

const threads = [
  {
    id: "11111111-1111-1111-1111-111111111111",
    title: "PTO question",
    createdAt: "2026-05-28T12:00:00.000Z",
  },
];

describe("ConversationSidebar", () => {
  it("renders conversations and handles selection", () => {
    const onSelectThread = vi.fn();
    render(
      <ConversationSidebar
        threads={threads}
        activeThreadId={null}
        loading={false}
        deletingThreadId={null}
        onNewThread={vi.fn()}
        onSelectThread={onSelectThread}
        onDeleteThread={vi.fn()}
      />,
    );

    fireEvent.click(screen.getByText("PTO question"));
    expect(onSelectThread).toHaveBeenCalledWith(threads[0].id);
  });

  it("calls delete without selecting the thread", () => {
    const onSelectThread = vi.fn();
    const onDeleteThread = vi.fn();
    render(
      <ConversationSidebar
        threads={threads}
        activeThreadId={null}
        loading={false}
        deletingThreadId={null}
        onNewThread={vi.fn()}
        onSelectThread={onSelectThread}
        onDeleteThread={onDeleteThread}
      />,
    );

    fireEvent.click(
      screen.getByRole("button", { name: "Delete PTO question" }),
    );
    expect(onDeleteThread).toHaveBeenCalledWith(threads[0].id);
    expect(onSelectThread).not.toHaveBeenCalled();
  });
});
