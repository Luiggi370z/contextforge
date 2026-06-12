import { useEffect, useState } from "react";
import { useChat } from "../hooks/useChat";
import { ConversationSidebar } from "./ConversationSidebar";
import { DemoPrompts } from "./DemoPrompts";
import { EvalPanel } from "./EvalPanel";
import { MessageList } from "./MessageList";
import { PipelineDiagram } from "./PipelineDiagram";

function EvalModal({ onClose }: { onClose: () => void }) {
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-6">
      <button
        type="button"
        aria-label="Close retrieval eval"
        className="absolute inset-0 cursor-default bg-black/60"
        onClick={onClose}
      />
      <dialog
        open
        aria-label="Retrieval eval"
        className="relative m-0 max-h-[85vh] w-full max-w-4xl overflow-y-auto rounded-xl border border-slate-700 bg-[#0f1419] p-4 text-slate-100 shadow-2xl"
      >
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-white">Retrieval eval</h2>
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg border border-slate-600 px-2 py-1 text-sm text-slate-300 hover:bg-slate-800"
          >
            Close
          </button>
        </div>
        <p className="mt-2 text-xs leading-relaxed text-slate-400">
          Runs a fixed set of golden questions against the indexed corpus and
          scores how well retrieval ranks the documents known to contain each
          answer (recall, MRR, nDCG). Answers are only as good as the chunks
          handed to the model — if retrieval misses, generation hallucinates or
          abstains. Run it after uploading documents or changing retrieval
          settings to catch ranking regressions before they reach the chat.
        </p>
        <EvalPanel />
      </dialog>
    </div>
  );
}

export function Chat() {
  const chat = useChat();
  const [evalOpen, setEvalOpen] = useState(false);
  const [pipelineOpen, setPipelineOpen] = useState(true);

  const conversationEmpty =
    chat.messages.length === 0 && !chat.threadLoading && !chat.loading;

  const inputForm = (
    <form
      className="flex w-full gap-2"
      onSubmit={(event) => {
        event.preventDefault();
        void chat.sendMessage(chat.input);
      }}
    >
      <input
        className="flex-1 rounded-xl border border-slate-600 bg-slate-900 px-4 py-2 outline-none focus:border-sky-500"
        value={chat.input}
        onChange={(event) => chat.setInput(event.target.value)}
        placeholder="Ask about your documents..."
        disabled={chat.loading}
      />
      <button
        type="submit"
        disabled={chat.loading}
        className="rounded-xl bg-sky-600 px-5 py-2 font-medium disabled:opacity-50"
      >
        Send
      </button>
    </form>
  );

  // Live events while streaming; afterwards the path stored on the last answer.
  const lastAssistantEvents = [...chat.messages]
    .reverse()
    .find(
      (message) =>
        message.role === "assistant" &&
        (message.pipelineEvents?.length ?? 0) > 0,
    )?.pipelineEvents;
  const diagramEvents = chat.loading
    ? chat.pipelineEvents
    : lastAssistantEvents;

  return (
    <div className="flex h-screen bg-[#0f1419] text-slate-100">
      <ConversationSidebar
        threads={chat.threads}
        activeThreadId={chat.threadId}
        loading={chat.threadLoading}
        deletingThreadId={chat.deletingThreadId}
        onNewThread={chat.startNewThread}
        onSelectThread={(id) => void chat.loadThread(id)}
        onDeleteThread={(id) => void chat.deleteThread(id)}
      />

      <div className="flex min-w-0 flex-1 flex-col gap-3 p-4">
        <header className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-700 pb-3">
          <div>
            <h1 className="text-xl font-semibold text-white">ContextForge</h1>
            <p className="text-sm text-slate-400">
              Agentic RAG chat — routes, citations, debug
            </p>
          </div>
          <div className="flex items-center gap-2">
            <a
              href="#/documents"
              className="rounded-lg border border-slate-600 px-3 py-1.5 text-sm hover:bg-slate-800"
            >
              Documents
            </a>
            <button
              type="button"
              onClick={() => setEvalOpen(true)}
              className="rounded-lg border border-slate-600 px-3 py-1.5 text-sm hover:bg-slate-800"
            >
              Retrieval eval
            </button>
          </div>
        </header>

        {evalOpen && <EvalModal onClose={() => setEvalOpen(false)} />}

        {conversationEmpty ? (
          <div className="flex flex-1 flex-col items-center justify-center gap-6">
            <div className="w-full max-w-2xl">
              <DemoPrompts
                disabled={chat.loading}
                onSelect={(text) => {
                  chat.setInput(text);
                  void chat.sendMessage(text);
                }}
              />
            </div>
            <div className="w-full max-w-2xl">{inputForm}</div>
          </div>
        ) : (
          <>
            <DemoPrompts
              disabled={chat.loading}
              onSelect={(text) => {
                chat.setInput(text);
                void chat.sendMessage(text);
              }}
            />

            {diagramEvents && diagramEvents.length > 0 && (
              <section className="rounded-lg border border-slate-700 bg-slate-900/60">
                <button
                  type="button"
                  onClick={() => setPipelineOpen((value) => !value)}
                  aria-expanded={pipelineOpen}
                  className="flex w-full items-center justify-between px-3 py-2 text-left font-mono text-[11px] uppercase tracking-wide text-slate-400 hover:text-slate-200"
                >
                  <span>RAG pipeline</span>
                  <span>{pipelineOpen ? "Hide" : "View"}</span>
                </button>
                {pipelineOpen && (
                  <div className="px-3 pb-3">
                    <PipelineDiagram
                      events={diagramEvents}
                      finished={!chat.loading}
                    />
                  </div>
                )}
              </section>
            )}

            <MessageList
              messages={chat.messages}
              loading={chat.loading}
              threadLoading={chat.threadLoading}
              pipelineEvents={chat.pipelineEvents}
            />

            {inputForm}
          </>
        )}
      </div>
    </div>
  );
}
