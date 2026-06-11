import { useChat } from "../hooks/useChat";
import { ConversationSidebar } from "./ConversationSidebar";
import { DemoPrompts } from "./DemoPrompts";
import { EvalPanel } from "./EvalPanel";
import { MessageList } from "./MessageList";

export function Chat() {
  const chat = useChat();

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
            <label className="flex cursor-pointer items-center gap-2 rounded-lg border border-slate-600 px-3 py-1.5 text-sm hover:bg-slate-800">
              Upload
              <input
                type="file"
                accept=".md,.txt,.markdown"
                className="hidden"
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) chat.uploadDocument(file);
                }}
              />
            </label>
            <button
              type="button"
              onClick={() => chat.setShowDebug((value) => !value)}
              className="rounded-lg border border-slate-600 px-3 py-1.5 text-sm hover:bg-slate-800"
            >
              Debug {chat.showDebug ? "on" : "off"}
            </button>
          </div>
        </header>

        {chat.uploadStatus && (
          <p className="text-sm text-emerald-400">{chat.uploadStatus}</p>
        )}

        {chat.showDebug && <EvalPanel />}

        <DemoPrompts
          disabled={chat.loading}
          onSelect={(text) => {
            chat.setInput(text);
            void chat.sendMessage(text);
          }}
        />

        <MessageList
          messages={chat.messages}
          showDebug={chat.showDebug}
          loading={chat.loading}
          threadLoading={chat.threadLoading}
          streamStage={chat.streamStage}
        />

        <form
          className="flex gap-2"
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
      </div>
    </div>
  );
}
