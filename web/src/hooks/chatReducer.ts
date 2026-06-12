import type {
  ChatMessage,
  PipelineEvent,
  QueryMetadata,
  Thread,
} from "../types";
import type { Citation } from "../types";

export interface ChatState {
  threads: Thread[];
  threadId: string | null;
  messages: ChatMessage[];
  input: string;
  loading: boolean;
  pipelineEvents: PipelineEvent[];
  threadLoading: boolean;
}

export const initialChatState: ChatState = {
  threads: [],
  threadId: null,
  messages: [],
  input: "",
  loading: false,
  pipelineEvents: [],
  threadLoading: false,
};

export type ChatAction =
  | { type: "threads_loaded"; threads: Thread[] }
  | { type: "input_changed"; input: string }
  | { type: "new_thread" }
  | { type: "thread_load_start"; threadId: string }
  | { type: "thread_loaded"; messages: ChatMessage[] }
  | { type: "thread_load_end" }
  | { type: "thread_deleted"; threadId: string }
  | { type: "send_start"; userMessage: string }
  | { type: "stream_stage"; event: PipelineEvent }
  | { type: "stream_token"; assistantContent: string }
  | {
      type: "send_success";
      threadId?: string;
      answer: string;
      metadata: QueryMetadata;
      citations: Citation[];
    }
  | { type: "send_error"; message: string }
  | { type: "send_end" };

function updateLastAssistant(
  messages: ChatMessage[],
  content: string,
  extras?: Partial<ChatMessage>,
): ChatMessage[] {
  if (messages.length === 0) return messages;
  const copy = [...messages];
  const lastIndex = copy.length - 1;
  const last = copy[lastIndex];
  if (last?.role !== "assistant") return messages;
  copy[lastIndex] = { ...last, content, ...extras };
  return copy;
}

export function chatReducer(state: ChatState, action: ChatAction): ChatState {
  switch (action.type) {
    case "threads_loaded":
      return { ...state, threads: action.threads };
    case "input_changed":
      return { ...state, input: action.input };
    case "new_thread":
      return { ...state, threadId: null, messages: [] };
    case "thread_load_start":
      return {
        ...state,
        threadId: action.threadId,
        threadLoading: true,
        messages: [],
      };
    case "thread_loaded":
      return { ...state, messages: action.messages };
    case "thread_load_end":
      return { ...state, threadLoading: false };
    case "thread_deleted": {
      const wasActive = state.threadId === action.threadId;
      return {
        ...state,
        threads: state.threads.filter(
          (thread) => thread.id !== action.threadId,
        ),
        threadId: wasActive ? null : state.threadId,
        messages: wasActive ? [] : state.messages,
        threadLoading: false,
      };
    }
    case "send_start":
      return {
        ...state,
        input: "",
        loading: true,
        pipelineEvents: [],
        messages: [
          ...state.messages,
          { role: "user", content: action.userMessage },
          { role: "assistant", content: "" },
        ],
      };
    case "stream_stage":
      return {
        ...state,
        pipelineEvents: [...state.pipelineEvents, action.event],
      };
    case "stream_token":
      return {
        ...state,
        messages: updateLastAssistant(state.messages, action.assistantContent),
      };
    case "send_success":
      return {
        ...state,
        threadId: action.threadId ?? state.threadId,
        messages: updateLastAssistant(state.messages, action.answer, {
          metadata: action.metadata,
          citations: action.citations,
          // Keep the traversed path inspectable on the finished message.
          pipelineEvents: state.pipelineEvents,
        }),
      };
    case "send_error":
      return {
        ...state,
        messages: updateLastAssistant(state.messages, action.message),
      };
    case "send_end":
      return { ...state, loading: false, pipelineEvents: [] };
    default:
      return state;
  }
}
