export type RouteKind = "direct" | "single_hop_rag" | "multi_hop" | "unknown";

export interface Citation {
  documentId?: string;
  chunkId?: string;
  snippet: string;
  score?: number;
}

export interface QueryMetadata {
  route: RouteKind;
  abstained: boolean;
  nodesVisited: string[];
  retrievalScores: number[];
  graphCheckpointEnabled?: boolean;
}

export interface QueryResponse {
  answer: string;
  threadId?: string;
  citations: Citation[];
  metadata: QueryMetadata;
}

export interface Thread {
  id: string;
  title: string | null;
  createdAt: string;
}

export interface ThreadMessage {
  id: string;
  role: string;
  content: string;
  createdAt: string;
}

export interface ThreadDetail extends Thread {
  messages: ThreadMessage[];
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  metadata?: QueryMetadata;
  citations?: Citation[];
}

export interface DemoPrompt {
  label: string;
  text: string;
  hint: string;
}
