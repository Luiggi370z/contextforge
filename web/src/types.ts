export type RouteKind = "direct" | "single_hop_rag" | "multi_hop" | "unknown";

export interface Citation {
  document_id?: string;
  chunk_id?: string;
  snippet: string;
  score?: number;
}

export interface QueryMetadata {
  route: RouteKind;
  abstained: boolean;
  nodes_visited: string[];
  retrieval_scores: number[];
  graph_checkpoint_enabled?: boolean;
}

export interface QueryResponse {
  answer: string;
  thread_id?: string;
  citations: Citation[];
  metadata: QueryMetadata;
}

export interface Thread {
  id: string;
  title: string | null;
  created_at: string;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  metadata?: QueryMetadata;
  citations?: Citation[];
}
