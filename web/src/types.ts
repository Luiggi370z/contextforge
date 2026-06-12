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
  metadata?: QueryMetadata;
  citations?: Citation[];
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

export interface DocumentInfo {
  id: string;
  filename: string;
  contentType: string;
  status: string;
  sizeBytes?: number | null;
  createdAt: string;
}

export interface DocumentList {
  items: DocumentInfo[];
  total: number;
}

export interface IngestionJob {
  id: string;
  filename: string;
  status: "queued" | "processing" | "completed" | "failed";
  progress: number;
  documentId?: string;
  error?: string;
}

export interface EvalRun {
  id: string;
  label: string;
  config: Record<string, unknown>;
  metrics: Record<string, number>;
  rowCount: number;
  createdAt: string;
}

export interface EvalRunList {
  items: EvalRun[];
  total: number;
}
