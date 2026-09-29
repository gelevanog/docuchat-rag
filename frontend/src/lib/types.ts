export type DocumentStatus = "pending" | "processing" | "ready" | "failed";

export interface DocumentItem {
  id: string;
  filename: string;
  title: string;
  file_type: string;
  size_bytes: number;
  status: DocumentStatus;
  error: string | null;
  page_count: number | null;
  chunk_count: number;
  created_at: string;
  updated_at: string;
}

export interface UploadResult {
  document: DocumentItem;
  duplicate: boolean;
}

/** A retrieved chunk, numbered the way the model saw it ("[id]" in the answer). */
export interface Source {
  id: number;
  chunk_id: string;
  document_id: string;
  document_title: string;
  filename: string;
  page: number | null;
  heading: string | null;
  snippet: string;
  content: string;
  /** Reciprocal Rank Fusion score of the hybrid search. */
  score: number;
  /** Re-ranker relevance in [0, 1]; null (or absent in older messages) when re-ranking is off. */
  rerank_score?: number | null;
}

export interface ConversationSummary {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface StoredMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  rewritten_query: string | null;
  citations: Source[] | null;
  created_at: string;
}

export interface ConversationDetail extends ConversationSummary {
  messages: StoredMessage[];
}

export interface ChatMeta {
  conversation_id: string;
  user_message_id: string;
  rewritten_query: string;
}

export interface ChatDone {
  message_id: string;
  answer: string;
  citations: Source[];
}

/** Message as rendered by the chat panel. */
export interface UIMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  /** Everything retrieved for this answer (resolves [n] chips while streaming). */
  sources: Source[];
  /** Only the sources the final answer cites. */
  citations: Source[];
  rewrittenQuery?: string | null;
  status: "searching" | "streaming" | "done" | "error";
  error?: string;
}
