import { SSEParser } from "./sse";
import type {
  ChatDone,
  ChatMeta,
  ConversationDetail,
  ConversationSummary,
  DocumentItem,
  Source,
  UploadResult,
} from "./types";

export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(
  /\/$/,
  "",
);

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, init);
  if (!response.ok) {
    throw new ApiError(response.status, await errorDetail(response));
  }
  return (response.status === 204 ? undefined : await response.json()) as T;
}

async function errorDetail(response: Response): Promise<string> {
  try {
    const body = await response.json();
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) return body.detail.map((d: { msg: string }) => d.msg).join("; ");
  } catch {
    // not JSON
  }
  return `Request failed (${response.status})`;
}

export const api = {
  listDocuments: () => request<DocumentItem[]>("/api/documents"),

  uploadDocument: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<UploadResult>("/api/documents", { method: "POST", body: form });
  },

  deleteDocument: (id: string) => request<void>(`/api/documents/${id}`, { method: "DELETE" }),

  listConversations: () => request<ConversationSummary[]>("/api/conversations"),

  getConversation: (id: string) => request<ConversationDetail>(`/api/conversations/${id}`),

  deleteConversation: (id: string) =>
    request<void>(`/api/conversations/${id}`, { method: "DELETE" }),
};

export interface ChatHandlers {
  onMeta: (meta: ChatMeta) => void;
  onSources: (sources: Source[]) => void;
  onToken: (text: string) => void;
  onDone: (done: ChatDone) => void;
  onError: (detail: string) => void;
}

export interface ChatParams {
  message: string;
  conversationId?: string | null;
  documentIds?: string[];
  signal?: AbortSignal;
}

/** POST /api/chat and dispatch the Server-Sent Events as they arrive. */
export async function streamChat(params: ChatParams, handlers: ChatHandlers): Promise<void> {
  const response = await fetch(`${API_URL}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify({
      message: params.message,
      conversation_id: params.conversationId ?? null,
      document_ids: params.documentIds?.length ? params.documentIds : null,
    }),
    signal: params.signal,
  });
  if (!response.ok || !response.body) {
    throw new ApiError(response.status, await errorDetail(response));
  }

  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  const parser = new SSEParser();
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    for (const { event, data } of parser.push(value)) {
      const payload = JSON.parse(data);
      switch (event) {
        case "meta":
          handlers.onMeta(payload as ChatMeta);
          break;
        case "sources":
          handlers.onSources((payload as { sources: Source[] }).sources);
          break;
        case "token":
          handlers.onToken((payload as { text: string }).text);
          break;
        case "done":
          handlers.onDone(payload as ChatDone);
          break;
        case "error":
          handlers.onError((payload as { detail: string }).detail);
          break;
      }
    }
  }
}
