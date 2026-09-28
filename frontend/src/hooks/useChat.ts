"use client";

import { useCallback, useRef, useState } from "react";

import { api, streamChat } from "@/lib/api";
import type { UIMessage } from "@/lib/types";

let localId = 0;
const nextId = () => `local-${++localId}`;

export function useChat(onConversationChanged: () => void) {
  const [messages, setMessages] = useState<UIMessage[]>([]);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [sending, setSending] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const patchMessage = useCallback((id: string, patch: (m: UIMessage) => Partial<UIMessage>) => {
    setMessages((all) => all.map((m) => (m.id === id ? { ...m, ...patch(m) } : m)));
  }, []);

  const send = useCallback(
    async (text: string, documentIds: string[]) => {
      const question = text.trim();
      if (!question || sending) return;

      const assistantId = nextId();
      setMessages((all) => [
        ...all,
        { id: nextId(), role: "user", content: question, sources: [], citations: [], status: "done" },
        { id: assistantId, role: "assistant", content: "", sources: [], citations: [], status: "searching" },
      ]);
      setSending(true);
      const controller = new AbortController();
      abortRef.current = controller;

      try {
        await streamChat(
          { message: question, conversationId, documentIds, signal: controller.signal },
          {
            onMeta: (meta) => {
              setConversationId(meta.conversation_id);
              patchMessage(assistantId, () => ({
                rewrittenQuery: meta.rewritten_query !== question ? meta.rewritten_query : null,
              }));
            },
            onSources: (sources) => patchMessage(assistantId, () => ({ sources, status: "streaming" })),
            onToken: (token) => patchMessage(assistantId, (m) => ({ content: m.content + token })),
            onDone: (done) =>
              patchMessage(assistantId, () => ({
                content: done.answer,
                citations: done.citations,
                status: "done",
              })),
            onError: (detail) => patchMessage(assistantId, () => ({ status: "error", error: detail })),
          },
        );
      } catch (error) {
        if (!controller.signal.aborted) {
          patchMessage(assistantId, () => ({ status: "error", error: (error as Error).message }));
        }
      } finally {
        setSending(false);
        abortRef.current = null;
        onConversationChanged();
      }
    },
    [conversationId, sending, patchMessage, onConversationChanged],
  );

  const reset = useCallback(() => {
    abortRef.current?.abort();
    setMessages([]);
    setConversationId(null);
  }, []);

  const load = useCallback(async (id: string) => {
    abortRef.current?.abort();
    const conversation = await api.getConversation(id);
    setConversationId(conversation.id);
    // The rewritten query is stored on the user turn; show it on the answer that used it.
    let rewrittenQuery: string | null = null;
    setMessages(
      conversation.messages.map((m): UIMessage => {
        if (m.role === "user") rewrittenQuery = m.rewritten_query;
        return {
          id: m.id,
          role: m.role,
          content: m.content,
          sources: m.citations ?? [],
          citations: m.citations ?? [],
          rewrittenQuery: m.role === "assistant" ? rewrittenQuery : null,
          status: "done",
        };
      }),
    );
  }, []);

  return { messages, conversationId, sending, send, reset, load };
}
