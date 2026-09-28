"use client";

import { useCallback, useEffect, useState } from "react";

import { api } from "@/lib/api";
import type { ConversationSummary } from "@/lib/types";

export function useConversations() {
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);

  // On failure keep the last known list; the document list surfaces connectivity problems.
  const refresh = useCallback(
    () => api.listConversations().then(setConversations, () => undefined),
    [],
  );

  useEffect(() => {
    let active = true;
    api.listConversations().then(
      (items) => active && setConversations(items),
      () => undefined,
    );
    return () => {
      active = false;
    };
  }, []);

  const remove = useCallback(
    async (id: string) => {
      setConversations((all) => all.filter((c) => c.id !== id));
      await api.deleteConversation(id);
      await refresh();
    },
    [refresh],
  );

  return { conversations, refresh, remove };
}
