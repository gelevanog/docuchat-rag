"use client";

import { useCallback, useMemo, useState } from "react";

import { useChat } from "@/hooks/useChat";
import { useConversations } from "@/hooks/useConversations";
import { useDocuments } from "@/hooks/useDocuments";
import type { DocumentItem } from "@/lib/types";
import { ChatPanel } from "./ChatPanel";
import { Sidebar } from "./Sidebar";

export function DocuChatApp() {
  const docs = useDocuments();
  const conversations = useConversations();
  const refreshConversations = conversations.refresh;
  const chat = useChat(useCallback(() => void refreshConversations(), [refreshConversations]));
  const [selection, setSelection] = useState<Set<string>>(new Set());

  // Ignore selections for documents that were deleted or are no longer ready.
  const readyIds = useMemo(
    () => new Set(docs.documents.filter((d) => d.status === "ready").map((d) => d.id)),
    [docs.documents],
  );
  const selected = useMemo(
    () => new Set([...selection].filter((id) => readyIds.has(id))),
    [selection, readyIds],
  );

  const toggle = (id: string) =>
    setSelection((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  const deleteDocument = (doc: DocumentItem) => {
    if (window.confirm(`Delete "${doc.title}" and its indexed chunks?`)) void docs.remove(doc.id);
  };

  const deleteConversation = (id: string) => {
    if (id === chat.conversationId) chat.reset();
    void conversations.remove(id);
  };

  const scopeLabel =
    selected.size === 0
      ? `Searching all documents (${readyIds.size})`
      : `Searching ${selected.size} selected document${selected.size > 1 ? "s" : ""}`;

  return (
    <div className="flex h-dvh overflow-hidden">
      <Sidebar
        documents={docs.documents}
        documentsLoaded={docs.loaded}
        uploading={docs.uploading}
        notice={docs.notice}
        onDismissNotice={docs.dismissNotice}
        onUpload={(files) => void docs.upload(files)}
        onDeleteDocument={deleteDocument}
        selected={selected}
        onToggleDocument={toggle}
        onClearSelection={() => setSelection(new Set())}
        conversations={conversations.conversations}
        activeConversationId={chat.conversationId}
        onSelectConversation={(id) => void chat.load(id)}
        onDeleteConversation={deleteConversation}
        onNewChat={chat.reset}
      />
      <ChatPanel
        messages={chat.messages}
        sending={chat.sending}
        scopeLabel={scopeLabel}
        hasReadyDocuments={readyIds.size > 0}
        onSend={(text) => void chat.send(text, [...selected])}
      />
    </div>
  );
}
