"use client";

import type { UploadNotice } from "@/hooks/useDocuments";
import { cn } from "@/lib/format";
import type { ConversationSummary, DocumentItem } from "@/lib/types";
import { ConversationList } from "./ConversationList";
import { DocumentList } from "./DocumentList";
import { LogoIcon, PlusIcon } from "./icons";
import { UploadDropzone } from "./UploadDropzone";

interface Props {
  documents: DocumentItem[];
  documentsLoaded: boolean;
  uploading: number;
  notice: UploadNotice | null;
  onDismissNotice: () => void;
  onUpload: (files: File[]) => void;
  onDeleteDocument: (doc: DocumentItem) => void;
  selected: Set<string>;
  onToggleDocument: (id: string) => void;
  onClearSelection: () => void;
  conversations: ConversationSummary[];
  activeConversationId: string | null;
  onSelectConversation: (id: string) => void;
  onDeleteConversation: (id: string) => void;
  onNewChat: () => void;
}

export function Sidebar(props: Props) {
  return (
    <aside className="flex w-80 shrink-0 flex-col border-r border-zinc-200 bg-white">
      <div className="flex h-14 shrink-0 items-center justify-between border-b border-zinc-200 px-4">
        <div className="flex items-center gap-2">
          <div className="flex size-7 items-center justify-center rounded-lg bg-indigo-600 text-white">
            <LogoIcon />
          </div>
          <span className="font-semibold tracking-tight text-zinc-900">DocuChat</span>
        </div>
        <button
          type="button"
          onClick={props.onNewChat}
          className="inline-flex items-center gap-1 rounded-lg border border-zinc-200 px-2.5 py-1.5 text-xs font-medium text-zinc-700 transition hover:bg-zinc-50"
        >
          <PlusIcon width={14} height={14} /> New chat
        </button>
      </div>

      <div className="flex-1 space-y-6 overflow-y-auto p-4">
        <section className="space-y-3">
          <UploadDropzone onFiles={props.onUpload} uploading={props.uploading} />
          {props.notice && (
            <div
              role="status"
              className={cn(
                "flex items-start justify-between gap-2 rounded-lg px-3 py-2 text-xs",
                props.notice.kind === "error" ? "bg-rose-50 text-rose-700" : "bg-zinc-100 text-zinc-700",
              )}
            >
              <span>{props.notice.text}</span>
              <button type="button" onClick={props.onDismissNotice} aria-label="Dismiss" className="font-semibold">
                ×
              </button>
            </div>
          )}
        </section>

        <section>
          <div className="mb-2 flex items-center justify-between px-1">
            <h2 className="text-xs font-semibold tracking-wide text-zinc-500 uppercase">Documents</h2>
            {props.selected.size > 0 && (
              <button type="button" onClick={props.onClearSelection} className="text-xs text-indigo-600 hover:underline">
                Clear filter ({props.selected.size})
              </button>
            )}
          </div>
          {props.documentsLoaded ? (
            <DocumentList
              documents={props.documents}
              selected={props.selected}
              onToggle={props.onToggleDocument}
              onDelete={props.onDeleteDocument}
            />
          ) : (
            <p className="px-1 text-sm text-zinc-400">Loading…</p>
          )}
        </section>

        <section>
          <h2 className="mb-2 px-1 text-xs font-semibold tracking-wide text-zinc-500 uppercase">Chats</h2>
          <ConversationList
            conversations={props.conversations}
            activeId={props.activeConversationId}
            onSelect={props.onSelectConversation}
            onDelete={props.onDeleteConversation}
          />
        </section>
      </div>
    </aside>
  );
}
