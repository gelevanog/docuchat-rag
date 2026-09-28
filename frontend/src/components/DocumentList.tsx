"use client";

import { cn, formatBytes } from "@/lib/format";
import type { DocumentItem } from "@/lib/types";
import { FileIcon, TrashIcon } from "./icons";
import { StatusBadge } from "./StatusBadge";

const TYPE_LABELS: Record<string, string> = { markdown: "md", text: "txt" };

export function DocumentList({
  documents,
  selected,
  onToggle,
  onDelete,
}: {
  documents: DocumentItem[];
  selected: Set<string>;
  onToggle: (id: string) => void;
  onDelete: (doc: DocumentItem) => void;
}) {
  if (!documents.length) {
    return <p className="px-1 text-sm text-zinc-500">No documents yet. Upload a few to get started.</p>;
  }
  return (
    <ul className="space-y-1">
      {documents.map((doc) => {
        const ready = doc.status === "ready";
        const isSelected = selected.has(doc.id);
        return (
          <li
            key={doc.id}
            className={cn(
              "group flex items-start gap-2.5 rounded-lg px-2 py-2 transition",
              isSelected ? "bg-indigo-50" : "hover:bg-zinc-100",
            )}
          >
            <input
              type="checkbox"
              aria-label={`Limit questions to ${doc.title}`}
              title={ready ? "Limit questions to selected documents" : "Available once ready"}
              className="mt-1 size-3.5 accent-indigo-600"
              checked={isSelected}
              disabled={!ready}
              onChange={() => onToggle(doc.id)}
            />
            <FileIcon className="mt-0.5 shrink-0 text-zinc-400" />
            <div className="min-w-0 flex-1">
              <p className="truncate text-sm font-medium text-zinc-800" title={doc.filename}>
                {doc.title}
              </p>
              <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 text-[11px] text-zinc-500">
                <StatusBadge status={doc.status} />
                <span className="uppercase">{TYPE_LABELS[doc.file_type] ?? doc.file_type}</span>
                <span>{formatBytes(doc.size_bytes)}</span>
                {ready && <span>{doc.chunk_count} chunks</span>}
              </div>
              {doc.status === "failed" && doc.error && (
                <p className="mt-1 line-clamp-2 text-xs text-rose-600" title={doc.error}>
                  {doc.error}
                </p>
              )}
            </div>
            <button
              type="button"
              onClick={() => onDelete(doc)}
              aria-label={`Delete ${doc.title}`}
              className="rounded p-1 text-zinc-400 opacity-0 transition hover:bg-zinc-200 hover:text-rose-600 focus:opacity-100 group-hover:opacity-100"
            >
              <TrashIcon />
            </button>
          </li>
        );
      })}
    </ul>
  );
}
