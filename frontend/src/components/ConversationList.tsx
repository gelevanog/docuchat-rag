"use client";

import { cn } from "@/lib/format";
import type { ConversationSummary } from "@/lib/types";
import { ChatIcon, TrashIcon } from "./icons";

export function ConversationList({
  conversations,
  activeId,
  onSelect,
  onDelete,
}: {
  conversations: ConversationSummary[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
}) {
  if (!conversations.length) {
    return <p className="px-1 text-sm text-zinc-500">Your conversations will appear here.</p>;
  }
  return (
    <ul className="space-y-0.5">
      {conversations.map((c) => (
        <li key={c.id} className="group relative">
          <button
            type="button"
            onClick={() => onSelect(c.id)}
            className={cn(
              "flex w-full items-center gap-2 rounded-lg px-2 py-1.5 pr-8 text-left text-sm transition",
              c.id === activeId ? "bg-zinc-200/70 font-medium text-zinc-900" : "text-zinc-600 hover:bg-zinc-100",
            )}
          >
            <ChatIcon className="shrink-0 text-zinc-400" />
            <span className="truncate">{c.title}</span>
          </button>
          <button
            type="button"
            onClick={() => onDelete(c.id)}
            aria-label={`Delete conversation ${c.title}`}
            className="absolute top-1/2 right-1 -translate-y-1/2 rounded p-1 text-zinc-400 opacity-0 transition hover:text-rose-600 focus:opacity-100 group-hover:opacity-100"
          >
            <TrashIcon />
          </button>
        </li>
      ))}
    </ul>
  );
}
