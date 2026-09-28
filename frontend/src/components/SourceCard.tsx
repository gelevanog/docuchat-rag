"use client";

import type { Ref } from "react";

import { cn, sourceLocation } from "@/lib/format";
import type { Source } from "@/lib/types";
import { FileIcon } from "./icons";

export function SourceCard({
  source,
  expanded,
  onToggle,
  ref,
}: {
  source: Source;
  expanded: boolean;
  onToggle: () => void;
  ref?: Ref<HTMLDivElement>;
}) {
  const location = sourceLocation(source);
  return (
    <div
      ref={ref}
      className={cn(
        "rounded-xl border bg-white text-sm transition",
        expanded ? "border-indigo-300 shadow-sm ring-2 ring-indigo-100" : "border-zinc-200 hover:border-zinc-300",
      )}
    >
      <button
        type="button"
        onClick={onToggle}
        aria-expanded={expanded}
        className="flex w-full items-start gap-3 p-3 text-left"
      >
        <span className="mt-0.5 flex size-5 shrink-0 items-center justify-center rounded-md bg-indigo-100 text-[11px] font-semibold text-indigo-700">
          {source.id}
        </span>
        <span className="min-w-0 flex-1">
          <span className="flex items-center gap-1.5 font-medium text-zinc-800">
            <FileIcon className="shrink-0 text-zinc-400" width={14} height={14} />
            <span className="truncate">{source.document_title}</span>
          </span>
          {location && <span className="mt-0.5 block truncate text-xs text-zinc-500">{location}</span>}
          {!expanded && <span className="mt-1.5 line-clamp-2 text-zinc-600">{plain(source.snippet)}</span>}
        </span>
      </button>
      {expanded && (
        <div className="border-t border-zinc-100 px-3 pt-2 pb-3">
          <p className="max-h-72 overflow-y-auto whitespace-pre-line text-zinc-700">{plain(source.content)}</p>
          <p className="mt-2 text-[11px] text-zinc-400">
            {source.filename} · relevance {source.score.toFixed(4)}
          </p>
        </div>
      )}
    </div>
  );
}

/** Source text is raw document text; drop Markdown emphasis markers for display. */
function plain(text: string): string {
  return text.replace(/(\*\*|__)(.+?)\1/g, "$2");
}
