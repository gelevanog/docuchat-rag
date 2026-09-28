"use client";

import { useRef, useState } from "react";

import type { UIMessage } from "@/lib/types";
import { AnswerText } from "./AnswerText";
import { LogoIcon, SearchIcon, SpinnerIcon } from "./icons";
import { SourceCard } from "./SourceCard";

export function MessageItem({ message }: { message: UIMessage }) {
  const [active, setActive] = useState<number | null>(null);
  const cardRefs = useRef(new Map<number, HTMLDivElement>());

  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-2xl rounded-br-md bg-zinc-900 px-4 py-2.5 whitespace-pre-wrap text-white">
          {message.content}
        </div>
      </div>
    );
  }

  const cards = message.status === "done" ? message.citations : [];
  const openCitation = (id: number) => {
    setActive((current) => (current === id ? null : id));
    requestAnimationFrame(() =>
      cardRefs.current.get(id)?.scrollIntoView({ behavior: "smooth", block: "nearest" }),
    );
  };

  return (
    <div className="flex gap-3">
      <div className="mt-0.5 flex size-7 shrink-0 items-center justify-center rounded-lg bg-indigo-600 text-white">
        <LogoIcon />
      </div>
      <div className="min-w-0 flex-1 space-y-3">
        {message.rewrittenQuery && (
          <p className="flex items-center gap-1.5 text-xs text-zinc-500">
            <SearchIcon width={12} height={12} />
            Searched for: <span className="italic">{message.rewrittenQuery}</span>
          </p>
        )}

        {message.status === "searching" && (
          <p className="flex items-center gap-2 text-sm text-zinc-500">
            <SpinnerIcon /> Searching your documents…
          </p>
        )}

        {message.content && (
          <AnswerText
            text={message.content}
            sources={message.status === "done" ? message.citations : message.sources}
            activeCitation={active}
            onCitation={openCitation}
          />
        )}
        {message.status === "streaming" && (
          <span className="inline-block h-4 w-1.5 animate-pulse rounded-sm bg-indigo-500 align-middle" />
        )}

        {message.status === "error" && (
          <p className="rounded-lg bg-rose-50 px-3 py-2 text-sm text-rose-700">
            {message.error ?? "Something went wrong."}
          </p>
        )}

        {cards.length > 0 && (
          <div className="space-y-2 pt-1">
            <p className="text-xs font-medium tracking-wide text-zinc-500 uppercase">Sources</p>
            <div className="grid gap-2 sm:grid-cols-2">
              {cards.map((source) => (
                <SourceCard
                  key={source.id}
                  ref={(el) => {
                    if (el) cardRefs.current.set(source.id, el);
                    else cardRefs.current.delete(source.id);
                  }}
                  source={source}
                  expanded={active === source.id}
                  onToggle={() => setActive((current) => (current === source.id ? null : source.id))}
                />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
