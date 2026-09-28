"use client";

import { useEffect, useRef } from "react";

import type { UIMessage } from "@/lib/types";
import { Composer } from "./Composer";
import { LogoIcon } from "./icons";
import { MessageItem } from "./MessageItem";

const SUGGESTIONS = [
  "How many vacation days do full-time employees get?",
  "What is the refund policy for annual subscriptions?",
  "What should I do if my laptop is stolen?",
];

export function ChatPanel({
  messages,
  sending,
  scopeLabel,
  hasReadyDocuments,
  onSend,
}: {
  messages: UIMessage[];
  sending: boolean;
  scopeLabel: string;
  hasReadyDocuments: boolean;
  onSend: (text: string) => void;
}) {
  const bottom = useRef<HTMLDivElement>(null);
  const last = messages.at(-1);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages.length, last?.content.length, last?.status]);

  return (
    <section className="flex min-w-0 flex-1 flex-col bg-zinc-50/60">
      <header className="flex h-14 shrink-0 items-center justify-between border-b border-zinc-200 bg-white/80 px-6 backdrop-blur">
        <h1 className="text-sm font-semibold text-zinc-800">Ask your documents</h1>
        <span className="rounded-full bg-zinc-100 px-2.5 py-1 text-xs text-zinc-600">{scopeLabel}</span>
      </header>

      <div className="flex-1 overflow-y-auto">
        <div className="mx-auto w-full max-w-3xl space-y-8 px-6 py-8">
          {messages.length === 0 ? (
            <EmptyState hasReadyDocuments={hasReadyDocuments} onPick={onSend} disabled={sending} />
          ) : (
            messages.map((m) => <MessageItem key={m.id} message={m} />)
          )}
          <div ref={bottom} />
        </div>
      </div>

      <div className="shrink-0 px-6 pb-6">
        <div className="mx-auto w-full max-w-3xl">
          <Composer
            onSend={onSend}
            disabled={sending}
            placeholder={hasReadyDocuments ? "Ask a question about your documents…" : "Upload a document to start"}
          />
          <p className="mt-2 text-center text-[11px] text-zinc-400">
            Answers are generated only from your documents and cite their sources. Press Enter to send,
            Shift+Enter for a new line.
          </p>
        </div>
      </div>
    </section>
  );
}

function EmptyState({
  hasReadyDocuments,
  onPick,
  disabled,
}: {
  hasReadyDocuments: boolean;
  onPick: (text: string) => void;
  disabled: boolean;
}) {
  return (
    <div className="flex flex-col items-center pt-16 text-center">
      <div className="flex size-12 items-center justify-center rounded-2xl bg-indigo-600 text-white shadow-sm">
        <LogoIcon width={24} height={24} />
      </div>
      <h2 className="mt-5 text-xl font-semibold text-zinc-900">Chat with your documents</h2>
      <p className="mt-2 max-w-md text-sm text-zinc-500">
        Upload PDFs, Word files or Markdown, then ask questions. Every answer links back to the exact
        passage it came from.
      </p>
      {hasReadyDocuments && (
        <div className="mt-8 grid w-full max-w-xl gap-2">
          {SUGGESTIONS.map((s) => (
            <button
              key={s}
              type="button"
              disabled={disabled}
              onClick={() => onPick(s)}
              className="rounded-xl border border-zinc-200 bg-white px-4 py-3 text-left text-sm text-zinc-700 transition hover:border-indigo-300 hover:bg-indigo-50/50"
            >
              {s}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
