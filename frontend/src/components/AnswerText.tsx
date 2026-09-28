import { Fragment, type ReactNode } from "react";

import { cn } from "@/lib/format";
import type { Source } from "@/lib/types";

const CITATION = /\[(\d+(?:\s*,\s*\d+)*)\]/g;
const BOLD = /\*\*([^*]+)\*\*/g;

interface Props {
  text: string;
  sources: Source[];
  activeCitation: number | null;
  onCitation: (id: number) => void;
}

/**
 * Renders an answer as paragraphs / bullet lists (a deliberately small Markdown subset) and
 * turns citation markers like [1] or [1, 2] into clickable chips linked to source cards.
 */
export function AnswerText({ text, sources, activeCitation, onCitation }: Props) {
  const known = new Set(sources.map((s) => s.id));

  const inline = (line: string, key: string): ReactNode[] => {
    const nodes: ReactNode[] = [];
    let last = 0;
    for (const match of line.matchAll(CITATION)) {
      nodes.push(...bold(line.slice(last, match.index), `${key}-t${last}`));
      const ids = match[1].split(",").map((n) => Number(n.trim()));
      ids.forEach((id, i) => {
        const source = sources.find((s) => s.id === id);
        nodes.push(
          known.has(id) ? (
            <button
              key={`${key}-c${match.index}-${i}`}
              type="button"
              onClick={() => onCitation(id)}
              title={source ? `${source.document_title}${source.page ? `, p. ${source.page}` : ""}` : undefined}
              className={cn(
                "mx-0.5 inline-flex h-[18px] min-w-[18px] -translate-y-px items-center justify-center rounded-md px-1 align-middle text-[11px] font-semibold transition",
                activeCitation === id
                  ? "bg-indigo-600 text-white"
                  : "bg-indigo-100 text-indigo-700 hover:bg-indigo-200",
              )}
            >
              {id}
            </button>
          ) : (
            <span key={`${key}-c${match.index}-${i}`} className="text-zinc-400">
              [{id}]
            </span>
          ),
        );
      });
      last = (match.index ?? 0) + match[0].length;
    }
    nodes.push(...bold(line.slice(last), `${key}-t${last}`));
    return nodes;
  };

  const blocks = text.split(/\n{2,}/).filter((b) => b.trim());
  return (
    <div className="space-y-3 leading-relaxed text-zinc-800">
      {blocks.map((block, b) => {
        const lines = block.split("\n").filter((l) => l.trim());
        const isList = lines.every((l) => /^\s*([-*•]|\d+\.)\s+/.test(l));
        if (isList) {
          return (
            <ul key={b} className="list-disc space-y-1.5 pl-5 marker:text-zinc-400">
              {lines.map((l, i) => (
                <li key={i}>{inline(l.replace(/^\s*([-*•]|\d+\.)\s+/, ""), `${b}-${i}`)}</li>
              ))}
            </ul>
          );
        }
        return (
          <p key={b}>
            {lines.map((l, i) => (
              <Fragment key={i}>
                {i > 0 && <br />}
                {inline(l, `${b}-${i}`)}
              </Fragment>
            ))}
          </p>
        );
      })}
    </div>
  );
}

function bold(text: string, key: string): ReactNode[] {
  const nodes: ReactNode[] = [];
  let last = 0;
  for (const match of text.matchAll(BOLD)) {
    nodes.push(text.slice(last, match.index));
    nodes.push(
      <strong key={`${key}-b${match.index}`} className="font-semibold text-zinc-900">
        {match[1]}
      </strong>,
    );
    last = (match.index ?? 0) + match[0].length;
  }
  nodes.push(text.slice(last));
  return nodes.filter((n) => n !== "");
}
