import type { DocumentStatus } from "@/lib/types";
import { cn } from "@/lib/format";

const STYLES: Record<DocumentStatus, string> = {
  pending: "bg-zinc-100 text-zinc-600 ring-zinc-200",
  processing: "bg-amber-50 text-amber-700 ring-amber-200",
  ready: "bg-emerald-50 text-emerald-700 ring-emerald-200",
  failed: "bg-rose-50 text-rose-700 ring-rose-200",
};

export function StatusBadge({ status }: { status: DocumentStatus }) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium capitalize ring-1 ring-inset",
        STYLES[status],
      )}
    >
      {status === "processing" && <span className="size-1.5 animate-pulse rounded-full bg-amber-500" />}
      {status}
    </span>
  );
}
