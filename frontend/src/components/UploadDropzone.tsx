"use client";

import { useRef, useState, type DragEvent } from "react";

import { cn } from "@/lib/format";
import { SpinnerIcon, UploadIcon } from "./icons";

const ACCEPT = ".pdf,.docx,.md,.markdown,.txt";

export function UploadDropzone({
  onFiles,
  uploading,
}: {
  onFiles: (files: File[]) => void;
  uploading: number;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const onDrop = (event: DragEvent) => {
    event.preventDefault();
    setDragging(false);
    const files = Array.from(event.dataTransfer.files);
    if (files.length) onFiles(files);
  };

  return (
    <button
      type="button"
      onClick={() => input.current?.click()}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
      className={cn(
        "flex w-full flex-col items-center gap-1.5 rounded-xl border border-dashed px-4 py-5 text-center transition",
        dragging
          ? "border-indigo-400 bg-indigo-50 text-indigo-700"
          : "border-zinc-300 bg-white text-zinc-500 hover:border-indigo-300 hover:text-zinc-700",
      )}
    >
      {uploading > 0 ? <SpinnerIcon width={20} height={20} /> : <UploadIcon width={20} height={20} />}
      <span className="text-sm font-medium text-zinc-700">
        {uploading > 0 ? `Uploading ${uploading} file${uploading > 1 ? "s" : ""}…` : "Drop files or click to upload"}
      </span>
      <span className="text-xs">PDF, DOCX, Markdown or TXT</span>
      <input
        ref={input}
        type="file"
        accept={ACCEPT}
        multiple
        hidden
        onChange={(e) => {
          const files = Array.from(e.target.files ?? []);
          if (files.length) onFiles(files);
          e.target.value = "";
        }}
      />
    </button>
  );
}
