"use client";

import { useCallback, useEffect, useState } from "react";

import { api } from "@/lib/api";
import type { DocumentItem } from "@/lib/types";

const POLL_MS = 1500;

export interface UploadNotice {
  kind: "info" | "error";
  text: string;
}

export function useDocuments() {
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [uploading, setUploading] = useState(0);
  const [notice, setNotice] = useState<UploadNotice | null>(null);

  const apply = useCallback((docs: DocumentItem[]) => {
    setDocuments(docs);
    setLoaded(true);
  }, []);
  const fail = useCallback((error: unknown) => {
    setLoaded(true);
    setNotice({ kind: "error", text: `Could not load documents: ${(error as Error).message}` });
  }, []);

  const refresh = useCallback(() => api.listDocuments().then(apply, fail), [apply, fail]);

  useEffect(() => {
    let active = true;
    api.listDocuments().then(
      (docs) => active && apply(docs),
      (error) => active && fail(error),
    );
    return () => {
      active = false;
    };
  }, [apply, fail]);

  // Poll while any document is still being ingested.
  const busy = documents.some((d) => d.status === "pending" || d.status === "processing");
  useEffect(() => {
    if (!busy) return;
    const timer = setInterval(() => void refresh(), POLL_MS);
    return () => clearInterval(timer);
  }, [busy, refresh]);

  const upload = useCallback(
    async (files: File[]) => {
      setNotice(null);
      setUploading((n) => n + files.length);
      const messages: UploadNotice[] = [];
      await Promise.all(
        files.map(async (file) => {
          try {
            const result = await api.uploadDocument(file);
            if (result.duplicate) {
              messages.push({
                kind: "info",
                text: `"${file.name}" is already in the library as "${result.document.title}".`,
              });
            }
          } catch (error) {
            messages.push({ kind: "error", text: `${file.name}: ${(error as Error).message}` });
          } finally {
            setUploading((n) => n - 1);
          }
        }),
      );
      setNotice(messages.find((m) => m.kind === "error") ?? messages[0] ?? null);
      await refresh();
    },
    [refresh],
  );

  const remove = useCallback(
    async (id: string) => {
      setDocuments((docs) => docs.filter((d) => d.id !== id));
      try {
        await api.deleteDocument(id);
      } catch (error) {
        setNotice({ kind: "error", text: `Delete failed: ${(error as Error).message}` });
      }
      await refresh();
    },
    [refresh],
  );

  return { documents, loaded, uploading, notice, dismissNotice: () => setNotice(null), upload, remove };
}
