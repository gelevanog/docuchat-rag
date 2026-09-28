export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function sourceLocation(source: { page: number | null; heading: string | null }): string {
  const parts = [];
  if (source.page !== null) parts.push(`p. ${source.page}`);
  if (source.heading) parts.push(source.heading);
  return parts.join(" · ");
}

export function cn(...classes: (string | false | null | undefined)[]): string {
  return classes.filter(Boolean).join(" ");
}
