import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "DocuChat - chat with your documents",
  description: "Ask questions about your documents and get answers with citations to the source.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="h-full bg-white text-zinc-900">{children}</body>
    </html>
  );
}
