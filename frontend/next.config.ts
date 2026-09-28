import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Emit a self-contained server in .next/standalone for a small Docker image.
  output: "standalone",
  poweredByHeader: false,
};

export default nextConfig;
