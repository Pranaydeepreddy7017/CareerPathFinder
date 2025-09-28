import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  images: {
    domains: ["cdn-icons-png.flaticon.com", "i.pravatar.cc"], // ✅ add domains here
  },
};

export default nextConfig;
