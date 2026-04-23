import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => ({
  plugins: [react()],
  server: {
    port: 5173,
    // Dev only — proxy /api to local Flask
    proxy: {
      "/api": {
        target:      "http://localhost:5000",
        changeOrigin: true,
      },
    },
  },
  build: {
    sourcemap: mode === "development",
  },
}));
