import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The FastAPI backend runs on 127.0.0.1:8000. In dev we proxy /api to it so the
// browser talks to a single origin and we avoid CORS entirely.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
});
