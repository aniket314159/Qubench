import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";
import tailwindcss from "@tailwindcss/vite";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    // api.js calls "/api/...". This strips the prefix and forwards to uvicorn,
    // so no component ever holds a hostname or a port, and there is no CORS
    // preflight in development.
    //
    //   ./run.sh all                 backend on 8000, this on 5173
    //   VITE_MOCK=true npm run dev   fixtures only, no backend needed
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ""),
      },
    },
  },
});
