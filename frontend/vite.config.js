import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  envPrefix: ["VITE_", "BACKEND_"],
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/compare": {
        target: "http://localhost:8000",
        changeOrigin: true
      }
    }
  }
});
