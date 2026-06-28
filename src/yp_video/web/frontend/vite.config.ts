import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { fileURLToPath, URL } from 'node:url';

// Every backend route is mounted under /api on the FastAPI app (REST + SSE +
// media). In dev that single prefix is proxied to the running yp-app on :8080;
// in production FastAPI serves the built SPA and /api from the same origin.
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': { target: 'http://localhost:8080', changeOrigin: true },
    },
  },
  build: {
    // FastAPI serves this directory at the end of the migration.
    outDir: 'dist',
    emptyOutDir: true,
  },
});
