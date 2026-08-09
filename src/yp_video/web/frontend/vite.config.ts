import { defineConfig, searchForWorkspaceRoot } from 'vite';
import react from '@vitejs/plugin-react';
import { fileURLToPath, URL } from 'node:url';

// Repo-root contracts/ — the generated JSON schemas the train forms read
// (defaults, bounds, enum options, descriptions). Outside the frontend root,
// so the dev server must be allowed to serve it (server.fs.allow below).
const contractsDir = fileURLToPath(new URL('../../../../contracts', import.meta.url));

// Every backend route is mounted under /api on the FastAPI app (REST + SSE +
// media). In dev that single prefix is proxied to the running yp-app on :8080;
// in production FastAPI serves the built SPA and /api from the same origin.
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
      '@contracts': contractsDir,
    },
  },
  server: {
    port: 5173,
    fs: { allow: [searchForWorkspaceRoot(process.cwd()), contractsDir] },
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
