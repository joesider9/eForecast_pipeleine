import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The FastAPI application serves the built app from webui/dist; during development Vite proxies /api to it.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { '/api': { target: 'http://localhost:8000', changeOrigin: true } } },
  build: { outDir: 'dist', sourcemap: false, chunkSizeWarningLimit: 1500 },
})
