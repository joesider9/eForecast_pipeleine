// A private browser pass. The repository config previews `dist`, which is the build the lab's
// service actually serves; this one previews a throwaway directory instead, so a browser check
// can never depend on -- or disturb -- what the colleagues are using.
import base from './playwright.config'
import { defineConfig } from '@playwright/test'

const directory = process.env.EFORECAST_PREVIEW_DIR || 'dist-private-a13'
const port = Number(process.env.EFORECAST_PREVIEW_PORT || 5176)

export default defineConfig({
  ...base,
  use: { ...base.use, baseURL: `http://127.0.0.1:${port}` },
  webServer: {
    command: `npm run preview -- --host 127.0.0.1 --port ${port} --strictPort --outDir ${directory}`,
    url: `http://127.0.0.1:${port}`, reuseExistingServer: false, timeout: 60_000,
  },
})
