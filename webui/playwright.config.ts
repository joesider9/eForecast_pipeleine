import { existsSync } from 'node:fs'
import { defineConfig } from '@playwright/test'

const executablePath = process.env.EFORECAST_CHROME ||
  (!process.env.CI && existsSync('/opt/google/chrome/chrome') ? '/opt/google/chrome/chrome' : undefined)

export default defineConfig({
  testDir: './e2e', workers: 1, fullyParallel: false, timeout: 30_000,
  use: {
    baseURL: 'http://127.0.0.1:5175', viewport: { width: 1440, height: 1000 },
    headless: true, trace: 'retain-on-failure', screenshot: 'only-on-failure',
    launchOptions: { executablePath, args: ['--disable-dev-shm-usage'] },
  },
  webServer: {
    command: 'npm run preview -- --host 127.0.0.1 --port 5175 --strictPort',
    url: 'http://127.0.0.1:5175', reuseExistingServer: false, timeout: 30_000,
  },
})
