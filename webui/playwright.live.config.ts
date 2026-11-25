import { existsSync } from 'node:fs'
import { defineConfig } from '@playwright/test'

// Started and torn down by tests/test_run_integration.py, with disposable workspaces/stores.
// Deliberately separate from the deterministic HTTP-fixture browser suite.
const baseURL = process.env.EFORECAST_TEST_LIVE_URL
if (!baseURL || new URL(baseURL).hostname !== '127.0.0.1') {
  throw new Error('Run this live-service gate through tests/test_run_integration.py')
}

export default defineConfig({
  testDir: './e2e-live', workers: 1, fullyParallel: false, timeout: 120_000,
  outputDir: process.env.EFORECAST_TEST_BROWSER_OUTPUT,
  use: {
    baseURL, viewport: { width: 1440, height: 1000 }, headless: true,
    actionTimeout: 15_000, navigationTimeout: 30_000,
    trace: 'retain-on-failure', screenshot: 'only-on-failure',
    launchOptions: { executablePath: process.env.EFORECAST_CHROME ||
      (existsSync('/opt/google/chrome/chrome') ? '/opt/google/chrome/chrome' : undefined),
      args: ['--disable-dev-shm-usage'] },
  },
})
