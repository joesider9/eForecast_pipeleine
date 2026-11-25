import { readFile, writeFile, unlink } from 'node:fs/promises'
import { join } from 'node:path'
import { test, expect } from '@playwright/test'

test('real CPU studies survive restart; Stop cleans up; Continue retains completed models', async ({ page, request }, info) => {
  const root = process.env.EFORECAST_TEST_INTEGRATION_DIR!
  const errors: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  await page.goto('/cases/demo')
  const save = page.getByRole('button', { name: 'Save', exact: true })
  await expect(save).toBeEnabled()
  await save.click(); await expect(save).toBeDisabled()
  await page.getByRole('tab', { name: 'Runs', exact: true }).click()
  const data = page.locator('.MuiCard-root').filter({ hasText: '1 · Datasets' })
  const stages = data.getByRole('combobox', { name: 'stages' })
  await stages.fill(''); await stages.press('Backspace'); await stages.fill('all')
  await page.getByRole('option', { name: 'all', exact: true }).click(); await stages.press('Escape')
  await data.getByRole('checkbox', { name: /^serial/ }).check()
  const built = page.waitForResponse(r => r.request().method() === 'POST' && r.url().endsWith('/runs'))
  await data.getByRole('button', { name: 'Run datasets', exact: true }).click()
  const dataReply = await built
  expect(dataReply.status()).toBe(201)
  const dataId = (await dataReply.json()).id
  await page.getByRole('button', { name: 'Close', exact: true }).click()
  await expect.poll(async () => (await (await request.get(`/api/cases/demo/runs/${dataId}`)).json()).cleanup,
    { timeout: 90_000 }).toBe('complete')
  await page.getByRole('button', { name: 'Refresh', exact: true }).click()
  const training = page.locator('.MuiCard-root').filter({ hasText: '2 · Training' })
  await training.getByRole('checkbox', { name: /CPU methods/ }).check()
  await training.getByLabel('Independent studies', { exact: true }).fill('2')
  await training.getByLabel('Threads per study', { exact: true }).fill('2')
  const submitted = page.waitForResponse(r => r.request().method() === 'POST' && r.url().endsWith('/runs'))
  await training.getByRole('button', { name: 'Run training', exact: true }).click()
  const response = await submitted
  expect(response.status()).toBe(201)
  const runId = (await response.json()).id
  await page.getByRole('button', { name: 'Close', exact: true }).click()
  await expect.poll(async () => {
    try { return JSON.parse(await readFile(join(root, 'compute-barrier.json'), 'utf8')).run_id }
    catch { return '' }
  }, { timeout: 90_000 }).toBe(runId)
  await page.getByRole('button', { name: 'Refresh', exact: true }).click()
  // The run's own report: how its studies share the CPUs, and the study that is fitting now.
  const report = page.getByRole('group', { name: 'Run resources' }).filter({ hasText: 'Independent studies at a time' })
  await expect(report).toContainText('CPU studies · running')
  await expect(report).toContainText('Independent studies at a time: 2 (asked: 2)')
  await expect(report).toContainText('global / RF · running')
  await expect.poll(async () => (await (await request.get(`/api/cases/demo/runs/${runId}`)).json())
    .resources?.compute?.progress?.verified_complete ?? 0, { timeout: 60_000 }).toBe(1)
  await page.screenshot({ path: info.outputPath('live-compute-studies.png'), fullPage: true })
  await writeFile(join(root, 'compute-restart-request.json'), '{}')
  await expect.poll(async () => {
    try { return JSON.parse(await readFile(join(root, 'compute-restart-done.json'), 'utf8')).run_id }
    catch { return '' }
  }, { timeout: 45_000 }).toBe(runId)
  await page.reload()
  await page.getByRole('tab', { name: 'Runs', exact: true }).click()
  await page.getByRole('button', { name: 'Stop', exact: true }).click()
  await expect.poll(async () => {
    const run = await (await request.get(`/api/cases/demo/runs/${runId}`)).json()
    return [run.outcome?.kind, run.cleanup]
  }, { timeout: 60_000 }).toEqual(['cancelled', 'complete'])
  await unlink(join(root, 'fault.json'))
  await page.reload()
  await page.getByRole('tab', { name: 'Runs', exact: true }).click()
  await page.getByRole('checkbox', { name: /CPU methods/ }).check()
  const resumed = page.waitForResponse(r => r.request().method() === 'POST' && r.url().endsWith('/runs'))
  await page.getByRole('button', { name: 'Run training', exact: true }).click()
  const retry = await resumed
  expect(retry.status()).toBe(201)
  const resumedId = (await retry.json()).id
  await page.getByRole('button', { name: 'Close', exact: true }).click()
  await expect.poll(async () => {
    const run = await (await request.get(`/api/cases/demo/runs/${resumedId}`)).json()
    return [run.outcome?.kind, run.cleanup]
  }, { timeout: 90_000 }).toEqual(['success', 'complete'])
  await page.getByRole('button', { name: 'Refresh', exact: true }).click()
  await page.screenshot({ path: info.outputPath('live-compute-resumed.png'), fullPage: true })
  expect(errors).toEqual([])
})
