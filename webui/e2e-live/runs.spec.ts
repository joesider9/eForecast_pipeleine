import { readFile, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { test, expect } from '@playwright/test'

test('real submission reply loss, revision conflict, reuse, reload and descendant cancellation', async ({ page, request }, info) => {
  const errors: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  await page.route(/fonts\.(googleapis|gstatic)\.com/, route => route.abort())
  await page.goto('/cases/demo')
  // The production schema fills missing optional project defaults in this sparse fixture.
  // Accept those visible draft changes through the actual save flow before requesting a run.
  const save = page.getByRole('button', { name: 'Save', exact: true })
  await expect(save).toBeVisible()
  await page.getByRole('tab', { name: 'Runs', exact: true }).click()
  if (await save.isEnabled()) {
    await save.click()
    await expect(page.getByText('saved', { exact: true })).toBeVisible()
  }
  const card = page.locator('.MuiCard-root', { hasText: '1 · Datasets and clustering' })
  const stages = card.getByRole('combobox', { name: 'stages' })
  // Replace the real parser's default multi-select with the small actual target producer.
  await stages.fill('')
  await stages.press('Backspace')
  await stages.fill('target')
  await page.getByRole('option', { name: 'target', exact: true }).click()
  await stages.press('Escape')
  await card.getByRole('checkbox', { name: /^serial/ }).check()

  const accepted: { id: string; client_key: string; expected_revision: string }[] = []
  // This is a transport fault, not an HTTP fixture: fetch reaches the live API and commits;
  // only its first response is dropped. The retry goes to that API without interception.
  await page.route('**/api/cases/demo/runs', async route => {
    if (route.request().method() !== 'POST') { await route.continue(); return }
    await page.unroute('**/api/cases/demo/runs')
    const response = await route.fetch()
    expect(response.status()).toBe(201)
    const body = await response.json()
    accepted.push({ ...route.request().postDataJSON(), id: body.id })
    await route.abort('failed')
  })
  await card.getByRole('button', { name: 'Run datasets', exact: true }).click()
  await expect(page.getByRole('alert').filter({ hasText: /Failed to fetch|NetworkError/i })).toBeVisible()
  await expect.poll(async () => (await (await request.get(`/api/cases/demo/runs/${accepted[0]?.id}`)).json()).cleanup,
    { timeout: 60_000 }).toBe('complete')
  await expect(card.getByRole('button', { name: 'Run datasets', exact: true })).toBeEnabled({ timeout: 20_000 })
  const retry = page.waitForResponse(response => response.request().method() === 'POST' && response.url().endsWith('/runs'))
  await card.getByRole('button', { name: 'Run datasets', exact: true }).click()
  const retried = await retry
  expect((await retried.json()).id).toBe(accepted[0].id)
  expect(retried.request().postDataJSON().client_key).toBe(accepted[0].client_key)
  expect(retried.request().postDataJSON().expected_revision).toBe(accepted[0].expected_revision)
  await page.getByRole('button', { name: 'Close', exact: true }).click()

  // A genuinely stale programmatic client must be refused by the other API process.
  const conflict = await request.post(`${process.env.EFORECAST_TEST_LIVE_OTHER_URL}/api/cases/demo/runs`, {
    data: { script: 'datasets', options: { stages: ['target'] }, client_key: 'stale-browser-client',
      expected_revision: 'stale', action: 'continue' },
  })
  expect(conflict.status()).toBe(409)

  await card.getByRole('button', { name: 'Preview', exact: true }).click()
  const plan = page.getByRole('dialog')
  await expect(plan.getByText('Ready to run')).toBeVisible()
  await expect(plan.getByText('reuse', { exact: true })).toBeVisible()
  const second = page.waitForResponse(response => response.request().method() === 'POST' && response.url().endsWith('/runs'))
  await plan.getByRole('button', { name: 'Start', exact: true }).click()
  const secondId = (await (await second).json()).id
  await page.getByRole('button', { name: 'Close', exact: true }).click()
  await expect.poll(async () => (await (await request.get(`/api/cases/demo/runs/${secondId}`)).json()).cleanup,
    { timeout: 60_000 }).toBe('complete')
  await writeFile(join(process.env.EFORECAST_TEST_INTEGRATION_DIR!, 'restart-request.json'), '{}')
  await expect.poll(async () => {
    try { return JSON.parse(await readFile(join(process.env.EFORECAST_TEST_INTEGRATION_DIR!, 'restart-done.json'), 'utf8')).restarted }
    catch { return false }
  }, { timeout: 30_000 }).toBe(true)
  await page.reload()
  await page.getByRole('tab', { name: 'Runs', exact: true }).click()
  const row = page.getByRole('row').filter({ has: page.getByRole('button', { name: /^Stages/ }) }).first()
  await row.getByRole('button', { name: /^Stages/ }).click()
  await expect(page.getByRole('cell', { name: 'reused', exact: true })).toBeVisible()
  await page.screenshot({ path: info.outputPath('live-reuse.png'), animations: 'disabled' })

  // Only the test entry point understands this local barrier. Cancellation uses the real UI,
  // durable intent, supervisor census, TERM/KILL escalation and independent cleanup collector.
  await writeFile(join(process.env.EFORECAST_TEST_INTEGRATION_DIR!, 'fault.json'), JSON.stringify({ boundary: 'descendants' }))
  const third = page.waitForResponse(response => response.request().method() === 'POST' && response.url().endsWith('/runs'))
  await card.getByRole('button', { name: 'Run datasets', exact: true }).click()
  const thirdId = (await (await third).json()).id
  await page.getByRole('button', { name: 'Close', exact: true }).click()
  await expect.poll(async () => (await (await request.get(`/api/cases/demo/runs/${thirdId}`)).json()).state,
    { timeout: 60_000 }).toBe('running')
  // Root running precedes the fault worker reaching its descendant barrier.
  await expect.poll(async () => {
    try { return JSON.parse(await readFile(join(process.env.EFORECAST_TEST_INTEGRATION_DIR!, 'barrier.json'), 'utf8')).run_id }
    catch { return '' }
  }, { timeout: 60_000 }).toBe(thirdId)
  await page.getByRole('button', { name: 'Stop', exact: true }).click()
  await expect.poll(async () => {
    const record = await (await request.get(`/api/cases/demo/runs/${thirdId}`)).json()
    return [record.outcome?.kind, record.cleanup]
  }, { timeout: 60_000 }).toEqual(['cancelled', 'complete'])
  await page.reload()
  await page.getByRole('tab', { name: 'Runs', exact: true }).click()
  await expect(page.getByRole('row').filter({ has: page.getByRole('button', { name: /^Stages/ }) }).first()).toContainText('stopped')

  // A14-03: the cancelled run is exactly what a colleague asks about, so the report is taken
  // from the real UI against the real services -- not from a fixture. The download is asserted
  // rather than the clipboard: a headless browser's clipboard is a permission, the file is the
  // artefact a maintainer actually receives.
  await page.getByRole('row').filter({ has: page.getByRole('button', { name: /^Stages/ }) })
    .first().getByRole('button', { name: 'Log', exact: true }).click()
  const download = page.waitForEvent('download')
  await page.getByTestId('download-support-report').click()
  const report = await (await download).createReadStream();
  const body = await new Promise<string>((resolve, reject) => {
    let text = ''
    report.on('data', chunk => { text += chunk })
    report.on('end', () => resolve(text))
    report.on('error', reject)
  })
  expect(body).toContain('eforecast support report')
  expect(body).toContain(`run      ${thirdId}`)
  expect(body).toContain('case     demo')
  expect(body).toMatch(/doctor {3}(ready|needs attention|cannot start)/)
  expect(body).not.toMatch(/password|secret|token=/i)
  await page.getByTestId('copy-support-report').click()
  await expect(page.getByTestId('support-notice')).toContainText(/copied|Download support report/)
  // The notice is dismissible, so it offers a Close of its own: name the drawer's.
  await page.getByTestId('close-log').click()

  // §4.1: the installation's health, against the real doctor on this machine rather than a
  // fixture. The chip must state a verdict it actually read -- "Health unknown" here would
  // mean the route is unreachable from the page -- and the dialog must carry a remedy, since
  // a finding a colleague cannot act on is why this surface was needed in the first place.
  const chip = page.getByTestId('health-chip')
  await expect(chip).toBeVisible()
  await expect(chip).toHaveAttribute('data-verdict', /ready|needs attention|cannot start/)
  await chip.click()
  const dialog = page.getByRole('dialog')
  await expect(dialog).toContainText('bin/eforecast doctor')
  await expect(dialog.getByText(/^Do: /).first()).toBeVisible()
  await page.getByTestId('health-close').click()

  expect(errors).toEqual([])
  await page.screenshot({ path: info.outputPath('live-cancelled.png'), animations: 'disabled' })
})
