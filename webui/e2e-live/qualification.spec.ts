import { test, expect } from '@playwright/test'
import { writeFile } from 'node:fs/promises'

test('A12 qualification records actual PV target click-to-result separately from model compute', async ({ page }, info) => {
  test.setTimeout(180_000)
  const request = page.context().request
  const errors: string[] = []; page.on('pageerror', e => errors.push(e.message))
  await page.goto('/cases/a12latency')
  const save = page.getByRole('button', { name: 'Save', exact: true })
  await expect(save).toBeVisible(); if (await save.isEnabled()) await save.click()
  await expect(save).toBeDisabled()
  const before = (await (await request.get('/api/cases/a12latency')).json()).revision
  await page.getByRole('tab', { name: 'Runs', exact: true }).click()
  const data = page.locator('.MuiCard-root').filter({ hasText: '1 · Datasets' })
  const stages = data.getByRole('combobox', { name: 'stages' })
  await stages.fill(''); await stages.press('Backspace'); await stages.fill('target')
  await page.getByRole('option', { name: 'target', exact: true }).click(); await stages.press('Escape')
  await data.getByRole('checkbox', { name: /^serial/ }).check()
  await data.getByRole('combobox', { name: 'action' }).click()
  await page.getByRole('option', { name: 'rebuild', exact: true }).click()
  const rows: object[] = []
  for (let repeat = 1; repeat <= 3; repeat++) {
    const start = data.getByRole('button', { name: 'Run datasets', exact: true }); await expect(start).toBeEnabled()
    const response = page.waitForResponse(r => r.request().method() === 'POST' && r.url().endsWith('/runs'))
    const clicked = Date.now(); await start.click()
    const acceptedResponse = await response; expect(acceptedResponse.status()).toBe(201)
    const received = Date.now(); const accepted = await acceptedResponse.json()
    await page.getByRole('button', { name: 'Close', exact: true }).click()
    let final: Awaited<ReturnType<typeof acceptedResponse.json>>
    await expect.poll(async () => {
      const response = await request.get(`/api/cases/a12latency/runs/${accepted.id}`)
      expect(response.status()).toBe(200); final = await response.json()
      return [final.outcome?.kind, final.cleanup]
    }, { timeout: 120_000, intervals: [250] }).toEqual(['success', 'complete'])
    const observed = Date.now()
    expect(final!.input_snapshot_status).toBe('pinned')
    // API completion may precede the UI's next poll. Measure both boundaries and
    // capture the finished view, rather than calling a stale running row ready.
    await expect(start).toBeEnabled()
    await expect(page.getByRole('button', { name: 'Stop', exact: true })).toHaveCount(0)
    rows.push({ repeat, run_id: accepted.id, workload: 'real PV target dataset stage; not a trained-model throughput cell',
      click_to_acceptance_response_seconds: (received - clicked) / 1000,
      click_to_observed_cleanup_seconds: (observed - clicked) / 1000,
      click_to_rendered_ready_seconds: (Date.now() - clicked) / 1000,
      api_acceptance_to_outcome_seconds: final!.outcome.observed_at - final!.accepted,
      polling_interval_ms: 250, outcome: final!.outcome, stages: final!.stages })
  }
  expect((await (await request.get('/api/cases/a12latency')).json()).revision).toBe(before)
  expect(errors).toEqual([])
  await writeFile(info.outputPath('latency.json'), JSON.stringify({ rows, errors }, null, 2))
  await page.screenshot({ path: info.outputPath('qualification-runs.png'), fullPage: true, animations: 'disabled' })
})
