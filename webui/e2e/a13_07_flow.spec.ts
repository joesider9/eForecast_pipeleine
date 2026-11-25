// A13-07 / A13-04 / A13-06 in a real browser: the readiness panel's field-linked findings, the
// six-group header, and the deliberate approval of a consequential run -- driven by keyboard
// where the task asks for it.
//
// Deterministic HTTP fixtures, as the existing browser workflows use. This proves the DOM, the
// focus behaviour and the keyboard path; it does not launch training or claim backend coverage.
import { test, expect, type BrowserContext } from '@playwright/test'
import { caseSummary, formSchema, previewFixture, runFixture, scripts, snapshot } from '../src/test/fixtures'
import type { CaseResponse, Document, Readiness } from '../src/api/client'

const working: Document = {
  project: {
    name: 'Aggiti', type: 'pv', rated_power: 100, coordinates: [38.0, 23.0],
    filename: '/data/aggiti/pv_ts_long.csv', ts_resolution: 'H', horizon_type: 'long_term',
    horizon: 0, site_timezone: 'Europe/Athens', local_timezone: 'EET', version_model: 1,
    is_global: false, is_fuzzy: true, country: 'Greece',
    nwp_models: [{ model: 'ecmwf', resolution: 0.1, path_nwp_source: '/media/old/HHD2' }],
  },
  data: { variables: [{ name: 'Flux' }, { name: 'Cloud' }], target_variable: { source: 'power' } },
  methods: { methods_cpu: { CatBoost: true, Chronos2: true }, cluster_methods: { CNN: true },
             method_params: { CatBoost: { iterations: 1000, spare_knob: 'keep me' } } },
  folders: { sys_folder: '/media/models' }, jobs: { n_cpus: 16 },
  extension: { legacy_zero: 0 },
}

const finding = (over: Partial<Readiness['findings'][number]> = {}) => ({
  check: 'target.coverage', field: 'project.filename', severity: 'ok' as const,
  checked: 'the first and last timestamp of the target series',
  finding: '16560 rows from 2024-07-11 to 2026-05-31',
  consequence: 'this is the period a run can learn from', remedy: 'Nothing to do.', ...over,
})

const readiness = (over: Partial<Readiness> = {}): Readiness => ({
  verdict: 'ready', checked_at: '2026-10-25 20:00:00 EET', checked_at_epoch: 1,
  inputs_digest: 'digest-1', findings: [finding()],
  counts: { ok: 1, attention: 0, blocking: 0 },
  coverage: { first: '2024-07-11 00:00:00', last: '2026-05-31 23:00:00', rows: 16560,
              span_days: 690, truncated: false },
  bounded: { max_index_rows: 500000 }, ...over,
})

const blocked = readiness({
  verdict: 'cannot start',
  counts: { ok: 0, attention: 0, blocking: 1 },
  findings: [finding({
    check: 'target.file', field: 'project.filename', severity: 'blocking',
    checked: 'the main time series file not_here.csv',
    finding: '/data/north/not_here.csv does not exist or is not a file',
    consequence: 'the target cannot be read, so the run stops at its first stage',
    remedy: 'Correct the path, or point it at a copy of the series on this machine.',
  })],
})

async function boundary(context: BrowserContext) {
  const cases = new Map<string, CaseResponse>([['working', snapshot(structuredClone(working), 'v1', 'working')]])
  const submissions: Record<string, unknown>[] = []
  let readinessCalls = 0
  await context.route(/fonts\.(googleapis|gstatic)\.com/, route => route.fulfill({ body: '', contentType: 'text/css' }))
  await context.route('**/api/**', async route => {
    const request = route.request()
    const path = new URL(request.url()).pathname
    const body = request.postData() ? request.postDataJSON() as Record<string, unknown> : {}
    const send = (value: unknown, status = 200) => route.fulfill({ json: value, status })
    if (path === '/api/me') return send({ user: 'local', auth: false, role: null })
    if (path === '/api/schema') return send(formSchema)
    if (path === '/api/options') return send({ timezones: ['UTC', 'Europe/Athens'], ts_resolutions: ['15min', 'H', 'D'] })
    if (path === '/api/scripts') return send(scripts)
    if (path === '/api/readiness') {
      readinessCalls += 1
      const document = body.document as Document
      // the real check refuses a file that is not there; the fixture mirrors that rule
      const filename = String((document.project as Record<string, unknown>)?.filename ?? '')
      return send(filename.includes('not_here') ? blocked : readiness())
    }
    if (path === '/api/cases') {
      if (request.method() === 'POST') {
        const name = String(body.name)
        const source = body.copy_from ? cases.get(String(body.copy_from))!.document : structuredClone(working)
        const copied = structuredClone(source) as Document
        // the backend gives a copy its own project name (A13-06)
        if (body.copy_from) (copied.project as Record<string, unknown>).name = name
        cases.set(name, snapshot(copied, 'v2', name))
        return send({ main: 'main.yaml', parts: [], note: '' }, 201)
      }
      return send([...cases].map(([name, value]) => caseSummary(name, value.document.project?.name as string)))
    }
    const parts = path.split('/').filter(Boolean)
    const name = decodeURIComponent(parts[2] ?? '')
    const saved = cases.get(name)
    if (parts[1] === 'cases' && saved) {
      if (parts[3] === 'runs' && parts[4] === 'preview') return send(previewFixture())
      if (parts[3] === 'runs' && request.method() === 'POST') {
        submissions.push(body)
        return send({ ...runFixture(), state: 'pending' }, 201)
      }
      if (parts[3] === 'runs') return send([])
      if (request.method() === 'PUT') {
        cases.set(name, snapshot(body.document as Document, 'v3', name))
        return send({ main: 'main.yaml', parts: [], note: '', document: body.document, revision: 'v3' })
      }
      return send(saved)
    }
    if (parts[1] === 'cases') return send({ detail: 'case not found' }, 404)
    return send([])
  })
  return { submissions, calls: () => readinessCalls }
}

test('a colleague copies a case, meets a field-linked refusal, and approves what will be replaced',
  async ({ page, context }) => {
  const { submissions } = await boundary(context)
  await page.goto('/')

  // ---- copy a working case -------------------------------------------------------------
  await page.getByRole('button', { name: 'New case' }).click()
  await page.getByRole('textbox', { name: /^Name/ }).fill('north_site')
  await page.getByRole('combobox', { name: 'Start from' }).click()
  await page.getByRole('option', { name: 'Copy of working' }).click()
  await page.getByRole('button', { name: 'Create', exact: true }).click()

  // ---- creating routes straight into the editor; the six groups are its first tab -------
  await expect(page.getByRole('heading', { name: 'north_site' })).toBeVisible()
  await page.getByRole('tab', { name: 'Everyday settings' }).click()
  await expect(page.getByTestId('case-header')).toBeVisible()
  for (const group of ['site', 'resolution', 'horizon', 'timezone', 'input-sources', 'model-scope']) {
    await expect(page.getByTestId(`header-group-${group}`)).toBeVisible()
  }
  // the copy carries its own site name, not the source's
  await expect(page.getByTestId('header-site-name')).toHaveValue('north_site')

  // ---- point it at a file that is not there, and read the refusal -----------------------
  await page.getByTestId('header-filename').fill('/data/north/not_here.csv')
  await page.getByRole('tab', { name: 'Review & save' }).click()
  await page.getByTestId('readiness-check').click()
  const verdict = page.getByTestId('readiness-verdict')
  await expect(verdict).toHaveAttribute('data-verdict', 'cannot start')
  await expect(verdict).toContainText('Cannot start')

  // the finding is tied to the field a colleague has to change, and carries a remedy
  const refusal = page.getByTestId('readiness-finding-target.file')
  await expect(refusal).toHaveAttribute('data-severity', 'blocking')
  await expect(refusal).toContainText('not_here.csv does not exist')
  await expect(refusal).toContainText('What to do: Correct the path')
  await expect(refusal.getByTestId('readiness-field')).toHaveText('project.filename')

  // ---- correct it, and confirm the refusal is not still standing ------------------------
  await page.getByRole('tab', { name: 'Everyday settings' }).click()
  await page.getByTestId('header-filename').fill('/data/aggiti/pv_ts_long.csv')
  await page.getByRole('tab', { name: 'Review & save' }).click()
  // Leaving the tab unmounts the panel, so the earlier verdict is gone rather than shown
  // against a document it no longer describes. That is the property that matters: a refusal
  // from before the edit must never read as current.
  await expect(page.getByTestId('readiness-verdict')).toHaveCount(0)
  await expect(page.getByTestId('readiness-idle')).toBeVisible()

  await page.getByTestId('readiness-check').click()
  await expect(page.getByTestId('readiness-verdict')).toHaveAttribute('data-verdict', 'ready')
  await expect(page.getByTestId('readiness-stale')).toHaveCount(0)

  // ---- review what the run would replace, and approve it from the keyboard --------------
  await page.getByRole('tab', { name: 'Runs' }).click()
  await page.getByRole('button', { name: 'Preview' }).first().click()
  const dialog = page.getByRole('dialog')
  await expect(dialog.getByTestId('run-consequences')).toHaveAttribute('data-consequential', 'true')
  await expect(dialog.getByTestId('run-consequences')).toContainText('dataset_row_data.pickle')
  await expect(dialog.getByTestId('consequences-recovery')).toContainText('not kept anywhere')

  const start = dialog.getByTestId('plan-start')
  await expect(start).toBeDisabled()
  await expect(dialog.getByTestId('plan-refusal')).toContainText('Accept the summary')
  await expect(start).toContainText('Replace 1 file(s) and start')

  // the acceptance is reachable and operable by keyboard alone
  const accept = dialog.getByTestId('accept-consequences')
  await accept.focus()
  await expect(accept).toBeFocused()
  await page.keyboard.press('Space')
  await expect(accept).toBeChecked()
  await expect(start).toBeEnabled()

  await start.focus()
  await page.keyboard.press('Enter')
  await expect.poll(() => submissions.length).toBe(1)
  expect(submissions[0].action).toBe('continue')
})

test('declining the summary starts nothing', async ({ page, context }) => {
  const { submissions } = await boundary(context)
  await page.goto('/cases/working')
  await page.getByRole('tab', { name: 'Runs' }).click()
  await page.getByRole('button', { name: 'Preview' }).first().click()
  const dialog = page.getByRole('dialog')
  await expect(dialog.getByTestId('plan-start')).toBeDisabled()
  await dialog.getByRole('button', { name: 'Close' }).click()
  await expect(page.getByRole('dialog')).toHaveCount(0)
  expect(submissions).toHaveLength(0)
})
