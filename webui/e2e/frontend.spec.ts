import { readFile } from 'node:fs/promises'
import { test, expect, type BrowserContext } from '@playwright/test'
import { caseDocument, caseSummary, formSchema, previewFixture, resourceFixture, runFixture, scripts, snapshot, stageFixture } from '../src/test/fixtures'
import type { ResourceControls } from '../src/api/resourceContract.generated'
import type { CaseResponse, Document, Run } from '../src/api/client'

const experiments = { mlp1: { row_all: [{ type: 'dense', params: { size: { tag: 'fixed', value: 16 } }, enabled: true }],
  output: [{ type: 'concatenate', params: {}, enabled: true }] } }
const catalogue = { aliases: {}, merges: ['concatenate'], branches: [{ name: 'row_all', description: 'Row features' }],
  layers: [{ name: 'dense', label: 'Dense', kind: 'dense', inputs: ['row_all'], output: 'features', size: 'units',
    size_type: 'int', description: 'Dense layer', params: [{ name: 'size', type: 'int', default: 16, help: 'Number of units' }] }] }

// These browser workflows exercise the production build with deterministic HTTP
// fixtures. They do not launch training or claim backend end-to-end coverage.
async function boundary(context: BrowserContext, options: { empty?: boolean; log?: string; finished?: boolean; auth?: boolean;
  lostSubmissionReply?: boolean; held?: boolean } = {}) {
  const cases = new Map<string, CaseResponse>()
  const runs = new Map<string, Run[]>()
  const unexpected: string[] = []
  let version = 1
  let log = options.log ?? 'training started\n'
  let maxRead = 0
  let sessionFailed = false
  let sessionChecks = 0
  let currentUser: string | null = options.auth ? 'alice' : 'local'
  const calls: string[] = []
  const submissions: Record<string, unknown>[] = []
  const previews: Record<string, unknown>[] = []
  const document = { ...structuredClone(caseDocument), experiments, methods: { method_params: { MLP: { experiment_tag: ['mlp1'] } } } }
  if (!options.empty) cases.set('demo', snapshot(document))
  if (options.finished) runs.set('demo', [runFixture('finished')])
  if (options.held) runs.set('demo', [{ ...runFixture('lost'), state: 'unknown', cleanup: 'unknown',
    report_error: 'report collection: unreadable journal',
    stages: [stageFixture('target'), stageFixture('row', { state: 'unknown', outcome: null, readiness: { status: 'unverified' } })] }])
  const resolutions: Record<string, unknown>[] = []
  await context.route(/fonts\.(googleapis|gstatic)\.com/, route => route.fulfill({ body: '', contentType: 'text/css' }))
  await context.route('**/api/**', async route => {
    const request = route.request()
    const url = new URL(request.url())
    const path = url.pathname
    calls.push(`${request.method()} ${path}`)
    const body = request.postData() ? request.postDataJSON() as Record<string, unknown> : {}
    const send = (value: unknown, status = 200) => route.fulfill({ json: value, status })
    if (path === '/api/me') {
      sessionChecks += 1
      return sessionFailed ? send({ detail: 'temporary session outage' }, 503) : send({ user: currentUser, auth: !!options.auth,
        role: currentUser ? currentUser === 'bob' ? 'user' : 'admin' : null })
    }
    if (path === '/api/logout') { currentUser = null; return send({ user: null }) }
    if (path === '/api/login') { currentUser = String(body.name); return send({ user: currentUser }) }
    if (options.auth && !currentUser) return send({ detail: 'not logged in' }, 401)
    if (path === '/api/doctor') return send({ verdict: 'ready', counts: { ok: 1 }, findings: [],
      checked_at: 1793180000, age_s: 0, stale: false, examined: 1, case: url.searchParams.get('case'),
      scope: 'maintainer' })
    if (path === '/api/schema') return send(formSchema)
    if (path === '/api/options') return send({ flags: { cluster_methods: ['CNN', 'TIMM', 'LSTM', 'TRANSFORMER', 'MLP', 'CrossViVit', 'Distributed'] } })
    if (path === '/api/scripts') return send(scripts)
    if (path === '/api/layers') return send(catalogue)
    if (path === '/api/graph/normalize') return send(body.experiments)
    if (path === '/api/graph/shapes') return send({ shapes: { row_all: [1, 3] }, steps: 1 })
    if (path === '/api/cases') {
      if (options.auth && currentUser !== 'alice') return send([])
      if (request.method() === 'POST') {
        const name = String(body.name)
        cases.set(name, snapshot((body.document as Document | undefined) ?? structuredClone(document), `v${++version}`, name))
        return send({ main: 'main.yaml', parts: [], note: '' }, 201)
      }
      return send([...cases].map(([name, value]) => ({ ...caseSummary(name, value.document.project?.name), owner: options.auth ? 'alice' : null })))
    }
    const parts = path.split('/').filter(Boolean)
    const name = decodeURIComponent(parts[2] ?? '')
    const saved = cases.get(name)
    if (parts[1] === 'cases' && saved) {
      if (options.auth && currentUser !== 'alice') return send({ detail: 'case not visible' }, 403)
      if (parts[3] === 'runs') {
        if (parts[4] === 'preview') {
          previews.push(body)
          // The server answers a CPU training preview with the study settings the run would use:
          // an override of the request where there is one, the saved case otherwise.
          const asked = body.options as ResourceControls & { cpu?: boolean }
          return send(previewFixture(asked.cpu ? { resource_controls: {
            controls: { n_cpus: 4, cpu_workers: asked.cpu_workers ?? null, cpu_study_threads: asked.cpu_study_threads ?? null,
              cpu_study_device: asked.cpu_study_device ?? null },
            note: 'Requested settings. How many studies run at once and with how many threads is decided when the run starts, '
              + 'from the CPUs the case uses and the studies it has.',
          } } : {}))
        }
        if (parts[5] === 'resolve') {
          resolutions.push(body)
          const ended = { ...runFixture('failed'), case: name, state: 'terminal' as const, cleanup: 'complete' as const,
            outcome: { kind: 'failure', detail: `resolved by alice without confirmed evidence: ${String(body.reason)}`,
                       observed_at: 3, returncode: null } }
          runs.set(name, [ended]); return send(ended)
        }
        if (parts[5] === 'log') {
          if (parts[6] === 'download') return route.fulfill({ body: log, headers: {
            'Content-Type': 'text/plain', 'Content-Disposition': `attachment; filename="${name}-${parts[4]}.log"`,
          } })
          const offset = Number(url.searchParams.get('offset'))
          const next = Math.min(log.length, offset + 256 * 1024)
          maxRead = Math.max(maxRead, next)
          return send({ text: log.slice(offset, next), offset, next, size: log.length,
            status: runs.get(name)?.[0]?.status ?? 'finished', generation: '1', truncated: next < log.length })
        }
        if (parts[5] === 'stop') {
          const stopped = { ...runFixture('stopped'), case: name }
          runs.set(name, [stopped]); log += 'stop completed\n'; return send(stopped)
        }
        if (request.method() === 'POST') {
          submissions.push(body)
          if (options.lostSubmissionReply && submissions.length === 1) return send({ detail: 'submission response lost' }, 503)
          const started = { ...runFixture(), case: name }
          runs.set(name, [started]); return send(started, 201)
        }
        if (request.method() === 'DELETE') { runs.set(name, []); return route.fulfill({ status: 204 }) }
        return send(runs.get(name) ?? [])
      }
      if (request.method() === 'PUT') {
        if (body.expected_revision !== saved.revision) return send({ detail: 'Case changed', ...saved }, 409)
        const next = snapshot(body.document as Document, `v${++version}`, name)
        cases.set(name, next)
        return send({ ...next, main: 'main.yaml', parts: [], note: '' })
      }
      return send(saved)
    }
    unexpected.push(`${request.method()} ${path}`)
    return send({ detail: 'Unhandled fixture route' }, 404)
  })
  return { cases, runs, unexpected, calls, submissions, previews, resolutions, readBytes: () => maxRead,
    sessionFailure: (failed: boolean) => { sessionFailed = failed }, sessionChecks: () => sessionChecks }
}

for (const runtime_ids of [{}, { TimesFM: 'foundation' }]) {
  test(`runtime configuration is automatic and preserves ${JSON.stringify(runtime_ids)} on save`, async ({ page, context }) => {
    const server = await boundary(context)
    const productionSchema = JSON.parse(await readFile(
      new URL('../../schema/experiment_config.schema.json', import.meta.url), 'utf8',
    )) as { properties: Record<string, unknown>; $defs: Record<string, unknown> }
    const methods = { env_name: 'pytorch_env', path_env: '/old/pytorch_env/bin', runtime_ids, backend: 'TORCH', CVs: 5 }
    server.cases.get('demo')!.document.methods = methods
    await context.route('**/api/schema', route => route.fulfill({ json: {
      ...formSchema, properties: { ...formSchema.properties, methods: productionSchema.properties.methods },
      $defs: { ...productionSchema.$defs, ...formSchema.$defs },
    } }))
    await page.goto('/cases/demo')
    await page.getByRole('tab', { name: 'Methods', exact: true }).click()
    const folds = page.getByRole('spinbutton', { name: /^CVs/i })
    await expect(folds).toHaveValue('5')
    await expect(page.getByRole('textbox', { name: /Legacy interpreter|Legacy environment|Runtime Ids/i })).toHaveCount(0)
    await expect(page.getByText('Runtime Ids', { exact: true })).toHaveCount(0)
    // The sparse fixture can acquire ordinary schema defaults; runtime values must survive the edit.
    await folds.fill('7')
    await page.getByRole('button', { name: 'Save', exact: true }).click()
    await expect(page.getByText('Case saved.', { exact: true })).toBeVisible()
    expect(server.cases.get('demo')!.document.methods).toMatchObject({ ...methods, CVs: 7 })
    expect(server.runs.size).toBe(0)
    expect(server.unexpected).toEqual([])
  })
}

for (const policy of [undefined, 'legacy_merge']) {
  test(`NWP policy ${policy ?? 'default'} is not selectable and survives a delay edit`, async ({ page, context }) => {
    const server = await boundary(context)
    const productionSchema = JSON.parse(await readFile(
      new URL('../../schema/experiment_config.schema.json', import.meta.url), 'utf8',
    )) as { properties: Record<string, unknown>; $defs: Record<string, unknown> }
    server.cases.get('demo')!.document.data = {
      nwp_availability_hours: 3, ...(policy ? { nwp_selection_policy: policy } : {}),
    }
    // Exercise the real Data schema, keeping unrelated sections on their small fixtures.
    await context.route('**/api/schema', route => route.fulfill({ json: {
      ...formSchema, properties: { ...formSchema.properties, data: productionSchema.properties.data },
      $defs: { ...productionSchema.$defs, ...formSchema.$defs },
    } }))
    await page.goto('/cases/demo')
    await page.getByRole('tab', { name: 'Data', exact: true }).click()
    const delay = page.getByRole('spinbutton', { name: /Nwp Availability Hours/i })
    await expect(delay).toHaveValue('3')
    await expect(page.getByRole('combobox', { name: /selection policy/i })).toHaveCount(0)
    await expect(page.getByText('Nwp Selection Policy', { exact: true })).toHaveCount(0)
    await expect(page.getByText(/This saved case uses legacy weather selection/)).toHaveCount(policy ? 1 : 0)
    await delay.fill('4')
    await page.getByRole('button', { name: 'Save', exact: true }).click()
    await expect(page.getByText('Case saved.', { exact: true })).toBeVisible()
    expect(server.cases.get('demo')!.document.data).toMatchObject({
      nwp_availability_hours: 4, nwp_selection_policy: policy ?? 'as_of',
    })
    expect(server.runs.size).toBe(0)
    expect(server.unexpected).toEqual([])
  })
}

test('resource policy survives saving without an editable Jobs tab', async ({ page, context }, info) => {
  const server = await boundary(context)
  const jobs = { n_cpus: 8, n_jobs: 4, n_gpus: 0, cpu_workers: 2, cpu_study_threads: null,
    cpu_study_device: 'cpu', n_jobs_mlp: 3, intra_op: 2, model_cache_mode: 'offline' }
  server.cases.get('demo')!.document.jobs = jobs
  await page.goto('/cases/demo')
  await expect(page.getByRole('textbox', { name: 'Project name' })).toBeVisible()
  await expect(page.getByRole('tab', { name: 'Jobs', exact: true })).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Save', exact: true })).toBeDisabled()
  await page.getByRole('textbox', { name: 'Project name' }).fill('edited site')
  await page.getByRole('tab', { name: 'Data', exact: true }).click()
  await page.getByRole('button', { name: 'Save', exact: true }).click()
  await expect(page.getByText('Case saved.', { exact: true })).toBeVisible()
  expect(server.cases.get('demo')!.document.jobs).toEqual(jobs)
  expect(server.runs.size).toBe(0)
  expect(server.unexpected).toEqual([])
  await page.getByRole('tab', { name: 'Runs', exact: true }).click()
  await page.screenshot({ path: info.outputPath('editor-without-jobs.png'), fullPage: true, animations: 'disabled' })
})

// A case file of an older version still names settings this one does not have. The run page
// offers none of them, starts the run from its button and sends only what the run takes.
test('a deep training run of a case that stores settings of an older version starts with the current options only', async ({ page, context }, info) => {
  const server = await boundary(context)
  server.cases.get('demo')!.document.jobs = { n_cpus: 4, execution_profile: 'a'.repeat(64), profile_resources: true,
    array_reader: 'prefer-mapped', cache_max_bytes: 2000000, cpu_study_ram_mb: 4500, cpu_wait_timeout: 60 }
  await page.goto('/cases/demo')
  await page.getByRole('tab', { name: 'Runs', exact: true }).click()
  const training = page.locator('.MuiCard-root', { hasText: '2 · Training' })
  await expect(training).not.toContainText(/profile|cache|reader|admi|reserv/i)
  await page.screenshot({ path: info.outputPath('runs-current-options.png'), fullPage: true, animations: 'disabled' })
  await training.getByRole('button', { name: 'Run training', exact: true }).click()
  await expect(page.getByLabel('Run log')).toBeVisible()
  expect(server.previews).toEqual([])
  expect(server.submissions).toHaveLength(1)
  expect(server.submissions[0].options).toEqual({ log_level: 'INFO' })
  expect(Object.keys(server.submissions[0]).sort()).toEqual(['action', 'client_key', 'expected_revision', 'options', 'script'])
  await page.getByRole('button', { name: 'Close', exact: true }).click()
  // The row of the run offers its log and its end, and the page asked for nothing else about it.
  await expect(page.getByRole('button', { name: 'Stop', exact: true })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Cache', exact: true })).toHaveCount(0)
  expect(server.calls.filter(call => /\/(cache|profiles)$/.test(call))).toEqual([])
  expect(server.calls.some(call => call.startsWith('DELETE') || call.startsWith('PUT'))).toBe(false)
  expect(server.unexpected).toEqual([])
})

test('submission retry preserves its key and saved revision in the production browser build', async ({ page, context }, info) => {
  const server = await boundary(context, { lostSubmissionReply: true })
  await page.goto('/cases/demo')
  await page.getByRole('tab', { name: 'Runs', exact: true }).click()
  await page.getByRole('button', { name: 'Run datasets', exact: true }).click()
  await expect(page.getByText('submission response lost', { exact: true })).toBeVisible()
  await page.getByRole('button', { name: 'Run datasets', exact: true }).click()
  await expect(page.getByLabel('Run log')).toContainText('training started')
  expect(server.submissions).toHaveLength(2)
  expect(server.submissions[0].client_key).toMatch(/^[0-9a-f]{32}$/)
  expect(server.submissions[1].client_key).toBe(server.submissions[0].client_key)
  expect(server.submissions[1].expected_revision).toBe(server.cases.get('demo')?.revision)
  expect(server.unexpected).toEqual([])
  await page.screenshot({ path: info.outputPath('submission-retry.png'), animations: 'disabled' })
})

test('logout Cancel retains the draft, then Save and log out commits before leaving the account', async ({ page, context }, info) => {
  const server = await boundary(context, { auth: true })
  await page.goto('/cases/demo')
  const name = page.getByRole('textbox', { name: 'Project name' })
  await name.fill('saved through logout')
  await page.getByLabel('Metadata YAML').fill('note: saved\nenabled: false\nthreshold: 0\ninterval_coverage: []\n')
  await page.getByRole('button', { name: 'Log out', exact: true }).click()
  const prompt = page.getByRole('dialog', { name: 'Unsaved changes before logging out' })
  await expect(prompt).toBeVisible()
  expect(server.calls.filter(call => call === 'POST /api/logout')).toHaveLength(0)
  await page.screenshot({ path: info.outputPath('logout-draft.png'), animations: 'disabled' })
  await prompt.getByRole('button', { name: 'Cancel', exact: true }).press('Enter')
  await expect(name).toHaveValue('saved through logout')
  expect(server.calls.some(call => call.startsWith('PUT '))).toBe(false)
  expect(server.calls.some(call => call === 'POST /api/logout')).toBe(false)
  await page.getByRole('button', { name: 'Log out', exact: true }).click()
  await prompt.getByRole('button', { name: 'Save and log out' }).press('Enter')
  await expect(page.getByRole('button', { name: 'Sign in', exact: true })).toBeVisible()
  expect(server.calls.filter(call => call === 'PUT /api/cases/demo' || call === 'POST /api/logout')).toEqual(['PUT /api/cases/demo', 'POST /api/logout'])
  expect(server.cases.get('demo')?.document.project).toMatchObject({ name: 'saved through logout',
    metadata: { note: 'saved', enabled: false, threshold: 0, interval_coverage: [] } })
  await page.getByRole('textbox', { name: 'User', exact: true }).fill('bob')
  await page.getByLabel(/^Password/).fill('password')
  await page.getByRole('button', { name: 'Sign in', exact: true }).click()
  await expect(page.getByText(/No cases yet/)).toBeVisible()
  await expect(page.getByRole('textbox', { name: 'Project name' })).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Users', exact: true })).toHaveCount(0)
  expect(server.unexpected).toEqual([])
})

test('logout guards unfinished YAML and explicit discard clears private draft state', async ({ page, context }, info) => {
  const server = await boundary(context, { auth: true })
  await page.goto('/cases/demo')
  const yaml = page.getByLabel('Metadata YAML')
  await yaml.fill('private unfinished YAML: [')
  await page.getByRole('button', { name: 'Log out', exact: true }).click()
  const prompt = page.getByRole('dialog', { name: 'Unsaved changes before logging out' })
  await expect(prompt.getByRole('button', { name: 'Save and log out' })).toBeDisabled()
  await expect(prompt.getByText(/Unfinished YAML cannot be saved/)).toBeVisible()
  expect(server.calls.filter(call => call === 'POST /api/logout')).toHaveLength(0)
  await page.screenshot({ path: info.outputPath('logout-unfinished-yaml.png'), animations: 'disabled' })
  await prompt.getByRole('button', { name: 'Cancel', exact: true }).click()
  await expect(yaml).toHaveText('private unfinished YAML: [')
  await page.getByRole('button', { name: 'Log out', exact: true }).click()
  await prompt.getByRole('button', { name: 'Discard and log out' }).click()
  await expect(page.getByRole('button', { name: 'Sign in', exact: true })).toBeVisible()
  expect(server.calls.filter(call => call.startsWith('PUT '))).toHaveLength(0)
  expect(server.calls.filter(call => call === 'POST /api/logout')).toHaveLength(1)
  await page.getByRole('textbox', { name: 'User', exact: true }).fill('alice')
  await page.getByLabel(/^Password/).fill('password')
  await page.getByRole('button', { name: 'Sign in', exact: true }).click()
  await page.getByRole('link', { name: 'demo', exact: true }).click()
  await expect(yaml).toContainText('preserve me')
  await expect(page.getByText(/Fix or discard unfinished YAML/)).toHaveCount(0)
  expect(server.unexpected).toEqual([])
})

test('a background session-check failure retains real editor text and recovers safely', async ({ page, context }, info) => {
  const server = await boundary(context)
  await page.goto('/cases/demo')
  const name = page.getByRole('textbox', { name: 'Project name' })
  await name.fill('unsaved before outage')
  const yaml = page.getByLabel('Metadata YAML')
  await expect(yaml).toBeVisible()
  // Keep the real CodeMirror buffer through a failed session refresh as well as
  // the case-level buffer's tab transitions tested below.
  await yaml.fill('unfinished: [')
  server.sessionFailure(true)
  await page.evaluate(() => window.dispatchEvent(new Event('visibilitychange')))
  await expect.poll(server.sessionChecks).toBeGreaterThanOrEqual(2)
  await expect(page.getByText(/Your session could not be rechecked/)).toBeVisible()
  await expect(name).toHaveValue('unsaved before outage')
  await expect(yaml).toHaveText('unfinished: [')
  await expect(page.getByRole('button', { name: 'Save', exact: true })).toBeDisabled()
  await expect(page.getByRole('button', { name: 'Save as new case' })).toBeDisabled()
  await name.fill('edited during outage')
  server.sessionFailure(false)
  await page.getByRole('button', { name: 'Retry session check' }).click()
  await expect(page.getByText(/Your session could not be rechecked/)).toHaveCount(0)
  await expect(name).toHaveValue('edited during outage')
  await expect(yaml).toHaveText('unfinished: [')
  await expect(page.getByRole('button', { name: 'Save', exact: true })).toBeDisabled()
  await page.getByRole('button', { name: 'Discard YAML edits' }).click()
  await expect(page.getByRole('button', { name: 'Save', exact: true })).toBeEnabled()
  await page.screenshot({ path: info.outputPath('session-recovered.png'), animations: 'disabled' })
  expect(server.cases.get('demo')?.document.project.name).toBe('baseline')
  expect(server.unexpected).toEqual([])
})

test('unfinished YAML survives real tab and navigation changes and guards saved/run actions', async ({ page, context }, info) => {
  const server = await boundary(context)
  await page.goto('/cases/demo')
  const yaml = page.getByLabel('Metadata YAML')
  await yaml.fill('unfinished: [')
  await expect(page.getByText('unsaved changes', { exact: true })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Save', exact: true })).toBeDisabled()
  await expect(page.getByRole('button', { name: 'Validate', exact: true })).toBeDisabled()
  await expect(page.getByRole('button', { name: 'Save as new case' })).toBeDisabled()
  await page.getByRole('tab', { name: 'Data', exact: true }).click()
  await page.getByRole('tab', { name: 'Project', exact: true }).click()
  await expect(yaml).toHaveText('unfinished: [')
  await page.getByRole('button', { name: 'eforecast configuration' }).click()
  await expect(page.getByRole('dialog', { name: 'Leave unsaved changes?' })).toBeVisible()
  await page.getByRole('button', { name: 'Stay', exact: true }).click()
  await expect(yaml).toHaveText('unfinished: [')
  await page.getByRole('tab', { name: 'Runs', exact: true }).click()
  await expect(page.getByText(/Fix or discard unfinished YAML/)).toBeVisible()
  await expect(page.getByRole('button', { name: 'Run training' })).toBeDisabled()
  await page.getByRole('tab', { name: 'Experiments', exact: true }).click()
  await page.getByRole('button', { name: 'yaml', exact: true }).click()
  const experiment = page.getByLabel('Experiment YAML')
  await experiment.fill('row_all: [')
  await page.getByRole('tab', { name: 'Project', exact: true }).click()
  await expect(yaml).toHaveText('unfinished: [')
  await page.getByRole('button', { name: 'Discard YAML edits' }).click()
  await expect(page.getByRole('button', { name: 'Validate', exact: true })).toBeDisabled()
  await page.getByRole('tab', { name: 'Experiments', exact: true }).click()
  await page.getByRole('button', { name: 'yaml', exact: true }).click()
  await expect(experiment).toHaveText('row_all: [')
  await expect(page.getByRole('button', { name: 'Dry build' })).toBeDisabled()
  await page.screenshot({ path: info.outputPath('unfinished-yaml.png'), animations: 'disabled' })
  await page.getByRole('button', { name: 'Discard YAML edits' }).click()
  await page.getByRole('tab', { name: 'Project', exact: true }).click()
  await yaml.fill('# retained across tabs\nnote: repaired\nenabled: false\nthreshold: 0\ninterval_coverage: []\n')
  await page.getByRole('tab', { name: 'Data', exact: true }).click()
  await page.getByRole('tab', { name: 'Project', exact: true }).click()
  await expect(yaml).toContainText('# retained across tabs')
  await page.getByRole('button', { name: 'Save', exact: true }).click()
  await expect(page.getByText('Case saved.', { exact: true })).toBeVisible()
  expect(server.cases.get('demo')?.document.project.metadata).toEqual({ note: 'repaired', enabled: false, threshold: 0, interval_coverage: [] })
  expect(server.runs.size).toBe(0)
  expect(server.unexpected).toEqual([])
})

test('invalid experiment structure retains the last graph and becomes editable/savable after repair', async ({ page, context }, info) => {
  const server = await boundary(context)
  const errors: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  await page.goto('/cases/demo')
  await page.getByRole('textbox', { name: 'Project name' }).fill('retained project draft')
  await page.getByRole('tab', { name: 'Experiments', exact: true }).click()
  await page.getByRole('button', { name: 'yaml', exact: true }).click()
  const yaml = page.getByLabel('Experiment YAML')
  await yaml.fill('row_all: null\noutput: []\n')
  await expect(page.getByText('row_all: expected a list of layers.', { exact: true })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Save', exact: true })).toBeDisabled()
  await expect(page.getByRole('button', { name: 'Dry build' })).toBeDisabled()
  await page.getByRole('button', { name: 'graph', exact: true }).click()
  await expect(page.getByRole('button', { name: 'dense layer 1' })).toBeVisible()
  await expect(page.getByText(/This graph shows the last valid value/)).toBeVisible()
  await page.screenshot({ path: info.outputPath('last-valid-graph.png'), animations: 'disabled' })
  await page.getByRole('tab', { name: 'Project', exact: true }).click()
  await expect(page.getByRole('textbox', { name: 'Project name' })).toHaveValue('retained project draft')
  await page.getByRole('tab', { name: 'Experiments', exact: true }).click()
  await page.getByRole('button', { name: 'yaml', exact: true }).click()
  await expect(yaml).toHaveText('row_all: null\noutput: []\n', { useInnerText: true })
  await yaml.fill('row_all:\n  - type: dense\n    params:\n      size: {tag: choice, value: 2}\noutput: []\n')
  await expect(page.getByText('row_all[0].params.size.value: expected a list for choice.', { exact: true })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Save', exact: true })).toBeDisabled()
  expect(server.cases.get('demo')?.document.experiments.mlp1.row_all[0].params.size.value).toBe(16)
  await yaml.fill('row_all:\n  - type: dense\n    enabled: false\n    params:\n      size: {tag: fixed, value: 32}\n      settings: {tag: fixed, value: {toString: 2}}\noutput: []\n')
  await expect(page.getByRole('button', { name: 'Save', exact: true })).toBeEnabled()
  await page.getByRole('button', { name: 'graph', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'Row features' })).toBeVisible()
  await expect(page.getByText('in (1 × 3)', { exact: true })).toBeVisible()
  await page.getByRole('button', { name: 'dense layer 1' }).press('Enter')
  await expect(page.getByRole('textbox', { name: 'settings value' })).toHaveValue('{"toString":2}')
  await page.getByRole('button', { name: 'Save', exact: true }).click()
  await expect(page.getByText('Case saved.', { exact: true })).toBeVisible()
  expect(server.cases.get('demo')?.document.project.name).toBe('retained project draft')
  expect(server.cases.get('demo')?.document.experiments.mlp1.row_all[0]).toMatchObject({ enabled: false,
    params: { size: { tag: 'fixed', value: 32 }, settings: { tag: 'fixed', value: { toString: 2 } } } })
  expect(errors).toEqual([])
  expect(server.unexpected).toEqual([])
})

test('create, edit, save, start, cancel and inspect using the keyboard', async ({ page, context }, info) => {
  const server = await boundary(context, { empty: true })
  await page.goto('/')
  await page.getByRole('button', { name: 'New case' }).focus()
  await page.keyboard.press('Enter')
  await page.getByRole('textbox', { name: /^Name/ }).fill('solar_demo')
  await page.getByRole('button', { name: 'Create', exact: true }).focus()
  await page.keyboard.press('Enter')
  const name = page.getByRole('textbox', { name: 'Project name' })
  await name.focus(); await page.keyboard.press('ControlOrMeta+A'); await page.keyboard.type('Solar demo')
  await page.getByRole('button', { name: 'Save', exact: true }).focus(); await page.keyboard.press('Enter')
  await expect(page.getByText('Case saved.', { exact: true })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Save', exact: true })).toBeDisabled()
  expect(server.cases.get('solar_demo')?.document.project.name).toBe('Solar demo')
  await page.getByRole('tab', { name: 'Runs', exact: true }).focus(); await page.keyboard.press('Enter')
  const start = page.getByRole('button', { name: 'Run training' })
  // focus() does not wait for the scripts/session checks that enable this button.
  await expect(start).toBeEnabled(); await start.focus(); await expect(start).toBeFocused()
  await page.keyboard.press('Enter')
  await expect(page.getByLabel('Run log')).toContainText('training started')
  await page.keyboard.press('Escape')
  await page.getByRole('button', { name: 'Stop', exact: true }).focus(); await page.keyboard.press('Enter')
  await expect(page.getByRole('button', { name: 'Delete', exact: true })).toBeVisible()
  await page.getByRole('button', { name: 'Log', exact: true }).focus(); await page.keyboard.press('Enter')
  await expect(page.getByLabel('Run log')).toContainText('stop completed')
  await expect(page.getByText(/bytes · complete/)).toBeVisible()
  await page.screenshot({ path: info.outputPath('run-log.png'), animations: 'disabled' })
  expect(server.unexpected).toEqual([])
})

test('a completed 3 MiB log drains fully, keeps a bounded view and downloads in full', async ({ page, context }, info) => {
  const size = 3 * 1024 * 1024
  const marker = 'FINAL LOG LINE'
  const text = 'training line\n'.repeat(Math.ceil(size / 14)).slice(0, size - marker.length) + marker
  const server = await boundary(context, { log: text, finished: true })
  await page.goto('/cases/demo')
  await page.getByRole('tab', { name: 'Runs', exact: true }).click()
  await page.getByRole('button', { name: 'Log', exact: true }).click()
  await expect(page.getByLabel('Run log')).toContainText(marker)
  await expect(page.getByText(/bytes · complete/)).toBeVisible()
  expect(server.readBytes()).toBe(size)
  const retained = await page.getByLabel('Run log').textContent() ?? ''
  expect(retained.length).toBeLessThanOrEqual(512 * 1024)
  expect(retained.split('\n').length).toBeLessThanOrEqual(4000)
  const downloading = page.waitForEvent('download')
  await page.getByRole('link', { name: 'Download full log' }).click()
  const download = await downloading
  const path = info.outputPath('full.log')
  await download.saveAs(path)
  expect((await readFile(path)).length).toBe(size)
  await page.screenshot({ path: info.outputPath('bounded-log.png'), animations: 'disabled' })
  expect(server.unexpected).toEqual([])
})

test('two tabs surface a stale save and preserve the draft for save-as-new', async ({ page, context }, info) => {
  const server = await boundary(context)
  const other = await context.newPage()
  await Promise.all([page.goto('/cases/demo'), other.goto('/cases/demo')])
  await expect(other.getByRole('textbox', { name: 'Project name' })).toHaveValue('baseline')
  await page.getByRole('textbox', { name: 'Project name' }).fill('first tab saved')
  await page.getByRole('button', { name: 'Save', exact: true }).click()
  await expect(page.getByText('Case saved.', { exact: true })).toBeVisible()
  await other.getByRole('textbox', { name: 'Project name' }).fill('my local draft')
  await other.getByRole('button', { name: 'Save', exact: true }).click()
  await expect(other.getByText(/The saved case changed. Your draft is retained/)).toBeVisible()
  await expect(other.getByRole('textbox', { name: 'Project name' })).toHaveValue('my local draft')
  await expect(other.getByRole('button', { name: 'Save', exact: true })).toBeDisabled()
  await other.getByRole('button', { name: 'Review differences' }).click()
  await other.screenshot({ path: info.outputPath('conflict-review.png'), animations: 'disabled' })
  await other.getByRole('button', { name: 'Save as new case' }).click()
  await other.getByRole('textbox', { name: 'New case name' }).fill('my_copy')
  await other.getByRole('button', { name: 'Create case', exact: true }).click()
  await expect(other).toHaveURL(/\/cases\/my_copy$/)
  await expect(other.getByRole('textbox', { name: 'Project name' })).toHaveValue('my local draft')
  expect(server.cases.get('demo')?.document.project.name).toBe('first tab saved')
  expect(server.cases.get('my_copy')?.document.project.name).toBe('my local draft')
  expect(server.unexpected).toEqual([])
})

test('run inputs show pinned revisions and degraded status in the list and log', async ({ page, context }, info) => {
  const server = await boundary(context)
  const effective = 'a'.repeat(64)
  const raw = 'b'.repeat(64)
  server.runs.set('demo', [
    { ...runFixture('finished'), id: 'pinned-run', input_snapshot_status: 'pinned', effective_revision: effective,
      config_revision: raw, effective_options: { n_jobs: 3, global_method: 'MLP' } },
    { ...runFixture('finished'), id: 'degraded-run', input_snapshot_status: 'degraded', config_frozen: true,
      normalized_error: 'Experiment definition could not be pinned: dependency unavailable' },
  ])
  await page.goto('/cases/demo')
  await page.getByRole('tab', { name: 'Runs', exact: true }).click()
  await expect(page.getByText(`Effective revision: ${effective}`, { exact: true })).toBeVisible()
  await expect(page.getByText(`Raw case revision: ${raw}`, { exact: true })).toBeVisible()
  const row = page.getByRole('row').filter({ hasText: 'Degraded inputs: using live configuration.' })
  await expect(row).toBeVisible()
  await page.screenshot({ path: info.outputPath('run-inputs.png'), animations: 'disabled' })
  await row.getByRole('button', { name: 'Log', exact: true }).click()
  const inputs = page.getByRole('group', { name: 'Run input provenance' })
  await expect(inputs).toContainText('Degraded inputs: using live configuration.')
  await inputs.getByText('Input details', { exact: true }).press('Enter')
  await expect(inputs).toContainText('Experiment definition could not be pinned: dependency unavailable')
  await page.screenshot({ path: info.outputPath('degraded-inputs-log.png'), animations: 'disabled' })
  expect(server.unexpected).toEqual([])
})

test('custom deep methods assign existing and new experiments on the Experiments tab and save them together', async ({ page, context }, info) => {
  const server = await boundary(context)
  const saved = server.cases.get('demo')!
  saved.document = { ...saved.document,
    methods: { backend: 'TORCH', global_methods: { MLP: true, CNN: false }, method_params: {
      MLP: { experiment_tag: { choices: ['mlp1'] }, n_trials: 4, extension: { keep: 0 } },
      CNN: { experiment_tag: 'mlp1', filters: 16 }, CatBoost: { depth: 6 }, TimesFM: { context_length: 96 },
      Chronos2: {}, TiRex: {}, RF: {}, clustering: {}, GRAPH: {},
    } },
    experiments: { ...structuredClone(experiments), alternative: { row_all: [], output: [] } },
  }
  const before = structuredClone(saved.document)
  await page.goto('/cases/demo')
  await page.getByRole('tab', { name: 'Experiments', exact: true }).click()
  await page.getByRole('combobox', { name: 'method', exact: true }).click()
  await expect(page.getByRole('option')).toHaveText(['CNN', 'MLP'])
  await page.getByRole('option', { name: 'MLP', exact: true }).click()
  await expect(page.getByRole('button', { name: 'Save', exact: true })).toBeDisabled()
  await page.getByRole('combobox', { name: 'experiment', exact: true }).click()
  await page.getByRole('option', { name: 'alternative', exact: true }).click()
  await page.getByRole('button', { name: 'Assign to MLP', exact: true }).click()
  await expect(page.getByRole('group', { name: 'Experiments assigned to MLP' })).toContainText('alternative')
  await page.getByRole('button', { name: 'New', exact: true }).click()
  await page.getByRole('dialog').getByRole('textbox', { name: 'tag', exact: true }).fill('lab_model')
  await page.getByRole('button', { name: 'Create & assign', exact: true }).click()
  await expect(page.getByRole('group', { name: 'Experiments assigned to MLP' })).toContainText('lab_model')
  await page.getByRole('button', { name: 'Save', exact: true }).click()
  await expect(page.getByText('Case saved.', { exact: true })).toBeVisible()
  const result = server.cases.get('demo')!.document
  expect(result.methods.method_params.MLP.experiment_tag).toEqual({ choices: ['mlp1', 'alternative', 'lab_model'] })
  expect(result.methods.method_params.MLP.extension).toEqual({ keep: 0 })
  expect(result.experiments.lab_model.row_all[0].type).toBe('dense')
  for (const family of ['CNN', 'CatBoost', 'TimesFM', 'Chronos2', 'TiRex', 'RF', 'clustering', 'GRAPH']) {
    expect(result.methods.method_params[family]).toEqual(before.methods.method_params[family])
  }
  expect(result.methods.global_methods).toEqual(before.methods.global_methods)
  await page.reload()
  await page.getByRole('tab', { name: 'Experiments', exact: true }).click()
  await page.getByRole('combobox', { name: 'method', exact: true }).click()
  await page.getByRole('option', { name: 'MLP', exact: true }).click()
  await expect(page.getByRole('group', { name: 'Experiments assigned to MLP' })).toContainText('lab_model')
  await page.getByRole('group', { name: 'Experiments assigned to MLP' }).getByRole('button', { name: 'lab_model', exact: true }).click()
  await expect(page.getByRole('button', { name: 'Assign to MLP', exact: true })).toBeDisabled()
  await page.screenshot({ path: info.outputPath('method-experiments.png'), animations: 'disabled' })
  expect(server.unexpected).toEqual([])
})

test('branches can be dragged without editing the experiment and retain their layout across tabs', async ({ page, context }, info) => {
  const server = await boundary(context)
  const original = structuredClone(server.cases.get('demo')!.document.experiments)
  await page.goto('/cases/demo')
  await page.getByRole('tab', { name: 'Experiments', exact: true }).click()
  await expect(page.getByText('in (1 × 3)', { exact: true })).toBeVisible()
  const node = page.locator('.react-flow__node[data-id="row_all"]')
  const position = () => node.evaluate(element => {
    const matrix = new DOMMatrixReadOnly(getComputedStyle(element).transform)
    return { x: matrix.m41, y: matrix.m42 }
  })
  const before = await position()
  const title = await node.getByText('row_all', { exact: true }).boundingBox()
  expect(title).not.toBeNull()
  await page.mouse.move(title!.x + 30, title!.y + title!.height / 2)
  await page.mouse.down()
  await page.mouse.move(title!.x + 150, title!.y + title!.height / 2 + 85, { steps: 12 })
  await page.mouse.up()
  await expect.poll(async () => (await position()).x).toBeGreaterThan(before.x + 50)
  await expect.poll(async () => (await position()).y).toBeGreaterThan(before.y + 40)
  const moved = await position()
  await expect(page.getByRole('button', { name: 'Save', exact: true })).toBeDisabled()
  // Selecting a layer rebuilds node data; it must not reset the dragged position.
  await node.getByRole('button', { name: 'dense layer 1' }).press('Enter')
  await expect(page.getByRole('textbox', { name: 'size value' })).toBeVisible()
  await expect.poll(position).toEqual(moved)
  await page.getByRole('button', { name: 'yaml', exact: true }).click()
  await page.getByRole('button', { name: 'graph', exact: true }).click()
  await expect.poll(position).toEqual(moved)
  await page.getByRole('tab', { name: 'Data', exact: true }).click()
  await page.getByRole('tab', { name: 'Experiments', exact: true }).click()
  await expect.poll(position).toEqual(moved)
  // A real layer edit remains independent of the layout and saves no coordinates.
  await node.getByRole('button', { name: 'dense layer 1' }).press('Enter')
  await page.getByRole('textbox', { name: 'size value' }).fill('32')
  await expect.poll(position).toEqual(moved)
  await page.getByRole('button', { name: 'Save', exact: true }).click()
  await expect(page.getByText('Case saved.', { exact: true })).toBeVisible()
  const expected = structuredClone(original)
  expected.mlp1.row_all[0].params.size.value = 32
  expect(server.cases.get('demo')!.document.experiments).toEqual(expected)
  await expect.poll(position).toEqual(moved)
  await page.screenshot({ path: info.outputPath('branch-dragged.png'), animations: 'disabled' })
  await page.getByRole('button', { name: 'Reset layout', exact: true }).click()
  await expect.poll(position).toEqual(before)
  await expect(page.getByRole('button', { name: 'Save', exact: true })).toBeDisabled()
  expect(server.unexpected).toEqual([])
})

for (const height of [1000, 720]) test(`multi-branch edge dragging keeps the camera stable and recentering recovers every node (${height}px)`, async ({ page, context }, info) => {
  await page.setViewportSize({ width: 1440, height })
  const server = await boundary(context)
  // Normalized transformer2 from the existing case: nine NWP layers, a calendar
  // branch, merge and head. No workspace or model runtime is needed by this test.
  const transformer2 = JSON.parse(await readFile(new URL('./fixtures/transformer2.json', import.meta.url), 'utf8'))
  server.cases.get('demo')!.document = { ...server.cases.get('demo')!.document,
    methods: { global_methods: { TRANSFORMER: true }, method_params: { TRANSFORMER: { experiment_tag: ['transformer2'] } } },
    experiments: { transformer2 },
  }
  const errors: string[] = []
  page.on('pageerror', error => errors.push(String(error)))
  const canvas = page.locator('.react-flow')
  const nodes = page.locator('.react-flow__node')
  const viewport = page.locator('.react-flow__viewport')
  const camera = () => viewport.evaluate(element => getComputedStyle(element).transform)
  const positions = () => nodes.evaluateAll(elements => elements.map(element => ({
    id: element.getAttribute('data-id'), transform: (element as HTMLElement).style.transform,
  })))
  const visibleNodes = () => nodes.evaluateAll(elements => {
    const rect = document.querySelector('.react-flow')!.getBoundingClientRect()
    return elements.filter(element => {
      const box = element.getBoundingClientRect()
      return box.left >= rect.left - 1 && box.right <= rect.right + 1 &&
        box.top >= rect.top - 1 && box.bottom <= rect.bottom + 1
    }).length
  })
  await page.goto('/cases/demo')
  await page.getByRole('tab', { name: 'Experiments', exact: true }).click()
  await expect(nodes).toHaveCount(4)
  await expect.poll(visibleNodes).toBe(4)
  await canvas.scrollIntoViewIfNeeded()
  const originalCamera = await camera()
  const originalPositions = await positions()
  const title = await page.locator('.react-flow__node[data-id="nwp"]').getByText('nwp', { exact: true }).boundingBox()
  const bounds = (await canvas.boundingBox())!
  await page.mouse.move(title!.x + 30, title!.y + title!.height / 2)
  await page.mouse.down()
  await page.mouse.move(bounds.x + bounds.width + 20, bounds.y - 20, { steps: 25 })
  await page.waitForTimeout(1600) // deliberate edge hold; previously panned the whole graph away
  await page.mouse.up()
  expect(await camera()).toBe(originalCamera)
  await expect(nodes).toHaveCount(4)
  const movedPositions = await positions()
  expect(movedPositions).not.toEqual(originalPositions)
  await page.getByRole('button', { name: 'Fit graph', exact: true }).click()
  await expect.poll(visibleNodes).toBe(4)
  expect(await positions()).toEqual(movedPositions)

  // Deliberate background panning remains available. Recovery must recenter the
  // viewport as well as resetting positions, even after all nodes leave view.
  await canvas.scrollIntoViewIfNeeded()
  const pane = (await canvas.boundingBox())!
  await page.mouse.move(pane.x + 10, pane.y + pane.height - 10)
  await page.mouse.down()
  await page.mouse.move(pane.x + 1800, pane.y + pane.height + 1200, { steps: 12 })
  await page.mouse.up()
  await expect.poll(visibleNodes).toBe(0)
  await page.getByRole('button', { name: 'Reset layout', exact: true }).click()
  await expect.poll(visibleNodes).toBe(4)
  expect(await positions()).toEqual(originalPositions)
  await expect(page.getByRole('button', { name: 'Save', exact: true })).toBeDisabled()
  await expect(page.getByText('The experiment editor could not render.', { exact: false })).toHaveCount(0)
  expect(errors).toEqual([])
  expect(server.cases.get('demo')!.document.experiments).toEqual({ transformer2 })
  await page.screenshot({ path: info.outputPath('recovered-multibranch.png'), animations: 'disabled' })
  expect(server.unexpected).toEqual([])
})

test('the real graph canvas supports keyboard layer selection and editing', async ({ page, context }, info) => {
  const server = await boundary(context)
  await page.goto('/cases/demo')
  await page.getByRole('tab', { name: 'Experiments', exact: true }).click()
  // The initial catalogue/shapes responses replace the provisional nodes. Wait
  // for that visible state before testing keyboard selection on the real graph.
  await expect(page.getByRole('heading', { name: 'Row features' })).toBeVisible()
  await expect(page.getByText('in (1 × 3)', { exact: true })).toBeVisible()
  const layer = page.getByRole('button', { name: 'dense layer 1' })
  await layer.press('Enter')
  await page.getByRole('textbox', { name: 'size value' }).fill('32')
  await page.getByRole('button', { name: 'Save', exact: true }).click()
  await expect(page.getByText('Case saved.', { exact: true })).toBeVisible()
  expect(server.cases.get('demo')?.document.experiments.mlp1.row_all[0].params.size.value).toBe(32)
  await expect(page.getByRole('button', { name: 'dense layer 1' })).toBeVisible()
  await page.screenshot({ path: info.outputPath('graph-canvas.png'), animations: 'disabled' })
  await page.getByRole('button', { name: 'yaml', exact: true }).click()
  await expect(page.getByLabel('Experiment YAML')).toContainText('32')
  await page.screenshot({ path: info.outputPath('experiment-yaml.png'), animations: 'disabled' })
  expect(server.unexpected).toEqual([])
})

test('a parameter that takes branch names starts empty, offers the branches, and shows the 64 an older editor stored', async ({ page, context }, info) => {
  const server = await boundary(context)
  // The graph all three cases of the first user carried: the cross attention of mlp1 with the
  // number the editor used to start `attention_query` at. The last route registered answers first.
  const merge = { name: 'cross_attention', label: 'Cross attention (merge)', kind: 'merge', inputs: ['branches'], output: 'flat',
    size: 'd_model', size_type: 'int', description: 'The flat branches query the token branches.',
    params: [{ name: 'size', type: 'int', default: null, help: 'd_model' }, { name: 'heads', type: 'int', default: 4, help: 'attention heads' },
      { name: 'attention_query', type: 'list', default: null, help: 'branch names used as queries (default: the flat branches)' },
      { name: 'd_model', type: 'int', default: null, help: 'token width; the size by default' }] }
  await context.route('**/api/layers', route => route.fulfill({ json: { ...catalogue, merges: ['cross_attention'], layers: [...catalogue.layers, merge] } }))
  const dense = (size: number) => ({ type: 'dense', params: { size: { tag: 'fixed', value: size } }, enabled: true })
  const graph = (query: unknown) => ({ mlp1: { nwp: [dense(128)], row_calendar: [dense(64)],
    output: [{ type: 'cross_attention', enabled: true,
      params: { heads: { tag: 'fixed', value: 4 }, attention_query: { tag: 'fixed', value: query } } }, dense(64)] } })
  server.cases.get('demo')!.document = { ...server.cases.get('demo')!.document, experiments: graph(64) }
  const errors: string[] = []
  page.on('pageerror', error => errors.push(String(error)))
  await page.goto('/cases/demo')
  await page.getByRole('tab', { name: 'Experiments', exact: true }).click()
  await page.getByRole('button', { name: 'cross_attention layer 1' }).press('Enter')
  // what is stored is shown as it is, and said to be what it is
  const remark = page.getByRole('alert').filter({ hasText: 'names no branch' })
  await expect(remark).toHaveText('64 names no branch; the layer reads this as not set')
  const names = page.getByRole('combobox', { name: 'attention_query names' })
  await page.screenshot({ path: info.outputPath('stored-64.png'), animations: 'disabled' })
  await names.click()
  await names.press('Backspace')                                   // the chip goes, and with it the remark
  await expect(remark).toHaveCount(0)
  await expect(page.getByRole('option')).toHaveText(['nwp', 'row_calendar'])   // the branches of this graph, not the merge
  await page.getByRole('option', { name: 'row_calendar', exact: true }).click()
  await page.getByRole('button', { name: 'Save', exact: true }).click()
  await expect(page.getByText('Case saved.', { exact: true })).toBeVisible()
  expect(server.cases.get('demo')!.document.experiments).toEqual(graph(['row_calendar']))
  // a parameter without a default is added empty: nothing is made up for it
  await page.getByRole('button', { name: 'cross_attention layer 1' }).press('Enter')
  await page.getByRole('combobox').filter({ hasText: 'add a parameter…' }).click()
  await page.getByRole('option', { name: 'd_model', exact: true }).click()
  await expect(page.getByRole('textbox', { name: 'd_model value' })).toHaveValue('')
  await page.getByRole('button', { name: 'Save', exact: true }).click()
  await expect(page.getByText('Case saved.', { exact: true })).toBeVisible()
  expect(server.cases.get('demo')!.document.experiments.mlp1.output[0].params.d_model).toEqual({ tag: 'fixed', value: null })
  await page.screenshot({ path: info.outputPath('names-as-chips.png'), animations: 'disabled' })
  expect(errors).toEqual([])
  expect(server.unexpected).toEqual([])
})

test('a preview is approved, started, survives a reload and can end a held case', async ({ page, context }, info) => {
  const server = await boundary(context)
  await page.goto('/cases/demo')
  await page.getByRole('tab', { name: 'Runs', exact: true }).click()

  // what the run would do, before it does it
  await page.locator('.MuiCard-root', { hasText: '1 · Datasets and clustering' }).getByRole('button', { name: 'Preview' }).click()
  const plan = page.getByRole('dialog')
  await expect(plan.getByText('Ready to run')).toBeVisible()
  await expect(plan.getByText('content and input identities match the receipt')).toBeVisible()
  await expect(plan.getByText('1 artifact(s) would be replaced')).toBeVisible()
  await plan.getByTestId('accept-consequences').click()
  await plan.getByTestId('plan-start').click()
  await expect(page.getByLabel('Run log')).toContainText('training started')
  expect(server.previews[0].action).toBe('continue')
  expect(server.submissions[0].action).toBe('continue')
  await page.screenshot({ path: info.outputPath('run-preview.png'), animations: 'disabled' })

  // the same run after a reload, and its stages
  await page.reload()
  await page.getByRole('tab', { name: 'Runs', exact: true }).click()
  await expect(page.getByRole('cell', { name: 'training' }).first()).toBeVisible()
  expect(server.unexpected).toEqual([])
})

test('a held case says so and is ended only with a reason', async ({ page, context }, info) => {
  const server = await boundary(context, { held: true })
  await page.goto('/cases/demo')
  await page.getByRole('tab', { name: 'Runs', exact: true }).click()
  await expect(page.getByText(/this case is held/)).toBeVisible()
  await page.getByRole('button', { name: 'Stages (2)' }).click()
  await expect(page.getByText('unverified')).toBeVisible()

  await page.getByRole('button', { name: 'Resolve', exact: true }).click()
  const dialog = page.getByRole('dialog')
  await expect(dialog.getByRole('button', { name: 'Resolve', exact: true })).toBeDisabled()
  await dialog.getByLabel('reason').fill('the host was rebuilt')
  await dialog.getByRole('button', { name: 'Resolve', exact: true }).click()
  await expect(page.getByText(/resolved by alice without confirmed evidence/)).toBeVisible()
  expect(server.resolutions[0].reason).toBe('the host was rebuilt')
  expect(server.unexpected).toEqual([])
  await page.screenshot({ path: info.outputPath('held-case.png'), animations: 'disabled' })
})

test('typed resource controls, guarded preview and the report of the studies of a run', async ({ page, context }, info) => {
  const server = await boundary(context, { auth: true })
  await page.goto('/cases/demo')
  await page.getByRole('tab', { name: 'Runs', exact: true }).click()
  await page.getByRole('checkbox', { name: /CPU methods/ }).check()
  await page.getByLabel('Independent studies', { exact: true }).fill('2')
  await page.getByLabel('Foundation device', { exact: true }).fill('auto')
  const training = page.locator('.MuiCard-root').filter({ hasText: '2 · Training' })
  await training.getByRole('button', { name: 'Preview' }).click()
  const dialog = page.getByRole('dialog', { name: 'training · continue' })
  // What was typed is what is asked for; what was left blank is left to the run.
  const requested = dialog.getByRole('group', { name: 'Requested study resources' })
  await expect(requested).toContainText('Independent studies at a time: 2 · threads per study: automatic · CPUs for the studies: 4')
  await expect(requested).toContainText('Foundation device: auto')
  expect(server.previews[0].options).toEqual({ log_level: 'INFO', cpu: true, cpu_workers: 2, cpu_study_device: 'auto' })
  await dialog.getByTestId('accept-consequences').click()
  await expect(dialog.getByTestId('plan-start')).toBeEnabled()
  server.sessionFailure(true)
  await page.evaluate(() => window.dispatchEvent(new Event('visibilitychange')))
  await expect(dialog.getByTestId('plan-start')).toBeDisabled()
  expect(server.submissions).toHaveLength(0)
  await page.screenshot({ path: info.outputPath('resource-preview-session-guard.png'), animations: 'disabled' })
  await dialog.getByRole('button', { name: 'Close', exact: true }).click()
  server.sessionFailure(false)
  await page.getByRole('button', { name: 'Retry session check' }).click()
  await expect(page.getByText(/Your session could not be rechecked/)).toHaveCount(0)
  await page.setViewportSize({ width: 430, height: 950 })
  const controls = page.locator('fieldset').filter({ hasText: 'Study resources · optional overrides' })
  await controls.scrollIntoViewIfNeeded()
  const bounds = await controls.boundingBox()
  expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(430)
  await page.screenshot({ path: info.outputPath('resource-controls-mobile.png'), animations: 'disabled' })
  await page.setViewportSize({ width: 1440, height: 1000 })
  server.runs.set('demo', [{ ...runFixture(), resources: resourceFixture() }])
  await page.getByRole('button', { name: 'Refresh', exact: true }).click()
  const resources = page.getByRole('group', { name: 'Run resources' })
  await expect(resources).toContainText('CPU studies · running')
  await expect(resources).toContainText('Independent studies at a time: 2 (asked: automatic)')
  await expect(resources).toContainText('threads per study: 1 or more (asked: automatic) · CPUs for the studies: 2 (asked: 4)')
  await expect(resources).toContainText('Threads in use by running work: 2')
  await expect(resources).toContainText("Memory held by this run's processes (PSS): 1536 MiB · measured")
  await expect(resources).toContainText('global / Chronos2 · running · 1 thread · device asked auto, used cuda')
  await expect(resources).toContainText('rule-b / RF · waiting — its turn has not come')
  await expect(resources).toContainText('Verified models: 2')
  await expect(resources).not.toContainText(/admi|reserv|grant|allocat|ledger|fence/i)
  await resources.scrollIntoViewIfNeeded()
  await page.screenshot({ path: info.outputPath('resource-report-desktop.png'), fullPage: true, animations: 'disabled' })
  expect(server.cases.get('demo')?.document.jobs).toEqual({ n_cpus: 4 })
  expect(server.unexpected).toEqual([])
})
