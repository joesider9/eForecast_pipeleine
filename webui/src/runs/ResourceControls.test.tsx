import { useState } from 'react'
import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import RunsPanel from './RunsPanel'
import ResourceReport, { ResourcePreviewReport } from './ResourceReport'
import { IdentityContext, SessionConfirmedContext } from '../session/context'
import { caseDocument, previewFixture, resourceFixture, runFixture, scripts } from '../test/fixtures'
import { json, mockHttp } from '../test/http'
import { renderPage } from '../test/render'
import type { RunResources } from '../api/client'

// The note the server sends with a preview (eforecast/config/resource_contract.py).
const note = 'Requested settings. How many studies run at once and with how many threads is decided '
  + 'when the run starts, from the CPUs the case uses and the studies it has.'
// Words of the bookkeeping this pipeline no longer has; none of them belongs on the page.
const retiredWords = /admission|admitted|allocator|allocation|reserv|grant|fence|ledger|profile|cache/i

function boundary() {
  return mockHttp(({ url, method, body }) => {
    if (url.pathname === '/api/scripts') return json(scripts)
    if (url.pathname.endsWith('/preview')) return json(previewFixture({ resource_controls: {
      controls: { n_cpus: 4, cpu_workers: 2, cpu_study_threads: 1, cpu_study_device: 'auto' }, note,
    } }))
    if (url.pathname.endsWith('/log')) return json({ text: '', offset: 0, next: 0, size: 0, status: 'running' })
    if (method === 'POST') return json({ ...runFixture(), options: body.options, state: 'pending' }, 201)
    return json([])
  })
}
const card = () => within(screen.getByText('2 · Training').closest('.MuiCard-root')!)

describe('typed run resource controls', () => {
  it.each(['pending', 'complete'] as const)('matches the case claim when cleanup is %s', async cleanup => {
    const run = { ...runFixture('finished'), state: 'terminal' as const, cleanup }
    mockHttp(({ url }) => url.pathname === '/api/scripts' ? json(scripts) : json([run]))
    renderPage(<RunsPanel caseName="demo" document={caseDocument} dirty={false} />)
    await screen.findByText('finished' + (cleanup === 'pending' ? ' · cleanup pending' : ''))
    if (cleanup === 'pending') expect(screen.getByRole('button', { name: 'Run training' })).toBeDisabled()
    else expect(screen.getByRole('button', { name: 'Run training' })).toBeEnabled()
  })
  it('inherits an old case without materializing controls or saving anything', async () => {
    const calls = boundary()
    renderPage(<RunsPanel caseName="demo" document={caseDocument} dirty={false} />)
    const user = userEvent.setup()
    await screen.findByText('No run yet for this case.')
    await user.click(screen.getByRole('checkbox', { name: /CPU methods/ }))
    expect(screen.getByText(/The studies of a run use 4 CPUs between them/)).toBeInTheDocument()
    expect(screen.getByLabelText('Independent studies')).toHaveValue('')
    // The overrides a run takes are these three and the scope; a blank one is left to the run.
    const controls = within(screen.getByRole('group', { name: /Study resources/ }))
    expect(controls.getAllByRole('textbox').map(input => (input as HTMLInputElement).labels?.[0]?.textContent))
      .toEqual(['Independent studies', 'Threads per study', 'Foundation device'])
    expect(controls.getAllByText('Saved: automatic')).toHaveLength(2)
    expect(controls.getByText('Saved: family setting')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Run training' }))
    await waitFor(() => expect(calls.filter(c => c.method === 'POST' && c.url.pathname.endsWith('/runs'))).toHaveLength(1))
    expect(calls.find(c => c.method === 'POST')!.body.options).toEqual({ log_level: 'INFO', cpu: true })
    expect(calls.some(c => c.method === 'PUT' || c.method === 'PATCH')).toBe(false)
  })

  it('previews and submits identical numeric resource overrides with the saved revision', async () => {
    const calls = boundary()
    renderPage(<RunsPanel caseName="demo" document={caseDocument} dirty={false} expectedRevision="saved-v1" />)
    const user = userEvent.setup()
    await screen.findByText('No run yet for this case.')
    await user.click(screen.getByRole('checkbox', { name: /CPU methods/ }))
    for (const [label, value] of [['Independent studies', '2'], ['Threads per study', '1'], ['Foundation device', 'auto']]) {
      await user.type(screen.getByLabelText(label), value)
    }
    await user.click(card().getByRole('button', { name: 'Preview' }))
    const dialog = await screen.findByRole('dialog')
    const requested = await within(dialog).findByRole('group', { name: 'Requested study resources' })
    expect(requested).toHaveTextContent('Independent studies at a time: 2 · threads per study: 1 · CPUs for the studies: 4')
    expect(requested).toHaveTextContent('Foundation device: auto')
    expect(within(dialog).getByText(/decided when the run starts/)).toBeInTheDocument()
    // A13-07: this plan replaces a file, so the summary is accepted before starting.
    await user.click(within(dialog).getByTestId('accept-consequences'))
    await user.click(within(dialog).getByTestId('plan-start'))
    await waitFor(() => expect(calls.filter(c => c.url.pathname.endsWith('/runs') && c.method === 'POST')).toHaveLength(1))
    const preview = calls.find(c => c.url.pathname.endsWith('/preview'))!.body
    const submitted = calls.find(c => c.url.pathname.endsWith('/runs') && c.method === 'POST')!.body
    expect(submitted.options).toEqual(preview.options)
    expect(submitted.expected_revision).toBe('saved-v1')
    expect(submitted.options).toEqual({ log_level: 'INFO', cpu: true, cpu_workers: 2, cpu_study_threads: 1, cpu_study_device: 'auto' })
    expect(Object.keys(submitted).sort()).toEqual(['action', 'client_key', 'expected_revision', 'options', 'script'])
  })

  it('refuses invalid overrides and omits hidden options after a mode switch', async () => {
    const calls = boundary()
    renderPage(<RunsPanel caseName="demo" document={caseDocument} dirty={false} />)
    const user = userEvent.setup()
    await screen.findByText('No run yet for this case.')
    await user.type(screen.getByLabelText('gpus'), '0 1')
    await user.click(screen.getByRole('checkbox', { name: /CPU methods/ }))
    await user.type(screen.getByLabelText('Independent studies'), '0')
    await user.click(card().getByRole('button', { name: 'Preview' }))
    expect(await screen.findByText('Independent studies must be a positive integer')).toBeInTheDocument()
    expect(calls.some(c => c.method === 'POST')).toBe(false)
    await user.clear(screen.getByLabelText('Independent studies'))
    await user.type(screen.getByLabelText('Independent studies'), '2')
    await user.click(card().getByRole('button', { name: 'Preview' }))
    const dialog = await screen.findByRole('dialog')
    expect(calls.find(c => c.url.pathname.endsWith('/preview'))!.body.options).not.toHaveProperty('gpus')
    await user.click(within(dialog).getByRole('button', { name: 'Close' }))
    await user.click(screen.getByRole('checkbox', { name: /CPU methods/ }))
    await user.click(screen.getByRole('button', { name: 'Run training' }))
    await waitFor(() => expect(calls.some(c => c.method === 'POST' && c.url.pathname.endsWith('/runs'))).toBe(true))
    const options = calls.find(c => c.method === 'POST' && c.url.pathname.endsWith('/runs'))!.body.options
    expect(options).toHaveProperty('gpus', [0, 1])
    expect(options).not.toHaveProperty('cpu_workers')
  })

  // A case file written by an older version still holds settings this one does not have, and
  // the server hands the file over as it is stored. None of them is offered, sent or acted on:
  // the run starts straight from its button, as it does for any other case.
  it('starts a run of a case that still stores settings of an older version without sending them', async () => {
    const stored = { ...caseDocument, jobs: { ...caseDocument.jobs, execution_profile: 'a'.repeat(64), profile_resources: true,
      array_reader: 'prefer-mapped', cache_max_bytes: 2000000, cache_max_entries: 3, cache_batch_rows: 64, cache_buffer_mb: 32,
      cpu_study_ram_mb: 4500, cpu_study_vram_mb: 6000, cpu_wait_timeout: 60 } }
    const original = structuredClone(stored)
    const calls = boundary()
    renderPage(<RunsPanel caseName="demo" document={stored} dirty={false} expectedRevision="saved-v1" />)
    const user = userEvent.setup()
    await screen.findByText('No run yet for this case.')
    await user.click(screen.getByRole('checkbox', { name: /CPU methods/ }))
    expect(within(screen.getByRole('group', { name: /Study resources/ })).getAllByRole('textbox')).toHaveLength(3)
    // (the combine card speaks of its own cached predictions, so the training card is what is read)
    expect(screen.getByText('2 · Training').closest('.MuiCard-root')!.textContent).not.toMatch(retiredWords)
    await user.click(screen.getByRole('button', { name: 'Run training' }))
    await waitFor(() => expect(calls.filter(c => c.method === 'POST' && c.url.pathname.endsWith('/runs'))).toHaveLength(1))
    expect(calls.some(c => c.url.pathname.endsWith('/preview'))).toBe(false)
    const submitted = calls.find(c => c.method === 'POST' && c.url.pathname.endsWith('/runs'))!.body
    expect(submitted.options).toEqual({ log_level: 'INFO', cpu: true })
    expect(Object.keys(submitted).sort()).toEqual(['action', 'client_key', 'expected_revision', 'options', 'script'])
    expect(stored).toEqual(original)
    expect(calls.every(c => /^\/api\/(scripts|cases\/demo\/runs(\/[^/]+\/log)?)$/.test(c.url.pathname))).toBe(true)
  })

  it.each(['dirty', 'session', 'readOnly', 'tenant'] as const)('protects an open resource preview when %s changes', async mode => {
    const calls = boundary()
    function Harness() {
      const [changed, setChanged] = useState(false)
      return <IdentityContext.Provider value={mode === 'tenant' && changed ? 'tenant-b' : 'tenant-a'}>
        <SessionConfirmedContext.Provider value={!(mode === 'session' && changed)}>
          <button onClick={() => setChanged(true)}>Change guard</button>
          <RunsPanel caseName="demo" document={caseDocument} dirty={mode === 'dirty' && changed} readOnly={mode === 'readOnly' && changed} />
        </SessionConfirmedContext.Provider>
      </IdentityContext.Provider>
    }
    renderPage(<Harness />)
    const user = userEvent.setup()
    await screen.findByText('No run yet for this case.')
    await user.click(screen.getByRole('checkbox', { name: /CPU methods/ }))
    await user.click(card().getByRole('button', { name: 'Preview' }))
    const dialog = await screen.findByRole('dialog')
    await within(dialog).findByText('Ready to run')
    // The harness simulates a background session/revision/tenant update while the modal is open.
    screen.getByText('Change guard').click()
    if (mode === 'tenant') await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument())
    else await waitFor(() => expect(within(dialog).getByTestId('plan-start')).toBeDisabled())
    expect(calls.some(c => c.method === 'POST' && c.url.pathname.endsWith('/runs'))).toBe(false)
  })

  it('says automatic where the request leaves a number to the run', () => {
    renderPage(<ResourcePreviewReport value={{ controls: { n_cpus: 16, cpu_workers: null, cpu_study_threads: null, cpu_study_device: null }, note }} />)
    const requested = screen.getByRole('group', { name: 'Requested study resources' })
    expect(requested).toHaveTextContent('Independent studies at a time: automatic · threads per study: automatic · CPUs for the studies: 16')
    expect(requested).toHaveTextContent('Foundation device: family setting')
    expect(requested.textContent).not.toMatch(retiredWords)
  })
})

describe('resource reporting', () => {
  const now = () => Date.now() / 1000
  // What the trials on the GPUs report: no study list and no way of sharing CPUs, but a row
  // for each card. The server files this writer under 'legacy' as well.
  const trials = (): RunResources => ({
    run_id: 'run-2', updated: now(), phase: 'running',
    cpu: { physical_cores: 8, logical_cpus: 16, affinity_cpus: 16, cgroup_quota_cpus: null, effective_cpus: 16 },
    running: { trials: 3, cpu_threads: 6, worker_threads: 2 },
    observed: { pids: [5001, 5002, 5003], pss_mb: 9000.4, rss_mb: 12000, os_threads: 61, confidence: 'measured_pss' },
    devices: { 0: { key: 'GPU-aaaa', host: 'host-a', name: 'RTX 2080 Ti', total_mb: 11264, identified: true },
               1: { key: 'GPU-bbbb', host: 'host-a', name: 'RTX 2080 Ti', total_mb: 11264, identified: true } },
    gpus: { 0: { measured_used_mb: 5200, reserved_mb: 6000, charged_mb: 6400, free_mb: 4564, attribution: 'per_pid', telemetry: 'measured' },
            1: { measured_used_mb: null, reserved_mb: 3000, charged_mb: null, free_mb: 7964, attribution: 'unknown_conservative', telemetry: 'unavailable' } },
    compute: { version: 2, source: 'gpu_trials', updated: now(), phase: 'running', error: null, running_threads: 6,
      pss_mb: 9000.4, pss_confidence: 'measured_pss', dispatch: null, studies: [], progress: null, warnings: [] },
  })

  it('says how the CPU studies share the machine, what each one is doing and how many models are done', () => {
    renderPage(<ResourceReport resources={resourceFixture()} live />)
    const report = screen.getByRole('group', { name: 'Run resources' })
    expect(within(report).getByText('CPU studies · running')).toBeInTheDocument()
    // Automatic where the case names no number, and what the run made of it beside it.
    expect(within(report).getByText(/Independent studies at a time: 2 \(asked: automatic\)/))
      .toHaveTextContent('threads per study: 1 or more (asked: automatic) · CPUs for the studies: 2 (asked: 4) · 3 studies in all')
    expect(within(report).getByText('CPUs this run may use: 2 of 16')).toBeInTheDocument()
    expect(within(report).getByText('Threads in use by running work: 2')).toBeInTheDocument()
    expect(within(report).getByText("Memory held by this run's processes (PSS): 1536 MiB · measured")).toBeInTheDocument()
    expect(within(report).getByText('Studies (3)')).toBeInTheDocument()
    expect(within(report).getByText('global / Chronos2 · running · 1 thread · device asked auto, used cuda')).toBeInTheDocument()
    expect(within(report).getByText('rule-a / RF · running · 1 thread · device cpu')).toBeInTheDocument()
    expect(within(report).getByText('Shallow models use CPU; a foundation device preference does not apply.')).toBeInTheDocument()
    // A study that is not running yet says why, and claims no threads or device.
    expect(within(report).getByText('rule-b / RF · waiting — its turn has not come: the studies that run at once are running')).toBeInTheDocument()
    expect(within(report).getByText(/Verified models: 2 · 1.00 per minute over 120 s · 1 pruned · 0 failed/))
      .toHaveTextContent('models reused from an earlier run and pruned trials are not in the rate')
    expect(within(report).queryByText(/not updated/)).not.toBeInTheDocument()
    expect(report.textContent).not.toMatch(retiredWords)
  })

  it('shows the numbers the case asked for when it asked for them', () => {
    const resources = resourceFixture()
    resources.compute = { ...resources.compute, dispatch: { requested_workers: 4, effective_workers: 2, requested_threads: 8,
      threads_per_study: 1, requested_total_threads: 4, total_threads: 2, trials_per_study: 1, studies: 1 } }
    renderPage(<ResourceReport resources={resources} live />)
    expect(screen.getByText(/Independent studies at a time: 2 \(asked: 4\)/))
      .toHaveTextContent('threads per study: 1 (asked: 8) · CPUs for the studies: 2 (asked: 4) · 1 study in all')
  })

  it('reads the trials on the GPUs and each card they use', () => {
    renderPage(<ResourceReport resources={trials()} live />)
    const report = screen.getByRole('group', { name: 'Run resources' })
    expect(within(report).getByText('GPU trials · running')).toBeInTheDocument()
    expect(within(report).getByText('Threads in use by running work: 6')).toBeInTheDocument()
    expect(within(report).getByText("Memory held by this run's processes (PSS): 9000 MiB · measured")).toBeInTheDocument()
    expect(within(report).getByText('GPU 0: 5200 MiB in use · 6000 MiB planned for running trials · 4564 MiB free for another trial · memory counted per process')).toBeInTheDocument()
    expect(within(report).getByText('GPU 1: memory in use not measured · 3000 MiB planned for running trials · 7964 MiB free for another trial'
      + ' · memory per process unknown, so every trial counts in full')).toBeInTheDocument()
    expect(within(report).queryByText(/Independent studies|Studies \(|Verified models/)).not.toBeInTheDocument()
    expect(within(report).queryByText(/At most/)).not.toBeInTheDocument()
    expect(report.textContent).not.toMatch(retiredWords)
  })

  it('says when a run starts fewer trials at once because the machine ran out of memory', () => {
    const resources = trials()
    resources.compute = { ...resources.compute, trial_limit: 5 }
    renderPage(<ResourceReport resources={resources} live />)
    expect(screen.getByText('At most 5 trials at a time now: the machine ran out of memory with more running')).toBeInTheDocument()
  })

  it('names the worker of a single study and its end', () => {
    renderPage(<ResourceReport live={false} resources={{ updated: now(), phase: 'study_finished', running: { studies: 1, cpu_threads: 4 },
      compute: { version: 2, source: 'worker', updated: now(), phase: 'study_finished', error: null, running_threads: 4, pss_mb: 812.6,
        pss_confidence: 'partial_or_unavailable', dispatch: null, progress: null, warnings: [],
        studies: [{ id: 'run-3//models/global/TiRex', method: 'TiRex', cluster: 'global', state: 'study_finished', threads: 4,
          requested_device: 'cuda:1', resolved_device: 'cpu', normalization: 'No GPU matches the preference; the study runs on the CPU.', reason: null }] } }} />)
    expect(screen.getByText('CPU study worker · study finished')).toBeInTheDocument()
    expect(screen.getByText("Memory held by this run's processes (PSS): 813 MiB · partial or unavailable")).toBeInTheDocument()
    expect(screen.getByText('global / TiRex · finished · 4 threads · device asked cuda:1, used cpu')).toBeInTheDocument()
    expect(screen.getByText('No GPU matches the preference; the study runs on the CPU.')).toBeInTheDocument()
  })

  it('keeps degraded measurements unknown, shows what the server could not read and marks a live snapshot that stopped moving', () => {
    const resources = resourceFixture()
    resources.compute = { ...resources.compute, updated: 1, running_threads: null, pss_mb: null, pss_confidence: null,
      error: 'the studies stopped reporting', studies: [], progress: null,
      warnings: ['Study telemetry is unreadable.', 'Completion-rate telemetry is unavailable.'] }
    const { unmount } = renderPage(<ResourceReport resources={resources} live />)
    expect(screen.getByText('CPU studies · running · not updated for over 2 minutes')).toBeInTheDocument()
    expect(screen.getByText('the studies stopped reporting')).toBeInTheDocument()
    expect(screen.getByText('Threads in use by running work: unknown')).toBeInTheDocument()
    expect(screen.getByText("Memory held by this run's processes (PSS): unknown · measurement unavailable")).toBeInTheDocument()
    expect(screen.getByText('Study telemetry is unreadable.')).toBeInTheDocument()
    expect(screen.getByText('Completion-rate telemetry is unavailable.')).toBeInTheDocument()
    expect(screen.queryByText(/Verified models|Studies \(/)).not.toBeInTheDocument()
    unmount()
    // The last snapshot of a run that has ended is simply its last one.
    renderPage(<ResourceReport resources={resources} live={false} />)
    expect(screen.getByText('CPU studies · running')).toBeInTheDocument()
  })

  it('renders nothing but the fact when there is no snapshot or it cannot be read', () => {
    const empty = renderPage(<ResourceReport live />)
    expect(screen.getByText('No resource snapshot')).toBeInTheDocument()
    empty.unmount()
    const blank = renderPage(<ResourceReport resources={{}} live />)
    expect(screen.getByText('No resource measurements in this record')).toBeInTheDocument()
    blank.unmount()
    // What the server answers for a status file it could not parse.
    renderPage(<ResourceReport live resources={{ error: 'Resource snapshot unreadable', compute: { version: 2, source: 'legacy',
      updated: null, phase: null, error: 'Resource snapshot unreadable', running_threads: null, pss_mb: null, pss_confidence: null,
      dispatch: null, studies: [], progress: null, warnings: [] } }} />)
    expect(screen.getByText('Resource snapshot unreadable')).toBeInTheDocument()
    expect(screen.queryByRole('group', { name: 'Run resources' })).not.toBeInTheDocument()
  })

  // A run recorded by an older version: the file names what was set aside for it and who else
  // had claims, and the reading kept with it has fields this version does not know. What it
  // still says -- threads, memory, the cards -- is shown; the rest is not read.
  it('reads an older record for what it still says, whatever else it holds', () => {
    const old: unknown = { updated: 1, phase: 'succeeded', registry: { mode: 'strict', state: 'ok' },
      reservations: { cpu_threads: 6, host_mb: 2048, worker_threads: 2 }, observed: { pss_mb: 512, rss_mb: 900, os_threads: 12, confidence: 'measured_pss' },
      foreign_claims: [{ run_id: 'other', cpu_threads: 2, host_mb: 100, gpu_exclusive: ['0'] }],
      gpus: { 0: { measured_used_mb: 800, reserved_mb: 2000, charged_mb: 2000, free_mb: -40, attribution: 'per_pid', telemetry: 'measured' } },
      compute: { version: 1, source: 'legacy', allocator_state: 'ok', updated: 1, phase: 'succeeded', reserved_threads: 6, reserved_ram_mb: 2048,
        pss_mb: 512, pss_confidence: 'measured_pss', limits: { requested_workers: 4, effective_workers: 2 },
        queue: { status: 'recorded', reason: 'FIFO', tickets: [{ id: 't', task_id: 'global/RF', age_seconds: 42 }] },
        grants: [{ allocation_id: 'allocation-1', fence: 3, host: 'host-a', device_uuid: 'GPU-physical-uuid', cpu_threads: 2 }],
        studies: [{ id: 'global:RF', method: 'RF', cluster: 'global', state: 'waiting', estimated_threads: 2,
          wait: { kind: 'compute', reason: 'RAM capacity occupied by another run', seconds: 12 },
          grant: { allocation_id: 'allocation-2', fence: 4 } }, null, 'unreadable'],
        progress: { verified_complete: 'many' }, warnings: [7, 'One study entry is unreadable.'] } }
    renderPage(<ResourceReport resources={old as RunResources} live={false} />)
    const report = screen.getByRole('group', { name: 'Run resources' })
    expect(within(report).getByText('Older record · succeeded')).toBeInTheDocument()
    expect(within(report).getByText('Threads in use by running work: unknown')).toBeInTheDocument()
    expect(within(report).getByText("Memory held by this run's processes (PSS): 512 MiB · measured")).toBeInTheDocument()
    expect(within(report).getByText('global / RF · waiting')).toBeInTheDocument()
    expect(within(report).getByText('Studies (1)')).toBeInTheDocument()
    expect(within(report).getByText('GPU 0: 800 MiB in use · 2000 MiB planned for running trials · 0 MiB free for another trial · memory counted per process')).toBeInTheDocument()
    expect(within(report).getByText('One study entry is unreadable.')).toBeInTheDocument()
    expect(within(report).queryByText(/Verified models|Independent studies/)).not.toBeInTheDocument()
    expect(report.textContent).not.toMatch(retiredWords)
  })

  it('keeps malformed telemetry from crashing the editor', () => {
    // Deliberately malformed optional wire data, with no reading from the server beside it.
    const raw: unknown = { phase: 7, gpus: { 0: null, 1: { attribution: {}, reserved_mb: 'bad' }, 2: [] },
      cpu: { physical_cores: {}, logical_cpus: [], effective_cpus: '16' }, running: 'three', observed: { confidence: {}, pss_mb: null },
      compute: 'unreadable', dispatch: [], studies: {}, progress: 3 }
    renderPage(<ResourceReport resources={raw as RunResources} live />)
    expect(screen.getByText('Older record · time of the last update unknown')).toBeInTheDocument()
    expect(screen.getByText('GPU 0: measurement unavailable')).toBeInTheDocument()
    expect(screen.getByText('GPU 1: memory in use not measured')).toBeInTheDocument()
    expect(screen.getByText('GPU 2: measurement unavailable')).toBeInTheDocument()
    expect(screen.getByText('Threads in use by running work: unknown')).toBeInTheDocument()
    expect(screen.getByText("Memory held by this run's processes (PSS): unknown · measurement unavailable")).toBeInTheDocument()
  })

  it('shows a term it has no words for as it was written, whatever the term is', () => {
    // Names every object answers to are words like any other when a snapshot uses them.
    const odd: unknown = { updated: now(), gpus: { 0: { attribution: 'constructor' } },
      compute: { source: 'hasOwnProperty', updated: now(), phase: 'constructor', pss_confidence: 'toString',
        studies: [{ id: 's', cluster: 'global', method: 'RF', state: 'valueOf' }] } }
    renderPage(<ResourceReport resources={odd as RunResources} live />)
    expect(screen.getByText('Older record · constructor')).toBeInTheDocument()
    expect(screen.getByText('global / RF · valueOf')).toBeInTheDocument()
    expect(screen.getByText("Memory held by this run's processes (PSS): unknown · toString")).toBeInTheDocument()
    expect(screen.getByText('GPU 0: memory in use not measured · constructor')).toBeInTheDocument()
  })

  it('reads a record that carries no reading from the server by what it wrote itself', () => {
    const { compute: _compute, ...written } = trials()
    renderPage(<ResourceReport resources={written} live={false} />)
    expect(screen.getByText('GPU trials · running')).toBeInTheDocument()
    expect(screen.getByText('Threads in use by running work: 6')).toBeInTheDocument()
    expect(screen.getByText("Memory held by this run's processes (PSS): 9000 MiB · measured")).toBeInTheDocument()
    expect(screen.getByText('CPUs this run may use: 16')).toBeInTheDocument()
  })
})
