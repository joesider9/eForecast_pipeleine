import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import RunsPanel from './RunsPanel'
import { caseDocument, resourceFixture, runFixture, scripts, stageFixture } from '../test/fixtures'
import { json, mockHttp } from '../test/http'
import { renderPage } from '../test/render'
import type { Run } from '../api/client'

describe('run actions and failures', () => {
  it('reuses the submission key after a lost response and sends the saved revision', async () => {
    let attempts = 0
    const accepted = { ...runFixture(), state: 'pending' as const, input_snapshot_status: 'pending' as const }
    const calls = mockHttp(({ url, method }) => {
      if (url.pathname === '/api/scripts') return json(scripts)
      if (url.pathname.endsWith('/log')) return json({ text: '', offset: 0, next: 0, size: 0, status: 'running' })
      if (method === 'POST') return ++attempts === 1 ? json({ detail: 'response lost' }, 503) : json(accepted, 201)
      return json([])
    })
    renderPage(<RunsPanel caseName="demo" document={caseDocument} dirty={false} expectedRevision="saved-v1" />)
    const user = userEvent.setup()
    await screen.findByText('No run yet for this case.')
    await user.click(screen.getByRole('button', { name: 'Run datasets' }))
    await screen.findByText('response lost')
    await user.click(screen.getByRole('button', { name: 'Run datasets' }))
    expect(await screen.findByText('Preparing accepted inputs.')).toBeInTheDocument()
    const submitted = calls.filter(call => call.method === 'POST')
    expect(submitted).toHaveLength(2)
    expect(submitted[0].body.client_key).toEqual(expect.any(String))
    expect(submitted[1].body.client_key).toBe(submitted[0].body.client_key)
    expect(submitted[1].body.expected_revision).toBe('saved-v1')
  })
  it.each(['pinned', 'degraded', 'raw', 'legacy'] as const)('shows %s input provenance in the run list and log', async (mode) => {
    const run = runFixture('finished')
    if (mode === 'pinned') {
      run.input_snapshot_status = 'pinned'; run.effective_revision = 'effective-123'; run.config_revision = 'raw-456'
      run.effective_options = { serial: true, stage: ['row'] }
    } else if (mode === 'degraded') {
      run.input_snapshot_status = 'degraded'; run.config_frozen = true
      run.normalized_error = 'Experiment module could not be resolved'
    } else if (mode === 'raw') run.config_frozen = true
    mockHttp(({ url }) => url.pathname === '/api/scripts' ? json(scripts) : url.pathname.endsWith('/log') ?
      json({ text: 'done', offset: 0, next: 4, size: 4, status: 'finished', truncated: false }) : json([run]))
    renderPage(<RunsPanel caseName="demo" document={caseDocument} dirty={false} />)
    const label = mode === 'pinned' ? 'Inputs pinned' : mode === 'degraded' ? 'Degraded inputs: using live configuration.' :
      mode === 'raw' ? 'Raw files archived; effective inputs unverified.' : 'Input provenance unavailable for this older run.'
    expect(await screen.findByText(label)).toBeInTheDocument()
    if (mode === 'pinned') {
      expect(screen.getByText('Effective revision: effective-123')).toBeInTheDocument()
      expect(screen.getByText('Raw case revision: raw-456')).toBeInTheDocument()
    }
    await userEvent.setup().click(screen.getByRole('button', { name: 'Log' }))
    await screen.findByText('done')
    const groups = screen.getAllByRole('group', { name: 'Run input provenance' })
    const drawer = groups[groups.length - 1]
    expect(within(drawer).getByText(label)).toBeInTheDocument()
    if (mode === 'degraded') {
      await userEvent.setup().click(within(drawer).getByText('Input details'))
      expect(within(drawer).getByText('Experiment module could not be resolved')).toBeVisible()
    }
  })

  it('distinguishes failed run loading from an empty case and supports retry', async () => {
    let failed = true
    mockHttp(({ url }) => url.pathname === '/api/scripts' ? json(scripts) : failed ? json({ detail: 'database unavailable' }, 503) : json([]))
    renderPage(<RunsPanel caseName="demo" document={caseDocument} dirty={false} />)
    await screen.findByText(/Could not load runs/)
    expect(screen.queryByText('No run yet for this case.')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Run training' })).toBeDisabled()
    failed = false
    await userEvent.setup().click(screen.getByRole('button', { name: 'Retry' }))
    await screen.findByText('No run yet for this case.')
    expect(screen.getByRole('button', { name: 'Run training' })).toBeEnabled()
  })

  it('starts, inspects, stops and deletes a run through the HTTP boundary', async () => {
    let runs: Run[] = []
    const calls = mockHttp(({ url, method }) => {
      if (url.pathname === '/api/scripts') return json(scripts)
      if (url.pathname.endsWith('/log')) return json({ text: 'training output', offset: 0, next: 15, size: 15, status: 'running', truncated: false })
      if (url.pathname.endsWith('/stop')) { runs = [runFixture('stopped')]; return json(runs[0]) }
      if (method === 'DELETE') { runs = []; return new Response(null, { status: 204 }) }
      if (method === 'POST') { runs = [runFixture()]; return json(runs[0], 201) }
      return json(runs)
    })
    renderPage(<RunsPanel caseName="demo" document={caseDocument} dirty={false} />)
    const user = userEvent.setup()
    await screen.findByText('No run yet for this case.')
    await user.click(screen.getByRole('button', { name: 'Run training' }))
    await screen.findByText('training output')
    await user.click(screen.getByRole('button', { name: 'Close' }))
    await user.click(await screen.findByRole('button', { name: 'Stop' }))
    await user.click(await screen.findByRole('button', { name: 'Delete' }))
    await screen.findByText('No run yet for this case.')
    expect(calls.some(call => call.url.pathname.endsWith('/stop'))).toBe(true)
    expect(calls.some(call => call.method === 'DELETE')).toBe(true)
  })

  it.each([['running', 'Stop'], ['finished', 'Delete']] as const)('shows a failed %s run action', async (status, action) => {
    mockHttp(({ url, method }) => url.pathname === '/api/scripts' ? json(scripts) : method === 'GET' ? json([runFixture(status)]) : json({ detail: 'permission denied' }, 403))
    renderPage(<RunsPanel caseName="demo" document={caseDocument} dirty={false} />)
    await userEvent.setup().click(await screen.findByRole('button', { name: action }))
    expect(await screen.findByText(`${action} failed: permission denied`)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: action })).toBeEnabled()
  })

  it('prevents launch of unsaved or read-only cases', async () => {
    const calls = mockHttp(({ url }) => url.pathname === '/api/scripts' ? json(scripts) : json([]))
    const { unmount } = renderPage(<RunsPanel caseName="demo" document={caseDocument} dirty />)
    await screen.findByText('No run yet for this case.')
    for (const name of ['Run datasets', 'Run training', 'Run combine']) expect(screen.getByRole('button', { name })).toBeDisabled()
    unmount()
    renderPage(<RunsPanel caseName="demo" document={caseDocument} dirty={false} readOnly />)
    await screen.findByText('No run yet for this case.')
    expect(screen.getByRole('button', { name: 'Run training' })).toBeDisabled()
    // a preview resolves the case on the server, so it is refused with the run it would lead to
    expect(screen.getAllByRole('button', { name: 'Preview' }).every(button => button.hasAttribute('disabled'))).toBe(true)
    expect(calls.every(call => call.method === 'GET')).toBe(true)
  })

  it('retains resource reporting and accurately describes combine roles', async () => {
    const run = runFixture('finished')
    run.resources = resourceFixture()
    mockHttp(({ url }) => url.pathname === '/api/scripts' ? json(scripts) : json([run]))
    renderPage(<RunsPanel caseName="demo" document={caseDocument} dirty={false} />)
    const report = await screen.findByRole('group', { name: 'Run resources' })
    expect(within(report).getByText('CPU studies · running')).toBeInTheDocument()
    expect(within(report).getByText('Threads in use by running work: 2')).toBeInTheDocument()
    expect(within(report).getByText(/rule-b \/ RF · waiting — its turn has not come/)).toBeInTheDocument()
    expect(screen.getByText(/A separate calibration slice fits prediction intervals/)).toBeInTheDocument()
    expect(screen.queryByText(/first half/)).not.toBeInTheDocument()
    await waitFor(() => expect(screen.getByRole('button', { name: 'Run combine' })).toBeEnabled())
  })

  it('opens and hides the stages of a run, and asks the server for nothing else about it', async () => {
    const run = { ...runFixture('finished'), stages: [stageFixture('target')] }
    const calls = mockHttp(({ url }) => url.pathname === '/api/scripts' ? json(scripts) : json([run]))
    renderPage(<RunsPanel caseName="demo" document={caseDocument} dirty={false} />)
    const user = userEvent.setup()
    await user.click(await screen.findByRole('button', { name: 'Stages (1)' }))
    expect(await screen.findByText('target')).toBeVisible()
    await user.click(screen.getByRole('button', { name: 'Hide stages' }))
    expect(screen.queryByText('target')).not.toBeInTheDocument()
    // The row's actions are its stages, its log and its end; the page reads the run list and the scripts.
    const row = within(screen.getByText('training').closest('tr')!)
    expect(row.getAllByRole('button').map(button => button.textContent)).toEqual(['Stages (1)', 'Log', 'Delete'])
    expect(new Set(calls.map(call => `${call.method} ${call.url.pathname}`))).toEqual(new Set(['GET /api/scripts', 'GET /api/cases/demo/runs']))
  })

  // A run accepted by an older version was recorded with what that version kept about it: where
  // it stood in a queue, what had been set aside for it, which throughput settings it was
  // pinned to. The record still loads; none of that is read, and what the row says of the run
  // is what it says of any other.
  it('shows a run recorded by an older version without what this version no longer has', async () => {
    const old: unknown = { ...runFixture('running'), state: 'queued', pid: null,
      profile_error: 'the profile store could not be read', profile_selection: { id: 'pinned-version-1', status: 'fallback',
        requested: { execution_profile: 'a'.repeat(64), profile_resources: true }, reason: 'Exact workload must match.' },
      options: { cpu: true, execution_profile: 'a'.repeat(64), array_reader: 'prefer-mapped', cpu_study_ram_mb: 4500, cpu_wait_timeout: 60 },
      dispatch: { status: 'waiting', reserved_bytes: 268435456, reservations: [
        { key: 'run:1', kind: 'run', state: 'waiting', reason: 'no supervisor slot is free', ticket: 1, wait_kind: 'supervisor_slot',
          held_by: ['a run started by hand holds /data/models/pv_park'], failure: { type: 'OSError', errno: 5, detail: 'disk vanished' },
          roots: [{ id: 'host-a:259', path: '/models', bytes: 268435456, actual_bytes: 268435456 }] }] },
      resources: { registry: { mode: 'strict', state: 'ok' }, reservations: { cpu_threads: 4, host_mb: 2048 },
        foreign_claims: [{ run_id: 'other', cpu_threads: 2, host_mb: 100 }] } }
    mockHttp(({ url }) => url.pathname === '/api/scripts' ? json(scripts) : json([old]))
    renderPage(<RunsPanel caseName="demo" document={caseDocument} dirty={false} />)
    const row = (await screen.findByText('queued')).closest('tr')!
    expect(within(row).getByText('Preparing accepted inputs.')).toBeInTheDocument()
    expect(within(row).getByText('No resource measurements in this record')).toBeInTheDocument()
    expect(row.textContent).not.toMatch(/supervisor|slot|promised|scratch|strict|disk vanished|pinned-version|store could not/)
    // It has not ended, so it holds the case like any run that has not.
    expect(screen.getByRole('button', { name: 'Run training' })).toBeDisabled()
  })
})
