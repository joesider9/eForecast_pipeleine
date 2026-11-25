import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import RunsPanel from './RunsPanel'
import { caseDocument, previewFixture, runFixture, scripts, stageFixture } from '../test/fixtures'
import { json, mockHttp } from '../test/http'
import { renderPage } from '../test/render'
import type { Run } from '../api/client'

const base = (extra: (call: { url: URL; method: string; body: Record<string, unknown> }) => Response | undefined) =>
  mockHttp((call) => extra(call) ?? (call.url.pathname === '/api/scripts' ? json(scripts) :
    call.url.pathname.endsWith('/log') ? json({ text: '', offset: 0, next: 0, size: 0, status: 'running' }) : json([])))

describe('previewing a run before starting it', () => {
  it('shows what would be reused and what would be replaced, and starts with the same action', async () => {
    const accepted = { ...runFixture(), state: 'pending' as const }
    const calls = base(({ url, method }) => {
      if (url.pathname.endsWith('/runs/preview')) return json(previewFixture())
      if (method === 'POST' && url.pathname.endsWith('/runs')) return json(accepted, 201)
      return undefined
    })
    renderPage(<RunsPanel caseName="demo" document={caseDocument} dirty={false} expectedRevision="saved-v1" />)
    const user = userEvent.setup()
    await screen.findByText('No run yet for this case.')

    await user.click(within(screen.getByText('1 · Datasets and clustering').closest('.MuiCard-root')!)
      .getByRole('button', { name: 'Preview' }))

    const dialog = await screen.findByRole('dialog')
    expect(within(dialog).getByText('Ready to run')).toBeInTheDocument()
    expect(within(dialog).getByText('content and input identities match the receipt')).toBeInTheDocument()
    expect(within(dialog).getByText('1 artifact(s) would be replaced')).toBeInTheDocument()
    expect(within(dialog).getByText('reuse')).toBeInTheDocument()

    // A13-07: this plan replaces a file, so the control is named after that and the summary is
    // accepted first. The action submitted is unchanged.
    await user.click(within(dialog).getByTestId('accept-consequences'))
    await user.click(within(dialog).getByTestId('plan-start'))
    await waitFor(() => expect(calls.some(call => call.method === 'POST' && call.url.pathname.endsWith('/runs'))).toBe(true))
    const submitted = calls.find(call => call.method === 'POST' && call.url.pathname.endsWith('/runs'))!
    const previewed = calls.find(call => call.url.pathname.endsWith('/runs/preview'))!
    expect(previewed.body.action).toBe('continue')
    expect(submitted.body.action).toBe('continue')
    expect(submitted.body.expected_revision).toBe('saved-v1')
  })

  it('previews and submits the action the user chose', async () => {
    const calls = base(({ url, method }) => {
      if (url.pathname.endsWith('/runs/preview')) return json(previewFixture({ status: 'ready' }))
      if (method === 'POST' && url.pathname.endsWith('/runs')) return json({ ...runFixture(), state: 'pending' as const }, 201)
      return undefined
    })
    renderPage(<RunsPanel caseName="demo" document={caseDocument} dirty={false} />)
    const user = userEvent.setup()
    await screen.findByText('No run yet for this case.')
    const card = within(screen.getByText('1 · Datasets and clustering').closest('.MuiCard-root')!)
    await user.click(card.getByLabelText('action'))
    await user.click(await screen.findByRole('option', { name: 'rebuild' }))
    await user.click(card.getByRole('button', { name: 'Preview' }))
    const dialog = await screen.findByRole('dialog')
    await user.click(within(dialog).getByTestId('accept-consequences'))
    await user.click(within(dialog).getByTestId('plan-start'))
    await waitFor(() => expect(calls.some(call => call.method === 'POST' && call.url.pathname.endsWith('/runs'))).toBe(true))
    expect(calls.find(call => call.url.pathname.endsWith('/runs/preview'))!.body.action).toBe('rebuild')
    expect(calls.find(call => call.method === 'POST' && call.url.pathname.endsWith('/runs'))!.body.action).toBe('rebuild')
  })

  it('refuses to start a blocked plan and names the prerequisite', async () => {
    base(({ url }) => url.pathname.endsWith('/runs/preview') ? json(previewFixture({
      status: 'blocked',
      stage_plan: { ...previewFixture().stage_plan!, blocked: [{ reason: 'target requires source: required payload is missing' }] },
    })) : undefined)
    renderPage(<RunsPanel caseName="demo" document={caseDocument} dirty={false} />)
    const user = userEvent.setup()
    await screen.findByText('No run yet for this case.')
    await user.click(within(screen.getByText('1 · Datasets and clustering').closest('.MuiCard-root')!)
      .getByRole('button', { name: 'Preview' }))
    const dialog = await screen.findByRole('dialog')
    expect(within(dialog).getByText('This run cannot start yet')).toBeInTheDocument()
    expect(within(dialog).getByText('target requires source: required payload is missing')).toBeInTheDocument()
    expect(within(dialog).getByTestId('plan-start')).toBeDisabled()
  })

  it('offers to restore an interrupted publication and reports what it did', async () => {
    const interrupted = previewFixture({
      status: 'verify',
      resume: { ...previewFixture().resume!, interrupted: [{ stage: 'target', canonical: 'intact', recoverable: true,
        reason: 'every output still has the bytes of the receipt this publication replaced',
        scratch: '/models/.stage-artifacts/attempts/abc', marker: '/models/.stage-artifacts/receipts/target.json' }] },
    })
    const calls = base(({ url, method }) => {
      if (url.pathname.endsWith('/runs/preview')) return json(interrupted)
      if (method === 'POST' && url.pathname.endsWith('/stages/recover')) return json([
        { stage: 'target', canonical: 'intact', recoverable: true, reason: 'ok', scratch: null, marker: 'm',
          action: 'restored', detail: 'restored; the stage is verified again' }])
      return undefined
    })
    renderPage(<RunsPanel caseName="demo" document={caseDocument} dirty={false} manageable />)
    const user = userEvent.setup()
    await screen.findByText('No run yet for this case.')
    await user.click(within(screen.getByText('1 · Datasets and clustering').closest('.MuiCard-root')!)
      .getByRole('button', { name: 'Preview' }))
    const dialog = await screen.findByRole('dialog')
    expect(within(dialog).getByText('A publication of this case was interrupted')).toBeInTheDocument()
    await user.click(within(dialog).getByRole('button', { name: 'Restore' }))
    expect(await within(dialog).findByText(/restored; the stage is verified again/)).toBeInTheDocument()
    expect(calls.find(call => call.url.pathname.endsWith('/stages/recover'))!.body.apply).toBe(true)
  })
})

describe('a run that cannot simply be called finished', () => {
  const held = (): Run => ({ ...runFixture('finished'), state: 'terminal', cleanup: 'pending',
    stages: [stageFixture('target'), stageFixture('row', { state: 'unknown', outcome: null, readiness: { status: 'unverified' } })] })

  it('says the cleanup is unresolved and that the case is held', async () => {
    base(({ url }) => url.pathname.endsWith('/runs') ? json([held()]) : undefined)
    renderPage(<RunsPanel caseName="demo" document={caseDocument} dirty={false} />)
    expect(await screen.findByText('finished · cleanup pending')).toBeInTheDocument()
    expect(screen.getByText(/this case is held/)).toBeInTheDocument()
  })

  it('shows each stage with its own outcome and artifact verdict', async () => {
    base(({ url }) => url.pathname.endsWith('/runs') ? json([held()]) : undefined)
    renderPage(<RunsPanel caseName="demo" document={caseDocument} dirty={false} />)
    const user = userEvent.setup()
    await user.click(await screen.findByRole('button', { name: 'Stages (2)' }))
    expect(await screen.findByText('target observed')).toBeInTheDocument()
    expect(screen.getByText('verified')).toBeInTheDocument()
    expect(screen.getByText('unverified')).toBeInTheDocument()
  })

  it('ends a lost run only with a reason, and sends it', async () => {
    const lost: Run = { ...runFixture('lost'), state: 'unknown', cleanup: 'unknown',
      report_error: 'report collection: unreadable journal' }
    const calls = base(({ url, method }) => {
      if (url.pathname.endsWith('/resolve') && method === 'POST') return json({ ...lost, state: 'terminal', cleanup: 'complete' })
      if (url.pathname.endsWith('/runs')) return json([lost])
      return undefined
    })
    renderPage(<RunsPanel caseName="demo" document={caseDocument} dirty={false} manageable />)
    const user = userEvent.setup()
    await user.click(await screen.findByRole('button', { name: 'Resolve' }))
    const dialog = await screen.findByRole('dialog')
    expect(within(dialog).getByText('report collection: unreadable journal')).toBeInTheDocument()
    expect(within(dialog).getByRole('button', { name: 'Resolve' })).toBeDisabled()
    await user.type(within(dialog).getByLabelText('reason'), 'the host was rebuilt')
    await user.click(within(dialog).getByRole('button', { name: 'Resolve' }))
    await waitFor(() => expect(calls.some(call => call.url.pathname.endsWith('/resolve'))).toBe(true))
    expect(calls.find(call => call.url.pathname.endsWith('/resolve'))!.body.reason).toBe('the host was rebuilt')
  })
})

describe('the submission key', () => {
  it('survives a reload, so a retry reaches the run that was already accepted', async () => {
    window.localStorage.clear()
    let attempts = 0
    const calls = base(({ url, method }) => {
      if (method === 'POST' && url.pathname.endsWith('/runs')) {
        return ++attempts === 1 ? json({ detail: 'response lost' }, 503) : json({ ...runFixture(), state: 'pending' as const }, 201)
      }
      return undefined
    })
    const first = renderPage(<RunsPanel caseName="demo" document={caseDocument} dirty={false} expectedRevision="saved-v1" />)
    const user = userEvent.setup()
    await screen.findByText('No run yet for this case.')
    await user.click(screen.getByRole('button', { name: 'Run datasets' }))
    await screen.findByText('response lost')
    first.unmount()                                   // the page is reloaded

    renderPage(<RunsPanel caseName="demo" document={caseDocument} dirty={false} expectedRevision="saved-v1" />)
    await screen.findByText('No run yet for this case.')
    await user.click(screen.getByRole('button', { name: 'Run datasets' }))
    await waitFor(() => expect(calls.filter(call => call.method === 'POST' && call.url.pathname.endsWith('/runs'))).toHaveLength(2))
    const posts = calls.filter(call => call.method === 'POST' && call.url.pathname.endsWith('/runs'))
    expect(posts[1].body.client_key).toBe(posts[0].body.client_key)
  })
})

describe('the corrections asked for in review', () => {
  // R10-02: the dialog may not start what the card refuses, and a failed plan approves nothing
  it('refuses to approve a plan it could not resolve', async () => {
    const calls = base(({ url }) => url.pathname.endsWith('/runs/preview') ? json({ detail: 'the case did not resolve' }, 400) : undefined)
    renderPage(<RunsPanel caseName="demo" document={caseDocument} dirty={false} />)
    const user = userEvent.setup()
    await screen.findByText('No run yet for this case.')
    await user.click(within(screen.getByText('1 · Datasets and clustering').closest('.MuiCard-root')!)
      .getByRole('button', { name: 'Preview' }))
    const dialog = await screen.findByRole('dialog')
    expect(await within(dialog).findByText(/could not be resolved/)).toBeInTheDocument()
    expect(within(dialog).getByTestId('plan-start')).toBeDisabled()
    expect(within(dialog).getByText(/nothing can be approved/)).toBeInTheDocument()
    expect(calls.filter(call => call.method === 'POST' && call.url.pathname.endsWith('/runs'))).toHaveLength(0)
  })

  it('does not offer a preview at all while the editor has unsaved changes', async () => {
    base(() => undefined)
    renderPage(<RunsPanel caseName="demo" document={caseDocument} dirty />)
    await screen.findByText('Save the case first: a run reads the saved files, not the editor.')
    for (const button of screen.getAllByRole('button', { name: 'Preview' })) expect(button).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Run datasets' })).toBeDisabled()
  })

  // R10-03: the UI offers exactly what the server accepts, from whoever it accepts it
  it('does not offer the repairs to a writer who does not manage the tenant', async () => {
    const lost: Run = { ...runFixture('lost'), state: 'unknown', cleanup: 'unknown' }
    base(({ url }) => url.pathname.endsWith('/runs') ? json([lost]) : undefined)
    renderPage(<RunsPanel caseName="demo" document={caseDocument} dirty={false} />)
    expect(await screen.findByText(/this case is held/)).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Resolve' })).not.toBeInTheDocument()
  })

  it('does not offer resolution for a run that can still finish by itself', async () => {
    const settling: Run = { ...runFixture('finished'), state: 'terminal', cleanup: 'pending' }
    base(({ url }) => url.pathname.endsWith('/runs') ? json([settling]) : undefined)
    renderPage(<RunsPanel caseName="demo" document={caseDocument} dirty={false} manageable />)
    expect(await screen.findByText('finished · cleanup pending')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Resolve' })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Delete' })).toBeDisabled()
  })

  it('keeps an observed success in the words it uses about resolution', async () => {
    const held: Run = { ...runFixture('finished'), state: 'terminal', cleanup: 'unknown',
      outcome: { kind: 'success', detail: 'all good', observed_at: 2, returncode: 0 } }
    base(({ url }) => url.pathname.endsWith('/runs') ? json([held]) : undefined)
    renderPage(<RunsPanel caseName="demo" document={caseDocument} dirty={false} manageable />)
    const user = userEvent.setup()
    await user.click(await screen.findByRole('button', { name: 'Resolve' }))
    const dialog = await screen.findByRole('dialog')
    expect(within(dialog).getByText(/observed success is kept as it is/)).toBeInTheDocument()
    expect(within(dialog).queryByText(/recorded as a failure/)).not.toBeInTheDocument()
  })

  // R10-04: a run with no process yet is not shown as running, and it holds the case all the
  // same. Nothing queues a run behind another's resources any more, so there is no further
  // reason to give beside the state.
  it('shows a run that has no process yet as queued, and the case as taken', async () => {
    const parked: Run = { ...runFixture('running'), state: 'queued', pid: null }
    base(({ url }) => url.pathname.endsWith('/runs') ? json([parked]) : undefined)
    renderPage(<RunsPanel caseName="demo" document={caseDocument} dirty={false} />)
    expect(await screen.findByText('queued')).toBeInTheDocument()
    expect(screen.queryByText('running')).not.toBeInTheDocument()
    expect(screen.getByText('A training run is in progress; one run per case at a time.')).toBeInTheDocument()
    for (const name of ['Run datasets', 'Run training', 'Run combine']) expect(screen.getByRole('button', { name })).toBeDisabled()
  })
})

describe('a consequential run is approved deliberately (A13-07)', () => {
  const openPlan = async (preview: unknown, action?: string) => {
    const calls = base(({ url, method }) => {
      if (url.pathname.endsWith('/runs/preview')) return json(preview)
      if (method === 'POST' && url.pathname.endsWith('/runs')) return json({ ...runFixture(), state: 'pending' }, 201)
      return undefined
    })
    renderPage(<RunsPanel caseName="demo" document={caseDocument} dirty={false} expectedRevision="saved-v1" />)
    const user = userEvent.setup()
    await screen.findByText('No run yet for this case.')
    const card = within(screen.getByText('1 · Datasets and clustering').closest('.MuiCard-root')!)
    if (action) {
      await user.click(card.getByLabelText('action'))
      await user.click(await screen.findByRole('option', { name: action }))
    }
    await user.click(card.getByRole('button', { name: 'Preview' }))
    return { user, calls, dialog: await screen.findByRole('dialog') }
  }

  const started = (calls: { method: string; url: URL }[]) =>
    calls.some(call => call.method === 'POST' && call.url.pathname.endsWith('/runs'))

  it('refuses to start until the summary of what will be replaced is accepted', async () => {
    const { user, calls, dialog } = await openPlan(previewFixture())
    expect(within(dialog).getByTestId('plan-start')).toBeDisabled()
    expect(within(dialog).getByTestId('plan-refusal'))
      .toHaveTextContent('Accept the summary of what will be replaced first.')
    // the control is named after what it does, not after the action it was reached by
    expect(within(dialog).getByTestId('plan-start')).toHaveTextContent('Replace 1 file(s) and start')

    await user.click(within(dialog).getByTestId('accept-consequences'))
    expect(within(dialog).getByTestId('plan-start')).toBeEnabled()
    expect(started(calls)).toBe(false)
  })

  it('declining and closing starts nothing at all', async () => {
    const { user, calls, dialog } = await openPlan(previewFixture())
    await user.click(within(dialog).getByRole('button', { name: 'Close' }))
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
    expect(started(calls)).toBe(false)
  })

  it('keeps a routine run that replaces nothing simple', async () => {
    const routine = previewFixture({
      resume: { action: 'continue', verified: true, plan_id: 'plan-identity', script: 'datasets',
        interrupted: [], totals: { reuse: 2, artifacts_to_replace: 0 },
        stages: [{ key: 'target', scope: 'global', decision: 'reuse', reason: 'identities match', dependencies: [] },
                 { key: 'row', scope: 'global', decision: 'reuse', reason: 'identities match', dependencies: [] }] },
    })
    const { user, calls, dialog } = await openPlan(routine)
    expect(within(dialog).queryByTestId('accept-consequences')).toBeNull()
    expect(within(dialog).getByTestId('plan-start')).toHaveTextContent('Start')
    expect(within(dialog).getByTestId('plan-start')).toBeEnabled()
    await user.click(within(dialog).getByTestId('plan-start'))
    await waitFor(() => expect(started(calls)).toBe(true))
  })

  it('names a rebuild after what it replaces', async () => {
    const { dialog } = await openPlan(previewFixture({
      resume: { ...previewFixture().resume!, action: 'rebuild' } }), 'rebuild')
    expect(within(dialog).getByTestId('plan-start')).toHaveTextContent('Replace 1 file(s) and rebuild')
  })

  it('an acceptance does not carry over to a plan that replaces different files', async () => {
    let answered = 0
    const calls = base(({ url, method }) => {
      if (url.pathname.endsWith('/runs/preview')) {
        answered += 1
        const resume = previewFixture().resume!
        return json(previewFixture({ resume: { ...resume, plan_id: `plan-${answered}`,
          stages: [{ ...resume.stages[1], affected: [`/models/DATA/file${answered}.pickle`] }] } }))
      }
      if (method === 'POST' && url.pathname.endsWith('/runs')) return json({ ...runFixture(), state: 'pending' }, 201)
      return undefined
    })
    renderPage(<RunsPanel caseName="demo" document={caseDocument} dirty={false} expectedRevision="saved-v1" />)
    const user = userEvent.setup()
    await screen.findByText('No run yet for this case.')
    const card = within(screen.getByText('1 · Datasets and clustering').closest('.MuiCard-root')!)
    await user.click(card.getByRole('button', { name: 'Preview' }))
    let dialog = await screen.findByRole('dialog')
    await user.click(within(dialog).getByTestId('accept-consequences'))
    expect(within(dialog).getByTestId('plan-start')).toBeEnabled()

    // close and re-plan: the files are different, so the earlier acceptance must not stand
    await user.click(within(dialog).getByRole('button', { name: 'Close' }))
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
    await user.click(card.getByLabelText('action'))
    await user.click(await screen.findByRole('option', { name: 'rebuild' }))
    await user.click(card.getByRole('button', { name: 'Preview' }))
    dialog = await screen.findByRole('dialog')
    await waitFor(() => expect(within(dialog).getByTestId('run-consequences'))
      .toHaveTextContent('/models/DATA/file2.pickle'))
    expect(within(dialog).getByTestId('accept-consequences')).not.toBeChecked()
    expect(within(dialog).getByTestId('plan-start')).toBeDisabled()
    expect(started(calls)).toBe(false)
  })

  it('shows the consequences and the stage table together, not one instead of the other', async () => {
    const { dialog } = await openPlan(previewFixture())
    expect(within(dialog).getByTestId('run-consequences')).toBeInTheDocument()
    expect(within(dialog).getByText('1 artifact(s) would be replaced')).toBeInTheDocument()
    expect(within(dialog).getByText('content and input identities match the receipt')).toBeInTheDocument()
  })

  it('the acceptance is reachable from the keyboard', async () => {
    const { user, dialog } = await openPlan(previewFixture())
    const box = within(dialog).getByTestId('accept-consequences')
    box.focus()
    expect(box).toHaveFocus()
    await user.keyboard(' ')
    expect(box).toBeChecked()
    expect(within(dialog).getByTestId('plan-start')).toBeEnabled()
  })
})

// The configuration is no longer the one the datasets or the models were built from, and the
// run would replace them. The server starts no such run unasked; the preview is where what
// goes is read and accepted, by the name of the case.
describe('a run that replaces what another configuration built', () => {
  const changed = () => previewFixture({
    changes: { stages: [{ key: 'row', reason: 'row configuration changed; rebuild this stage', files: 1 }],
      families: { TRANSFORMER: { units: 8, trials: 101, clusters: ['global'] } } },
    since: { record: {}, changes: [{ path: 'data.variables[Cloud].lags', change: 'changed', before: [0, 1], after: [0, 1, 2] }] },
  })

  it('is started from its preview only, with the name of the case typed', async () => {
    const calls = base(({ url, method }) => {
      if (url.pathname.endsWith('/runs/preview')) return json(changed())
      if (method === 'POST' && url.pathname.endsWith('/runs')) return json({ ...runFixture(), state: 'pending' as const }, 201)
      return undefined
    })
    renderPage(<RunsPanel caseName="demo" document={caseDocument} dirty={false} expectedRevision="saved-v1" />)
    const user = userEvent.setup()
    await screen.findByText('No run yet for this case.')
    await user.click(within(screen.getByText('1 · Datasets and clustering').closest('.MuiCard-root')!)
      .getByRole('button', { name: 'Preview' }))
    const dialog = await screen.findByRole('dialog')
    const panel = within(dialog).getByTestId('configuration-changes')
    expect(panel).toHaveTextContent('The configuration is not the one this was built from')
    expect(panel).toHaveTextContent('data.variables[Cloud].lags')
    expect(panel).toHaveTextContent('row: 1 file(s)')
    expect(panel).toHaveTextContent('TRANSFORMER: 8 studies, 101 trial folder(s)')
    // a tick does not accept this: the checkbox is not even offered
    expect(within(dialog).queryByTestId('accept-consequences')).not.toBeInTheDocument()
    const start = within(dialog).getByTestId('plan-start')
    expect(start).toBeDisabled()
    expect(within(dialog).getByTestId('plan-refusal')).toHaveTextContent('Type the name of the case')
    await user.type(within(dialog).getByTestId('typed-confirmation'), 'dem')
    expect(start).toBeDisabled()
    await user.type(within(dialog).getByTestId('typed-confirmation'), 'o')
    expect(start).toBeEnabled()
    await user.click(start)
    await waitFor(() => expect(calls.some(call => call.method === 'POST' && call.url.pathname.endsWith('/runs'))).toBe(true))
    const submitted = calls.find(call => call.method === 'POST' && call.url.pathname.endsWith('/runs'))!
    expect(submitted.body.accept_changes).toBe(true)
    expect(submitted.body.action).toBe('continue')
    expect(submitted.body.expected_revision).toBe('saved-v1')
  })

  it('opens its preview when the run button is pressed, instead of starting or failing', async () => {
    let started = 0
    const calls = base(({ url, method, body }) => {
      if (url.pathname.endsWith('/runs/preview')) return json(changed())
      if (method === 'POST' && url.pathname.endsWith('/runs')) {
        started += 1
        return body.accept_changes ? json({ ...runFixture(), state: 'pending' as const }, 201)
          : json({ detail: 'The configuration of demo is not the one this was built from.', preview: changed() }, 428)
      }
      return undefined
    })
    renderPage(<RunsPanel caseName="demo" document={caseDocument} dirty={false} />)
    const user = userEvent.setup()
    await screen.findByText('No run yet for this case.')
    await user.click(screen.getByRole('button', { name: 'Run datasets' }))
    const dialog = await screen.findByRole('dialog')
    expect(within(dialog).getByTestId('configuration-changes')).toBeInTheDocument()
    expect(screen.queryByRole('alert', { name: /failed/ })).not.toBeInTheDocument()
    expect(started).toBe(1)
    await user.type(within(dialog).getByTestId('typed-confirmation'), 'demo')
    await user.click(within(dialog).getByTestId('plan-start'))
    await waitFor(() => expect(started).toBe(2))
    const posts = calls.filter(call => call.method === 'POST' && call.url.pathname.endsWith('/runs'))
    expect(posts[0].body).not.toHaveProperty('accept_changes')
    expect(posts[1].body.accept_changes).toBe(true)
    // the answer is part of what names the request: the refused one and the accepted one are two
    expect(posts[1].body.client_key).not.toBe(posts[0].body.client_key)
  })

  it('still starts straight from its button when nothing is to be accepted', async () => {
    const calls = base(({ url, method }) =>
      method === 'POST' && url.pathname.endsWith('/runs') ? json({ ...runFixture(), state: 'pending' as const }, 201) : undefined)
    renderPage(<RunsPanel caseName="demo" document={caseDocument} dirty={false} />)
    const user = userEvent.setup()
    await screen.findByText('No run yet for this case.')
    await user.click(screen.getByRole('button', { name: 'Run datasets' }))
    await waitFor(() => expect(calls.some(call => call.method === 'POST' && call.url.pathname.endsWith('/runs'))).toBe(true))
    expect(calls.some(call => call.url.pathname.endsWith('/preview'))).toBe(false)
    expect(calls.find(call => call.method === 'POST')!.body).not.toHaveProperty('accept_changes')
  })

  it('says in the stage table which studies are trained from the beginning', async () => {
    base(({ url }) => url.pathname.endsWith('/runs/preview') ? json(previewFixture({
      changes: { stages: [], families: { MLP: { units: 1, trials: 4, clusters: ['global'] } } },
      resume: { action: 'continue', verified: false, plan_id: 'plan-2', script: 'training', interrupted: [],
        totals: { reuse: 0, resume: 1, verify: 0, rebuild: 0, blocked: 0, skip: 0, artifacts_to_replace: 5 },
        stages: [{ key: 'training', scope: 'global', decision: 'resume', dependencies: [], cause: 'configuration',
          reason: 'trained from the beginning, because the configuration it was trained under has changed',
          affected: ['/models/global/MLP/test_0'],
          units: { totals: {}, units: [
            { cluster: 'global', method: 'MLP', reuse: 0, redo: 0, other_revision: 0, retrain: true, discard: 4,
              budget: { done: 0, wanted: 4, missing: 4 } },
            { cluster: 'global', method: 'CNN', reuse: 2, redo: 0, other_revision: 0, budget: { done: 2, wanted: 2, missing: 0 } }] } }] },
    })) : undefined)
    renderPage(<RunsPanel caseName="demo" document={caseDocument} dirty={false} />)
    const user = userEvent.setup()
    await screen.findByText('No run yet for this case.')
    await user.click(within(screen.getByText('2 · Training').closest('.MuiCard-root')!).getByRole('button', { name: 'Preview' }))
    const dialog = await screen.findByRole('dialog')
    expect(dialog).toHaveTextContent('global/MLP: trained from the beginning, 4 trial folder(s) removed')
    expect(dialog).toHaveTextContent('global/CNN: 2/2 trials, reuse 2, redo 0')
  })

  it('says how many transfers each cluster has still to train', async () => {
    // No study wants them, so no budget counts them: a plan used to show "3/3 trials" for a
    // cluster with every one of its transfers untrained.
    base(({ url }) => url.pathname.endsWith('/runs/preview') ? json(previewFixture({
      resume: { action: 'continue', verified: false, plan_id: 'plan-3', script: 'training', interrupted: [],
        totals: { reuse: 0, resume: 1, verify: 0, rebuild: 0, blocked: 0, skip: 0, artifacts_to_replace: 0 },
        stages: [{ key: 'training', scope: 'global', decision: 'resume', dependencies: [],
          reason: '0 trial(s) still wanted, 0 unfinished to redo, 5 transfer trial(s) to train, 9 compatible trial(s) kept',
          units: { totals: {}, units: [
            { cluster: 'global', method: 'LSTM', reuse: 3, redo: 0, other_revision: 0, transfer: null,
              budget: { done: 3, wanted: 3, missing: 0 } },
            { cluster: 'RBF_rule_0', method: 'LSTM', reuse: 4, redo: 0, other_revision: 0,
              budget: { done: 3, wanted: 3, missing: 0 }, transfer: { sources: 3, done: 1, pending: 2, later: 0 } },
            { cluster: 'RBF_rule_1', method: 'LSTM', reuse: 3, redo: 0, other_revision: 0,
              budget: { done: 3, wanted: 3, missing: 0 }, transfer: { sources: 3, done: 0, pending: 3, later: 2 } },
            { cluster: 'RBF_rule_2', method: 'LSTM', reuse: 6, redo: 0, other_revision: 0,
              budget: { done: 3, wanted: 3, missing: 0 }, transfer: { sources: 3, done: 3, pending: 0, later: 0 } },
            { cluster: 'RBF_rule_0', method: 'RF', reuse: 1, redo: 0, other_revision: 0,
              budget: { done: 1, wanted: 1, missing: 0 }, transfer: { sources: 0, done: 0, pending: 0, later: 0 } }] } }] },
    })) : undefined)
    renderPage(<RunsPanel caseName="demo" document={caseDocument} dirty={false} />)
    const user = userEvent.setup()
    await screen.findByText('No run yet for this case.')
    await user.click(within(screen.getByText('2 · Training').closest('.MuiCard-root')!).getByRole('button', { name: 'Preview' }))
    const dialog = await screen.findByRole('dialog')
    expect(dialog).toHaveTextContent('5 transfer trial(s) to train')
    expect(dialog).toHaveTextContent('global/LSTM: 3/3 trials, reuse 3, redo 0 ·')
    expect(dialog).toHaveTextContent('RBF_rule_0/LSTM: 3/3 trials, reuse 4, redo 0, transfers 1/3 trained, 2 to train ·')
    expect(dialog).toHaveTextContent(
      'RBF_rule_1/LSTM: 3/3 trials, reuse 3, redo 0, transfers 0/3 trained, 3 to train, up to 2 more after the global study')
    expect(dialog).toHaveTextContent('RBF_rule_2/LSTM: 3/3 trials, reuse 6, redo 0, transfers 3/3 trained ·')
    // a family that trains none says nothing of them
    expect(dialog).toHaveTextContent(/RBF_rule_0\/RF: 1\/1 trials, reuse 1, redo 0(?!, transfers)/)
  })
})
