// A13-04: the panel must give a non-specialist something to act on, and must never present a
// stale or failed check as a current verdict.
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import { json, mockHttp } from '../test/http'
import { renderPage } from '../test/render'
import type { Readiness, ReadinessFinding, Severity } from '../api/client'
import ReadinessPanel from './ReadinessPanel'

const finding = (over: Partial<ReadinessFinding> = {}): ReadinessFinding => ({
  check: 'target.coverage', field: 'project.filename', severity: 'ok',
  checked: 'the first and last timestamp of the target series',
  finding: '16560 rows from 2024-07-11 to 2026-05-31', consequence: 'this is the period a run can learn from',
  remedy: 'Nothing to do.', ...over,
})

const report = (over: Partial<Readiness> = {}): Readiness => {
  const findings = over.findings ?? [finding()]
  const counts = { ok: 0, attention: 0, blocking: 0 } as Record<Severity, number>
  findings.forEach(f => { counts[f.severity] += 1 })
  return {
    verdict: 'ready', checked_at: '2026-10-25 19:30:00 EET', checked_at_epoch: 1792000000,
    inputs_digest: 'digest-1', findings,
    coverage: { first: '2024-07-11 00:00:00', last: '2026-05-31 23:00:00', rows: 16560, span_days: 690, truncated: false },
    bounded: { max_index_rows: 500000 }, ...over, counts: over.counts ?? counts,
  }
}

const doc = { project: { name: 'pv_park', filename: '/data/target.csv' }, data: {}, methods: {} }

function boundary(payload: Readiness | { status: number; detail: string }) {
  return mockHttp(({ url }) => {
    if (url.pathname === '/api/readiness') {
      if ('status' in payload) return json({ detail: payload.detail }, payload.status)
      return json(payload)
    }
    return json({})
  })
}

describe('ReadinessPanel', () => {
  it('checks nothing until asked, and says what the check will do', () => {
    boundary(report())
    renderPage(<ReadinessPanel document={doc} />)
    expect(screen.getByTestId('readiness-idle')).toHaveTextContent(/trains nothing and changes nothing/)
    expect(screen.queryByTestId('readiness-verdict')).toBeNull()
  })

  it('shows one of three outcomes with the time it was checked', async () => {
    boundary(report())
    renderPage(<ReadinessPanel document={doc} />)
    await userEvent.click(screen.getByTestId('readiness-check'))
    const verdict = await screen.findByTestId('readiness-verdict')
    expect(verdict).toHaveAttribute('data-verdict', 'ready')
    expect(verdict).toHaveTextContent('Ready to start')
    expect(screen.getByTestId('readiness-checked-at')).toHaveTextContent('2026-10-25 19:30:00 EET')
  })

  it('gives a blocking finding its field, its consequence and one thing to do', async () => {
    boundary(report({
      verdict: 'cannot start',
      findings: [finding({
        check: 'nwp.horizon', field: 'project.nwp_models.0.path_nwp_source', severity: 'blocking',
        finding: 'the archive stops 2026-05-31; this run needs forecasts through 2026-06-14',
        consequence: 'the last part of the period has no weather input',
        remedy: 'Extend the ecmwf archive to 2026-06-14, or choose an earlier end date.',
      })],
    }))
    renderPage(<ReadinessPanel document={doc} />)
    await userEvent.click(screen.getByTestId('readiness-check'))
    const verdict = await screen.findByTestId('readiness-verdict')
    expect(verdict).toHaveAttribute('data-verdict', 'cannot start')
    expect(verdict).toHaveTextContent('Cannot start')
    const item = screen.getByTestId('readiness-finding-nwp.horizon')
    expect(item).toHaveAttribute('data-severity', 'blocking')
    expect(item).toHaveTextContent('needs forecasts through 2026-06-14')
    expect(item).toHaveTextContent('What follows: the last part of the period has no weather input')
    expect(item).toHaveTextContent('What to do: Extend the ecmwf archive')
    expect(item).toHaveTextContent('project.nwp_models.0.path_nwp_source')
  })

  it('names the global fallback for a small cluster as attention, not as a failure', async () => {
    boundary(report({
      verdict: 'needs attention',
      findings: [finding({
        check: 'clusters.support', field: 'methods.cluster_methods', severity: 'attention',
        finding: 'cluster rule_0 has 120 samples (2 validation days, 2 test days), below the configured minimum',
        consequence: 'training for rule_0 falls back to the global scope',
        remedy: 'Accept the fallback, or widen the cluster so it has its own support.',
      })],
    }))
    renderPage(<ReadinessPanel document={doc} />)
    await userEvent.click(screen.getByTestId('readiness-check'))
    const verdict = await screen.findByTestId('readiness-verdict')
    expect(verdict).toHaveAttribute('data-verdict', 'needs attention')
    expect(screen.getByTestId('readiness-finding-clusters.support'))
      .toHaveTextContent('falls back to the global scope')
  })

  it('keeps the passed checks out of the way but reachable', async () => {
    boundary(report({ findings: [finding({ check: 'runtime.grib' }), finding({ check: 'target.lags' })] }))
    renderPage(<ReadinessPanel document={doc} />)
    await userEvent.click(screen.getByTestId('readiness-check'))
    await screen.findByTestId('readiness-verdict')
    expect(screen.getByTestId('readiness-ok-list')).not.toBeVisible()
    await userEvent.click(screen.getByTestId('readiness-toggle-ok'))
    await waitFor(() => expect(screen.getByTestId('readiness-ok-list')).toBeVisible())
    expect(screen.getByTestId('readiness-finding-runtime.grib')).toBeVisible()
    expect(screen.getByTestId('readiness-toggle-ok')).toHaveTextContent('Hide the 2 checks that passed')
  })

  it('says the verdict is out of date once the case changes, instead of showing it as current', async () => {
    boundary(report())
    const { rerender } = render(<ReadinessPanel document={doc} />)
    await userEvent.click(screen.getByTestId('readiness-check'))
    await screen.findByTestId('readiness-verdict')
    expect(screen.queryByTestId('readiness-stale')).toBeNull()
    // the same panel, a changed document: the answer on screen no longer describes it
    rerender(<ReadinessPanel document={{ ...doc, project: { ...doc.project, filename: '/data/other.csv' } }} />)
    await waitFor(() => expect(screen.getByTestId('readiness-stale'))
      .toHaveTextContent(/changed since this check. Re-check before starting/))
    // and re-checking against the new document clears it
    await userEvent.click(screen.getByTestId('readiness-check'))
    await waitFor(() => expect(screen.queryByTestId('readiness-stale')).toBeNull())
  })

  it('reports a failed check as a failed check, never as a verdict', async () => {
    boundary({ status: 400, detail: 'cannot check readiness: the target file is a directory' })
    renderPage(<ReadinessPanel document={doc} />)
    await userEvent.click(screen.getByTestId('readiness-check'))
    const error = await screen.findByTestId('readiness-error')
    expect(error).toHaveTextContent('the target file is a directory')
    expect(error).toHaveTextContent('This is not a verdict about the case')
    expect(screen.queryByTestId('readiness-verdict')).toBeNull()
  })

  it('cannot start a second check while one is in flight, and the later answer is the one shown', async () => {
    let answered = 0
    mockHttp(({ url }) => {
      if (url.pathname !== '/api/readiness') return json({})
      answered += 1
      const first = answered === 1
      const payload = report(first
        ? { verdict: 'cannot start', inputs_digest: 'older',
            findings: [finding({ check: 'older.answer', severity: 'blocking' })] }
        : { verdict: 'ready', inputs_digest: 'newer', findings: [finding({ check: 'newer.answer' })] })
      return first
        ? new Promise<Response>(resolve => setTimeout(() => resolve(json(payload)), 30))
        : json(payload)
    })
    const { rerender } = render(<ReadinessPanel document={doc} />)
    await userEvent.click(screen.getByTestId('readiness-check'))
    // in flight: the control is disabled, so two checks cannot overlap from the interface
    expect(screen.getByTestId('readiness-check')).toBeDisabled()
    expect(screen.getByTestId('readiness-busy')).toBeInTheDocument()
    await screen.findByTestId('readiness-finding-older.answer')
    expect(screen.getByTestId('readiness-check')).toBeEnabled()

    rerender(<ReadinessPanel document={{ ...doc, project: { ...doc.project, filename: '/data/new.csv' } }} />)
    await userEvent.click(screen.getByTestId('readiness-check'))
    const verdict = await screen.findByTestId('readiness-verdict')
    expect(verdict).toHaveAttribute('data-verdict', 'ready')
    await new Promise(resolve => setTimeout(resolve, 60))
    expect(screen.queryByTestId('readiness-finding-older.answer')).toBeNull()
    expect(screen.getByTestId('readiness-finding-newer.answer')).toBeInTheDocument()
    expect(answered).toBe(2)
  })

  it('shows the coverage it read, and says when it read only part of the file', async () => {
    boundary(report({ coverage: { first: '2024-01-01 00:00:00', last: '2024-02-01 00:00:00',
                                  rows: 500000, span_days: 31, truncated: true } }))
    renderPage(<ReadinessPanel document={doc} />)
    await userEvent.click(screen.getByTestId('readiness-check'))
    expect(await screen.findByTestId('readiness-coverage'))
      .toHaveTextContent('only the first 500000 rows were read')
  })

  it('renders nothing without a document', () => {
    boundary(report())
    renderPage(<ReadinessPanel document={null} />)
    expect(screen.queryByTestId('readiness-panel')).toBeNull()
  })
})
