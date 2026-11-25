// A14-02: the panel a colleague reads before using a number. Each test is a way it could lie.
import { screen, waitFor } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { json, mockHttp } from '../test/http'
import { renderPage } from '../test/render'
import type { ResultSummary } from '../api/client'
import ResultSummaryPanel from './ResultSummary'

const summary = (over: Partial<ResultSummary> = {}): ResultSummary => ({
  case: { name: 'wind', type: 'wind', horizon_type: 'short_term', path_model: '/m', model_exists: true },
  verdict: { state: 'experimental', sentence: 'These results are experimental: nobody has approved a version for routine use.',
             actions: [{ what: 'ask a maintainer to approve a version', why: 'nothing is approved, so these results are experimental' }] },
  model: { state: 'experimental', detail: 'no version is approved for this case', remedy: 'eforecast approve wind <bundle folder>' },
  headline: { served: { level: 'models', predictor: 'kmeans', cluster: 'all', mae: 1.4, rmse: 2.1, bias: 0.1, n: 154000 },
              best: { level: 'global', predictor: 'CNN', cluster: 'global', mae: 0.9, rmse: 1.9, bias: -0.1, n: 154000 },
              is_best: false, note: 'global/CNN scores better here. Switching to it because it won on the holdout would make the holdout a selection slice.' },
  scores: [
    { role: 'final', label: 'Final evaluation (holdout)', period: 'the holdout slice', estimates_accuracy: true,
      caption: 'The only numbers here that estimate accuracy on data the models never influenced.',
      source: '/m/Results/summary.csv', rows: [{ predictor: 'kmeans', level: 'models', cluster: 'all', mae: 1.4, rmse: 2.1, bias: 0.1, n: 154000 }], missing: null },
    { role: 'selection', label: 'Selection score (selection slice)', period: 'the selection slice', estimates_accuracy: false,
      caption: 'These numbers chose what was kept.', source: '/m/Results/selection.csv',
      rows: [{ cluster: 'global', method: 'CNN', trial: 'test_1', mae: 0.65, n: 142624, kept: true }], missing: null },
  ],
  coverage: { recorded: false, detail: 'no coverage table was written, so missing timestamps cannot be ruled out', slices: [], worst: [] },
  period: { declared: { evaluation_start: '2025-01-11 00:00', selection_fraction: 0.5, calibration_fraction: 0.3 },
            slices_present: ['holdout', 'selection'], rows_per_slice: {}, written_at: 1792000000, protocol: 'two slices in these artifacts: selection and holdout' },
  units: { rated: 100, rated_recorded: null, target: 'wind', metrics: 'percent of rated power', steps: 16, resolution: '15min', horizon_type: 'short_term' },
  freshness: { state: 'current', detail: 'the results are newer than the last change to the case file',
               evidence: 'file times', written_at: 1792000000, edited_at: 1791000000 },
  produced_by: { state: 'unknown', detail: 'the run history is not available here' },
  forecast: { case: 'wind', verified: true, approved_history: true, matches_active: true,
    forecast: { state: 'ok', issue_id: 'i-1', issued_at: 1792000000, age_seconds: 4320, detail: 'issued …', remedy: 'Nothing to do.' },
    attempt: { state: 'ok', detail: 'the last attempt published i-1' },
    covered: { first: '2026-10-27 00:00', last: '2026-10-27 23:00', rows: 24, columns: ['power'] },
    files: { latest: { path: '/p/latest.csv', exists: true }, status: { path: '/p/status.json', exists: false } } },
  incomplete: { unsupported_clusters: [], unfinished_trials: [], missing_artifacts: ['coverage'] },
  caveats: [],
  sources: { summary: { path: '/m/Results/summary.csv', exists: true, modified: 1792000000, size_bytes: 3019 } },
  ...over,
})

function serve(payload: ResultSummary | { status: number; detail: string }) {
  return mockHttp(({ url }) => {
    if (url.pathname === '/api/cases/wind/results') {
      if ('status' in payload) return json({ detail: payload.detail }, payload.status)
      return json(payload)
    }
    return json({})
  })
}

describe('ResultSummary', () => {
  it('says a finished result is experimental until someone approves a version', async () => {
    serve(summary())
    renderPage(<ResultSummaryPanel caseName="wind" />)
    const verdict = await screen.findByTestId('results-verdict')
    expect(verdict).toHaveAttribute('data-state', 'experimental')
    expect(verdict).toHaveTextContent('nobody has approved a version')
    expect(screen.getByTestId('model-state')).toHaveTextContent('experimental')
  })

  it('shows the predictor the service serves, and warns about taking the better row', async () => {
    serve(summary())
    renderPage(<ResultSummaryPanel caseName="wind" />)
    expect(await screen.findByTestId('results-headline')).toHaveTextContent('models / kmeans: MAE 1.40 percent of rated power')
    expect(screen.getByTestId('headline-note')).toHaveTextContent('selection slice')
  })

  it('marks every score with the question it answers', async () => {
    serve(summary())
    renderPage(<ResultSummaryPanel caseName="wind" />)
    const final = await screen.findByTestId('score-final')
    expect(final).toHaveAttribute('data-estimates', 'true')
    expect(screen.getByTestId('score-selection')).toHaveAttribute('data-estimates', 'false')
    expect(screen.getByTestId('score-selection')).toHaveTextContent('not an accuracy estimate')
  })

  it('keeps a failed attempt from making the published forecast look new', async () => {
    serve(summary({
      forecast: { ...summary().forecast,
        attempt: { state: 'rejected', issue_id: 'i-2', detail: 'the last attempt was not published (empty)' } },
      verdict: { state: 'attention', sentence: 'An approved version is published, but the last attempt did not publish.',
                 actions: [{ what: 'read the reason on the forecast line', why: 'the last attempt did not publish; the previous forecast stands' }] },
    }))
    renderPage(<ResultSummaryPanel caseName="wind" />)
    const line = await screen.findByTestId('forecast-line')
    expect(line).toHaveTextContent('Published forecast issued')
    expect(line).toHaveTextContent('this forecast is unchanged')
    expect(line.textContent).not.toMatch(/just now|refreshed/i)
  })

  it('does not present an unwritten coverage table as full coverage', async () => {
    serve(summary())
    renderPage(<ResultSummaryPanel caseName="wind" />)
    expect(await screen.findByTestId('results-gaps')).toHaveTextContent('cannot be ruled out')
  })

  it('shows a known input problem as a problem, not a footnote', async () => {
    serve(summary({ caveats: [{ code: 'dst_ambiguous', detail: 'the series are stored in local time in Europe/Athens',
                                consequence: 'those hours are shifted or dropped on import',
                                remedy: 'an owner decision: store the series in UTC, or accept two hours a year knowingly' }] }))
    renderPage(<ResultSummaryPanel caseName="wind" />)
    expect(await screen.findByTestId('results-caveats')).toHaveTextContent('Europe/Athens')
  })

  it('reports a failed request as a failure, never as an empty result', async () => {
    serve({ status: 503, detail: 'the run store is unavailable' })
    renderPage(<ResultSummaryPanel caseName="wind" />)
    await waitFor(() => expect(screen.getByTestId('results-error')).toHaveTextContent('run store is unavailable'))
    expect(screen.queryByTestId('results-headline')).toBeNull()
  })
})
