// A14-02: the two sentences that must not blur into one.
import { describe, expect, it } from 'vitest'
import type { PublicationAnswer, ResultSummary, ScoreBlock } from '../api/client'
import { age, forecastLine, headlineLine, metric, roleBadge } from './resultSummary'

type Forecast = ResultSummary['forecast']

const published: PublicationAnswer = {
  state: 'ok', issue_id: 'i-1', issued_at: 1000, age_seconds: 4320,
  detail: 'issued 2026-10-27 06:00 UTC (1.2 h ago)', remedy: 'Nothing to do.',
}

const forecast = (over: Partial<Forecast> = {}): Forecast => ({
  case: 'wind', verified: true, approved_history: true, matches_active: true,
  forecast: published,
  attempt: { state: 'ok', detail: 'the last attempt published i-1' },
  covered: { first: '2026-10-27 00:00', last: '2026-10-27 23:00', rows: 24, columns: ['power'] },
  files: { latest: { path: '/p/latest.csv', exists: true },
           status: { path: '/p/status.json', exists: false } },
  ...over,
})

const block = (over: Partial<ScoreBlock>): ScoreBlock => ({
  role: 'final', label: 'Final evaluation (holdout)', period: 'the holdout slice',
  estimates_accuracy: true, caption: '', source: null, rows: [], missing: null, ...over,
})

const summary = (headline: ResultSummary['headline']): ResultSummary => ({
  units: { rated: 100, rated_recorded: null, target: 'wind', metrics: 'percent of rated power',
           steps: 16, resolution: '15min', horizon_type: 'short_term' },
  headline,
} as ResultSummary)

describe('forecastLine', () => {
  it('reports a published forecast with what it covers', () => {
    const line = forecastLine(forecast())
    expect(line.tone).toBe('success')
    expect(line.text).toContain('2026-10-27 00:00 to 2026-10-27 23:00')
  })

  it('keeps a failed attempt and the published forecast in separate sentences', () => {
    const line = forecastLine(forecast({
      attempt: { state: 'rejected', issue_id: 'i-2',
                 detail: 'the last attempt was not published (empty)' },
    }))
    expect(line.tone).toBe('warning')
    expect(line.text).toContain('this forecast is unchanged')
    expect(line.text).toContain('(empty)')
    expect(line.text).not.toMatch(/just now|updated|refreshed/i)
  })

  it('says plainly when nothing was ever published', () => {
    const line = forecastLine(forecast({ forecast: { state: 'none', detail: 'none' } }))
    expect(line.text).toContain('no forecast to use yet')
    expect(line.tone).toBe('warning')
  })

  it('does not guess when the store could not be read', () => {
    const line = forecastLine({ state: 'unknown', detail: 'the store is not reachable',
                                files: forecast().files })
    expect(line.text).toBe('the store is not reachable')
    expect(line.tone).toBe('warning')
  })

  it('calls an old forecast old', () => {
    const line = forecastLine(forecast({
      forecast: { ...published, state: 'stale', age_seconds: 200000 },
    }))
    expect(line.text).toContain('old')
    expect(line.tone).toBe('warning')
  })
})

describe('labels', () => {
  it('never calls a selection score an accuracy estimate', () => {
    expect(roleBadge(block({ role: 'selection', estimates_accuracy: false })))
      .toContain('not an accuracy estimate')
    expect(roleBadge(block({ estimates_accuracy: true }))).toBe('estimates accuracy')
  })

  it('shows a missing number as a dash rather than a zero', () => {
    expect(metric(null)).toBe('—')
    expect(metric(undefined)).toBe('—')
    expect(metric(1.4)).toBe('1.40')
  })

  it('reads ages the way a person says them', () => {
    expect(age(300)).toBe('5 min ago')
    expect(age(5400)).toBe('1.5 h ago')
    expect(age(3 * 86400)).toBe('3 days ago')
  })
})

describe('headlineLine', () => {
  const served = { level: 'models', predictor: 'kmeans', cluster: 'all',
                   mae: 1.4, rmse: 2.1, bias: 0.1, n: 154000 }

  it('reports the predictor the service serves, in its units', () => {
    const line = headlineLine(summary({ served, best: served, is_best: true }))
    expect(line!.text).toContain('models / kmeans: MAE 1.40 percent of rated power')
    expect(line!.note).toBeUndefined()
  })

  it('passes on the warning when a better row exists', () => {
    const line = headlineLine(summary({
      served, is_best: false,
      best: { level: 'global', predictor: 'CNN', cluster: 'global', mae: 0.9, rmse: 1.9, bias: -0.1, n: 154000 },
      note: 'global/CNN scores better here, over the same period. Switching to it because it '
        + 'won on the holdout would make the holdout a selection slice.',
    }))
    expect(line!.note).toContain('selection slice')
  })

  it('returns nothing when there is nothing recorded', () => {
    expect(headlineLine(summary(null))).toBeNull()
  })
})


describe('verified publication labels', () => {
  it.each([
    { verified: false }, { approved_history: false }, { matches_active: false },
  ])('keeps an unverified or mismatched publication from looking usable: %j', (over) => {
    const line = forecastLine(forecast(over))
    expect(line.tone).toBe('warning')
    expect(line.text).toContain('do not match')
  })
  it('does not format infinity as a score', () => {
    expect(metric(Infinity)).toBe('—')
  })
})
