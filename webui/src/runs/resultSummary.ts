// A14-02: the words a result is allowed to be described with.
//
// The rules live here rather than in the panel because they are the deliverable: a score that
// chose a model may never be labelled an estimate of its accuracy, and a failed attempt may
// never make the published forecast look newly generated. A pure module can be tested for
// exactly that, and the panel below it only arranges what this returns.
import type { ResultSummary, ScoreBlock, VerdictState } from '../api/client'

export type Tone = 'success' | 'warning' | 'error' | 'info'

export const VERDICT_TONE: Record<VerdictState, Tone> = {
  usable: 'success', attention: 'warning', experimental: 'info', none: 'info',
}
export const VERDICT_LABEL: Record<VerdictState, string> = {
  usable: 'Approved and published',
  attention: 'Needs attention',
  experimental: 'Experimental',
  none: 'Nothing evaluated yet',
}

// The one label that must never drift: what a number is evidence of.
export const roleBadge = (block: ScoreBlock) =>
  block.estimates_accuracy ? 'estimates accuracy' : 'chose the model, not an accuracy estimate'

export function metric(value: number | null | undefined, digits = 2): string {
  return value === null || value === undefined || !Number.isFinite(value) ? '—' : value.toFixed(digits)
}

export function when(epoch: number | null | undefined): string {
  if (!epoch) return 'unknown'
  return new Date(epoch * 1000).toLocaleString()
}

export function age(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return ''
  const hours = seconds / 3600
  if (hours < 1) return `${Math.max(1, Math.round(seconds / 60))} min ago`
  if (hours < 48) return `${hours.toFixed(1)} h ago`
  return `${Math.round(hours / 24)} days ago`
}

/** The forecast, in one line that keeps the two answers apart. */
export function forecastLine(forecast: ResultSummary['forecast']): { text: string; tone: Tone } {
  if (!forecast || forecast.state === 'unknown') {
    return { text: forecast?.detail || 'The publication status could not be read.', tone: 'warning' }
  }
  const published = forecast.forecast
  const attempt = forecast.attempt
  if (!published || published.state === 'none') {
    return {
      text: 'Nothing has ever passed the publication checks for this case, so there is no '
        + 'forecast to use yet.', tone: 'warning',
    }
  }
  if (!forecast.verified || !forecast.approved_history || !forecast.matches_active) {
    return { text: 'A forecast is recorded, but its verified payload and active approval do not match. '
      + 'Keep it as historical output until a qualified forecast is published.', tone: 'warning' }
  }
  const covered = forecast.covered?.first && forecast.covered?.last
    ? ` covering ${forecast.covered.first} to ${forecast.covered.last}` : ''
  const head = published.state === 'stale'
    ? `The published forecast is old: issued ${age(published.age_seconds)}${covered}.`
    : `Published forecast issued ${age(published.age_seconds)}${covered}.`
  if (attempt && attempt.state === 'rejected') {
    // Deliberately a second sentence about a different thing. Merging them is the mistake.
    return {
      text: `${head} The most recent attempt did not publish, so this forecast is unchanged `
        + `— ${attempt.detail}.`,
      tone: 'warning',
    }
  }
  return { text: head, tone: published.state === 'stale' ? 'warning' : 'success' }
}

/** What the colleague actually receives, and the trap in picking the better row. */
export function headlineLine(summary: ResultSummary): { text: string; note?: string } | null {
  const headline = summary.headline
  if (!headline) return null
  const unit = summary.units.metrics
  if (!headline.served) {
    return { text: 'The predictor this case serves is not in the recorded evaluation.',
             note: headline.note }
  }
  const served = headline.served
  return {
    text: `${served.level} / ${served.predictor}: MAE ${metric(served.mae)} ${unit}`
      + `, RMSE ${metric(served.rmse)}, bias ${metric(served.bias)} (holdout)`,
    note: headline.note,
  }
}
