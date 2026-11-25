// §4.1: the installation's health, where a colleague is standing.
//
// Each test is a way a one-word chip could mislead: a stale reading shown as current, a broken
// health check shown as a healthy machine, an unreachable route shown as either verdict, a
// colleague told to fix something that is not theirs, or a maintainer's remedy lost on the way
// to the page.
import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import { json, mockHttp } from '../test/http'
import { renderPage } from '../test/render'
import type { Health, HealthFinding } from '../api/client'
import HealthChip from './HealthChip'
import { ageOf } from './health'

const finding = (over: Partial<HealthFinding> = {}): HealthFinding => ({
  check: 'backup', severity: 'attention', who: 'maintainer',
  checked: 'whether an approved bundle is copied anywhere else',
  finding: 'no backup destination is configured',
  consequence: 'rollback assumes the previous bundle is still on a disk that did not fail',
  remedy: 'Set EFORECAST_BACKUP_DIR to a path on a separate physical disk.', ...over,
})

const reading = (over: Partial<Health> = {}): Health => ({
  verdict: 'needs attention', counts: { ok: 23, attention: 1 },
  findings: over.findings ?? [finding()],
  checked_at: 1793180000, age_s: 4, stale: false, examined: 24, case: null,
  scope: 'maintainer', ...over,
})

function boundary(payload: Health | { status: number; detail: string }, seen?: URL[]) {
  return mockHttp(({ url }) => {
    if (url.pathname === '/api/doctor') {
      seen?.push(url)
      if ('status' in payload) return json({ detail: payload.detail }, payload.status)
      return json(payload)
    }
    return json({})
  })
}

describe('HealthChip', () => {
  it('says the verdict in one word, and how many things need attention', async () => {
    boundary(reading())
    renderPage(<HealthChip />)
    const chip = await screen.findByTestId('health-chip')
    expect(chip).toHaveAttribute('data-verdict', 'needs attention')
    expect(chip).toHaveTextContent('Needs attention (1)')
  })

  it('a ready installation is not decorated with a count', async () => {
    boundary(reading({ verdict: 'ready', counts: { ok: 24 }, findings: [] }))
    renderPage(<HealthChip />)
    expect(await screen.findByTestId('health-chip')).toHaveTextContent('Installation ready')
    expect(screen.getByTestId('health-chip')).not.toHaveTextContent('(0)')
  })

  it('shows what was checked, what follows and the one thing to do', async () => {
    boundary(reading())
    renderPage(<HealthChip />)
    await userEvent.click(await screen.findByTestId('health-chip'))
    const entry = await screen.findByTestId('health-finding-backup')
    expect(entry).toHaveTextContent('no backup destination is configured')
    expect(entry).toHaveTextContent('rollback assumes the previous bundle')
    expect(entry).toHaveTextContent('Set EFORECAST_BACKUP_DIR')
    expect(entry).toHaveTextContent("the maintainer's")
  })

  it('a stale reading is labelled stale rather than presented as current', async () => {
    boundary(reading({ stale: true, age_s: 7200 }))
    renderPage(<HealthChip />)
    const chip = await screen.findByTestId('health-chip')
    expect(chip).toHaveAttribute('data-stale', 'true')
    await userEvent.click(chip)
    expect(await screen.findByTestId('health-stale')).toHaveTextContent('2 h ago')
  })

  it('a health check that broke says so, and still shows the last reading', async () => {
    boundary(reading({ stale: true, unavailable: 'the health check could not run (OSError: disk gone)' }))
    renderPage(<HealthChip />)
    await userEvent.click(await screen.findByTestId('health-chip'))
    const alert = await screen.findByTestId('health-unavailable')
    expect(alert).toHaveTextContent('disk gone')
    expect(alert).toHaveTextContent('last reading taken')
    expect(screen.getByTestId('health-finding-backup')).toBeInTheDocument()
  })

  it('an unreachable route claims neither verdict', async () => {
    boundary({ status: 500, detail: 'nope' })
    renderPage(<HealthChip />)
    const chip = await screen.findByTestId('health-chip')
    expect(chip).toHaveAttribute('data-verdict', 'unknown')
    expect(chip).toHaveTextContent('Health unknown')
  })

  it('a reply that is not a reading claims neither verdict, and leaves the page standing', async () => {
    // What a browser test's catch-all route answered, and what a service of another version
    // may: the chip read `counts` of it and the whole application went down.
    for (const reply of [[], {}, { verdict: 'ready' }, { verdict: 'ready', counts: { ok: 1 } }, 'ok', null]) {
      mockHttp(({ url }) => url.pathname === '/api/doctor' ? json(reply) : json({}))
      const page = renderPage(<HealthChip />)
      const chip = await screen.findByTestId('health-chip')
      expect(chip).toHaveAttribute('data-verdict', 'unknown')
      expect(chip).toHaveTextContent('Health unknown')
      page.unmount()
    }
  })

  it('a colleague sees their own findings and who the rest belong to', async () => {
    boundary(reading({
      scope: 'colleague', maintainer_findings: 9,
      findings: [finding({ check: 'artifacts', who: 'colleague',
                           finding: 'no model root was given, so none was checked',
                           remedy: 'Run the doctor with --case <name>.' })],
    }))
    renderPage(<HealthChip />)
    await userEvent.click(await screen.findByTestId('health-chip'))
    expect(await screen.findByTestId('health-finding-artifacts')).toHaveTextContent('yours to do')
    expect(screen.getByTestId('health-maintainer-count')).toHaveTextContent('9 other findings are the')
    expect(screen.queryByTestId('health-recheck')).toBeNull()
  })

  it('only a maintainer can spend the machine on a fresh reading', async () => {
    const seen: URL[] = []
    boundary(reading(), seen)
    renderPage(<HealthChip />)
    await userEvent.click(await screen.findByTestId('health-chip'))
    await userEvent.click(await screen.findByTestId('health-recheck'))
    await waitFor(() => expect(seen.some((url) => url.searchParams.get('refresh') === 'true')).toBe(true))
  })

  it('nothing needing attention is said, not left blank', async () => {
    boundary(reading({ verdict: 'ready', counts: { ok: 24 }, findings: [] }))
    renderPage(<HealthChip />)
    await userEvent.click(await screen.findByTestId('health-chip'))
    expect(await screen.findByTestId('health-nothing')).toBeInTheDocument()
  })
})

describe('ageOf', () => {
  it('says how old a reading is the way somebody would say it', () => {
    expect(ageOf(null)).toBe('never')
    expect(ageOf(4)).toBe('just now')
    expect(ageOf(600)).toBe('10 min ago')
    expect(ageOf(7200)).toBe('2 h ago')
    expect(ageOf(400_000)).toBe('5 days ago')
  })
})
