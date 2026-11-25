import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import LogDrawer from './LogDrawer'
import type { Run } from '../api/client'
import { deferred, json, mockHttp } from '../test/http'

const finished: Run = { id: 'run-1', case: 'demo', script: 'training', options: {}, command: [],
  pid: 1, started: 1, finished: 2, status: 'finished', returncode: 0, log: '/logs/demo.log' }

describe('run log pagination', () => {
  it('drains a multi-megabyte terminal log and bounds the rendered tail', async () => {
    const page = 256 * 1024
    const size = page * 12
    let next = 0
    mockHttp(({ url }) => {
      const offset = Number(url.searchParams.get('offset'))
      next = Math.min(size, offset + page)
      const text = offset === size ? '' : next === size ? 'x'.repeat(next - offset - 9) + 'FINAL END' : 'x'.repeat(next - offset)
      return json({ text, offset, next, size, status: 'finished', truncated: next < size })
    })
    render(<LogDrawer caseName="demo" run={finished} onClose={vi.fn()} />)
    await waitFor(() => expect(next).toBe(size))
    expect(await screen.findByText(/FINAL END/)).toBeInTheDocument()
    expect(document.querySelector('pre')!.textContent!.length).toBeLessThanOrEqual(512 * 1024 + 200)
    expect(await screen.findByText(/bytes · complete/)).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Download full log' })).toHaveAttribute('href', '/api/cases/demo/runs/run-1/log/download')
  })

  it('bounds line retention independently of character retention', async () => {
    mockHttp(({ url }) => {
      const offset = Number(url.searchParams.get('offset'))
      const text = 'line\n'.repeat(10000) + 'last marker'
      return json({ text: offset ? '' : text, offset, next: text.length, size: text.length, status: 'finished', truncated: false })
    })
    render(<LogDrawer caseName="demo" run={finished} onClose={vi.fn()} />)
    await screen.findByText(/last marker/)
    expect(document.querySelector('pre')!.textContent!.split('\n').length).toBeLessThanOrEqual(4000)
    expect(screen.getByText(/Earlier output is omitted/)).toBeInTheDocument()
  })

  it('surfaces log errors and resumes from the existing offset on retry', async () => {
    let failed = true
    mockHttp(({ url }) => {
      if (failed) return json({ detail: 'temporarily unavailable' }, 503)
      const offset = Number(url.searchParams.get('offset'))
      return json({ text: offset ? '' : 'reconnected', offset, next: 11, size: 11, status: 'finished', truncated: false })
    })
    render(<LogDrawer caseName="demo" run={finished} onClose={vi.fn()} />)
    await screen.findByText(/Could not read the log/)
    expect(screen.queryByText('(empty)')).not.toBeInTheDocument()
    failed = false
    await userEvent.setup().click(screen.getByRole('button', { name: 'Retry now' }))
    expect(await screen.findByText('reconnected')).toBeInTheDocument()
  })

  it('cancels a delayed old-case page when the same run id is opened in another case', async () => {
    const previous = deferred<Response>()
    const calls = mockHttp(({ url }) => url.pathname.includes('/demo/') ? previous.promise : json({
      text: url.searchParams.get('offset') === '0' ? 'new case log' : '', offset: Number(url.searchParams.get('offset')),
      next: 12, size: 12, status: 'finished', truncated: false,
    }))
    const view = render(<LogDrawer caseName="demo" run={finished} onClose={vi.fn()} />)
    view.rerender(<LogDrawer caseName="other" run={finished} onClose={vi.fn()} />)
    await screen.findByText('new case log')
    await act(async () => previous.resolve(json({ text: 'OLD PRIVATE LOG', offset: 0, next: 15, size: 15, status: 'finished', truncated: false })))
    expect(screen.queryByText('OLD PRIVATE LOG')).not.toBeInTheDocument()
    expect(calls[0].signal?.aborted).toBe(true)
  })

  it('restarts at byte zero after a file generation changes', async () => {
    const offsets: number[] = []
    mockHttp(({ url }) => {
      const offset = Number(url.searchParams.get('offset')); offsets.push(offset)
      if (offsets.length === 1) return json({ text: 'OLD', offset: 0, next: 3, size: 6, status: 'running', truncated: true, generation: 'old' })
      if (offsets.length === 2) return json({ text: 'bad', offset: 3, next: 6, size: 6, status: 'finished', truncated: false, generation: 'new' })
      return json({ text: offset ? '' : 'new file', offset, next: 8, size: 8, status: 'finished', truncated: false, generation: 'new' })
    })
    render(<LogDrawer caseName="demo" run={finished} onClose={vi.fn()} />)
    expect(await screen.findByText('new file')).toBeInTheDocument()
    expect(offsets.slice(0, 3)).toEqual([0, 3, 0])
    expect(document.querySelector('pre')).not.toHaveTextContent('OLD')
    expect(screen.getByText(/The log was replaced or truncated/)).toBeInTheDocument()
  })

  it('handles truncation that keeps the same inode', async () => {
    let count = 0
    mockHttp(() => ++count === 1 ? json({ text: 'OLD', offset: 0, next: 3, size: 6, status: 'running', truncated: true }) :
      json({ text: count === 2 ? 'OK' : '', offset: count === 2 ? 0 : 2, next: 2, size: 2, status: 'finished', truncated: false }))
    render(<LogDrawer caseName="demo" run={finished} onClose={vi.fn()} />)
    expect(await screen.findByText('OK')).toBeInTheDocument()
    expect(document.querySelector('pre')).not.toHaveTextContent('OLD')
  })

  it('stops rather than spinning on a non-progressing page', async () => {
    mockHttp(() => json({ text: '', offset: 0, next: 0, size: 10, status: 'finished', truncated: true }))
    render(<LogDrawer caseName="demo" run={finished} onClose={vi.fn()} />)
    expect(await screen.findByText(/without valid forward progress/)).toBeInTheDocument()
  })
})

describe('the support report a colleague sends (A14-03)', () => {
  const serve = (report: string | { status: number; body: string }) => mockHttp(({ url }) => {
    if (url.pathname.endsWith('/support')) {
      if (typeof report === 'string') return new Response(report, { status: 200, headers: { 'Content-Type': 'text/plain' } })
      return new Response(report.body, { status: report.status, headers: { 'Content-Type': 'text/plain' } })
    }
    return json({ text: 'log line\n', offset: 0, next: 9, size: 9, status: 'failed', truncated: false })
  })

  it('copies the report and says what was copied', async () => {
    serve('eforecast support report (version 1)\ncase     demo\n')
    const writeText = vi.fn().mockResolvedValue(undefined)
    Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true })
    render(<LogDrawer caseName="demo" run={finished} onClose={vi.fn()} />)
    await userEvent.click(await screen.findByTestId('copy-support-report'))
    await waitFor(() => expect(writeText).toHaveBeenCalledWith(expect.stringContaining('support report')))
    expect(await screen.findByTestId('support-notice')).toHaveTextContent(/copied \(\d+ characters\)/)
  })

  it('offers the download when the browser will not let it copy', async () => {
    serve('eforecast support report (version 1)\n')
    Object.defineProperty(navigator, 'clipboard', { value: undefined, configurable: true })
    render(<LogDrawer caseName="demo" run={finished} onClose={vi.fn()} />)
    await userEvent.click(await screen.findByTestId('copy-support-report'))
    expect(await screen.findByTestId('support-notice')).toHaveTextContent('Use Download support report')
    expect(screen.getByTestId('download-support-report'))
      .toHaveAttribute('href', '/api/cases/demo/runs/run-1/support?text=true')
  })

  it('reports a server failure as a failure, never as an empty report', async () => {
    serve({ status: 503, body: '{"detail":"the run store is unavailable"}' })
    Object.defineProperty(navigator, 'clipboard', { value: { writeText: vi.fn() }, configurable: true })
    render(<LogDrawer caseName="demo" run={finished} onClose={vi.fn()} />)
    await userEvent.click(await screen.findByTestId('copy-support-report'))
    expect(await screen.findByTestId('support-notice')).toHaveTextContent('run store is unavailable')
  })
})
