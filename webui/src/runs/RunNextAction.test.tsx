import { screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import RunsPanel from './RunsPanel'
import { caseDocument, runFixture, scripts } from '../test/fixtures'
import { json, mockHttp } from '../test/http'
import { renderPage } from '../test/render'
import type { Run } from '../api/client'

const show = (run: Run) => {
  mockHttp(({ url }) => {
    if (url.pathname === '/api/scripts') return json(scripts)
    if (url.pathname.endsWith('/log')) return json({ text: '', offset: 0, next: 0, size: 0, status: 'running' })
    if (url.pathname.endsWith('/runs')) return json([run])
    return json([])
  })
  renderPage(<RunsPanel caseName="demo" document={caseDocument} dirty={false} />)
}

describe('a run says what to do about it', () => {
  it('offers one action for a failure, naming what to correct', async () => {
    show({ ...runFixture('failed'), id: 'r-7', cleanup: 'not_required',
           preparation_error: 'the target column Load is not in the file' })
    const action = await screen.findByTestId('next-action-r-7')
    expect(action).toHaveTextContent(/Correct the preparation/)
    expect(await screen.findByText(/No result was produced/)).toBeInTheDocument()
  })

  it('refuses to suggest a retry while cleanup is still running', async () => {
    show({ ...runFixture('failed'), id: 'r-8', cleanup: 'running' })
    const action = await screen.findByTestId('next-action-r-8')
    expect(action).toHaveTextContent(/Wait for cleanup/)
    expect(action).not.toHaveTextContent(/Correct|support report/)
  })

  it('says nothing for a run that is simply working', async () => {
    show({ ...runFixture('running'), id: 'r-9', stages: [] })
    await screen.findByText('training')            // the row is on screen
    expect(screen.queryByTestId('next-action-r-9')).toBeNull()
  })

  it('does not repeat the outcome detail the row already shows', async () => {
    const detail = 'the worker exited with code 9'
    show({ ...runFixture('failed'), id: 'r-10', cleanup: 'complete',
           outcome: { kind: 'failure', detail, observed_at: 1, returncode: 9 } })
    await screen.findByTestId('next-action-r-10')
    expect(screen.getAllByText(detail)).toHaveLength(1)
  })

  it('quotes the run id when there is nothing to correct', async () => {
    show({ ...runFixture('failed'), id: 'r-11', cleanup: 'complete' })
    expect(await screen.findByTestId('next-action-r-11')).toHaveTextContent(/r-11/)
  })
})
