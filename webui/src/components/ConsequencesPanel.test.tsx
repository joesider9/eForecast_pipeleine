// Review & save: what the draft would cost, asked before saving -- and for one draft only.
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import ConsequencesPanel from './ConsequencesPanel'
import { caseDocument, consequencesFixture } from '../test/fixtures'
import { deferred, json, mockHttp } from '../test/http'

describe('what saving the draft would cost', () => {
  it('asks of the saved case whether what is built is its own, and says what is still waiting', async () => {
    const waiting = consequencesFixture({ status: 'unchanged', changes: [], datasets: [], families: [], results: [],
      extract: null, consequential: false, digest: null,
      pending: { datasets: [{ stage: 'row', files: 1, because: [] }],
                 families: [{ method: 'TRANSFORMER', studies: 8, trials: 101, because: [] }] } })
    const calls = mockHttp(() => json(waiting))
    render(<ConsequencesPanel caseName="demo" document={caseDocument} dirty={false} />)
    expect(screen.getByTestId('consequences-check')).toHaveTextContent('Is what is built this configuration\'s?')
    expect(calls).toHaveLength(0)
    await userEvent.setup().click(screen.getByTestId('consequences-check'))
    const shown = await screen.findByTestId('consequences-waiting')
    expect(shown).toHaveTextContent('What is built is not this configuration\'s yet')
    expect(shown).toHaveTextContent('row: 1 file(s)')
    expect(shown).toHaveTextContent('TRANSFORMER: 8 studies, 101 trial folder(s)')
  })

  it('says so when everything that is built was built from the saved configuration', async () => {
    mockHttp(() => json(consequencesFixture({ status: 'unchanged', changes: [], datasets: [], families: [], results: [],
      extract: null, consequential: false, digest: null })))
    render(<ConsequencesPanel caseName="demo" document={caseDocument} dirty={false} />)
    await userEvent.setup().click(screen.getByTestId('consequences-check'))
    expect(await screen.findByTestId('consequences-whole')).toHaveTextContent(
      'Everything that is built was built from this configuration.')
  })

  it('asks the server for this draft and shows its answer', async () => {
    const calls = mockHttp(() => json(consequencesFixture()))
    render(<ConsequencesPanel caseName="demo" document={caseDocument} dirty />)
    await userEvent.setup().click(screen.getByTestId('consequences-check'))
    expect(await screen.findByTestId('consequences-families')).toHaveTextContent('TRANSFORMER')
    expect(calls).toHaveLength(1)
    expect(calls[0].url.pathname).toBe('/api/cases/demo/consequences')
    expect(calls[0].body.document).toEqual(caseDocument)
    expect(screen.getByTestId('consequences-check')).toHaveTextContent('Ask again')
  })

  it('says when the draft has moved on since the answer, instead of leaving it standing as current', async () => {
    mockHttp(() => json(consequencesFixture()))
    const { rerender } = render(<ConsequencesPanel caseName="demo" document={caseDocument} dirty />)
    await userEvent.setup().click(screen.getByTestId('consequences-check'))
    await screen.findByTestId('consequences-families')
    expect(screen.queryByTestId('consequences-outdated')).not.toBeInTheDocument()
    rerender(<ConsequencesPanel caseName="demo" document={{ ...caseDocument, jobs: { n_cpus: 9 } }} dirty />)
    expect(screen.getByTestId('consequences-outdated')).toHaveTextContent('The draft has changed since this was worked out')
  })

  it('says that it could not be worked out, in the server\'s words', async () => {
    mockHttp(() => json({ detail: 'cannot work out the consequences: the model folder is unreadable' }, 400))
    render(<ConsequencesPanel caseName="demo" document={caseDocument} dirty />)
    await userEvent.setup().click(screen.getByTestId('consequences-check'))
    expect(await screen.findByText(/the model folder is unreadable/)).toBeInTheDocument()
    expect(screen.queryByTestId('change-consequences')).not.toBeInTheDocument()
  })

  it('asks one question at a time', async () => {
    const answer = deferred<Response>()
    const calls = mockHttp(() => answer.promise)
    render(<ConsequencesPanel caseName="demo" document={caseDocument} dirty />)
    await userEvent.setup().click(screen.getByTestId('consequences-check'))
    expect(screen.getByTestId('consequences-check')).toBeDisabled()
    answer.resolve(json(consequencesFixture()))
    expect(await screen.findByTestId('consequences-families')).toBeInTheDocument()
    expect(screen.getByTestId('consequences-check')).toBeEnabled()
    expect(calls).toHaveLength(1)
  })
})
