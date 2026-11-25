import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useRef, useState } from 'react'
import { describe, expect, it, vi } from 'vitest'
import { BrowseDialog } from './PathField'
import SectionForm from '../forms/SectionForm'
import { deferred, json, mockHttp } from '../test/http'
import type { Document } from '../api/client'

describe('field request lifetimes', () => {
  it('keeps the current folder when a previous browse response arrives late', async () => {
    const first = deferred<Response>()
    const calls = mockHttp(({ url }) => url.searchParams.get('path') === '/first' ? first.promise :
      json({ path: '/second', parent: '/', dirs: ['second-child'], files: [], roots: ['/'] }))
    const view = render(<BrowseDialog open initial="/first" mode="dir" onClose={vi.fn()} onPick={vi.fn()} />)
    await waitFor(() => expect(calls).toHaveLength(1))
    view.rerender(<BrowseDialog open initial="/second" mode="dir" onClose={vi.fn()} onPick={vi.fn()} />)
    await screen.findByText('second-child')
    await act(async () => first.resolve(json({ path: '/first', parent: '/', dirs: ['first-secret-child'], files: [], roots: ['/'] })))
    expect(screen.queryByText('first-secret-child')).not.toBeInTheDocument()
    expect(calls[0].signal?.aborted).toBe(true)
  })

  it('does not replace newer case edits with an old apply-data-types response', async () => {
    const response = deferred<Response>()
    mockHttp(() => response.promise)
    const initial = { project: { name: 'old project' }, data: {}, methods: { base_data_types: {} } }
    function Parent() {
      const [document, setDocument] = useState<Document>(initial)
      const current = useRef(document)
      current.current = document
      return <>
        <button onClick={() => setDocument(previous => ({ ...previous, project: { name: 'new project' } }))}>Edit project</button>
        <SectionForm fullSchema={{ properties: { methods: { type: 'object', properties: {
          base_data_types: { type: 'object', additionalProperties: true },
        } } } }} section="methods" value={document.methods}
          onChange={methods => setDocument(previous => ({ ...previous, methods }))}
          context={{ document, options: null, setDocument, isCurrent: expected => current.current === expected }} />
        <output data-testid="document">{JSON.stringify(document)}</output>
      </>
    }
    render(<Parent />)
    const user = userEvent.setup()
    await user.click(screen.getByRole('button', { name: 'Apply to families, clusterer, combiners' }))
    await user.click(screen.getByRole('button', { name: 'Edit project' }))
    await act(async () => response.resolve(json({ document: initial, note: 'applied' })))
    await screen.findByText(/The draft changed while applying options/)
    expect(screen.getByTestId('document')).toHaveTextContent('new project')
  })
})
