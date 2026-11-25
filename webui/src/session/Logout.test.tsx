import { act, fireEvent, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import App from '../App'
import { caseDocument, caseSummary, formSchema, snapshot } from '../test/fixtures'
import { deferred, json, mockHttp, type HttpCall } from '../test/http'
import { renderPage } from '../test/render'
import { api, type Document } from '../api/client'

function serveEditor(override: (call: HttpCall) => Response | Promise<Response> | undefined = () => undefined,
  writable = true) {
  let identity: string | null = 'alice'
  let saved = snapshot()
  const calls = mockHttp(call => {
    const response = override(call)
    if (response) return response
    const path = call.url.pathname
    if (path === '/api/me') return json({ user: identity, auth: true, role: identity === 'alice' ? 'admin' : identity ? 'user' : null })
    if (path === '/api/logout') { identity = null; return json({ user: null }) }
    if (path === '/api/login') { identity = String(call.body.name); return json({ user: identity }) }
    if (path === '/api/cases') return json(identity === 'alice' ? [{ ...caseSummary(), writable }] : [])
    if (path === '/api/schema') return json(formSchema)
    if (path === '/api/options') return json({})
    if (path === '/api/cases/demo') {
      if (identity !== 'alice') return json({ detail: 'case not visible' }, 403)
      if (call.method === 'PUT') { saved = snapshot(call.body.document as Document, 'v2'); return json({ ...saved, main: 'main.yaml', parts: [], note: '' }) }
      return json(saved)
    }
    throw new Error(`Unexpected ${call.method} ${path}`)
  })
  return { calls, saved: () => saved, identity: (value: string | null) => { identity = value } }
}

describe('deliberate logout with a case draft', () => {
  it('asks before sending logout and Cancel retains the mounted private draft', async () => {
    const server = serveEditor()
    renderPage(<App />, { route: '*' })
    const user = userEvent.setup()
    const input = await screen.findByRole('textbox', { name: 'Project name' })
    fireEvent.change(input, { target: { value: 'keep this draft' } })
    await user.click(screen.getByRole('button', { name: 'Log out' }))
    expect(await screen.findByRole('dialog', { name: 'Unsaved changes before logging out' })).toBeInTheDocument()
    expect(server.calls.every(call => call.method === 'GET')).toBe(true)
    await user.click(screen.getByRole('button', { name: 'Cancel' }))
    expect(await screen.findByRole('textbox', { name: 'Project name' })).toBe(input)
    expect(input).toHaveValue('keep this draft')
    expect(server.calls.every(call => call.method === 'GET')).toBe(true)
  })

  it('logs out a clean editor without an extra prompt or save', async () => {
    const server = serveEditor()
    renderPage(<App />, { route: '*' })
    await screen.findByRole('textbox', { name: 'Project name' })
    await userEvent.setup().click(screen.getByRole('button', { name: 'Log out' }))
    expect(await screen.findByRole('button', { name: 'Sign in' })).toBeInTheDocument()
    expect(server.calls.filter(call => call.method !== 'GET').map(call => call.url.pathname)).toEqual(['/api/logout'])
  })

  it('discards unfinished-only YAML explicitly and keeps it out of the next account', async () => {
    const server = serveEditor()
    renderPage(<App />, { route: '*' })
    const user = userEvent.setup()
    await screen.findByRole('textbox', { name: 'Project name' })
    fireEvent.change(screen.getByRole('textbox', { name: 'YAML editor' }), { target: { value: 'alice private YAML: [' } })
    await user.click(screen.getByRole('button', { name: 'Log out' }))
    expect(await screen.findByRole('button', { name: 'Save and log out' })).toBeDisabled()
    expect(server.calls.every(call => call.method === 'GET')).toBe(true)
    await user.keyboard('{Escape}')
    expect(await screen.findByRole('textbox', { name: 'YAML editor' })).toHaveValue('alice private YAML: [')
    await user.click(screen.getByRole('button', { name: 'Log out' }))
    await user.click(await screen.findByRole('button', { name: 'Discard and log out' }))
    await user.type(await screen.findByRole('textbox', { name: 'User' }), 'bob')
    await user.type(screen.getByLabelText(/^Password/), 'password')
    await user.click(screen.getByRole('button', { name: 'Sign in' }))
    await screen.findByRole('button', { name: 'New case' })
    expect(screen.queryByDisplayValue('alice private YAML: [')).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'demo' })).not.toBeInTheDocument()
    expect(server.calls.filter(call => call.method === 'PUT')).toHaveLength(0)
    expect(server.calls.filter(call => call.url.pathname === '/api/logout')).toHaveLength(1)
  })

  it('saves the checked revision before logout and waits for confirmation', async () => {
    const saving = deferred<Response>()
    const server = serveEditor(call => call.method === 'PUT' ? saving.promise : undefined)
    renderPage(<App />, { route: '*' })
    const user = userEvent.setup()
    fireEvent.change(await screen.findByRole('textbox', { name: 'Project name' }), { target: { value: 'saved before logout' } })
    await user.click(screen.getByRole('button', { name: 'Log out' }))
    await user.click(await screen.findByRole('button', { name: 'Save and log out' }))
    await waitFor(() => expect(server.calls.some(call => call.method === 'PUT')).toBe(true))
    expect(server.calls.some(call => call.url.pathname === '/api/logout')).toBe(false)
    const submitted = server.calls.find(call => call.method === 'PUT')!
    expect(submitted.body.expected_revision).toBe('v1')
    expect((submitted.body.document as Document).extension).toEqual(caseDocument.extension)
    await act(async () => saving.resolve(json({ ...snapshot(submitted.body.document as Document, 'v2'), main: 'main.yaml', parts: [], note: '' })))
    await screen.findByRole('button', { name: 'Sign in' })
    expect(server.calls.filter(call => call.method !== 'GET').map(call => call.method)).toEqual(['PUT', 'POST'])
    expect(server.calls.filter(call => call.url.pathname === '/api/logout')).toHaveLength(1)
  })

  it.each([409, 422, 503, 'network'] as const)('keeps the draft and session when saving for logout fails (%s)', async failure => {
    const server = serveEditor(call => call.method === 'PUT' ? failure === 'network' ?
      Promise.reject(new TypeError('connection lost')) : json({ detail: 'save rejected',
        ...(failure === 409 ? snapshot({ ...caseDocument, project: { name: 'remote' } }, 'v2') : {}) }, failure) : undefined)
    renderPage(<App />, { route: '*' })
    const user = userEvent.setup()
    const input = await screen.findByRole('textbox', { name: 'Project name' })
    fireEvent.change(input, { target: { value: 'keep rejected draft' } })
    await user.click(screen.getByRole('button', { name: 'Log out' }))
    await user.click(await screen.findByRole('button', { name: 'Save and log out' }))
    expect(await screen.findByText(/Save did not complete. You are still signed in/)).toBeInTheDocument()
    expect(server.calls.some(call => call.url.pathname === '/api/logout')).toBe(false)
    await user.click(screen.getByRole('button', { name: 'Cancel' }))
    expect(await screen.findByRole('textbox', { name: 'Project name' })).toBe(input)
    expect(input).toHaveValue('keep rejected draft')
    expect(screen.queryByRole('button', { name: 'Sign in' })).not.toBeInTheDocument()
  })

  it.each(['parsed', 'raw'])('retains newer %s edits instead of logging out after a delayed save', async kind => {
    const saving = deferred<Response>()
    const server = serveEditor(call => call.method === 'PUT' ? saving.promise : undefined)
    renderPage(<App />, { route: '*' })
    const user = userEvent.setup()
    const input = await screen.findByRole('textbox', { name: 'Project name' })
    const yaml = screen.getByRole('textbox', { name: 'YAML editor' })
    fireEvent.change(input, { target: { value: 'submitted' } })
    await user.click(screen.getByRole('button', { name: 'Log out' }))
    await user.click(await screen.findByRole('button', { name: 'Save and log out' }))
    await waitFor(() => expect(server.calls.some(call => call.method === 'PUT')).toBe(true))
    // Models/presets can update the background draft while the modal is open.
    fireEvent.change(kind === 'parsed' ? input : yaml, { target: { value: kind === 'parsed' ? 'newer edit' : 'newer YAML: [' } })
    const submitted = server.calls.find(call => call.method === 'PUT')!.body.document as Document
    await act(async () => saving.resolve(json({ ...snapshot(submitted, 'v2'), main: 'main.yaml', parts: [], note: '' })))
    expect(await screen.findByText(/New edits arrived while saving/)).toBeInTheDocument()
    expect(server.calls.some(call => call.url.pathname === '/api/logout')).toBe(false)
    await user.click(screen.getByRole('button', { name: 'Cancel' }))
    expect(await screen.findByRole('textbox', { name: kind === 'parsed' ? 'Project name' : 'YAML editor' })).toHaveValue(kind === 'parsed' ? 'newer edit' : 'newer YAML: [')
  })

  it('cancels logout while its save is pending without undoing the already-submitted save', async () => {
    const saving = deferred<Response>()
    const server = serveEditor(call => call.method === 'PUT' ? saving.promise : undefined)
    renderPage(<App />, { route: '*' })
    const user = userEvent.setup()
    fireEvent.change(await screen.findByRole('textbox', { name: 'Project name' }), { target: { value: 'allowed save' } })
    await user.click(screen.getByRole('button', { name: 'Log out' }))
    await user.click(await screen.findByRole('button', { name: 'Save and log out' }))
    await waitFor(() => expect(server.calls.some(call => call.method === 'PUT')).toBe(true))
    await user.click(screen.getByRole('button', { name: 'Cancel' }))
    const submitted = server.calls.find(call => call.method === 'PUT')!.body.document as Document
    await act(async () => saving.resolve(json({ ...snapshot(submitted, 'v2'), main: 'main.yaml', parts: [], note: '' })))
    expect(await screen.findByText('Case saved.', { exact: true })).toBeInTheDocument()
    expect(server.calls.some(call => call.url.pathname === '/api/logout')).toBe(false)
    expect(await screen.findByRole('textbox', { name: 'Project name' })).toHaveValue('allowed save')
  })

  it('retains the draft when the logout request fails after explicit discard', async () => {
    const server = serveEditor(call => call.url.pathname === '/api/logout' ? json({ detail: 'logout unavailable' }, 503) : undefined)
    renderPage(<App />, { route: '*' })
    const user = userEvent.setup()
    const input = await screen.findByRole('textbox', { name: 'Project name' })
    fireEvent.change(input, { target: { value: 'retained after logout failure' } })
    await user.click(screen.getByRole('button', { name: 'Log out' }))
    await user.click(await screen.findByRole('button', { name: 'Discard and log out' }))
    expect(await screen.findByText(/Log out failed.*logout unavailable.*Your draft is retained/)).toBeInTheDocument()
    expect(await screen.findByRole('textbox', { name: 'Project name' })).toBe(input)
    expect(input).toHaveValue('retained after logout failure')
    await user.click(screen.getByRole('button', { name: 'Log out' }))
    expect(await screen.findByRole('dialog', { name: 'Unsaved changes before logging out' })).toBeInTheDocument()
    expect(server.calls.filter(call => call.url.pathname === '/api/logout')).toHaveLength(1)
  })

  it('blocks saving read-only drafts but offers explicit discard and cancel', async () => {
    const server = serveEditor(undefined, false)
    renderPage(<App />, { route: '*' })
    const user = userEvent.setup()
    fireEvent.change(await screen.findByRole('textbox', { name: 'Project name' }), { target: { value: 'local read-only draft' } })
    await user.click(screen.getByRole('button', { name: 'Log out' }))
    expect(await screen.findByRole('button', { name: 'Save and log out' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Discard and log out' })).toBeEnabled()
    expect(screen.getByRole('button', { name: 'Cancel' })).toBeEnabled()
    expect(server.calls.every(call => call.method === 'GET')).toBe(true)
  })

  it('pauses Save and log out while a session refresh is unconfirmed', async () => {
    let failed = false
    const server = serveEditor(call => call.url.pathname === '/api/me' && failed ? json({ detail: 'outage' }, 503) : undefined)
    const { client } = renderPage(<App />, { route: '*' })
    const user = userEvent.setup()
    fireEvent.change(await screen.findByRole('textbox', { name: 'Project name' }), { target: { value: 'retained draft' } })
    failed = true
    await act(async () => { await client.refetchQueries({ queryKey: ['me'] }) })
    await screen.findByText(/Your session could not be rechecked/)
    await user.click(screen.getByRole('button', { name: 'Log out' }))
    expect(await screen.findByRole('button', { name: 'Save and log out' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Discard and log out' })).toBeEnabled()
    expect(server.calls.every(call => call.method === 'GET')).toBe(true)
  })

  it('does not log a replacement account out when an older save completes', async () => {
    const saving = deferred<Response>()
    const server = serveEditor(call => call.method === 'PUT' ? saving.promise : undefined)
    const { client } = renderPage(<App />, { route: '*' })
    const user = userEvent.setup()
    fireEvent.change(await screen.findByRole('textbox', { name: 'Project name' }), { target: { value: 'alice private draft' } })
    await user.click(screen.getByRole('button', { name: 'Log out' }))
    await user.click(await screen.findByRole('button', { name: 'Save and log out' }))
    await waitFor(() => expect(server.calls.some(call => call.method === 'PUT')).toBe(true))
    server.identity('bob')
    await act(async () => { await client.refetchQueries({ queryKey: ['me'] }) })
    await screen.findByText('bob', { exact: true })
    const submitted = server.calls.find(call => call.method === 'PUT')!.body.document as Document
    await act(async () => saving.resolve(json({ ...snapshot(submitted, 'v2'), main: 'main.yaml', parts: [], note: '' })))
    expect(screen.queryByDisplayValue('alice private draft')).not.toBeInTheDocument()
    expect(server.calls.some(call => call.url.pathname === '/api/logout')).toBe(false)
    expect(screen.queryByText(/Log out failed/)).not.toBeInTheDocument()
  })

  it('prevents editing after logout is submitted and restores the draft if the request fails', async () => {
    const response = deferred<Response>()
    const server = serveEditor(call => call.url.pathname === '/api/logout' ? response.promise : undefined)
    renderPage(<App />, { route: '*' })
    const user = userEvent.setup()
    const input = await screen.findByRole('textbox', { name: 'Project name' })
    fireEvent.change(input, { target: { value: 'keep if logout fails' } })
    await user.click(screen.getByRole('button', { name: 'Log out' }))
    await user.click(await screen.findByRole('button', { name: 'Discard and log out' }))
    await screen.findByRole('dialog', { name: 'Logging out' })
    expect(screen.queryByRole('textbox', { name: 'Project name' })).not.toBeInTheDocument()
    await user.keyboard('later typing')
    expect(input).toHaveValue('keep if logout fails')
    await act(async () => response.resolve(json({ detail: 'logout rejected' }, 503)))
    expect(await screen.findByRole('textbox', { name: 'Project name' })).toBe(input)
    expect(input).toHaveValue('keep if logout fails')
    expect(server.calls.filter(call => call.url.pathname === '/api/logout')).toHaveLength(1)
  })

  it('waits for an existing save before offering a disposition and avoids a duplicate write', async () => {
    const saving = deferred<Response>()
    const server = serveEditor(call => call.method === 'PUT' ? saving.promise : undefined)
    renderPage(<App />, { route: '*' })
    const user = userEvent.setup()
    fireEvent.change(await screen.findByRole('textbox', { name: 'Project name' }), { target: { value: 'ordinary save' } })
    await user.click(screen.getByRole('button', { name: 'Save' }))
    await waitFor(() => expect(server.calls.some(call => call.method === 'PUT')).toBe(true))
    await user.click(screen.getByRole('button', { name: 'Log out' }))
    expect(await screen.findByRole('button', { name: 'Save and log out' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Discard and log out' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Cancel' })).toBeEnabled()
    const submitted = server.calls.find(call => call.method === 'PUT')!.body.document as Document
    await act(async () => saving.resolve(json({ ...snapshot(submitted, 'v2'), main: 'main.yaml', parts: [], note: '' })))
    await user.click(await screen.findByRole('button', { name: 'Log out' }))
    await screen.findByRole('button', { name: 'Sign in' })
    expect(server.calls.filter(call => call.method === 'PUT')).toHaveLength(1)
    expect(server.calls.filter(call => call.url.pathname === '/api/logout')).toHaveLength(1)
  })

  it.each(['expiry', 'unauthorized response'])('does not let a logout prompt prevent a confirmed %s', async reason => {
    let unauthorized = false
    const server = serveEditor(call => call.url.pathname === '/api/cases' && unauthorized ? json({ detail: 'expired' }, 401) : undefined)
    const { client } = renderPage(<App />, { route: '*' })
    const user = userEvent.setup()
    fireEvent.change(await screen.findByRole('textbox', { name: 'Project name' }), { target: { value: 'expired account draft' } })
    await user.click(screen.getByRole('button', { name: 'Log out' }))
    await screen.findByRole('dialog', { name: 'Unsaved changes before logging out' })
    if (reason === 'expiry') {
      server.identity(null)
      await act(async () => { await client.refetchQueries({ queryKey: ['me'] }) })
    } else {
      unauthorized = true
      await act(async () => { await api.cases().catch(() => {}) })
    }
    expect(await screen.findByRole('button', { name: 'Sign in' })).toBeInTheDocument()
    expect(screen.queryByDisplayValue('expired account draft')).not.toBeInTheDocument()
    expect(screen.queryByRole('dialog', { name: 'Unsaved changes before logging out' })).not.toBeInTheDocument()
    expect(server.calls.some(call => call.url.pathname === '/api/logout')).toBe(false)
  })
})
