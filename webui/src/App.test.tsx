import { act, fireEvent, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import App from './App'
import { caseSummary, formSchema, runFixture, scripts, snapshot } from './test/fixtures'
import { deferred, json, mockHttp } from './test/http'
import { renderPage } from './test/render'

describe('authentication transitions', () => {
  it('logs out, switches accounts and fetches the authoritative role before showing private cases', async () => {
    let user: string | null = 'alice'
    const calls = mockHttp(({ url, body }) => {
      if (url.pathname === '/api/me') return json({ user, auth: true, role: user === 'alice' ? 'admin' : user ? 'user' : null })
      if (url.pathname === '/api/logout') { user = null; return json({ user }) }
      if (url.pathname === '/api/login') { user = String(body.name); return json({ user }) }
      if (url.pathname === '/api/cases') return json([caseSummary('demo', `${user} private project`)])
      throw new Error(`Unexpected ${url}`)
    })
    renderPage(<App />, { path: '/', route: '*' })
    const interactions = userEvent.setup()
    await screen.findByText('alice private project')
    expect(screen.getByRole('button', { name: 'Users' })).toBeInTheDocument()
    await interactions.click(screen.getByRole('button', { name: 'Log out' }))
    await interactions.type(await screen.findByRole('textbox', { name: 'User' }), 'bob')
    await interactions.type(screen.getByLabelText(/^Password/), 'password')
    await interactions.click(screen.getByRole('button', { name: 'Sign in' }))
    await screen.findByText('bob private project')
    expect(screen.queryByText('alice private project')).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Users' })).not.toBeInTheDocument()
    expect(calls.filter(call => call.url.pathname === '/api/me')).toHaveLength(2)
  })

  it('keeps the trusted local auth-off path usable', async () => {
    mockHttp(({ url }) => url.pathname === '/api/me' ? json({ user: 'local', auth: false, role: 'admin' }) : json([]))
    renderPage(<App />, { path: '/', route: '*' })
    expect(await screen.findByRole('button', { name: 'New case' })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Log out' })).not.toBeInTheDocument()
  })

  it('shows a failed session check instead of pretending the user is logged out', async () => {
    mockHttp(() => json({ detail: 'session service unavailable' }, 503))
    renderPage(<App />, { path: '/', route: '*' })
    expect(await screen.findByText(/Could not check the current session/)).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Sign in' })).not.toBeInTheDocument()
  })

  it.each([false, true])('preserves a mounted editor through a failed session check and retry (dirty=%s)', async dirty => {
    const recovery = deferred<Response>()
    let sessionChecks = 0
    const calls = mockHttp(({ url }) => {
      if (url.pathname === '/api/me') {
        sessionChecks += 1
        if (sessionChecks === 2) return json({ detail: 'temporary outage' }, 503)
        if (sessionChecks === 3) return recovery.promise
        return json({ user: 'alice', auth: true, role: 'admin' })
      }
      if (url.pathname === '/api/cases') return json([caseSummary()])
      if (url.pathname === '/api/schema') return json(formSchema)
      if (url.pathname === '/api/options') return json({})
      if (url.pathname === '/api/cases/demo') return json(snapshot())
      throw new Error(`Unexpected ${url}`)
    })
    const { client } = renderPage(<App />, { route: '*' })
    const input = await screen.findByRole('textbox', { name: 'Project name' })
    await waitFor(() => expect(screen.getByRole('button', { name: 'Save as new case' })).toBeEnabled())
    if (dirty) fireEvent.change(input, { target: { value: 'valuable unsaved draft' } })

    await act(async () => { await client.refetchQueries({ queryKey: ['me'] }) })
    expect(await screen.findByText(/Your session could not be rechecked/)).toBeInTheDocument()
    expect(screen.getByRole('textbox', { name: 'Project name' })).toBe(input)
    expect(input).toHaveValue(dirty ? 'valuable unsaved draft' : 'baseline')
    expect(input).toBeEnabled()
    expect(screen.getByRole('button', { name: 'Save' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Save as new case' })).toBeDisabled()
    if (dirty) fireEvent.change(input, { target: { value: 'edited during outage' } })

    await userEvent.setup().click(screen.getByRole('button', { name: 'Retry session check' }))
    await waitFor(() => expect(sessionChecks).toBe(3))
    expect(screen.getByRole('button', { name: 'Retry session check' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Save as new case' })).toBeDisabled()
    expect(screen.getByRole('textbox', { name: 'Project name' })).toBe(input)
    await act(async () => recovery.resolve(json({ user: 'alice', auth: true, role: 'admin' })))
    await waitFor(() => expect(screen.queryByText(/Your session could not be rechecked/)).not.toBeInTheDocument())
    expect(screen.getByRole('textbox', { name: 'Project name' })).toBe(input)
    expect(input).toHaveValue(dirty ? 'edited during outage' : 'baseline')
    expect(screen.getByRole('button', { name: 'Save' }).hasAttribute('disabled')).toBe(!dirty)
    expect(screen.getByRole('button', { name: 'Save as new case' })).toBeEnabled()
    expect(calls.filter(call => call.url.pathname === '/api/cases/demo')).toHaveLength(1)
    expect(calls.every(call => call.method === 'GET')).toBe(true)
  })

  it.each(['other account', 'expired session', 'changed role'])('disposes of the previous draft after recovery confirms a %s', async change => {
    let checks = 0
    let identity = { user: 'alice' as string | null, auth: true, role: 'admin' as 'admin' | 'user' | null }
    mockHttp(({ url }) => {
      if (url.pathname === '/api/me') return ++checks === 2 ? json({ detail: 'outage' }, 503) : json(identity)
      if (url.pathname === '/api/cases') return json([caseSummary()])
      if (url.pathname === '/api/schema') return json(formSchema)
      if (url.pathname === '/api/options') return json({})
      if (url.pathname === '/api/cases/demo') return json(snapshot({ project: { name: `${identity.user} saved case`, metadata: { note: `${identity.user} metadata` } } }))
      throw new Error(`Unexpected ${url}`)
    })
    const { client } = renderPage(<App />, { route: '*' })
    fireEvent.change(await screen.findByRole('textbox', { name: 'Project name' }), { target: { value: 'alice private draft' } })
    fireEvent.change(screen.getByRole('textbox', { name: 'YAML editor' }), { target: { value: 'alice private YAML: [' } })
    await act(async () => { await client.refetchQueries({ queryKey: ['me'] }) })
    await screen.findByText(/Your session could not be rechecked/)
    identity = change === 'other account' ? { user: 'bob', auth: true, role: 'user' } :
      change === 'expired session' ? { user: null, auth: true, role: null } : { user: 'alice', auth: true, role: 'user' }
    await userEvent.setup().click(screen.getByRole('button', { name: 'Retry session check' }))
    if (change === 'expired session') {
      expect(await screen.findByRole('button', { name: 'Sign in' })).toBeInTheDocument()
      expect(screen.queryByRole('textbox', { name: 'Project name' })).not.toBeInTheDocument()
    } else {
      await waitFor(() => expect(screen.getByRole('textbox', { name: 'Project name' })).toHaveValue(`${identity.user} saved case`))
      expect(screen.queryByRole('button', { name: 'Users' })).not.toBeInTheDocument()
    }
    expect(screen.queryByDisplayValue('alice private draft')).not.toBeInTheDocument()
    expect(screen.queryByDisplayValue('alice private YAML: [')).not.toBeInTheDocument()
  })

  it('pauses an already-open copy dialog until the session is confirmed again', async () => {
    let failed = false
    const calls = mockHttp(({ url }) => {
      if (url.pathname === '/api/me') return failed ? Promise.reject(new TypeError('connection lost')) : json({ user: 'alice', auth: true, role: 'admin' })
      if (url.pathname === '/api/cases') return json([caseSummary()])
      if (url.pathname === '/api/schema') return json(formSchema)
      if (url.pathname === '/api/options') return json({})
      if (url.pathname === '/api/cases/demo') return json(snapshot())
      throw new Error(`Unexpected ${url}`)
    })
    const { client } = renderPage(<App />, { route: '*' })
    const user = userEvent.setup()
    await user.click(await screen.findByRole('button', { name: 'Save as new case' }))
    const name = screen.getByRole('textbox', { name: 'New case name' })
    fireEvent.change(name, { target: { value: 'retained_copy_name' } })
    expect(screen.getByRole('button', { name: 'Create case' })).toBeEnabled()
    failed = true
    await act(async () => { await client.refetchQueries({ queryKey: ['me'] }) })
    await screen.findByText(/Your session could not be rechecked/)
    expect(screen.getByRole('textbox', { name: 'New case name' })).toBe(name)
    expect(name).toHaveValue('retained_copy_name')
    expect(screen.getByRole('button', { name: 'Create case' })).toBeDisabled()
    await user.click(screen.getByRole('button', { name: 'Cancel' }))
    failed = false
    await user.click(await screen.findByRole('button', { name: 'Retry session check' }))
    await waitFor(() => expect(screen.queryByText(/Your session could not be rechecked/)).not.toBeInTheDocument())
    expect(calls.every(call => call.method === 'GET')).toBe(true)
  })

  it.each(['empty', 'running', 'finished'] as const)('pauses run actions during a session outage and restores the mounted panel (%s)', async status => {
    let failed = false
    const calls = mockHttp(({ url }) => {
      if (url.pathname === '/api/me') return failed ? json({ detail: 'outage' }, 503) : json({ user: 'alice', auth: true, role: 'admin' })
      if (url.pathname === '/api/cases') return json([caseSummary()])
      if (url.pathname === '/api/schema') return json(formSchema)
      if (url.pathname === '/api/options') return json({})
      if (url.pathname === '/api/cases/demo') return json(snapshot())
      if (url.pathname === '/api/scripts') return json(scripts)
      if (url.pathname === '/api/cases/demo/runs') return json(status === 'empty' ? [] : [runFixture(status)])
      throw new Error(`Unexpected ${url}`)
    })
    const { client } = renderPage(<App />, { route: '*' })
    const user = userEvent.setup()
    await screen.findByRole('textbox', { name: 'Project name' })
    await user.click(screen.getByRole('tab', { name: 'Runs' }))
    const action = status === 'empty' ? 'Run training' : status === 'running' ? 'Stop' : 'Delete'
    await waitFor(() => expect(screen.getByRole('button', { name: action })).toBeEnabled())
    const gpus = screen.getByRole('textbox', { name: 'gpus' })
    fireEvent.change(gpus, { target: { value: '0 1' } })
    failed = true
    await act(async () => { await client.refetchQueries({ queryKey: ['me'] }) })
    await screen.findByText(/Your session could not be rechecked/)
    expect(screen.getByRole('textbox', { name: 'gpus' })).toBe(gpus)
    expect(gpus).toHaveValue('0 1')
    expect(screen.getByRole('button', { name: action })).toBeDisabled()
    failed = false
    await user.click(screen.getByRole('button', { name: 'Retry session check' }))
    await waitFor(() => expect(screen.getByRole('button', { name: action })).toBeEnabled())
    expect(screen.getByRole('textbox', { name: 'gpus' })).toBe(gpus)
    expect(calls.every(call => call.method === 'GET')).toBe(true)
  })
})
