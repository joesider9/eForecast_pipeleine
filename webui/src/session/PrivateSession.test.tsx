import { act, render, screen, waitFor } from '@testing-library/react'
import { useQuery, useQueryClient, type QueryClient } from '@tanstack/react-query'
import { StrictMode } from 'react'
import { describe, expect, it } from 'vitest'
import PrivateSession from './PrivateSession'
import { api } from '../api/client'
import { deferred, json, mockHttp } from '../test/http'
import { snapshot } from '../test/fixtures'

describe('private query isolation', () => {
  it('discards private cache and delayed requests when identities change, even with identical case names', async () => {
    const oldResponse = deferred<Response>()
    const clients: QueryClient[] = []
    let requests = 0
    const calls = mockHttp(() => {
      requests += 1
      if (requests === 2) return oldResponse.promise
      return json(snapshot({ project: { name: requests === 1 ? 'ALICE PRIVATE' : 'BOB CASE' } }))
    })
    function Probe() {
      const client = useQueryClient()
      if (!clients.includes(client)) clients.push(client)
      const query = useQuery({ queryKey: ['case', 'demo'], queryFn: ({ signal }) => api.getCase('demo', signal) })
      return <div>{query.data?.document.project.name ?? 'loading'}</div>
    }
    const view = render(<PrivateSession identity="alice"><Probe /></PrivateSession>)
    await screen.findByText('ALICE PRIVATE')
    await act(async () => { void clients[0].invalidateQueries({ queryKey: ['case', 'demo'] }) })
    await waitFor(() => expect(requests).toBe(2))
    view.rerender(<PrivateSession identity="bob"><Probe /></PrivateSession>)
    expect(screen.queryByText('ALICE PRIVATE')).not.toBeInTheDocument()
    await screen.findByText('BOB CASE')
    await act(async () => oldResponse.resolve(json(snapshot({ project: { name: 'LATE ALICE SECRET' } }))))
    expect(screen.queryByText('LATE ALICE SECRET')).not.toBeInTheDocument()
    expect(calls[1].signal?.aborted).toBe(true)
    expect(clients[0].getQueryCache().getAll()).toHaveLength(0)
    expect(clients[1].getQueryData(['case', 'demo'])).toMatchObject({ document: { project: { name: 'BOB CASE' } } })
  })

  it('keeps a usable private cache under React StrictMode', async () => {
    mockHttp(() => json(snapshot({ project: { name: 'local case' } })))
    function Probe() {
      const query = useQuery({ queryKey: ['case', 'demo'], queryFn: ({ signal }) => api.getCase('demo', signal) })
      return <div>{query.data?.document.project.name ?? 'loading'}</div>
    }
    render(<StrictMode><PrivateSession identity="local"><Probe /></PrivateSession></StrictMode>)
    expect(await screen.findByText('local case')).toBeInTheDocument()
  })
})
