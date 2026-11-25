import { describe, expect, it, vi } from 'vitest'
import { advanceSession, api, ApiError } from './client'
import { deferred, json, mockHttp } from '../test/http'

describe('API boundary', () => {
  it('retains validation paths from a rejected save', async () => {
    mockHttp(() => json({ detail: 'validation failed', errors: [{ path: 'project.name', message: 'required' }] }, 422))
    await expect(api.saveCase('demo', {})).rejects.toMatchObject({
      status: 422, errors: [{ path: 'project.name', message: 'required' }],
    })
  })

  it('shows an HTTP failure even when a proxy returns HTML', async () => {
    mockHttp(() => new Response('<html>Unavailable</html>', { status: 503, statusText: 'Service Unavailable' }))
    await expect(api.cases()).rejects.toBeInstanceOf(ApiError)
  })

  it('forwards query cancellation and rejects a late response', async () => {
    const delayed = deferred<Response>()
    const calls = mockHttp(() => delayed.promise)
    const controller = new AbortController()
    const request = api.getCase('demo', controller.signal)
    const rejected = expect(request).rejects.toMatchObject({ name: 'AbortError' })
    controller.abort()
    delayed.resolve(json({ document: { private: 'alice' } }))
    await rejected
    expect(calls[0].signal?.aborted).toBe(true)
  })

  it('does not dispatch a delayed old-identity 401 into a new session', async () => {
    const delayed = deferred<Response>()
    mockHttp(() => delayed.promise)
    const unauthorized = vi.fn()
    window.addEventListener('eforecast:unauthorized', unauthorized)
    try {
      const request = api.cases()
      const rejected = expect(request).rejects.toMatchObject({ name: 'AbortError' })
      advanceSession()
      delayed.resolve(json({ detail: 'old expired cookie' }, 401))
      await rejected
      expect(unauthorized).not.toHaveBeenCalled()
    } finally { window.removeEventListener('eforecast:unauthorized', unauthorized) }
  })

  it('retains the current saved version in a conflict response', async () => {
    mockHttp(() => json({ detail: 'changed', revision: 'v2', document: { project: { name: 'remote' } } }, 409))
    await expect(api.saveCase('demo', {}, 'v1')).rejects.toMatchObject({
      status: 409, payload: { revision: 'v2', document: { project: { name: 'remote' } } },
    })
  })
})
