import { vi } from 'vitest'

export interface HttpCall {
  url: URL
  method: string
  body: Record<string, unknown>
  signal?: AbortSignal | null
}

export function mockHttp(handler: (call: HttpCall) => Response | Promise<Response>) {
  const calls: HttpCall[] = []
  vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const call: HttpCall = {
      url: new URL(String(input), 'http://localhost'), method: init?.method ?? 'GET',
      body: typeof init?.body === 'string' ? JSON.parse(init.body) : {}, signal: init?.signal,
    }
    calls.push(call)
    return handler(call)
  }))
  return calls
}

export const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), {
  status, headers: { 'Content-Type': 'application/json' },
})

export function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason: unknown) => void
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no })
  return { promise, resolve, reject }
}
