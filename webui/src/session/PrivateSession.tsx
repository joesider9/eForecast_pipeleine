import { useEffect, useRef, useState, type ReactNode } from 'react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { IdentityContext } from './context'

// The root client holds /me only. Each mounted identity gets a disposable private
// cache, so even identical case names can never reuse a previous account's data.
export default function PrivateSession({ identity, children }: { identity: string; children: ReactNode }) {
  return <SessionCache key={identity} identity={identity}>{children}</SessionCache>
}

function SessionCache({ identity, children }: { identity: string; children: ReactNode }) {
  const [client] = useState(() => new QueryClient({ defaultOptions: {
    queries: { retry: 1, refetchOnWindowFocus: false },
  } }))
  const mounted = useRef(false)
  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
      // React StrictMode replays cleanup/setup with the same client. Defer disposal
      // and cancel it on the replay rather than clearing a still-mounted cache.
      queueMicrotask(() => { if (!mounted.current) { void client.cancelQueries(); client.clear() } })
    }
  }, [client])
  return <IdentityContext.Provider value={identity}>
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  </IdentityContext.Provider>
}
