import { useCallback, useEffect, useRef } from 'react'

// Imperative requests have the same cancellation lifetime as their editor/panel.
export function useRequestScope() {
  const controller = useRef(new AbortController())
  useEffect(() => {
    if (controller.current.signal.aborted) controller.current = new AbortController()
    const current = controller.current
    return () => current.abort()
  }, [])
  return useCallback(() => controller.current.signal, [])
}
