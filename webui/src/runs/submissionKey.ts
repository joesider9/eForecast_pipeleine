// The key that makes a retry a retry.  A lost response, a reload during execution or a second
// click must reach the accepted run again instead of starting another one, so the key outlives
// the component: it belongs to the request, not to the page that happened to send it.
const PREFIX = 'eforecast.run-key'

const slot = (scope: string, caseName: string, fingerprint: string) =>
  `${PREFIX}.${scope}.${caseName}.${fingerprint}`

const random = () => {
  const bytes = new Uint8Array(16)
  crypto.getRandomValues(bytes)
  return Array.from(bytes, byte => byte.toString(16).padStart(2, '0')).join('')
}

// Browser storage is a convenience, never the authority: a private window, cleared site data or
// a storage exception simply means this request gets a new key and the server's own idempotency
// record decides the rest.
const read = (key: string) => { try { return window.localStorage.getItem(key) } catch { return null } }
const write = (key: string, value: string) => { try { window.localStorage.setItem(key, value) } catch { /* no storage */ } }
const drop = (key: string) => { try { window.localStorage.removeItem(key) } catch { /* no storage */ } }

export function submissionKey(scope: string, caseName: string, fingerprint: string) {
  const name = slot(scope, caseName, fingerprint)
  const existing = read(name)
  if (existing) return existing
  const fresh = random()
  write(name, fresh)
  return fresh
}

export function clearSubmissionKey(scope: string, caseName: string, fingerprint: string) {
  drop(slot(scope, caseName, fingerprint))
}

// A request is the same request when its script, its options, the action it was previewed with
// and the saved revision it was accepted against are the same. The server hashes the same four,
// and -- when it was given -- the answer to what the preview showed: a request that accepts
// what it replaces is another request than the one that did not.
export const requestFingerprint = (script: string, options: unknown, action: string, revision?: string,
  accept?: boolean) =>
  JSON.stringify(accept ? [script, options, action, revision ?? null, 'accepts'] : [script, options, action, revision ?? null])
