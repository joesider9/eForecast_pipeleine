import { useContext, useMemo, useState } from 'react'
import { fromYaml, toYaml } from '../lib/yaml'
import { YamlDraftContext, unfinishedYaml, type YamlBuffer } from './yamlDraftContext'

// The case owns these buffers. The local fallback keeps standalone form/graph
// consumers usable without introducing storage shared between cases/accounts.
export function useYamlBuffer(key: string, label: string, section: string, value: unknown,
  apply: (parsed: unknown) => void, check?: (parsed: unknown) => void) {
  const store = useContext(YamlDraftContext)
  const [local, setLocal] = useState<Record<string, YamlBuffer>>({})
  const buffers = store?.buffers ?? local
  const buffer = buffers[key]
  const applied = useMemo(() => toYaml(value), [value])
  // A real external value change refreshes valid text; an unfinished edit always
  // survives until it is repaired or explicitly discarded.
  const text = buffer && (buffer.error !== null || buffer.applied === applied) ? buffer.text : applied
  const change = (text: string) => {
    let parsed: unknown
    let error: string | null = null
    let parsedValue = applied
    try { parsed = fromYaml(text); check?.(parsed); parsedValue = toYaml(parsed) }
    catch (exception) { error = String(exception instanceof Error ? exception.message : exception).split('\n')[0] }
    const next: YamlBuffer = { text, error, applied: parsedValue, label, section }
    if (store) store.setBuffer(key, next)
    else setLocal(current => ({ ...current, [key]: next }))
    if (error === null) apply(parsed)
  }
  const discard = () => {
    if (store) store.discardBuffer(key)
    else setLocal(current => { const next = { ...current }; delete next[key]; return next })
  }
  return { text, error: buffer?.error ?? null, change, discard, hasErrors: unfinishedYaml(buffers).length > 0 }
}
