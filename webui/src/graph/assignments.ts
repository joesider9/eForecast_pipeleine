import type { Document } from '../api/client'

function payload(value: unknown): unknown {
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    for (const key of ['choices', 'values', 'range', 'value'] as const) {
      if (key in value) return (value as Record<string, unknown>)[key]
    }
  }
  return value
}

export function assignmentProblem(value: unknown): string | null {
  const raw = payload(value)
  if (raw == null || typeof raw === 'string' || Array.isArray(raw) && raw.every(tag => typeof tag === 'string')) return null
  return 'The stored experiment assignment needs repair in Methods before it can be changed here.'
}

// Keep the stored scalar/list/choices encoding unless the user's edit needs a
// scalar to become a list. Other family parameters and extension fields survive.
export function assignedTags(value: unknown): string[] {
  const raw = payload(value)
  const values = typeof raw === 'string' ? [raw] : Array.isArray(raw) ? raw : []
  return values.filter((tag): tag is string => typeof tag === 'string' && tag.length > 0)
}

export function changeAssignments(document: Document, family: string, change: (tags: string[]) => string[]): Document {
  const methods = document.methods ?? {}
  const params = methods.method_params ?? {}
  const current = params[family] ?? {}
  const value: unknown = current.experiment_tag
  if (assignmentProblem(value)) return document // preserve unrecognised stored data
  const tags = assignedTags(value)
  const next = change(tags)
  if (JSON.stringify(tags) === JSON.stringify(next)) return document
  let encoded: unknown = typeof value === 'string' && next.length === 1 ? next[0] : next
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    const key = ['choices', 'values', 'range', 'value'].find(key => key in value)
    if (key === 'value' && 'tag' in value && value.tag === 'fixed' && next.length !== 1) {
      encoded = { ...value, tag: 'choice', value: next }
    } else if (key) encoded = { ...value, [key]: typeof payload(value) === 'string' && next.length === 1 ? next[0] : next }
  }
  return { ...document, methods: { ...methods, method_params: { ...params,
    [family]: { ...current, experiment_tag: encoded },
  } } }
}

export function addAssignment(document: Document, family: string, tag: string): Document {
  return changeAssignments(document, family, tags => tags.includes(tag) ? tags : [...tags, tag])
}

export function renameAssignments(document: Document, families: readonly string[], from: string, to: string): Document {
  return families.reduce((next, family) => changeAssignments(next, family, tags => tags.map(tag => tag === from ? to : tag)), document)
}
