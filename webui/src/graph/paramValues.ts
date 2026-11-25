import { CatalogueParam, LayerParam, ParamTag, paramValueText } from './types'

export const isEmpty = (v: unknown) => v === null || v === undefined || v === '' || (Array.isArray(v) && v.length === 0)

// The names a list parameter holds, whatever was stored: a list, one name, or a value that is neither.
export const asNames = (v: unknown): string[] => Array.isArray(v) ? v.map(paramValueText) : isEmpty(v) ? [] : [paramValueText(v)]

const isBranch = (name: unknown, branches: string[]) => typeof name === 'string' && branches.some(b => b === name || b.split('/').pop() === name)

// The value to start from when a parameter is added or its tag changes.
// Nothing is made up for a parameter the catalogue gives no default: an empty field is a parameter
// that is not set, and the method's value stands. It used to start at 64 whatever it was, and a
// case that added `attention_query` (a list of branch names) lost every trial of that graph.
// `size` alone starts at a number, because a layer that takes one needs one.
export const seed = (spec: CatalogueParam | undefined, tag: ParamTag, previous?: unknown): unknown => {
  if (spec?.type === 'list') return Array.isArray(previous) ? previous : []
  const first: unknown = Array.isArray(previous) ? previous[0] : previous
  const base: unknown = first ?? spec?.default ?? (spec?.type === 'bool' ? true : spec?.name === 'size' ? 64 : null)
  if (tag === 'choice') return Array.isArray(previous) ? previous : base === null ? [] : [base]
  if (tag === 'range') {
    const low = typeof base === 'number' ? base : 1
    return Array.isArray(previous) && previous.length === 2 ? previous : [low, low * 4 || 4]
  }
  return base
}

// What a stored value is not, in the words the readiness report uses for the same finding.
// `size` has forms of its own ('linear', the [] of a layer without one) and is not judged here.
export const mismatch = (spec: CatalogueParam | undefined, param: LayerParam, branches: string[]): string | null => {
  if (!spec || spec.name === 'size') return null
  if (param.tag === 'choice' && isEmpty(param.value)) return 'add at least one value, or make it fixed'
  const values: unknown[] = spec.type === 'list' || param.tag === 'fixed' ? [param.value] : Array.isArray(param.value) ? param.value : []
  for (const value of values) {
    if (isEmpty(value)) continue
    if ((spec.type === 'int' || spec.type === 'float') && typeof value !== 'number') return `${paramValueText(value)} is not a number, and the layer takes a number here`
    if (spec.type === 'str' && typeof value !== 'string') return `${paramValueText(value)} is not text, and the layer takes text here`
    if (spec.type === 'bool' && typeof value !== 'boolean' && value !== 0 && value !== 1) return 'the layer takes true or false here'
    if (spec.type === 'list') {
      const names: unknown[] = Array.isArray(value) ? value : [value]
      if (names.some(n => typeof n !== 'string')) return `${names.map(paramValueText).join(', ')} names no branch; the layer reads this as not set`
      const strangers = names.filter(n => !isBranch(n, branches))
      if (strangers.length) return `${strangers.join(', ')} ${strangers.length === 1 ? 'is' : 'are'} not a branch of this graph`
    }
  }
  return null
}
