import type { StructuredSpec } from './StructuredParamField'

export type Mode = 'fixed' | 'choices' | 'range'

// How a stored parameter is read (the rule of the loader and of the Gradio editor): an explicit
// {choices: [...]} or {range: [lo, hi]}, else a plain list is *choices* for a set-valued key of the
// family and a *range* when it has two numbers, else a fixed value.
export function paramMode(value: any, family: string, key: string, setValued: Record<string, string[]>,
                          structured?: Record<string, StructuredSpec>): { mode: Mode; plain: any } {
  // a declared structured parameter is always one value and never searched: without this,
  // `quantile_levels: [0.1, 0.9]` was read as a range and `[0.1, 0.5, 0.9]` as a YAML blob
  if (structured && key in structured) return { mode: 'fixed', plain: value }
  if (value && typeof value === 'object' && !Array.isArray(value) && Object.keys(value).length === 1) {
    if ('choices' in value) return { mode: 'choices', plain: value.choices }
    if ('range' in value) return { mode: 'range', plain: value.range }
  }
  if (Array.isArray(value)) {
    if ((setValued[family] ?? []).includes(key)) return { mode: 'choices', plain: value }
    if (value.length === 2 && value.every(v => typeof v === 'number')) return { mode: 'range', plain: value }
  }
  return { mode: 'fixed', plain: value }
}

// The stored form: choices and ranges are written explicitly, so the loader never has to guess.
export function paramValue(mode: Mode, plain: any): any {
  if (mode === 'choices') return { choices: Array.isArray(plain) ? plain : [plain] }
  if (mode === 'range') return { range: plain }
  return plain
}

