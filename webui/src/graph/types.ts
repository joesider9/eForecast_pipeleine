// A layer parameter: fixed, a set of choices, or a numeric range. `choice` and `range` are the
// search dimensions Optuna explores; `fixed` is used as it stands.
export type ParamTag = 'fixed' | 'choice' | 'range'
export interface LayerParam { tag: ParamTag; value: any }

export interface LayerSpec {
  type: string
  params?: Record<string, LayerParam>       // every parameter, `size` included
  name?: string | null
  enabled?: boolean
}
export type Experiment = Record<string, LayerSpec[]>            // branch -> layers ('output' = the merge)
export type Experiments = Record<string, Experiment>
// Presentation only: never include these coordinates in the case configuration.
export type ExperimentLayouts = Record<string, Record<string, { x: number; y: number }>>

export interface CatalogueParam { name: string; type: string; default: any; help: string }
export interface CatalogueLayer {
  name: string; label: string; kind: string; inputs: string[]; output: string; size: string
  size_type: string | null; params: CatalogueParam[]; description: string
}
export interface Catalogue { layers: CatalogueLayer[]; branches: { name: string; description: string }[]; merges: string[]; aliases: Record<string, string> }

export interface BuiltLayer { index: number; type: string; configured?: string; output_shape: number[] | null; params: number; tokens?: number; query_branches?: string[]; kv_branches?: string[]; d_model?: number }
export interface DryBuild {
  ok: boolean; tag: string; total_params: number | null
  data_shapes: { shapes: Record<string, any>; steps: number; notes: string[] }
  branches: Record<string, { input?: any; output?: any; layers: BuiltLayer[]; error?: string; error_layer?: number | null }>
  merge: { layers: BuiltLayer[]; output?: any; error?: string } | null
  head: { type: string; n_out: number; params: number } | null
  warnings: string[]
}

export const fmtShape = (s: any): string => Array.isArray(s) ? `(${s.join(' × ')})` : s && typeof s === 'object' ? Object.entries(s).map(([k, v]) => `${k} (${(v as number[]).join(' × ')})`).join(', ') : ''
export const fmtParams = (n: number | null | undefined) => n === null || n === undefined ? '' : n >= 1e6 ? `${(n / 1e6).toFixed(2)} M` : n >= 1e3 ? `${(n / 1e3).toFixed(1)} k` : String(n)

// `{16, 32}` for choices, `[8 … 256]` for a range, the value itself when fixed.
// Fixed/choice values may be structured JSON under the backend's Any boundary.
// Avoid object-to-string coercion (including user keys named "toString").
export const paramValueText = (value: unknown): string =>
  value === null || value === undefined ? '' : typeof value === 'object' ? JSON.stringify(value) : String(value)

export const paramLabel = (p: LayerParam | undefined): string => {
  if (!p) return ''
  if (p.tag === 'choice') return `{${(p.value ?? []).map(paramValueText).join(', ')}}`
  if (p.tag === 'range') return `[${(p.value ?? []).map(paramValueText).join(' … ')}]`
  if (p.value === undefined || p.value === null) return ''
  if (Array.isArray(p.value)) return p.value.length ? `[${p.value.map(paramValueText).join(', ')}]` : ''
  return paramValueText(p.value)
}

// Everything but the size, for the second line of a node: `filters={12, 24} · pool=[1 … 3]`.
export const otherParamsLabel = (l: LayerSpec): string =>
  Object.keys(l.params ?? {}).filter(k => k !== 'size')
    .map(k => `${k}=${paramLabel(l.params![k])}`).join(' · ')
