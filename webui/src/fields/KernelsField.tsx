import { useState } from 'react'
import { FieldProps } from '@rjsf/utils'
import { Box, Button, Chip, Stack, TextField, Typography } from '@mui/material'
import { FormContext } from './context'

type Kernels = Record<string, number[][]>
const text = (k: number[]) => `[${k.join(', ')}]`
const parse = (t: string): number[] | null => {
  const nums = t.replace(/[[\]()]/g, '').split(/[ ,x×]+/).filter(Boolean).map(Number)
  return nums.length >= 2 && nums.every(n => Number.isInteger(n) && n > 0) ? nums : null
}

// The convolution kernels tried per data kind (methods.kernels): chips per group, presets, free entry.
export default function KernelsField(props: FieldProps) {
  const { formData, onChange, schema, formContext } = props
  const ctx = formContext as FormContext
  const presets = ctx?.options?.kernel_presets ?? {}
  const value: Kernels = formData && typeof formData === 'object' ? formData : {}
  const groups = [...new Set([...Object.keys(presets), ...Object.keys(value)])]
  const [drafts, setDrafts] = useState<Record<string, string>>({})
  const set = (group: string, kernels: number[][]) => onChange({ ...value, [group]: kernels })
  const add = (group: string) => { const k = parse(drafts[group] ?? ''); if (!k) return; set(group, [...(value[group] ?? []), k]); setDrafts({ ...drafts, [group]: '' }) }
  return (
    <Box sx={{ mb: 1.5 }}>
      <Stack direction="row" alignItems="center" spacing={1}>
        <Typography variant="subtitle2" sx={{ flex: 1 }}>{schema.title ?? 'Kernels'}</Typography>
        <Button size="small" onClick={() => onChange({ ...value, ...presets })}>Reset to presets</Button>
      </Stack>
      <Typography variant="caption" color="text.secondary">Kernel sizes tried by the convolutions, one list per data kind: (lags, lat, lon) for NWP, (frames, height, width) for images.</Typography>
      {groups.map(group => (
        <Stack key={group} direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap sx={{ mt: 1 }}>
          <Typography variant="body2" sx={{ width: 70 }}>{group}</Typography>
          {(value[group] ?? []).map((k, i) => <Chip key={i} size="small" label={text(k)} onDelete={() => set(group, value[group].filter((_, j) => j !== i))} />)}
          <TextField size="small" placeholder="e.g. 2, 3, 3" value={drafts[group] ?? ''} onChange={e => setDrafts({ ...drafts, [group]: e.target.value })}
            onKeyDown={e => { if (e.key === 'Enter') { e.preventDefault(); add(group) } }} sx={{ width: 140 }} />
          <Button size="small" disabled={!parse(drafts[group] ?? '')} onClick={() => add(group)}>Add</Button>
          {presets[group] && <Button size="small" onClick={() => set(group, presets[group])}>presets</Button>}
        </Stack>
      ))}
    </Box>
  )
}
