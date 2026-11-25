import { useRef, useState } from 'react'
import { FieldProps } from '@rjsf/utils'
import { Alert, Autocomplete, Box, Button, Dialog, DialogActions, DialogContent, DialogTitle, Stack, TextField, Typography } from '@mui/material'
import EditIcon from '@mui/icons-material/Edit'
import { api, isAbort } from '../api/client'
import { useRequestScope } from '../lib/useRequestScope'
import { FormContext } from './context'
import { fromYaml, toYaml } from '../lib/yaml'
import { lagsText, parseLags } from './lags'

interface Target { name: string; source: string; lags?: number | number[] | null; columns: string[]; transformer?: string | string[] | null; transformer_params?: Record<string, unknown> | null }

// The target variable: name, source column of the main file, horizon lags and columns (derived from the
// project when asked), transformer.
export default function TargetField(props: FieldProps) {
  const { formData, onChange, schema, formContext } = props
  const ctx = formContext as FormContext
  const target: Target = formData ?? { name: '', source: '', lags: [0], columns: ['Step_0'] }
  const [open, setOpen] = useState(false)
  return (
    <Box sx={{ mb: 1.5 }}>
      <Stack direction="row" spacing={1} alignItems="center">
        <TextField fullWidth size="small" label={schema.title ?? 'target variable'} InputProps={{ readOnly: true }}
          value={target.name ? `${target.name} (source ${target.source}, ${(target.columns ?? []).length} step(s), lags ${lagsText(target.lags)})` : 'not set'} />
        <Button variant="outlined" startIcon={<EditIcon />} onClick={() => setOpen(true)}>Edit</Button>
      </Stack>
      {open && <TargetDialog value={target} ctx={ctx} onClose={() => setOpen(false)} onSave={v => { onChange(v); setOpen(false) }} />}
    </Box>
  )
}

function TargetDialog({ value, ctx, onClose, onSave }: { value: Target; ctx: FormContext; onClose: () => void; onSave: (v: Target) => void }) {
  const signal = useRequestScope()
  const edit = useRef(0)
  const [deriveError, setDeriveError] = useState<string | null>(null)
  const [v, setV] = useState<Target>({ ...value, columns: [...(value.columns ?? [])] })
  const [lags, setLags] = useState(lagsText(value.lags))
  const [error, setError] = useState<string | null>(null)
  const [paramsText, setParamsText] = useState(toYaml(value.transformer_params ?? null))
  const set = (patch: Partial<Target>) => { edit.current += 1; setV(prev => ({ ...prev, ...patch })) }
  // the columns follow the lags: one Step_k per horizon step
  const onLags = (text: string) => {
    edit.current += 1
    setLags(text)
    try {
      const parsed = parseLags(text); const list = parsed === null ? [] : Array.isArray(parsed) ? parsed : [parsed]
      set({ lags: parsed, columns: list.map((_, i) => `Step_${i}`) }); setError(null)
    } catch (e: any) { setError(e.message) }
  }
  const derive = async () => {
    const startedEdit = edit.current
    const since = ctx.stamp?.()
    try {
      const facts = await api.derived(ctx.document, signal())
      if (ctx.isCurrent?.(ctx.document, since) === false || edit.current !== startedEdit) { setDeriveError('The target or case changed. Retry using the current values.'); return }
      set({ lags: facts.target_lags, columns: facts.target_columns }); setLags(lagsText(facts.target_lags)); setError(null); setDeriveError(null)
    } catch (error) { if (!isAbort(error)) setDeriveError(`Could not derive the target: ${String(error)}`) }
  }
  const transformers = Array.isArray(v.transformer) ? v.transformer : v.transformer ? [v.transformer] : []
  return (
    <Dialog open onClose={onClose} fullWidth maxWidth="sm">
      <DialogTitle>Target variable</DialogTitle>
      <DialogContent dividers>
        <Stack spacing={2} sx={{ mt: 0.5 }}>
          {deriveError && <Alert severity="warning">{deriveError}</Alert>}
          <Stack direction="row" spacing={1}>
            <TextField label="name" value={v.name} onChange={e => set({ name: e.target.value })} fullWidth />
            <TextField label="source column" value={v.source} onChange={e => set({ source: e.target.value })} fullWidth />
          </Stack>
          <Stack direction="row" spacing={1} alignItems="flex-start">
            <TextField label="lags (horizon steps)" value={lags} onChange={e => onLags(e.target.value)} error={!!error} helperText={error ?? 'e.g. 0..15'} fullWidth />
            <Button onClick={derive} sx={{ whiteSpace: 'nowrap', mt: 0.5 }}>From project</Button>
          </Stack>
          <TextField label="columns" value={(v.columns ?? []).join(', ')} onChange={e => set({ columns: e.target.value.split(',').map(s => s.trim()).filter(Boolean) })}
            helperText="set from the lags (one Step_k per step); editable if the file uses other names" />
          <Autocomplete multiple freeSolo options={ctx.options?.transformers ?? []} value={transformers}
            onChange={(_, items) => set({ transformer: items.length === 0 ? null : items.length === 1 ? items[0] : items })}
            renderInput={p => <TextField {...p} label="transformer" />} />
          {transformers.length > 0 && <TextField label="transformer parameters (YAML)" multiline minRows={2} value={paramsText}
            onChange={e => { setParamsText(e.target.value); try { set({ transformer_params: (fromYaml(e.target.value) as any) ?? null }) } catch { /* keep typing */ } }} />}
          <Typography variant="caption" color="text.secondary">Lags and columns have the same length; "From project" derives both from the horizon.</Typography>
        </Stack>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button variant="contained" disabled={!v.name || !v.source || !!error} onClick={() => onSave(v)}>Save</Button>
      </DialogActions>
    </Dialog>
  )
}
