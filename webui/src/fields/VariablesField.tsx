import { useState } from 'react'
import { FieldProps } from '@rjsf/utils'
import { Alert, Autocomplete, Box, Button, Checkbox, Chip, Dialog, DialogActions, DialogContent, DialogTitle, FormControlLabel, IconButton, MenuItem, Stack, Table, TableBody, TableCell, TableHead, TableRow, TextField, Typography } from '@mui/material'
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutline'
import AddIcon from '@mui/icons-material/Add'
import AutoFixHighIcon from '@mui/icons-material/AutoFixHigh'
import { api, isAbort, Document } from '../api/client'
import { useRequestScope } from '../lib/useRequestScope'
import { FormContext } from './context'
import { fromYaml, toYaml } from '../lib/yaml'
import { lagsText, parseLags } from './lags'

interface Variable {
  name: string; type?: string; source?: string; lags?: number | number[] | null; timezone?: string
  transformer?: string | string[] | null; transformer_params?: Record<string, unknown> | null
  bands?: string[] | null; nwp_provider?: string | null; use_diff_between_lags?: boolean
}

const blank = (): Variable => ({ name: '', type: 'nwp', source: 'grib', lags: [0], timezone: 'UTC', use_diff_between_lags: false })

// The input variables of a case as a table; a row opens the editor dialog.  "Presets" asks the server for
// the variables of the project's type, horizon and resolution and replaces the list.
export default function VariablesField(props: FieldProps) {
  const { formData, onChange, schema, formContext } = props
  const ctx = formContext as FormContext
  const signal = useRequestScope()
  const rows: Variable[] = Array.isArray(formData) ? formData : []
  const [editing, setEditing] = useState<{ index: number; value: Variable } | null>(null)
  const [message, setMessage] = useState<string | null>(null)
  const project = (ctx?.document?.project ?? {}) as Document

  const loadPresets = async () => {
    const since = ctx.stamp?.()
    try {
      const preset = await api.presets({
        problem_type: project.type ?? 'pv', horizon: project.horizon ?? 0, horizon_type: project.horizon_type ?? 'day_ahead',
        ts_resolution: project.ts_resolution ?? 'H', timezone: project.local_timezone ?? 'UTC',
      }, signal())
      if (ctx.isCurrent?.(ctx.document, since) === false) { setMessage('The draft changed while loading presets. Retry with the latest draft.'); return }
      onChange(preset)
      setMessage(`${preset.length} preset variables for a ${project.type} project (${project.horizon_type}, ${project.ts_resolution}).`)
    } catch (exc) { if (!isAbort(exc)) setMessage(`Presets: ${exc instanceof Error ? exc.message : String(exc)}`) }
  }
  const commit = (v: Variable) => {
    if (!editing) return
    const next = [...rows]
    if (editing.index === -1) next.push(v); else next[editing.index] = v
    onChange(next); setEditing(null)
  }
  return (
    <Box sx={{ mb: 1.5 }}>
      <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 0.5 }}>
        <Typography variant="subtitle2" sx={{ flex: 1 }}>{schema.title ?? 'Variables'} ({rows.length})</Typography>
        <Button size="small" startIcon={<AutoFixHighIcon />} onClick={loadPresets}>Presets for this project</Button>
        <Button size="small" startIcon={<AddIcon />} onClick={() => setEditing({ index: -1, value: blank() })}>Add variable</Button>
      </Stack>
      {message && <Alert severity="info" onClose={() => setMessage(null)} sx={{ mb: 1 }}>{message}</Alert>}
      <Table size="small">
        <TableHead><TableRow><TableCell>name</TableCell><TableCell>type</TableCell><TableCell>source</TableCell><TableCell>lags</TableCell><TableCell>provider</TableCell><TableCell>transformer</TableCell><TableCell /></TableRow></TableHead>
        <TableBody>
          {rows.map((v, i) => (
            <TableRow key={i} hover sx={{ cursor: 'pointer' }} onClick={() => setEditing({ index: i, value: { ...v } })}>
              <TableCell><strong>{v.name}</strong></TableCell>
              <TableCell><Chip size="small" label={v.type ?? 'nwp'} /></TableCell>
              <TableCell>{v.source}</TableCell>
              <TableCell><code>{lagsText(v.lags)}</code></TableCell>
              <TableCell>{v.nwp_provider ?? ''}</TableCell>
              <TableCell>{Array.isArray(v.transformer) ? v.transformer.join(', ') : v.transformer ?? ''}</TableCell>
              <TableCell align="right" onClick={e => e.stopPropagation()}>
                <IconButton size="small" onClick={() => onChange(rows.filter((_, j) => j !== i))}><DeleteOutlineIcon fontSize="small" /></IconButton>
              </TableCell>
            </TableRow>
          ))}
          {rows.length === 0 && <TableRow><TableCell colSpan={7}><Typography color="text.secondary">No variables. Load the presets or add one.</Typography></TableCell></TableRow>}
        </TableBody>
      </Table>
      {editing && <VariableDialog value={editing.value} isNew={editing.index === -1} options={ctx?.options} nwpProviders={(project.nwp_models ?? []).map((m: any) => m.model)}
        onClose={() => setEditing(null)} onSave={commit} />}
    </Box>
  )
}

function VariableDialog({ value, isNew, options, nwpProviders, onClose, onSave }: {
  value: Variable; isNew: boolean; options: FormContext['options']; nwpProviders: string[]; onClose: () => void; onSave: (v: Variable) => void
}) {
  const [v, setV] = useState<Variable>(value)
  const [lags, setLags] = useState(lagsText(value.lags))
  const [lagsError, setLagsError] = useState<string | null>(null)
  const [paramsText, setParamsText] = useState(toYaml(value.transformer_params ?? null))
  const [paramsError, setParamsError] = useState<string | null>(null)
  const set = (patch: Partial<Variable>) => setV({ ...v, ...patch })
  const onLags = (text: string) => { setLags(text); try { set({ lags: parseLags(text) }); setLagsError(null) } catch (e: any) { setLagsError(e.message) } }
  const onParams = (text: string) => { setParamsText(text); try { set({ transformer_params: (fromYaml(text) as any) ?? null }); setParamsError(null) } catch (e: any) { setParamsError(String(e.message).split('\n')[0]) } }
  const transformers = Array.isArray(v.transformer) ? v.transformer : v.transformer ? [v.transformer] : []
  return (
    <Dialog open onClose={onClose} fullWidth maxWidth="sm">
      <DialogTitle>{isNew ? 'New variable' : `Variable ${value.name}`}</DialogTitle>
      <DialogContent dividers>
        <Stack spacing={2} sx={{ mt: 0.5 }}>
          <Stack direction="row" spacing={1}>
            <Autocomplete freeSolo options={options?.variable_names?.[v.type ?? 'nwp'] ?? []} value={v.name} onChange={(_, n) => set({ name: String(n ?? '') })}
              onInputChange={(_, n, reason) => { if (reason === 'input') set({ name: n }) }} sx={{ flex: 1 }}
              renderInput={p => <TextField {...p} label="name" autoFocus helperText={`presets of the ${v.type ?? 'nwp'} type, or any name`} />} />
            <TextField select label="type" value={v.type ?? 'nwp'} onChange={e => { const t = e.target.value; const src = options?.variable_source_by_type?.[t]; setV({ ...v, type: t, ...(src && (!v.source || Object.values(options?.variable_source_by_type ?? {}).includes(v.source)) ? { source: src } : {}) }) }} sx={{ width: 160 }}>
              {(options?.variable_types ?? ['nwp', 'timeseries', 'calendar', 'image']).map(t => <MenuItem key={t} value={t}>{t}</MenuItem>)}
            </TextField>
          </Stack>
          <Stack direction="row" spacing={1}>
            <Autocomplete freeSolo options={options?.variable_sources ?? []} value={v.source ?? ''} onInputChange={(_, s) => set({ source: s })}
              renderInput={p => <TextField {...p} label="source" />} sx={{ flex: 1 }} />
            <TextField label="timezone" value={v.timezone ?? 'UTC'} onChange={e => set({ timezone: e.target.value })} sx={{ width: 160 }} />
          </Stack>
          <TextField label="lags" value={lags} onChange={e => onLags(e.target.value)} error={!!lagsError}
            helperText={lagsError ?? 'e.g. 0..15, -1,-2,-3, -24..-1, 0..15/4 (with a step), or empty'} />
          <Stack direction="row" spacing={1}>
            <Autocomplete multiple freeSolo options={options?.transformers ?? []} value={transformers}
              onChange={(_, items) => set({ transformer: items.length === 0 ? null : items.length === 1 ? items[0] : items })}
              renderInput={p => <TextField {...p} label="transformer" />} sx={{ flex: 1 }} />
            {v.type === 'nwp' && (
              <TextField select label="nwp provider" value={v.nwp_provider ?? ''} onChange={e => set({ nwp_provider: e.target.value || null })} sx={{ width: 180 }}>
                <MenuItem value="">(none)</MenuItem>
                {[...new Set([...nwpProviders, ...(options?.nwp_models ?? [])])].map(m => <MenuItem key={m} value={m}>{m}</MenuItem>)}
              </TextField>
            )}
          </Stack>
          {transformers.length > 0 && (
            <TextField label="transformer parameters (YAML)" multiline minRows={2} value={paramsText} onChange={e => onParams(e.target.value)}
              error={!!paramsError} helperText={paramsError ?? undefined} />
          )}
          {v.type === 'image' && (
            <TextField label="bands (comma separated)" value={(v.bands ?? []).join(', ')} onChange={e => set({ bands: e.target.value.split(',').map(s => s.trim()).filter(Boolean) || null })} />
          )}
          <FormControlLabel control={<Checkbox checked={!!v.use_diff_between_lags} onChange={e => set({ use_diff_between_lags: e.target.checked })} />} label="use differences between lags" />
        </Stack>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button variant="contained" disabled={!v.name.trim() || !!lagsError || !!paramsError} onClick={() => onSave(v)}>Save</Button>
      </DialogActions>
    </Dialog>
  )
}
