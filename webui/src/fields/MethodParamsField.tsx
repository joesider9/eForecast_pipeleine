import { useRef, useState } from 'react'
import { FieldProps } from '@rjsf/utils'
import { Alert, Autocomplete, Box, Button, Chip, Dialog, DialogActions, DialogContent, DialogTitle, IconButton, Stack, Table, TableBody, TableCell, TableHead, TableRow, TextField, ToggleButton, ToggleButtonGroup, Typography } from '@mui/material'
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutline'
import AddIcon from '@mui/icons-material/Add'
import { api, isAbort, Document } from '../api/client'
import { useRequestScope } from '../lib/useRequestScope'
import { FormContext } from './context'
import { fromYaml, toYaml } from '../lib/yaml'
import StructuredParamField, { StructuredSpec } from './StructuredParamField'
import { paramMode, paramValue, type Mode } from './methodParameters'

type Params = Record<string, any>

const scalarText = (v: any) => (v === null || v === undefined ? '' : typeof v === 'object' ? toYaml(v).trim() : String(v))
const parseScalar = (text: string) => { const t = text.trim(); if (t === '') return null; try { return fromYaml(t) } catch { return t } }

// The method families (methods.method_params): a table of families, each opening a dialog with one row
// per parameter and its search mode.  The clustering family is one of them.
export default function MethodParamsField(props: FieldProps) {
  const { formData, onChange, formContext } = props
  const ctx = formContext as FormContext
  const signal = useRequestScope()
  const [presetError, setPresetError] = useState<string | null>(null)
  const families: Record<string, Params> = formData ?? {}
  const methods = (ctx?.document?.methods ?? {}) as Document
  const enabled = new Set<string>([
    ...Object.entries(methods.methods_cpu ?? {}).filter(([, on]) => on).map(([k]) => k),
    ...Object.entries(methods.cluster_methods ?? {}).filter(([, on]) => on).map(([k]) => k),
    ...Object.entries(methods.global_methods ?? {}).filter(([, on]) => on).map(([k]) => k),
  ])
  const isFuzzy = Boolean((ctx?.document?.project as Document)?.is_fuzzy)
  const needsClusterer = isFuzzy && !('clustering' in families)
  const [editing, setEditing] = useState<string | null>(null)
  const [adding, setAdding] = useState('')
  const addFamily = async (name: string) => {
    let preset: Params = {}
    const since = ctx.stamp?.()
    try {
      preset = await api.familyPreset(name, ctx?.document ?? {}, signal())
      if (ctx.isCurrent?.(ctx.document, since) === false) { setPresetError('The draft changed while loading the family preset. Retry.'); return }
      setPresetError(null)
    } catch (error) { if (!isAbort(error)) setPresetError(`Could not load the family preset: ${String(error)}`); return }
    onChange({ ...families, [name]: preset }); setEditing(name)
  }
  const known = [...(ctx?.options?.families ?? []), 'clustering']
  const names = [...Object.keys(families), ...known.filter(k => !(k in families))]
  const structured = (ctx?.options?.structured_params ?? {}) as Record<string, StructuredSpec>
  const countSearched = (params: Params, family: string) =>
    Object.entries(params ?? {}).filter(([k, v]) => paramMode(v, family, k, ctx?.options?.set_valued_keys ?? {}, structured).mode !== 'fixed').length
  return (
    <Box sx={{ mb: 1.5 }}>
      {presetError && <Alert severity="error" onClose={() => setPresetError(null)}>{presetError}</Alert>}
      <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 0.5 }}>
        <Typography variant="subtitle2" sx={{ flex: 1 }}>Method families and their search spaces</Typography>
        <Autocomplete freeSolo size="small" options={known.filter(k => !(k in families))} inputValue={adding} onInputChange={(_, v) => setAdding(v)}
          renderInput={p => <TextField {...p} placeholder="family" />} sx={{ width: 180 }} />
        <Button size="small" startIcon={<AddIcon />} disabled={!adding.trim()} onClick={async () => {
          await addFamily(adding.trim()); setAdding('')
        }}>Add with presets</Button>
      </Stack>
      {needsClusterer && (
        <Alert severity="warning" sx={{ mb: 1 }}
               action={<Button size="small" onClick={() => addFamily('clustering')}>Add the clusterer</Button>}>
          The project has <code>is_fuzzy</code> on, so it is trained per cluster, but there is no
          <code> clustering</code> family. Training stops at the clusterer until one is added.
        </Alert>
      )}
      <Table size="small">
        <TableHead><TableRow><TableCell>family</TableCell><TableCell>status</TableCell><TableCell>parameters</TableCell><TableCell>searched</TableCell><TableCell /></TableRow></TableHead>
        <TableBody>
          {names.filter(n => n in families).map(name => (
            <TableRow key={name} hover sx={{ cursor: 'pointer' }} onClick={() => setEditing(name)}>
              <TableCell><strong>{name}</strong></TableCell>
              <TableCell>{name === 'clustering'
                ? (isFuzzy ? <Chip size="small" color="success" label="clusterer, enabled" /> : <Chip size="small" variant="outlined" label="clusterer, is_fuzzy is off" />)
                : enabled.has(name) ? <Chip size="small" color="success" label="enabled" /> : <Chip size="small" variant="outlined" label="not selected" />}</TableCell>
              <TableCell>{Object.keys(families[name] ?? {}).length}</TableCell>
              <TableCell>{countSearched(families[name], name)}</TableCell>
              <TableCell align="right" onClick={e => e.stopPropagation()}>
                <IconButton size="small" onClick={() => { const next = { ...families }; delete next[name]; onChange(next) }}><DeleteOutlineIcon fontSize="small" /></IconButton>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      {editing && families[editing] !== undefined && (
        <FamilyDialog family={editing} params={families[editing]} setValued={ctx?.options?.set_valued_keys ?? {}} document={ctx?.document ?? {}}
          structured={(ctx?.options?.structured_params ?? {}) as Record<string, StructuredSpec>}
          presetKeys={ctx?.options?.family_preset_keys?.[editing] ?? []}
          onClose={() => setEditing(null)} onSave={params => { onChange({ ...families, [editing]: params }); setEditing(null) }} />
      )}
    </Box>
  )
}

function FamilyDialog({ family, params, setValued, document, presetKeys, structured, onClose, onSave }: {
  family: string; params: Params; setValued: Record<string, string[]>; document: Document; presetKeys: string[]
  structured: Record<string, StructuredSpec>; onClose: () => void; onSave: (p: Params) => void
}) {
  const [rows, setRows] = useState(() => Object.entries(params ?? {}).map(([key, value]) => ({ key, ...paramMode(value, family, key, setValued, structured) })))
  const latestRows = useRef(rows)
  latestRows.current = rows
  const signal = useRequestScope()
  const [newKey, setNewKey] = useState('')
  const [presetNote, setPresetNote] = useState<string | null>(null)
  // "Fill presets" adds the parameters of the family preset that are missing; existing values are kept
  const fillPresets = async () => {
    try {
      const preset = await api.familyPreset(family, document, signal())
      const current = latestRows.current
      const present = new Set(current.map(row => row.key))
      const added = Object.entries(preset).filter(([key]) => !present.has(key)).map(([key, value]) => ({ key, ...paramMode(value, family, key, setValued, structured) }))
      setRows([...current, ...added]); setPresetNote(added.length ? `${added.length} preset parameter(s) added: ${added.map(row => row.key).join(', ')}` : 'every preset parameter is already present')
    } catch (error) { if (!isAbort(error)) setPresetNote(`Could not load presets: ${String(error)}`) }
  }
  const [yamlMode, setYamlMode] = useState(false)
  const [yamlText, setYamlText] = useState(toYaml(params))
  const [yamlError, setYamlError] = useState<string | null>(null)
  const update = (i: number, patch: Partial<{ key: string; mode: Mode; plain: any }>) => setRows(rows.map((r, j) => (j === i ? { ...r, ...patch } : r)))
  const switchMode = (i: number, mode: Mode) => {
    const r = rows[i]
    let plain = r.plain
    if (mode === 'choices') plain = Array.isArray(plain) ? plain : plain === null || plain === undefined ? [] : [plain]
    if (mode === 'range') plain = Array.isArray(plain) && plain.length === 2 ? plain : [0, 1]
    if (mode === 'fixed') plain = Array.isArray(plain) ? plain[0] ?? null : plain
    update(i, { mode, plain })
  }
  const result = (): Params => Object.fromEntries(rows.filter(r => r.key.trim()).map(r => [r.key.trim(), paramValue(r.mode, r.plain)]))
  const save = () => {
    if (yamlMode) { try { onSave((fromYaml(yamlText) as Params) ?? {}) } catch (e: any) { setYamlError(String(e.message).split('\n')[0]) } return }
    onSave(result())
  }
  return (
    <Dialog open onClose={onClose} fullWidth maxWidth="md">
      <DialogTitle>
        <Stack direction="row" alignItems="center" spacing={2}>
          <span style={{ flex: 1 }}>{family}</span>
          {presetKeys.length > 0 && !yamlMode && <Button size="small" onClick={fillPresets}>Fill presets</Button>}
          <ToggleButtonGroup size="small" exclusive value={yamlMode ? 'yaml' : 'table'} onChange={(_, v) => { if (v === 'yaml') setYamlText(toYaml(result())); if (v) setYamlMode(v === 'yaml') }}>
            <ToggleButton value="table">table</ToggleButton><ToggleButton value="yaml">yaml</ToggleButton>
          </ToggleButtonGroup>
        </Stack>
      </DialogTitle>
      <DialogContent dividers>
        {yamlMode ? (
          <Stack spacing={1}>
            <TextField multiline minRows={12} value={yamlText} onChange={e => { setYamlText(e.target.value); setYamlError(null) }} inputProps={{ style: { fontFamily: 'monospace', fontSize: 13 } }} />
            {yamlError && <Alert severity="error">{yamlError}</Alert>}
          </Stack>
        ) : (
          <Stack spacing={1}>
            {presetNote && <Alert severity="info" onClose={() => setPresetNote(null)} sx={{ py: 0 }}>{presetNote}</Alert>}
            <Typography variant="caption" color="text.secondary">
              fixed = one value; choices = the search picks one of the listed values; range = the search samples between the two numbers.
              Parameters {(setValued[family] ?? []).length ? `(${(setValued[family] ?? []).join(', ')})` : ''} are categorical for this family.
            </Typography>
            <Table size="small">
              <TableHead><TableRow><TableCell sx={{ width: 220 }}>parameter</TableCell><TableCell sx={{ width: 200 }}>mode</TableCell><TableCell>value</TableCell><TableCell /></TableRow></TableHead>
              <TableBody>
                {rows.map((r, i) => (
                  <TableRow key={i}>
                    <TableCell><TextField size="small" fullWidth value={r.key} onChange={e => update(i, { key: e.target.value })} /></TableCell>
                    <TableCell>
                      {structured[r.key] ? (
                        <Typography variant="caption" color="text.secondary">
                          {structured[r.key].label ?? 'a structured value'}, never searched
                        </Typography>
                      ) : (
                        <ToggleButtonGroup size="small" exclusive value={r.mode} onChange={(_, v) => v && switchMode(i, v)}>
                          <ToggleButton value="fixed">fixed</ToggleButton><ToggleButton value="choices">choices</ToggleButton><ToggleButton value="range">range</ToggleButton>
                        </ToggleButtonGroup>
                      )}
                    </TableCell>
                    <TableCell>
                      {structured[r.key] && (
                        <StructuredParamField value={r.plain} spec={structured[r.key]} family={family} document={document}
                                              onChange={v => update(i, { mode: 'fixed', plain: v })} />
                      )}
                      {!structured[r.key] && r.mode === 'fixed' && <TextField size="small" fullWidth value={scalarText(r.plain)} onChange={e => update(i, { plain: parseScalar(e.target.value) })} helperText={typeof r.plain === 'object' && r.plain !== null ? 'structured value (YAML)' : undefined} />}
                      {!structured[r.key] && r.mode === 'choices' && (
                        <Autocomplete multiple freeSolo size="small" options={[]} value={(r.plain ?? []).map(scalarText)}
                          onChange={(_, items) => update(i, { plain: (items as string[]).map(parseScalar) })}
                          renderTags={(tags, getTagProps) => tags.map((t, k) => <Chip size="small" label={t} {...getTagProps({ index: k })} />)}
                          renderInput={p => <TextField {...p} placeholder="type a value and press Enter" />} />
                      )}
                      {!structured[r.key] && r.mode === 'range' && (
                        <Stack direction="row" spacing={1}>
                          <TextField size="small" type="number" label="low" value={r.plain?.[0] ?? ''} onChange={e => update(i, { plain: [+e.target.value, r.plain?.[1] ?? 0] })} />
                          <TextField size="small" type="number" label="high" value={r.plain?.[1] ?? ''} onChange={e => update(i, { plain: [r.plain?.[0] ?? 0, +e.target.value] })} />
                        </Stack>
                      )}
                    </TableCell>
                    <TableCell align="right"><IconButton size="small" onClick={() => setRows(rows.filter((_, j) => j !== i))}><DeleteOutlineIcon fontSize="small" /></IconButton></TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
            <Stack direction="row" spacing={1} alignItems="center">
              <Autocomplete freeSolo size="small" options={presetKeys.filter(k => !rows.some(r => r.key === k))} inputValue={newKey} onInputChange={(_, v) => setNewKey(v)}
                renderInput={p => <TextField {...p} placeholder="new parameter (or pick a known one)" />} sx={{ width: 300 }} />
              <Button size="small" startIcon={<AddIcon />} disabled={!newKey.trim()} onClick={() => { setRows([...rows, { key: newKey.trim(), mode: 'fixed', plain: null }]); setNewKey('') }}>Add parameter</Button>
            </Stack>
          </Stack>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button variant="contained" onClick={save}>Apply</Button>
      </DialogActions>
    </Dialog>
  )
}
