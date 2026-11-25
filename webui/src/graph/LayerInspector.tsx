import { Alert, Autocomplete, Box, Button, Chip, Divider, FormControlLabel, IconButton, MenuItem, Select, Stack, Switch, TextField, ToggleButton, ToggleButtonGroup, Tooltip, Typography } from '@mui/material'
import { useState } from 'react'
import { asNames, mismatch, seed } from './paramValues'
import { Catalogue, CatalogueParam, LayerParam, LayerSpec, ParamTag, paramValueText } from './types'

const parseScalar = (t: string): any => { const s = t.trim(); if (s === '') return null; const n = Number(s); return Number.isNaN(n) ? s : n }

function ParamRow({ name, param, spec, branches, onChange, onRemove }: {
  name: string; param: LayerParam; spec?: CatalogueParam; branches: string[]
  onChange: (p: LayerParam) => void; onRemove: () => void
}) {
  const setTag = (tag: ParamTag) => onChange({ tag, value: seed(spec, tag, param.value) })
  const isBool = spec?.type === 'bool'
  const isList = spec?.type === 'list'
  const wrong = mismatch(spec, param, branches)
  return (
    <Box sx={{ border: '1px solid #e5e7eb', borderRadius: 1, p: 1, mt: 1 }}>
      <Stack direction="row" spacing={1} alignItems="center">
        <Typography variant="body2" sx={{ fontWeight: 600, flexGrow: 1 }}>{name}</Typography>
        <ToggleButtonGroup size="small" exclusive value={param.tag} onChange={(_, v) => v && setTag(v as ParamTag)}>
          <ToggleButton value="fixed" sx={{ py: 0, px: 1, fontSize: 11 }}>fixed</ToggleButton>
          <ToggleButton value="choice" sx={{ py: 0, px: 1, fontSize: 11 }} disabled={isList}>choice</ToggleButton>
          <ToggleButton value="range" sx={{ py: 0, px: 1, fontSize: 11 }} disabled={isBool || isList || spec?.type === 'str'}>range</ToggleButton>
        </ToggleButtonGroup>
        <Tooltip title="remove this parameter"><IconButton size="small" onClick={onRemove}>✕</IconButton></Tooltip>
      </Stack>

      {isList && (
        // a list of names is the value itself: chips, with the branches of this graph to pick from
        <Autocomplete multiple freeSolo size="small" options={branches} sx={{ mt: 1 }}
          value={asNames(param.value)}
          onChange={(_, items) => onChange({ tag: 'fixed', value: (items as string[]).map(v => v.trim()).filter(Boolean) })}
          renderTags={(tags, getTagProps) => tags.map((t, i) => { const { key, ...chip } = getTagProps({ index: i }); return <Chip key={key} size="small" label={t} {...chip} /> })}
          renderInput={p => <TextField {...p} label={`${name} names`} placeholder={asNames(param.value).length ? '' : 'default'} />} />
      )}

      {!isList && param.tag === 'fixed' && (isBool
        ? <Select size="small" fullWidth value={String(param.value ?? true)} sx={{ mt: 1 }}
                  onChange={e => onChange({ tag: 'fixed', value: e.target.value === 'true' })}>
            <MenuItem value="true">true</MenuItem><MenuItem value="false">false</MenuItem>
          </Select>
        : <TextField size="small" fullWidth label={`${name} value`} sx={{ mt: 1 }} value={paramValueText(param.value)}
                     placeholder={spec?.default === null || spec?.default === undefined ? 'default' : String(spec.default)}
                     onChange={e => onChange({ tag: 'fixed', value: parseScalar(e.target.value) })} />)}

      {!isList && param.tag === 'choice' && (
        <Autocomplete multiple freeSolo size="small" options={[]} sx={{ mt: 1 }}
          value={(param.value ?? []).map(paramValueText)}
          onChange={(_, items) => onChange({ tag: 'choice', value: (items as string[]).map(v => isBool ? v === 'true' : parseScalar(v)) })}
          renderTags={(tags, getTagProps) => tags.map((t, i) => { const { key, ...chip } = getTagProps({ index: i }); return <Chip key={key} size="small" label={t} {...chip} /> })}
          renderInput={p => <TextField {...p} label="values Optuna picks from (Enter adds)" />} />
      )}

      {!isList && param.tag === 'range' && (
        <Stack direction="row" spacing={1} sx={{ mt: 1 }}>
          <TextField size="small" type="number" label="low" value={param.value?.[0] ?? ''}
                     onChange={e => onChange({ tag: 'range', value: [+e.target.value, param.value?.[1] ?? 0] })} />
          <TextField size="small" type="number" label="high" value={param.value?.[1] ?? ''}
                     onChange={e => onChange({ tag: 'range', value: [param.value?.[0] ?? 0, +e.target.value] })} />
        </Stack>
      )}

      {spec?.help && <Typography variant="caption" color="text.secondary" component="div">{spec.help}</Typography>}
      {wrong && <Typography variant="caption" color="warning.main" component="div" role="alert">{wrong}</Typography>}
      {!spec && <Typography variant="caption" color="warning.main">not a parameter the catalogue knows for this layer</Typography>}
    </Box>
  )
}

// The right-hand panel: the layer's type, its parameters (each fixed, a set of choices or a range),
// an optional name, and the catalogue text that explains what the layer does.
export default function LayerInspector({ layer, catalogue, branch, branches = [], onChange, onRemove }: {
  layer: LayerSpec; catalogue: Catalogue | null; branch: string; branches?: string[]
  onChange: (l: LayerSpec) => void; onRemove: () => void
}) {
  const entry = catalogue?.layers.find(l => l.name === layer.type) ?? catalogue?.layers.find(l => catalogue.aliases[l.name] === layer.type)
  const [custom, setCustom] = useState<string | null>(null)
  const params = layer.params ?? {}
  const known = entry?.params ?? []
  const specOf = (name: string) => known.find(p => p.name === name)
  const available = known.filter(p => !(p.name in params)).map(p => p.name)

  const setParam = (name: string, param: LayerParam) => onChange({ ...layer, params: { ...params, [name]: param } })
  const removeParam = (name: string) => { const next = { ...params }; delete next[name]; onChange({ ...layer, params: next }) }
  const addParam = (name: string) => {
    const key = name.trim()
    if (!key || key in params) return
    setParam(key, { tag: 'fixed', value: seed(specOf(key), 'fixed') })
    setCustom(null)
  }
  // `size` first: it is the one the builder passes positionally, and the one that is always meant.
  // Then the catalogue's order; anything the catalogue does not know goes last.
  const rank = (name: string) => { const i = known.findIndex(p => p.name === name); return i < 0 ? 1e6 : i }
  const ordered = Object.keys(params).sort((a, b) => (a === 'size' ? -1 : b === 'size' ? 1 : rank(a) - rank(b)))
  const isMerge = branch === 'output'
  const choicesForType = (catalogue?.layers ?? []).filter(l => isMerge ? (l.kind === 'merge' || l.kind === 'dense' || l.kind === 'shape') : l.kind !== 'merge').map(l => l.name)

  return (
    <Stack spacing={1.5} sx={{ p: 2, width: 380, borderLeft: '1px solid #e5e7eb', height: '100%', overflow: 'auto', boxSizing: 'border-box' }}>
      <Typography variant="subtitle1">{entry?.label ?? layer.type}</Typography>
      <Autocomplete freeSolo size="small" options={choicesForType} value={layer.type} onChange={(_, v) => v && onChange({ ...layer, type: v as string })}
        onInputChange={(_, v, reason) => { if (reason === 'input' && v) onChange({ ...layer, type: v }) }} renderInput={p => <TextField {...p} label="layer type" />} />
      {entry && <Typography variant="body2" color="text.secondary">{entry.description}</Typography>}
      <Divider />

      <Typography variant="caption" color="text.secondary">
        parameters — <b>fixed</b> is used as it stands, <b>choice</b> and <b>range</b> are searched by Optuna
      </Typography>
      {ordered.map(name => (
        <ParamRow key={name} name={name} param={params[name]} spec={specOf(name)} branches={branches}
                  onChange={p => setParam(name, p)} onRemove={() => removeParam(name)} />
      ))}
      {!ordered.length && <Typography variant="caption" color="text.secondary">none set; the method's values are used</Typography>}

      <Select size="small" displayEmpty value="" onChange={e => {
        const picked = e.target.value as string
        if (picked === '__custom__') setCustom(''); else addParam(picked)
      }}>
        <MenuItem value="" disabled>add a parameter…</MenuItem>
        {available.map(name => <MenuItem key={name} value={name}>{name}</MenuItem>)}
        {!available.length && <MenuItem value="" disabled>every parameter of this layer is set</MenuItem>}
        <Divider />
        <MenuItem value="__custom__"><i>other…</i></MenuItem>
      </Select>
      {custom !== null && (
        <Stack direction="row" spacing={1}>
          <TextField size="small" autoFocus fullWidth label="parameter name" value={custom}
                     onChange={e => setCustom(e.target.value)}
                     onKeyDown={e => { if (e.key === 'Enter') { e.preventDefault(); addParam(custom) } }} />
          <Button size="small" onClick={() => addParam(custom)}>Add</Button>
          <Button size="small" onClick={() => setCustom(null)}>Cancel</Button>
        </Stack>
      )}

      <Divider />
      <TextField size="small" label="name (optional)" value={layer.name ?? ''} onChange={e => onChange({ ...layer, name: e.target.value || null })} />
      <FormControlLabel control={<Switch size="small" checked={layer.enabled !== false} onChange={e => onChange({ ...layer, enabled: e.target.checked })} />} label="enabled" />
      <Button color="error" variant="outlined" size="small" onClick={onRemove}>Remove layer</Button>
      {!entry && <Alert severity="warning">Unknown layer type for the catalogue; the dry build tells whether the builder accepts it.</Alert>}
    </Stack>
  )
}
