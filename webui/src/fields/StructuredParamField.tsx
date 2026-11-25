import { useMemo, useState } from 'react'
import { Autocomplete, Box, Button, Chip, IconButton, MenuItem, Select, Stack, TextField, Typography } from '@mui/material'
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutline'
import AddIcon from '@mui/icons-material/Add'
import { Document } from '../api/client'
import RbfVarImpField from './RbfVarImpField'

// A method-family parameter whose value is a structure, not a scalar. The kinds are declared
// server-side in options.STRUCTURED_PARAMS, so the editor never has to guess what a list means.
export interface StructuredSpec {
  kind: 'number_list' | 'option_list' | 'number_map' | 'rbf_var_imp'
  label?: string; help?: string
  minimum?: number; maximum?: number; sorted?: boolean; step?: number; presets?: number[][]
  options?: string[]; single?: boolean; keys?: string[]
}

const asNumbers = (value: unknown): number[] =>
  Array.isArray(value) ? value.map(Number).filter(v => !Number.isNaN(v)) : []

function NumberList({ value, spec, onChange }: { value: unknown; spec: StructuredSpec; onChange: (v: unknown) => void }) {
  const numbers = useMemo(() => asNumbers(value), [value])
  const [text, setText] = useState('')
  const outOfRange = numbers.filter(n =>
    (spec.minimum !== undefined && n < spec.minimum) || (spec.maximum !== undefined && n > spec.maximum))
  const unsorted = Boolean(spec.sorted) && numbers.some((n, i) => i > 0 && n <= numbers[i - 1])

  const commit = (next: number[]) => onChange(spec.sorted ? [...next].sort((a, b) => a - b) : next)
  const add = () => {
    const parsed = text.split(/[,\s]+/).map(Number).filter(v => !Number.isNaN(v))
    if (parsed.length) commit([...new Set([...numbers, ...parsed])])
    setText('')
  }
  return (
    <Stack spacing={0.5} sx={{ width: '100%' }}>
      <Stack direction="row" spacing={0.5} alignItems="center" flexWrap="wrap" useFlexGap>
        {numbers.map((n, i) => (
          <Chip key={i} size="small" label={n}
                color={(spec.minimum !== undefined && n < spec.minimum) || (spec.maximum !== undefined && n > spec.maximum) ? 'error' : 'default'}
                variant="outlined" onDelete={() => commit(numbers.filter((_, j) => j !== i))} />
        ))}
        {!numbers.length && <Typography variant="caption" color="text.secondary">none</Typography>}
      </Stack>
      <Stack direction="row" spacing={1} alignItems="center">
        <TextField size="small" sx={{ width: 190 }} value={text} placeholder="0.1 0.5 0.9"
                   onChange={e => setText(e.target.value)}
                   onKeyDown={e => { if (e.key === 'Enter') { e.preventDefault(); add() } }} />
        <Button size="small" startIcon={<AddIcon />} disabled={!text.trim()} onClick={add}>Add</Button>
        {(spec.presets ?? []).map((preset, i) => (
          <Chip key={i} size="small" variant="outlined" label={preset.join(', ')} onClick={() => commit(preset)} />
        ))}
      </Stack>
      {outOfRange.length > 0 && (
        <Typography variant="caption" color="error.main">
          outside {spec.minimum} … {spec.maximum}: {outOfRange.join(', ')}
        </Typography>
      )}
      {unsorted && <Typography variant="caption" color="warning.main">kept in ascending order</Typography>}
    </Stack>
  )
}

function OptionList({ value, spec, onChange }: { value: unknown; spec: StructuredSpec; onChange: (v: unknown) => void }) {
  const options = spec.options ?? []
  if (spec.single) {
    // a stored value outside the vocabulary stays offered, so opening the dialog cannot drop it
    const current = typeof value === 'string' ? value : ''
    const shown = options.includes(current) || !current ? options : [...options, current]
    return (
      <Stack spacing={0.5} sx={{ width: '100%' }}>
        <Select size="small" fullWidth value={current} onChange={e => onChange(e.target.value)}>
          {shown.map(o => <MenuItem key={o} value={o}>{o}</MenuItem>)}
        </Select>
        {current && !options.includes(current) && (
          <Typography variant="caption" color="error.main">not implemented: {current}</Typography>
        )}
      </Stack>
    )
  }
  const selected = Array.isArray(value) ? value.map(String) : value ? [String(value)] : []
  const unknown = selected.filter(v => !options.includes(v))
  return (
    <Stack spacing={0.5} sx={{ width: '100%' }}>
      <Autocomplete multiple size="small" options={options} value={selected}
        onChange={(_, items) => onChange(items as string[])}
        renderTags={(tags, getTagProps) => tags.map((t, i) => {
          const { key, ...rest } = getTagProps({ index: i })
          return <Chip key={key} size="small" label={t} color={options.includes(t) ? 'default' : 'error'} {...rest} />
        })}
        renderInput={p => <TextField {...p} placeholder={selected.length ? '' : 'pick one or more'} />} />
      {unknown.length > 0 && (
        <Typography variant="caption" color="error.main">not implemented: {unknown.join(', ')}</Typography>
      )}
    </Stack>
  )
}

function NumberMap({ value, spec, onChange }: { value: unknown; spec: StructuredSpec; onChange: (v: unknown) => void }) {
  const entries = useMemo(
    () => (value && typeof value === 'object' && !Array.isArray(value) ? Object.entries(value as Document) : []),
    [value])
  const commit = (next: [string, unknown][]) =>
    onChange(Object.fromEntries(next.filter(([k]) => String(k).trim())))
  const missing = (spec.keys ?? []).filter(k => !entries.some(([key]) => key === k))
  return (
    <Stack spacing={0.5} sx={{ width: '100%' }}>
      {entries.map(([key, v], i) => (
        <Stack key={i} direction="row" spacing={1} alignItems="center">
          <Autocomplete freeSolo size="small" sx={{ width: 170 }} options={spec.keys ?? []} value={key}
            onChange={(_, picked) => commit(entries.map((e, j) => (j === i ? [String(picked ?? ''), e[1]] : e)))}
            onInputChange={(_, text, reason) => { if (reason === 'input') commit(entries.map((e, j) => (j === i ? [text, e[1]] : e))) }}
            renderInput={p => <TextField {...p} placeholder="name" />} />
          <TextField size="small" type="number" sx={{ width: 120 }} value={v ?? ''}
                     onChange={e => commit(entries.map((entry, j) => (j === i ? [entry[0], e.target.value === '' ? null : Number(e.target.value)] : entry)))} />
          <IconButton size="small" onClick={() => commit(entries.filter((_, j) => j !== i))}>
            <DeleteOutlineIcon fontSize="small" />
          </IconButton>
        </Stack>
      ))}
      <Box>
        {missing.map(k => (
          <Chip key={k} size="small" variant="outlined" sx={{ mr: 0.5 }} label={`+ ${k}`}
                onClick={() => commit([...entries, [k, null]])} />
        ))}
        <Button size="small" startIcon={<AddIcon />} onClick={() => commit([...entries, ['', null]])}>Add</Button>
      </Box>
    </Stack>
  )
}

export default function StructuredParamField({ value, spec, family, document, onChange }: {
  value: unknown; spec: StructuredSpec; family: string; document: Document; onChange: (v: unknown) => void
}) {
  return (
    <Stack spacing={0.5} sx={{ width: '100%' }}>
      {spec.kind === 'number_list' && <NumberList value={value} spec={spec} onChange={onChange} />}
      {spec.kind === 'option_list' && <OptionList value={value} spec={spec} onChange={onChange} />}
      {spec.kind === 'number_map' && <NumberMap value={value} spec={spec} onChange={onChange} />}
      {spec.kind === 'rbf_var_imp' && (
        <RbfVarImpField value={value} document={document} flat={family !== 'clustering'} onChange={onChange} />
      )}
      {spec.help && spec.kind !== 'rbf_var_imp' && (
        <Typography variant="caption" color="text.secondary">{spec.help}</Typography>
      )}
    </Stack>
  )
}
