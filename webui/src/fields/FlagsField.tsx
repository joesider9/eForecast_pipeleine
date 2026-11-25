import { useState } from 'react'
import { FieldProps } from '@rjsf/utils'
import { Box, Checkbox, Chip, FormControlLabel, FormGroup, IconButton, Stack, TextField, Typography } from '@mui/material'
import AddIcon from '@mui/icons-material/Add'
import { FormContext } from './context'

// A dict {name: bool} (methods_cpu, cluster_methods, global_methods) as a row of checkboxes; unknown
// names found in the case are kept, and a new one can be added.
export default function FlagsField(props: FieldProps) {
  const { formData, onChange, name, schema, formContext } = props
  const ctx = formContext as FormContext
  const value: Record<string, boolean> = formData ?? {}
  const known = ctx?.options?.flags?.[name] ?? []
  const keys = [...known, ...Object.keys(value).filter(k => !known.includes(k))]
  const [extra, setExtra] = useState('')
  const toggle = (key: string) => onChange({ ...value, [key]: !value[key] })
  const add = () => { const k = extra.trim(); if (k && !(k in value)) onChange({ ...value, [k]: true }); setExtra('') }
  return (
    <Box sx={{ mb: 1.5 }}>
      <Typography variant="caption" sx={{ display: 'block', color: 'text.secondary' }}>{schema.title ?? name}</Typography>
      <Stack direction="row" alignItems="center" flexWrap="wrap" spacing={1}>
        <FormGroup row>
          {keys.map(key => (
            <FormControlLabel key={key} control={<Checkbox size="small" checked={!!value[key]} onChange={() => toggle(key)} />}
              label={known.includes(key) ? key : <Chip size="small" label={key} variant="outlined" />} />
          ))}
        </FormGroup>
        <TextField size="small" placeholder="other…" value={extra} onChange={e => setExtra(e.target.value)} sx={{ width: 140 }}
          onKeyDown={e => { if (e.key === 'Enter') { e.preventDefault(); add() } }} />
        <IconButton size="small" onClick={add} aria-label="add flag"><AddIcon fontSize="small" /></IconButton>
      </Stack>
    </Box>
  )
}
