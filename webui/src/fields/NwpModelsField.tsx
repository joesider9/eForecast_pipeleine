import { useState } from 'react'
import { FieldProps } from '@rjsf/utils'
import { Autocomplete, Box, Button, IconButton, MenuItem, Stack, Table, TableBody, TableCell, TableHead, TableRow, TextField, Typography } from '@mui/material'
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutline'
import FolderOpenIcon from '@mui/icons-material/FolderOpen'
import AddIcon from '@mui/icons-material/Add'
import { FormContext } from './context'
import { BrowseDialog } from './PathField'

interface NwpModel {
  model: string; resolution?: number | null; levels?: number | null
  path_nwp_source?: string | null; path_nwp_gdrive?: string | null
}

// The NWP providers of a project: model, resolution and levels (defaults filled by the schema when empty)
// and where the source files come from. A provider whose archive is kept in a shared Google Drive folder
// can read from there: a day that is not in the local folder is then downloaded when extract runs,
// converted to its daily pickle, and the downloaded files are removed.
export default function NwpModelsField(props: FieldProps) {
  const { formData, onChange, schema, formContext } = props
  const ctx = formContext as FormContext
  const rows: NwpModel[] = Array.isArray(formData) ? formData : []
  const known = ctx?.options?.nwp_models ?? ['ecmwf', 'skiron', 'gfs']
  const drives = ctx?.options?.nwp_gdrive ?? {}
  const resolutions = (ctx?.options?.nwp_resolutions ?? [0.25, 0.1, 0.05]).map(String)
  const levels = (ctx?.options?.nwp_levels ?? [2, 4, 8]).map(String)
  const num = (t: string) => { const v = Number(t); return t.trim() === '' || Number.isNaN(v) ? null : v }
  const [browsing, setBrowsing] = useState<number | null>(null)
  const update = (i: number, patch: Partial<NwpModel>) => onChange(rows.map((r, j) => (j === i ? { ...r, ...patch } : r)))
  const remove = (i: number) => onChange(rows.filter((_, j) => j !== i))
  const add = () => onChange([...rows, { model: known.find(m => !rows.some(r => r.model === m)) ?? 'ecmwf' }])
  // A row reads from Google Drive when it names a folder there, even one being typed. Going back to
  // the local folder takes the key out of the row instead of leaving a null behind in the case file.
  const fromDrive = (r: NwpModel) => r.path_nwp_gdrive !== undefined && r.path_nwp_gdrive !== null
  const local = (r: NwpModel): NwpModel => { const { path_nwp_gdrive: _unused, ...rest } = r; return rest }
  const setSource = (i: number, source: string) =>
    onChange(rows.map((r, j) => (j !== i ? r : source === 'gdrive' ? { ...r, path_nwp_gdrive: drives[r.model] ?? '' } : local(r))))
  // Only a provider with a shared folder downloads; a row moved to another provider keeps none.
  const setModel = (i: number, model: string) =>
    onChange(rows.map((r, j) => (j !== i ? r : { ...(drives[model] === undefined ? local(r) : r), model })))
  return (
    <Box sx={{ mb: 1.5 }}>
      <Stack direction="row" alignItems="center" spacing={1}>
        <Typography variant="caption" sx={{ color: 'text.secondary', flex: 1 }}>{schema.title ?? 'NWP models'}</Typography>
        <Button size="small" startIcon={<AddIcon />} onClick={add}>Add model</Button>
      </Stack>
      {rows.length > 0 && (
        <Table size="small">
          <TableHead><TableRow><TableCell>model</TableCell><TableCell>resolution (°)</TableCell><TableCell>levels (grid = 2·levels+1)</TableCell><TableCell>source</TableCell><TableCell /></TableRow></TableHead>
          <TableBody>
            {rows.map((r, i) => (
              <TableRow key={i}>
                <TableCell sx={{ width: 140, verticalAlign: 'top' }}>
                  <TextField select size="small" fullWidth value={r.model} onChange={e => setModel(i, e.target.value)}>
                    {[...known, ...(known.includes(r.model) ? [] : [r.model])].map(m => <MenuItem key={m} value={m}>{m}</MenuItem>)}
                  </TextField>
                </TableCell>
                <TableCell sx={{ width: 140, verticalAlign: 'top' }}><Autocomplete freeSolo size="small" options={resolutions} value={r.resolution === null || r.resolution === undefined ? '' : String(r.resolution)}
                  onChange={(_, v) => update(i, { resolution: num(String(v ?? '')) })} onInputChange={(_, v, reason) => { if (reason === 'input') update(i, { resolution: num(v) }) }}
                  renderInput={p => <TextField {...p} placeholder="default" />} /></TableCell>
                <TableCell sx={{ width: 120, verticalAlign: 'top' }}><Autocomplete freeSolo size="small" options={levels} value={r.levels === null || r.levels === undefined ? '' : String(r.levels)}
                  onChange={(_, v) => update(i, { levels: num(String(v ?? '')) })} onInputChange={(_, v, reason) => { if (reason === 'input') update(i, { levels: num(v) }) }}
                  renderInput={p => <TextField {...p} placeholder="default" />} /></TableCell>
                <TableCell>
                  <Stack spacing={1}>
                    <Stack direction="row" spacing={0.5}>
                      {(drives[r.model] !== undefined || fromDrive(r)) && (
                        <TextField select size="small" sx={{ width: 170, flexShrink: 0 }} value={fromDrive(r) ? 'gdrive' : 'folder'}
                          onChange={e => setSource(i, e.target.value)}
                          SelectProps={{ SelectDisplayProps: { 'aria-label': `${r.model} source` } }}>
                          <MenuItem value="folder">Local folder</MenuItem>
                          <MenuItem value="gdrive">Google Drive</MenuItem>
                        </TextField>
                      )}
                      <TextField size="small" fullWidth value={r.path_nwp_source ?? ''} onChange={e => update(i, { path_nwp_source: e.target.value || null })}
                        placeholder={fromDrive(r) ? 'folder the downloads pass through' : 'folder of the source files'}
                        inputProps={{ 'aria-label': `${r.model} folder` }} />
                      <IconButton size="small" aria-label={`browse for the ${r.model} folder`} onClick={() => setBrowsing(i)}><FolderOpenIcon fontSize="small" /></IconButton>
                    </Stack>
                    {fromDrive(r) && (
                      <TextField size="small" fullWidth label="Google Drive folder (link)" value={r.path_nwp_gdrive ?? ''}
                        onChange={e => update(i, { path_nwp_gdrive: e.target.value })}
                        placeholder="https://drive.google.com/drive/folders/…"
                        helperText="A day that is not in the folder above is downloaded from here when extract runs, converted to its daily pickle, and the downloaded files are removed. Days already in the folder are read from it and left alone." />
                    )}
                  </Stack>
                </TableCell>
                <TableCell align="right" sx={{ verticalAlign: 'top' }}><IconButton size="small" onClick={() => remove(i)}><DeleteOutlineIcon fontSize="small" /></IconButton></TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      )}
      <BrowseDialog open={browsing !== null} initial={browsing !== null ? String(rows[browsing]?.path_nwp_source ?? '') : ''} mode="dir"
        onClose={() => setBrowsing(null)} onPick={p => { if (browsing !== null) update(browsing, { path_nwp_source: p }); setBrowsing(null) }} />
    </Box>
  )
}
