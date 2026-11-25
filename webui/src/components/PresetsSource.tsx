import { useState } from 'react'
import { Button, IconButton, Paper, Stack, TextField, Typography } from '@mui/material'
import FolderOpenIcon from '@mui/icons-material/FolderOpen'
import { BrowseDialog } from '../fields/PathField'
import { DEFAULT_PRESETS_PATH } from '../fields/context'

// The case YAML the presets come from ("Default case" in the Experiments tab).  Remembered per case in
// the browser; the default is the repository's default.yaml link.
export default function PresetsSource({ value, onChange }: { value: string; onChange: (path: string) => void }) {
  const [open, setOpen] = useState(false)
  return (
    <Paper variant="outlined" sx={{ p: 1.5, mb: 2, bgcolor: '#f8fafc' }}>
      <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap>
        <Typography variant="subtitle2" sx={{ whiteSpace: 'nowrap' }}>Presets file</Typography>
        <TextField size="small" label="Presets file" value={value} onChange={e => onChange(e.target.value)} sx={{ flex: 1, minWidth: 320 }}
          helperText="the case YAML whose experiment templates are offered by Load presets → Default case" />
        <IconButton size="small" onClick={() => setOpen(true)} title="browse"><FolderOpenIcon fontSize="small" /></IconButton>
        <Button size="small" disabled={value === DEFAULT_PRESETS_PATH} onClick={() => onChange(DEFAULT_PRESETS_PATH)}>default.yaml</Button>
      </Stack>
      <BrowseDialog open={open} initial={value.replace(/\/[^/]*$/, '')} mode="file" onClose={() => setOpen(false)} onPick={p => { onChange(p); setOpen(false) }} />
    </Paper>
  )
}
