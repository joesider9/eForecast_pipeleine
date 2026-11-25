import { Box, MenuItem, Stack, TextField, Typography } from '@mui/material'
import { resourceControlMetadata, type ResourceControls } from '../api/resourceContract.generated'

import { labels, numeric, type StudyDraft } from './studyResources'

export default function StudyControls({ value, onChange, saved }: {
  value: StudyDraft; onChange: (value: StudyDraft) => void; saved: Record<string, unknown>
}) {
  const inherited = (key: keyof ResourceControls) => {
    const value = saved[key] ?? resourceControlMetadata[key].default
    return value == null ? 'automatic' : String(value)
  }
  return <Box component="fieldset" sx={{ border: 1, borderColor: 'divider', borderRadius: 1, m: 0, p: 1.5, minWidth: 0 }}>
    <Typography component="legend" variant="body2">Study resources · optional overrides</Typography>
    <Stack spacing={1.5}>
      <Typography variant="caption" color="text.secondary">
        The studies of a run use {inherited('n_cpus')} CPUs between them (saved case); trials within each study stay serial.
        Blank fields inherit the saved case. Automatic means as many studies at a time as there are, up to one per CPU,
        with the CPUs divided among the studies running.
      </Typography>
      <TextField select size="small" label="CPU scope" value={value.cpu_scope}
        onChange={e => onChange({ ...value, cpu_scope: e.target.value })}>
        <MenuItem value="">Saved case</MenuItem><MenuItem value="global">Global</MenuItem>
        <MenuItem value="clusters">Clusters</MenuItem><MenuItem value="both">Global and clusters</MenuItem>
      </TextField>
      <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 1.5 }}>
        {numeric.map(key => <TextField key={key} size="small" label={labels[key]} value={value[key]}
          onChange={e => onChange({ ...value, [key]: e.target.value })} inputProps={{ inputMode: 'numeric' }}
          placeholder={inherited(key)} helperText={`Saved: ${inherited(key)}`} title={resourceControlMetadata[key].description} />)}
        <TextField size="small" label="Foundation device" value={value.cpu_study_device}
          onChange={e => onChange({ ...value, cpu_study_device: e.target.value })}
          placeholder="cpu, auto, cuda:0 or GPU UUID" helperText={`Saved: ${saved.cpu_study_device ?? 'family setting'}`} />
      </Box>
      <Typography variant="caption" color="text.secondary">The device preference applies to foundation methods. Shallow estimators use CPU. A preference no visible card answers runs on the CPU.</Typography>
    </Stack>
  </Box>
}
