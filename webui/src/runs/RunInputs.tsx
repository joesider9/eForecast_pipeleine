import { Alert, Box, Stack, Typography } from '@mui/material'
import type { Run } from '../api/client'

export default function RunInputs({ run }: { run: Run }) {
  const pinned = run.input_snapshot_status === 'pinned' && !!run.effective_revision
  const degraded = run.input_snapshot_status === 'degraded' || run.config_frozen === false ||
    !!run.config_error || !!run.normalized_error
  const reason = run.normalized_error || run.config_error
  const preparing = run.state === 'pending' || run.state === 'preparing' || run.state === 'queued'
  return <Stack role="group" aria-label="Run input provenance" spacing={0.5} sx={{ minWidth: 230, maxWidth: 420, overflowWrap: 'anywhere' }}>
    {preparing ? <Typography variant="caption">Preparing accepted inputs.</Typography> :
      pinned ? <Typography variant="caption" color="success.main">Inputs pinned</Typography> :
      degraded ? <Alert severity="warning">{run.submission_key ? 'Degraded inputs: accepted raw configuration; effective inputs unverified.' : 'Degraded inputs: using live configuration.'}</Alert> :
        <Typography variant="caption" color="text.secondary">{run.config_frozen ?
          'Raw files archived; effective inputs unverified.' : 'Input provenance unavailable for this older run.'}</Typography>}
    {run.effective_revision && <Typography variant="caption">Effective revision: {run.effective_revision}</Typography>}
    {run.config_revision && <Typography variant="caption">Raw case revision: {run.config_revision}</Typography>}
    {run.preparation_error && <Alert severity="error">{run.preparation_error}</Alert>}
    {(reason || run.effective_options) && <Box component="details">
      <Box component="summary" sx={{ cursor: 'pointer', fontSize: 12 }}>Input details</Box>
      {reason && <Typography variant="caption" color="warning.main" sx={{ whiteSpace: 'pre-wrap' }}>{reason}</Typography>}
      {run.effective_options && <Box component="pre" sx={{ m: 0, fontSize: 11, whiteSpace: 'pre-wrap' }}>
        {JSON.stringify(run.effective_options, null, 2)}
      </Box>}
    </Box>}
  </Stack>
}
