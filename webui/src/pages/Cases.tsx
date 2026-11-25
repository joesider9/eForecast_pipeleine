import { useState } from 'react'
import { Link as RouterLink, useNavigate } from 'react-router-dom'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  Alert, Box, Button, Chip, Dialog, DialogActions, DialogContent, DialogTitle, FormControl, IconButton,
  InputLabel, MenuItem, Paper, Select, Stack, Table, TableBody, TableCell, TableHead, TableRow, TextField,
  Typography,
} from '@mui/material'
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutline'
import ShareIcon from '@mui/icons-material/Share'
import LockOutlinedIcon from '@mui/icons-material/LockOutlined'
import AddIcon from '@mui/icons-material/Add'
import { api, ApiError, CaseSummary } from '../api/client'
import { useSessionConfirmed } from '../session/context'

const when = (ts: number) => (ts ? new Date(ts * 1000).toLocaleString() : '')

export default function Cases() {
  const sessionConfirmed = useSessionConfirmed()
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const cases = useQuery({ queryKey: ['cases'], queryFn: ({ signal }) => api.cases(signal) })
  const [open, setOpen] = useState(false)
  const [name, setName] = useState('')
  const [source, setSource] = useState<string>('__defaults__')
  const [error, setError] = useState<string | null>(null)
  const [confirm, setConfirm] = useState<CaseSummary | null>(null)

  const create = useMutation({
    mutationFn: (input: { name: string; source: string }) => api.createCase(input.name, input.source === '__defaults__' ? {} : { copy_from: input.source }),
    onSuccess: (_, input) => { queryClient.invalidateQueries({ queryKey: ['cases'] }); setOpen(false); navigate(`/cases/${encodeURIComponent(input.name)}`) },
    onError: (exc) => setError(exc instanceof ApiError ? exc.message : String(exc)),
  })
  const share = useMutation({
    mutationFn: ({ n, shared }: { n: string; shared: boolean }) => api.shareCase(n, shared),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['cases'] }),
    onError: (exc) => setError(`Share failed: ${String(exc)}`),
  })
  const multiUser = (cases.data ?? []).some(c => c.owner)
  const remove = useMutation({
    mutationFn: (n: string) => api.deleteCase(n),
    onSuccess: () => { queryClient.invalidateQueries({ queryKey: ['cases'] }); setConfirm(null) },
    onError: (exc) => setError(`Delete failed: ${String(exc)}`),
  })

  return (
    <Stack spacing={2}>
      <Stack direction="row" alignItems="center" spacing={2}>
        <Typography variant="h6" sx={{ flex: 1 }}>Cases</Typography>
        <Button disabled={!sessionConfirmed} variant="contained" startIcon={<AddIcon />} onClick={() => { setName(''); setSource('__defaults__'); setError(null); setOpen(true) }}>
          New case
        </Button>
      </Stack>
      {cases.isError && <Alert severity="error">{String(cases.error)}</Alert>}
      {!open && error && <Alert severity="error" onClose={() => setError(null)}>{error}</Alert>}
      <Paper>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Case</TableCell><TableCell>Project</TableCell><TableCell>Type</TableCell>
              <TableCell>Horizon</TableCell><TableCell>State</TableCell>{multiUser && <TableCell>Owner</TableCell>}<TableCell>Modified</TableCell><TableCell />
            </TableRow>
          </TableHead>
          <TableBody>
            {(cases.data ?? []).map(c => (
              <TableRow key={c.name} hover sx={{ cursor: 'pointer' }} onClick={() => navigate(`/cases/${encodeURIComponent(c.name)}`)}>
                <TableCell><Button component={RouterLink} to={`/cases/${encodeURIComponent(c.name)}`} onClick={event => event.stopPropagation()}>{c.name}</Button></TableCell>
                <TableCell>{c.project ?? '—'}</TableCell>
                <TableCell>{c.type ?? '—'}</TableCell>
                <TableCell>{c.horizon_type ?? '—'}</TableCell>
                <TableCell>{c.complete ? <Chip size="small" color="success" label="complete" /> : <Chip size="small" color="warning" label={`parts: ${c.parts.join(', ') || 'none'}`} />}</TableCell>
                {multiUser && <TableCell>{c.owner ?? '—'} {c.shared && <Chip size="small" variant="outlined" label="shared" />}{!c.writable && <Chip size="small" icon={<LockOutlinedIcon />} label="read-only" sx={{ ml: 0.5 }} />}</TableCell>}
                <TableCell>{when(c.modified)}</TableCell>
                <TableCell align="right" onClick={e => e.stopPropagation()}>
                  {c.writable && c.owner && <IconButton disabled={!sessionConfirmed} size="small" color={c.shared ? 'primary' : 'default'} title={c.shared ? 'stop sharing' : 'share with the other users (read-only)'} onClick={() => share.mutate({ n: c.name, shared: !c.shared })}><ShareIcon fontSize="small" /></IconButton>}
                  {c.writable && <IconButton disabled={!sessionConfirmed} size="small" onClick={() => setConfirm(c)} aria-label={`delete ${c.name}`}><DeleteOutlineIcon fontSize="small" /></IconButton>}
                </TableCell>
              </TableRow>
            ))}
            {cases.data?.length === 0 && (
              <TableRow><TableCell colSpan={8}><Typography color="text.secondary" sx={{ py: 2 }}>No cases yet. Create one from the machine defaults or copy an existing case.</Typography></TableCell></TableRow>
            )}
          </TableBody>
        </Table>
      </Paper>

      <Dialog open={open} onClose={() => setOpen(false)} fullWidth maxWidth="xs">
        <DialogTitle>New case</DialogTitle>
        <DialogContent>
          <Stack spacing={2} sx={{ mt: 1 }}>
            <TextField label="Name" value={name} disabled={create.isPending} onChange={e => setName(e.target.value)} autoFocus
              helperText="letters, digits, '_', '-' and '.'" />
            <FormControl size="small">
              <InputLabel id="source">Start from</InputLabel>
              <Select labelId="source" label="Start from" value={source} onChange={e => setSource(e.target.value)}>
                <MenuItem value="__defaults__">Machine defaults (empty case)</MenuItem>
                {(cases.data ?? []).map(c => <MenuItem key={c.name} value={c.name}>Copy of {c.name}</MenuItem>)}
              </Select>
            </FormControl>
            {error && <Alert severity="error">{error}</Alert>}
          </Stack>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpen(false)}>Cancel</Button>
          <Button variant="contained" disabled={!sessionConfirmed || !name.trim() || create.isPending} onClick={() => create.mutate({ name: name.trim(), source })}>Create</Button>
        </DialogActions>
      </Dialog>

      <Dialog open={!!confirm} onClose={() => setConfirm(null)}>
        <DialogTitle>Delete case {confirm?.name}?</DialogTitle>
        <DialogContent><Typography>The case directory and its YAML files are removed. Trained models are not touched.</Typography></DialogContent>
        <DialogActions>
          <Button onClick={() => setConfirm(null)}>Cancel</Button>
          <Button disabled={!sessionConfirmed} color="error" variant="contained" onClick={() => confirm && remove.mutate(confirm.name)}>Delete</Button>
        </DialogActions>
      </Dialog>
      <Box />
    </Stack>
  )
}
