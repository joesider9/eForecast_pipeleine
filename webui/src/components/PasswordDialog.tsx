import { useState } from 'react'
import { Alert, Button, Dialog, DialogActions, DialogContent, DialogTitle, Stack, TextField } from '@mui/material'
import { api, ApiError } from '../api/client'
import { useSessionConfirmed } from '../session/context'

export default function PasswordDialog({ user, open, onClose }: { user: string; open: boolean; onClose: () => void }) {
  const sessionConfirmed = useSessionConfirmed()
  const [current, setCurrent] = useState(''); const [next, setNext] = useState(''); const [again, setAgain] = useState('')
  const [error, setError] = useState<string | null>(null); const [done, setDone] = useState(false)
  const submit = async () => {
    if (next !== again) { setError('The two passwords differ.'); return }
    try { await api.setPassword(user, next, current); setDone(true); setError(null); setTimeout(onClose, 800) } catch (e) { setError(e instanceof ApiError ? e.message : String(e)) }
  }
  return (
    <Dialog open={open} onClose={onClose} fullWidth maxWidth="xs">
      <DialogTitle>Change password</DialogTitle>
      <DialogContent>
        <Stack spacing={2} sx={{ mt: 1 }}>
          <TextField label="Current password" type="password" value={current} onChange={e => setCurrent(e.target.value)} autoFocus />
          <TextField label="New password" type="password" value={next} onChange={e => setNext(e.target.value)} helperText="at least 6 characters" />
          <TextField label="New password again" type="password" value={again} onChange={e => setAgain(e.target.value)} />
          {error && <Alert severity="error">{error}</Alert>}
          {done && <Alert severity="success">Password changed.</Alert>}
        </Stack>
      </DialogContent>
      <DialogActions><Button onClick={onClose}>Cancel</Button><Button variant="contained" disabled={!sessionConfirmed || next.length < 6} onClick={submit}>Save</Button></DialogActions>
    </Dialog>
  )
}
