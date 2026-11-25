import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Alert, Button, Chip, Dialog, DialogActions, DialogContent, DialogTitle, FormControl, InputLabel, MenuItem, Paper, Select, Stack, Table, TableBody, TableCell, TableHead, TableRow, TextField, Typography } from '@mui/material'
import { api, ApiError, Me } from '../api/client'
import { useSessionConfirmed } from '../session/context'

// Administrators: the users file of the workspace (name, role), new users, roles, password resets.
export default function Users({ me }: { me: Me }) {
  const sessionConfirmed = useSessionConfirmed()
  const qc = useQueryClient()
  const users = useQuery({ queryKey: ['users'], queryFn: ({ signal }) => api.users(signal) })
  const [error, setError] = useState<string | null>(null)
  const [form, setForm] = useState<{ name: string; password: string; role: string } | null>(null)
  const [reset, setReset] = useState<{ name: string; password: string } | null>(null)
  const refresh = () => qc.invalidateQueries({ queryKey: ['users'] })
  const fail = (e: unknown) => setError(e instanceof ApiError ? e.message : String(e))
  const create = useMutation({ mutationFn: () => api.createUser(form!.name.trim(), form!.password, form!.role), onSuccess: () => { refresh(); setForm(null); setError(null) }, onError: fail })
  const role = useMutation({ mutationFn: ({ name, role }: { name: string; role: string }) => api.setRole(name, role), onSuccess: refresh, onError: fail })
  const remove = useMutation({ mutationFn: (name: string) => api.deleteUser(name), onSuccess: refresh, onError: fail })
  const password = useMutation({ mutationFn: () => api.setPassword(reset!.name, reset!.password), onSuccess: () => { setReset(null); setError(null) }, onError: fail })
  if (!me.auth) return <Alert severity="info">Authentication is off (EFORECAST_AUTH=off): a single local user, no accounts to manage. Start the server with authentication on to add users.</Alert>
  return (
    <Stack spacing={2}>
      <Stack direction="row" alignItems="center" spacing={2}>
        <Typography variant="h6" sx={{ flex: 1 }}>Users</Typography>
        <Button disabled={!sessionConfirmed} variant="contained" onClick={() => setForm({ name: '', password: '', role: 'user' })}>New user</Button>
      </Stack>
      <Typography variant="body2" color="text.secondary">
        Every user owns the cases created under their account and sees the cases others marked as shared, read-only.
        Administrators see and edit every case. The same accounts can be managed on the server with <code>python -m eforecast.config.auth</code>.
      </Typography>
      {error && <Alert severity="error" onClose={() => setError(null)}>{error}</Alert>}
      {users.isError && <Alert severity="error" action={<Button onClick={() => users.refetch()}>Retry</Button>}>Could not load users: {String(users.error)}</Alert>}
      <Paper>
        <Table size="small">
          <TableHead><TableRow><TableCell>User</TableCell><TableCell>Role</TableCell><TableCell align="right" /></TableRow></TableHead>
          <TableBody>
            {(users.data ?? []).map(u => (
              <TableRow key={u.name} hover>
                <TableCell><strong>{u.name}</strong> {u.name === me.user && <Chip size="small" label="you" />}</TableCell>
                <TableCell>
                  <FormControl size="small" sx={{ minWidth: 120 }}>
                    <Select value={u.role} disabled={!sessionConfirmed || u.name === me.user} onChange={e => role.mutate({ name: u.name, role: e.target.value })}>
                      <MenuItem value="user">user</MenuItem><MenuItem value="admin">admin</MenuItem>
                    </Select>
                  </FormControl>
                </TableCell>
                <TableCell align="right">
                  <Button disabled={!sessionConfirmed} size="small" onClick={() => setReset({ name: u.name, password: '' })}>Reset password</Button>
                  <Button size="small" color="error" disabled={!sessionConfirmed || u.name === me.user} onClick={() => remove.mutate(u.name)}>Remove</Button>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </Paper>
      <Dialog open={!!form} onClose={() => setForm(null)} fullWidth maxWidth="xs">
        <DialogTitle>New user</DialogTitle>
        <DialogContent>
          <Stack spacing={2} sx={{ mt: 1 }}>
            <TextField label="Name" value={form?.name ?? ''} onChange={e => setForm(f => f && { ...f, name: e.target.value })} autoFocus helperText="letters, digits, '_', '-' and '.'" />
            <TextField label="Initial password" type="password" value={form?.password ?? ''} onChange={e => setForm(f => f && { ...f, password: e.target.value })} helperText="at least 6 characters; the user can change it later" />
            <FormControl size="small"><InputLabel>Role</InputLabel><Select label="Role" value={form?.role ?? 'user'} onChange={e => setForm(f => f && { ...f, role: e.target.value })}><MenuItem value="user">user</MenuItem><MenuItem value="admin">admin</MenuItem></Select></FormControl>
          </Stack>
        </DialogContent>
        <DialogActions><Button onClick={() => setForm(null)}>Cancel</Button><Button variant="contained" disabled={!sessionConfirmed || !form?.name.trim() || (form?.password.length ?? 0) < 6 || create.isPending} onClick={() => create.mutate()}>Create</Button></DialogActions>
      </Dialog>
      <Dialog open={!!reset} onClose={() => setReset(null)} fullWidth maxWidth="xs">
        <DialogTitle>Reset the password of {reset?.name}</DialogTitle>
        <DialogContent><TextField sx={{ mt: 1 }} fullWidth label="New password" type="password" value={reset?.password ?? ''} onChange={e => setReset(r => r && { ...r, password: e.target.value })} autoFocus /></DialogContent>
        <DialogActions><Button onClick={() => setReset(null)}>Cancel</Button><Button variant="contained" disabled={!sessionConfirmed || (reset?.password.length ?? 0) < 6} onClick={() => password.mutate()}>Save</Button></DialogActions>
      </Dialog>
    </Stack>
  )
}
