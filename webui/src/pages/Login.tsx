import { FormEvent, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useQueryClient } from '@tanstack/react-query'
import { Alert, Box, Button, Paper, Stack, TextField, Typography } from '@mui/material'
import { advanceSession, api, ApiError } from '../api/client'

export default function Login() {
  const [name, setName] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  const submit = async (event: FormEvent) => {
    event.preventDefault()
    setBusy(true); setError(null)
    try {
      const result = await api.login(name, password)
      advanceSession()
      const me = await api.me()
      if (!me.user || me.user !== result.user) throw new Error('The signed-in session could not be verified. Please retry.')
      // Login lives under the private boundary. Notify the root auth query using
      // an event rather than putting /me into that identity's disposable cache.
      window.dispatchEvent(new CustomEvent('eforecast:session', { detail: me }))
      queryClient.clear()
      navigate('/')
    } catch (exc) {
      setError(exc instanceof ApiError && exc.status === 401 ? 'Wrong user name or password.' : String(exc))
    } finally {
      setBusy(false)
    }
  }

  return (
    <Box sx={{ display: 'flex', justifyContent: 'center', mt: 8 }}>
      <Paper component="form" onSubmit={submit} sx={{ p: 3, width: 360 }}>
        <Stack spacing={2}>
          <Typography variant="h6">Sign in</Typography>
          <TextField label="User" value={name} onChange={e => setName(e.target.value)} autoFocus required />
          <TextField label="Password" type="password" value={password} onChange={e => setPassword(e.target.value)} required />
          {error && <Alert severity="error">{error}</Alert>}
          <Button type="submit" variant="contained" disabled={busy}>Sign in</Button>
          <Typography variant="caption" color="text.secondary">
            Users are created on the server with <code>python -m eforecast.config.auth add &lt;name&gt;</code>.
          </Typography>
        </Stack>
      </Paper>
    </Box>
  )
}
