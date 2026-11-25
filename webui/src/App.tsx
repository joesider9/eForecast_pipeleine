import { lazy, Suspense, useCallback, useEffect, useRef, useState } from 'react'
import { Navigate, Route, Routes, useLocation, useNavigate } from 'react-router-dom'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { Alert, AppBar, Box, Button, CircularProgress, Dialog, DialogContent, DialogTitle, Toolbar, Typography } from '@mui/material'
import { advanceSession, api, isAbort, type Me } from './api/client'
import HealthChip from './components/HealthChip'
import Login from './pages/Login'
import Cases from './pages/Cases'
import PasswordDialog from './components/PasswordDialog'
import PrivateSession from './session/PrivateSession'
import { SessionConfirmedContext } from './session/context'
import { LogoutGuardContext, type LogoutGuard } from './session/logoutGuard'

const CaseEditor = lazy(() => import('./pages/CaseEditor'))
const Users = lazy(() => import('./pages/Users'))
const sessionKey = (me: Me | undefined) => JSON.stringify([me?.user, me?.auth, me?.role])

// Routes: /login, / (the cases), /cases/:name (the editor).  The session comes from /api/me; a 401 raised by
// any call sends the user to the login page (event fired by the API client).
export default function App() {
  const navigate = useNavigate()
  const location = useLocation()
  const queryClient = useQueryClient()
  const me = useQuery({ queryKey: ['me'], queryFn: ({ signal }) => api.me(signal), refetchOnWindowFocus: true })
  const [busy, setBusy] = useState(false)
  const [password, setPassword] = useState(false)
  const [authError, setAuthError] = useState<string | null>(null)
  const [sendingLogout, setSendingLogout] = useState(false)
  const logoutRunning = useRef(false)
  const logoutGuard = useRef<LogoutGuard | null>(null)
  const registerLogoutGuard = useCallback((guard: LogoutGuard) => {
    logoutGuard.current = guard
    return () => { if (logoutGuard.current === guard) logoutGuard.current = null }
  }, [])

  useEffect(() => {
    const onUnauthorized = () => {
      advanceSession(); void queryClient.cancelQueries()
      queryClient.setQueryData(['me'], { user: null, auth: true, role: null })
      setPassword(false); navigate('/login')
    }
    const onSession = (event: Event) => {
      void queryClient.cancelQueries()
      queryClient.setQueryData(['me'], (event as CustomEvent<Me>).detail)
      setAuthError(null); setPassword(false); navigate('/')
    }
    window.addEventListener('eforecast:unauthorized', onUnauthorized)
    window.addEventListener('eforecast:session', onSession)
    return () => {
      window.removeEventListener('eforecast:unauthorized', onUnauthorized)
      window.removeEventListener('eforecast:session', onSession)
    }
  }, [navigate, queryClient])

  if (me.isLoading) return <Box sx={{ p: 6, textAlign: 'center' }}><CircularProgress /></Box>
  // A background failure keeps the last confirmed identity mounted. Returning
  // the error page here would dispose of its private cache and unsaved drafts.
  if (me.isError && !me.data) return <Alert severity="error" action={<Button onClick={() => me.refetch()}>Retry</Button>}>
    Could not check the current session: {String(me.error)}
  </Alert>
  const user = me.data?.user ?? null
  const loggedIn = !!user

  const logout = async () => {
    if (logoutRunning.current) return
    const actor = sessionKey(queryClient.getQueryData<Me>(['me']))
    const sameSession = () => sessionKey(queryClient.getQueryData<Me>(['me'])) === actor
    logoutRunning.current = true
    setBusy(true)
    setAuthError(null)
    try {
      if (logoutGuard.current && !await logoutGuard.current()) return
      if (!sameSession()) return
      setSendingLogout(true)
      await api.logout()
      if (!sameSession()) return
      advanceSession(); await queryClient.cancelQueries()
      if (!sameSession()) return
      queryClient.setQueryData(['me'], { user: null, auth: true, role: null })
      setPassword(false); navigate('/login')
    } catch (error) { if (sameSession() && !isAbort(error)) setAuthError(`Log out failed: ${String(error)}. Your draft is retained.`) }
    finally { logoutRunning.current = false; setBusy(false); setSendingLogout(false) }
  }

  return (
    <SessionConfirmedContext.Provider value={!me.isError}>
    <LogoutGuardContext.Provider value={registerLogoutGuard}>
    <Box sx={{ minHeight: '100%', display: 'flex', flexDirection: 'column' }}>
      <AppBar position="static" color="primary" elevation={0}>
        <Toolbar variant="dense" sx={{ gap: 2 }}>
          <Button color="inherit" sx={{ fontWeight: 600 }} onClick={() => navigate('/')}>
            eforecast configuration
          </Button>
          <Box sx={{ flex: 1 }} />
          {/* Whether the installation is working, next to who you are: the two questions a
              colleague asks before anything else, in the one place every page has. */}
          {loggedIn && <HealthChip />}
          {loggedIn && me.data?.auth && (
            <>
              <Typography variant="body2">{user}{me.data?.role === 'admin' ? ' (admin)' : ''}</Typography>
              {me.data?.role === 'admin' && <Button color="inherit" size="small" onClick={() => navigate('/users')}>Users</Button>}
              <Button color="inherit" size="small" onClick={() => setPassword(true)}>Password</Button>
              <Button color="inherit" size="small" disabled={busy} onClick={logout}>Log out</Button>
            </>
          )}
        </Toolbar>
      </AppBar>
      <Box sx={{ flex: 1, p: 2, maxWidth: 1400, width: '100%', mx: 'auto', boxSizing: 'border-box' }}>
        {authError && <Alert severity="error" onClose={() => setAuthError(null)}>{authError}</Alert>}
        {me.isError && <Alert severity="warning" sx={{ mb: 2 }}
          action={<Button disabled={me.isFetching} onClick={() => me.refetch()}>Retry session check</Button>}>
          Your session could not be rechecked. Your draft is retained and you can keep editing.
          Saving and other changes on the server are paused until the session check succeeds.
        </Alert>}
        <PrivateSession key={`${user}:${me.data?.auth}:${me.data?.role}`} identity={`${user ?? 'anonymous'}:${me.data?.auth}`}>
        <Suspense fallback={<Typography>Loading page…</Typography>}>
        <Routes>
          <Route path="/login" element={loggedIn ? <Navigate to="/" replace /> : <Login />} />
          <Route path="/" element={loggedIn ? <Cases /> : <Navigate to="/login" replace state={{ from: location }} />} />
          <Route path="/cases/:name" element={loggedIn ? <CaseEditor /> : <Navigate to="/login" replace />} />
          <Route path="/users" element={loggedIn && me.data?.role === 'admin' ? <Users me={me.data} /> : <Navigate to="/" replace />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
        </Suspense>
        </PrivateSession>
      </Box>
      {user && <PasswordDialog key={user} user={user} open={password} onClose={() => setPassword(false)} />}
      <Dialog open={sendingLogout}>
        <DialogTitle>Logging out</DialogTitle>
        <DialogContent><CircularProgress size={20} aria-label="Logging out" /></DialogContent>
      </Dialog>
    </Box>
    </LogoutGuardContext.Provider>
    </SessionConfirmedContext.Provider>
  )
}
