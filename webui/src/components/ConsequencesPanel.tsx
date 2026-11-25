// Review & save: what the draft would cost, asked before the Save button is pressed.
//
// The same answer the save itself gets when it needs accepting, available without trying to
// save. The answer is for one draft: once the draft has moved on, the panel says so and offers
// to ask again rather than leave an old answer standing as the current one.
import { useCallback, useRef, useState } from 'react'
import { Alert, Box, Button, CircularProgress, Stack, Typography } from '@mui/material'
import { api, ApiError, isAbort, type Consequences, type Document } from '../api/client'
import ChangeConsequences from './ChangeConsequences'

export default function ConsequencesPanel({ caseName, document, dirty, disabled }: {
  caseName: string; document: Document | null; dirty: boolean; disabled?: boolean
}) {
  const [report, setReport] = useState<Consequences | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [answered, setAnswered] = useState<string | null>(null)
  // Only the newest request may publish: a slow earlier answer must never overwrite a newer one.
  const latest = useRef(0)
  const current = document ? JSON.stringify(document) : null

  const ask = useCallback(async () => {
    if (!document) return
    const ticket = ++latest.current
    setBusy(true); setError(null); setAnswered(JSON.stringify(document))
    try {
      const result = await api.consequences(caseName, document)
      if (ticket === latest.current) setReport(result)
    } catch (caught) {
      if (ticket !== latest.current || isAbort(caught)) return
      setReport(null)
      setError(caught instanceof ApiError ? caught.message : String(caught))
    } finally {
      if (ticket === latest.current) setBusy(false)
    }
  }, [caseName, document])

  if (!document) return null
  const outdated = report !== null && answered !== current
  return (
    <Box data-testid="consequences-panel">
      <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1 }} flexWrap="wrap" useFlexGap>
        <Button variant="outlined" size="small" onClick={() => { void ask() }} disabled={busy || !!disabled}
          data-testid="consequences-check">
          {report || error ? 'Ask again' : dirty ? 'What does saving this draft cost?' : 'Is what is built this configuration\'s?'}
        </Button>
        {busy && <CircularProgress size={16} />}
        <Typography variant="caption" color="text.secondary">
          {dirty ? 'which datasets a later run builds again, which families it trains from the beginning'
            : 'a case builds one model, the model of one configuration: what was built from another is done again'}
        </Typography>
      </Stack>
      {error && <Alert severity="error">Could not work it out: {error}</Alert>}
      {outdated && <Alert severity="info" sx={{ mb: 1 }} data-testid="consequences-outdated">
        The draft has changed since this was worked out. Ask again for the draft as it is now.</Alert>}
      {report && <ChangeConsequences report={report} />}
    </Box>
  )
}
