// A13-04: what a colleague sees before starting a long run.
//
// Three outcomes and nothing else: ready, needs attention, cannot start. Each finding names the
// field it belongs to, what was checked, what was found, what follows from it and one thing to
// do. There is deliberately no evidence taxonomy, no storage forecast and no partition or
// scaler dashboard here -- those belong to the maintainer's reports.
//
// The verdict carries the document and runtime it describes. When the document has moved on,
// the panel says so and offers a re-check instead of showing a stale answer as current: a
// timestamp alone cannot authorise a verdict at submission.
import { useCallback, useRef, useState } from 'react'
import {
  Alert, AlertTitle, Box, Button, Chip, CircularProgress, Collapse, Divider,
  List, ListItem, ListItemText, Stack, Typography,
} from '@mui/material'
import { api, ApiError, Document, Readiness, ReadinessFinding, Severity, Verdict } from '../api/client'

const TONE: Record<Verdict, 'success' | 'warning' | 'error'> = {
  'ready': 'success', 'needs attention': 'warning', 'cannot start': 'error',
}
const HEADLINE: Record<Verdict, string> = {
  'ready': 'Ready to start',
  'needs attention': 'Needs attention',
  'cannot start': 'Cannot start',
}
const SEVERITY_TONE: Record<Severity, 'success' | 'warning' | 'error'> = {
  ok: 'success', attention: 'warning', blocking: 'error',
}
const SEVERITY_LABEL: Record<Severity, string> = {
  ok: 'checked', attention: 'needs attention', blocking: 'blocks the run',
}
const ORDER: Severity[] = ['blocking', 'attention', 'ok']

function Finding({ finding }: { finding: ReadinessFinding }) {
  return (
    <ListItem disableGutters alignItems="flex-start" sx={{ py: 0.5 }}
              data-testid={`readiness-finding-${finding.check}`} data-severity={finding.severity}>
      <ListItemText
        primary={
          <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap>
            <Chip size="small" label={SEVERITY_LABEL[finding.severity]} color={SEVERITY_TONE[finding.severity]}
                  variant={finding.severity === 'ok' ? 'outlined' : 'filled'} />
            <Typography component="span" variant="body2">{finding.finding}</Typography>
            {finding.field && (
              <Typography component="code" variant="caption" data-testid="readiness-field"
                          sx={{ color: 'text.secondary' }}>{finding.field}</Typography>
            )}
          </Stack>
        }
        secondary={
          <Box component="span" sx={{ display: 'block' }}>
            <Typography component="span" variant="caption" sx={{ display: 'block' }}>
              Checked: {finding.checked}
            </Typography>
            {finding.severity !== 'ok' && (
              <>
                <Typography component="span" variant="caption" sx={{ display: 'block' }}>
                  What follows: {finding.consequence}
                </Typography>
                <Typography component="span" variant="caption" sx={{ display: 'block', fontWeight: 600 }}>
                  What to do: {finding.remedy}
                </Typography>
              </>
            )}
          </Box>
        }
      />
    </ListItem>
  )
}

export default function ReadinessPanel(
  { document, path, caseName, disabled }: { document: Document | null; path?: string; caseName?: string; disabled?: boolean },
) {
  const [report, setReport] = useState<Readiness | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showChecked, setShowChecked] = useState(false)
  // Only the newest request may publish: a slow earlier answer must never overwrite a newer one.
  const latest = useRef(0)

  const check = useCallback(async () => {
    if (!document) return
    const ticket = ++latest.current
    setBusy(true)
    setError(null)
    try {
      const result = await api.readiness(document, path, caseName)
      if (ticket !== latest.current) return
      setReport(result)
    } catch (caught) {
      if (ticket !== latest.current) return
      if (caught instanceof DOMException && caught.name === 'AbortError') return
      setReport(null)
      setError(caught instanceof ApiError ? caught.message : String(caught))
    } finally {
      if (ticket === latest.current) setBusy(false)
    }
  }, [document, path, caseName])

  // An edit does not silently invalidate the verdict on screen, and it does not silently keep it
  // either: the panel remembers the document the answer was computed for and says when the case
  // has moved on. Re-checking is then the explicit next step.
  const current = document ? JSON.stringify(document) : null
  const [answered, setAnswered] = useState<string | null>(null)
  const outdated = report !== null && answered !== null && current !== answered

  const onCheck = () => { setAnswered(current); void check() }

  if (!document) return null
  return (
    <Box data-testid="readiness-panel">
      <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1 }} flexWrap="wrap" useFlexGap>
        <Button variant="outlined" size="small" onClick={onCheck} disabled={busy || !!disabled}
                data-testid="readiness-check">
          {report || error ? 'Re-check' : 'Check readiness'}
        </Button>
        {busy && <CircularProgress size={16} data-testid="readiness-busy" />}
        {report && (
          <Typography variant="caption" color="text.secondary" data-testid="readiness-checked-at">
            Checked at {report.checked_at}
          </Typography>
        )}
      </Stack>

      {error && (
        <Alert severity="error" data-testid="readiness-error">
          <AlertTitle>The readiness check could not run</AlertTitle>
          {error}
          <Typography variant="caption" sx={{ display: 'block', mt: 0.5 }}>
            This is not a verdict about the case. Re-check, or ask the maintainer.
          </Typography>
        </Alert>
      )}

      {report && (
        <Alert severity={TONE[report.verdict]} data-testid="readiness-verdict"
               data-verdict={report.verdict}>
          <AlertTitle>{HEADLINE[report.verdict]}</AlertTitle>
          {outdated && (
            <Typography variant="body2" sx={{ fontWeight: 600, mb: 1 }} data-testid="readiness-stale">
              The case has changed since this check. Re-check before starting.
            </Typography>
          )}
          {report.coverage && (
            <Typography variant="body2" data-testid="readiness-coverage">
              Target series: {report.coverage.rows} rows, {report.coverage.first} to {report.coverage.last}
              {report.coverage.truncated && ` (only the first ${report.bounded.max_index_rows} rows were read)`}
            </Typography>
          )}
          <List dense disablePadding>
            {ORDER.filter(s => s !== 'ok').map(severity =>
              report.findings.filter(f => f.severity === severity)
                .map(f => <Finding key={f.check + f.finding} finding={f} />))}
          </List>
          {report.counts.ok > 0 && (
            <>
              <Divider sx={{ my: 1 }} />
              <Button size="small" onClick={() => setShowChecked(v => !v)} data-testid="readiness-toggle-ok">
                {showChecked ? 'Hide' : 'Show'} the {report.counts.ok} checks that passed
              </Button>
              <Collapse in={showChecked}>
                <List dense disablePadding data-testid="readiness-ok-list">
                  {report.findings.filter(f => f.severity === 'ok')
                    .map(f => <Finding key={f.check + f.finding} finding={f} />)}
                </List>
              </Collapse>
            </>
          )}
        </Alert>
      )}

      {!report && !error && !busy && (
        <Typography variant="body2" color="text.secondary" data-testid="readiness-idle">
          Nothing is checked yet. The check reads the first rows of the target file, lists the weather
          archive and looks at the configured runtime. It trains nothing and changes nothing.
        </Typography>
      )}
    </Box>
  )
}
