// §4.1: whether this installation is working, on the page rather than in a terminal.
//
// Before this, the doctor had no surface outside a shell -- and of the findings on the lab's
// own machine, 34 of 35 were addressed to the maintainer. So a colleague who met a refusal had
// no way to see whether the installation was the problem, and every question became a message.
//
// The chip is one word. Behind it are the findings, each with what was checked, what follows
// from it and the one thing to do -- the same sentences `bin/eforecast doctor` prints, because
// two wordings of one check are two checks as far as anybody reading them is concerned.
//
// Three things it deliberately does not do: block the page while a reading is taken (the
// reading is cached; a stale one is labelled, never hidden), go blank when the health check
// itself breaks (the last thing known, with the reason, is more useful than nothing), and show
// a colleague this machine's paths and ports (the server decides that, not this component).
import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  Alert, Box, Button, Chip, CircularProgress, Dialog, DialogActions, DialogContent,
  DialogTitle, Divider, List, ListItem, ListItemText, Stack, Tooltip, Typography,
} from '@mui/material'
import { api, type Health, type HealthFinding, type Severity, type Verdict } from '../api/client'
import { ageOf } from './health'

const TONE: Record<Verdict, 'success' | 'warning' | 'error'> = {
  'ready': 'success', 'needs attention': 'warning', 'cannot start': 'error',
}
const LABEL: Record<Verdict, string> = {
  'ready': 'Installation ready', 'needs attention': 'Needs attention', 'cannot start': 'Must be fixed',
}
const SEVERITY_TONE: Record<Severity, 'success' | 'warning' | 'error'> = {
  ok: 'success', attention: 'warning', blocking: 'error',
}
const SEVERITY_LABEL: Record<Severity, string> = {
  ok: 'checked', attention: 'needs attention', blocking: 'must be fixed',
}
const ORDER: Severity[] = ['blocking', 'attention', 'ok']
const WHO: Record<string, string> = { colleague: 'yours to do', maintainer: "the maintainer's" }

function Finding({ finding }: { finding: HealthFinding }) {
  return (
    <ListItem disableGutters alignItems="flex-start" sx={{ py: 0.5 }}
              data-testid={`health-finding-${finding.check}`} data-severity={finding.severity}>
      <ListItemText
        primary={
          <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap>
            <Chip size="small" label={SEVERITY_LABEL[finding.severity]} color={SEVERITY_TONE[finding.severity]}
                  variant={finding.severity === 'ok' ? 'outlined' : 'filled'} />
            <Typography component="code" variant="caption" sx={{ color: 'text.secondary' }}>
              {finding.check}
            </Typography>
            <Typography component="span" variant="body2">{finding.finding}</Typography>
            {finding.who && (
              <Typography component="span" variant="caption" sx={{ color: 'text.secondary' }}>
                ({WHO[finding.who] ?? finding.who})
              </Typography>
            )}
          </Stack>
        }
        secondary={finding.severity === 'ok' ? null : (
          <Box component="span" sx={{ display: 'block' }}>
            {finding.consequence && (
              <Typography component="span" variant="caption" sx={{ display: 'block' }}>
                What follows: {finding.consequence}
              </Typography>
            )}
            {finding.remedy && (
              <Typography component="span" variant="caption" sx={{ display: 'block', fontWeight: 600 }}>
                Do: {finding.remedy}
              </Typography>
            )}
          </Box>
        )}
      />
    </ListItem>
  )
}

export function HealthDetail({ health }: { health: Health }) {
  const groups = ORDER
    .map((severity) => [severity, health.findings.filter((item) => item.severity === severity)] as const)
    .filter(([, items]) => items.length > 0)
  return (
    <>
      {health.unavailable && (
        <Alert severity="warning" sx={{ mb: 2 }} data-testid="health-unavailable">
          {health.unavailable}
          {health.checked_at && ' The findings below are the last reading taken.'}
        </Alert>
      )}
      {health.stale && !health.unavailable && (
        <Alert severity="info" sx={{ mb: 2 }} data-testid="health-stale">
          This reading was taken {ageOf(health.age_s)}; it may have changed since.
        </Alert>
      )}
      {health.scope === 'colleague' && (health.maintainer_findings ?? 0) > 0 && (
        <Alert severity="info" sx={{ mb: 2 }} data-testid="health-maintainer-count">
          {health.maintainer_findings} other {health.maintainer_findings === 1 ? 'finding is' : 'findings are'} the
          maintainer's. Send them this page's address, or the support report from the run that failed.
        </Alert>
      )}
      {groups.length === 0 && !health.unavailable && (
        <Typography variant="body2" data-testid="health-nothing">
          Nothing needs your attention.
        </Typography>
      )}
      {groups.map(([severity, items]) => (
        <Box key={severity} sx={{ mb: 1 }}>
          <Typography variant="subtitle2">{SEVERITY_LABEL[severity]}</Typography>
          <List dense disablePadding>
            {items.map((finding, index) => <Finding key={`${finding.check}-${index}`} finding={finding} />)}
          </List>
          <Divider />
        </Box>
      ))}
      <Typography variant="caption" sx={{ color: 'text.secondary' }}>
        The same checks as <code>bin/eforecast doctor</code>. {health.examined} checked,
        read {ageOf(health.age_s)}.
      </Typography>
    </>
  )
}

// A reply that is not a reading -- a service of another version answering this address, a
// proxy's own page -- is the health check being out of reach, not a verdict. It used to be read
// as a reading, and the first field that was not there took the whole page down with it.
const isReading = (data: unknown): data is Health => {
  const reading = data as Partial<Health> | null
  return !!reading && typeof reading === 'object' && !Array.isArray(reading) &&
    !!reading.counts && typeof reading.counts === 'object' && Array.isArray(reading.findings)
}

export default function HealthChip({ caseName }: { caseName?: string }) {
  const [open, setOpen] = useState(false)
  // Polled rather than fetched once: a colleague leaves this page open all day, and the point
  // of the chip is that it is true now. The server serves a cached reading, so this is cheap.
  const health = useQuery({
    queryKey: ['health', caseName ?? null],
    queryFn: ({ signal }) => api.health(caseName ? { case: caseName } : undefined, signal),
    refetchInterval: 60_000,
    refetchOnWindowFocus: true,
  })

  if (health.isLoading) {
    return <CircularProgress size={16} color="inherit" data-testid="health-loading" />
  }
  if (health.isError || !isReading(health.data)) {
    // The installation may be fine and only this route unreachable, so the chip says exactly
    // that rather than claiming either verdict.
    return (
      <Tooltip title="The health check could not be reached. Runs and pages are unaffected.">
        <Chip size="small" label="Health unknown" color="default" variant="outlined"
              data-testid="health-chip" data-verdict="unknown" onClick={() => health.refetch()} />
      </Tooltip>
    )
  }
  const data = health.data
  const verdict = data.verdict
  const needing = (data.counts.attention ?? 0) + (data.counts.blocking ?? 0)
  const label = verdict ? LABEL[verdict] : 'Health unknown'
  const summary = verdict === 'ready' || needing === 0 ? label : `${label} (${needing})`
  return (
    <>
      <Tooltip title={`Checked ${ageOf(data.age_s)}. Click for what was checked and what to do.`}>
        <Chip size="small" label={summary} data-testid="health-chip"
              data-verdict={verdict ?? 'unknown'} data-stale={data.stale ? 'true' : 'false'}
              color={verdict ? TONE[verdict] : 'default'}
              variant={data.stale || !verdict ? 'outlined' : 'filled'}
              onClick={() => setOpen(true)} />
      </Tooltip>
      <Dialog open={open} onClose={() => setOpen(false)} fullWidth maxWidth="md">
        <DialogTitle>Is this installation working?</DialogTitle>
        <DialogContent dividers>
          <HealthDetail health={data} />
        </DialogContent>
        <DialogActions>
          {data.scope === 'maintainer' && (
            <Button disabled={health.isFetching} data-testid="health-recheck"
                    onClick={() => api.health({ case: caseName, refresh: true })
                      .catch(() => undefined)   // a refresh that fails is not an unhandled error
                      .then(() => health.refetch())}>
              Check again now
            </Button>
          )}
          <Button onClick={() => setOpen(false)} data-testid="health-close">Close</Button>
        </DialogActions>
      </Dialog>
    </>
  )
}
