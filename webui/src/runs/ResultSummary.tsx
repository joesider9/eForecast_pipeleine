// A14-02: one page that answers "can I use this result?" without a data scientist in the room.
//
// Everything here is read from recorded artifacts by `eforecast/config/result_summary.py`;
// this file arranges it and adds no arithmetic of its own. Three things it is built to make
// impossible: presenting a selection score as accuracy, presenting a finished run as an
// approval, and presenting a failed attempt as a fresh forecast.
import { useQuery } from '@tanstack/react-query'
import {
  Alert, AlertTitle, Box, Chip, CircularProgress, Divider, Stack, Table, TableBody,
  TableCell, TableHead, TableRow, Tooltip, Typography,
} from '@mui/material'
import { api, ApiError, ResultSummary as Summary, ScoreBlock } from '../api/client'
import { VERDICT_LABEL, VERDICT_TONE, forecastLine, headlineLine, metric, roleBadge, when } from './resultSummary'

function Scores({ block }: { block: ScoreBlock }) {
  const columns = block.rows.length ? Object.keys(block.rows[0]) : []
  return (
    <Box data-testid={`score-${block.role}`} data-estimates={String(block.estimates_accuracy)}>
      <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap>
        <Typography variant="subtitle2">{block.label}</Typography>
        <Chip size="small" label={roleBadge(block)}
              color={block.estimates_accuracy ? 'success' : 'default'}
              variant={block.estimates_accuracy ? 'filled' : 'outlined'} />
        <Typography variant="caption" color="text.secondary">{block.period}</Typography>
      </Stack>
      <Typography variant="caption" sx={{ display: 'block', mb: 0.5 }}>{block.caption}</Typography>
      {block.missing && <Alert severity="info" sx={{ my: 0.5 }}>{block.missing}</Alert>}
      {!!block.rows.length && (
        <Box sx={{ overflowX: 'auto' }}>
          <Table size="small">
            <TableHead><TableRow>{columns.map(c => <TableCell key={c}>{c}</TableCell>)}</TableRow></TableHead>
            <TableBody>
              {block.rows.map((row, index) => (
                <TableRow key={index}>
                  {columns.map(c => <TableCell key={c}>{typeof row[c] === 'number' ? metric(row[c] as number) : String(row[c] ?? '')}</TableCell>)}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </Box>
      )}
      {block.source && <Typography variant="caption" color="text.secondary">{block.source}</Typography>}
    </Box>
  )
}

export default function ResultSummaryPanel({ caseName }: { caseName: string }) {
  const query = useQuery({
    queryKey: ['results', caseName],
    queryFn: ({ signal }) => api.results(caseName, signal),
  })
  if (query.isPending) return <Stack direction="row" spacing={1}><CircularProgress size={20} /><Typography>Reading the recorded results…</Typography></Stack>
  if (query.isError) {
    const error = query.error
    return <Alert severity="error" data-testid="results-error">
      <AlertTitle>The results could not be read</AlertTitle>
      {error instanceof ApiError ? error.message : String(error)}
    </Alert>
  }
  const data = query.data as Summary
  const forecast = forecastLine(data.forecast)
  const headline = headlineLine(data)
  return (
    <Stack spacing={2} data-testid="result-summary">
      <Alert severity={VERDICT_TONE[data.verdict.state]} data-testid="results-verdict"
             data-state={data.verdict.state}>
        <AlertTitle>{VERDICT_LABEL[data.verdict.state]}</AlertTitle>
        {data.verdict.sentence}
        {!!data.verdict.actions.length && (
          <Box component="ul" sx={{ m: 0, pl: 2 }}>
            {data.verdict.actions.map((action, index) => (
              <li key={index}><b>{action.what}</b> — {action.why}</li>
            ))}
          </Box>
        )}
      </Alert>

      <Box>
        <Typography variant="subtitle2">Recorded evaluation</Typography>
        {headline
          ? <>
              <Typography data-testid="results-headline">{headline.text}</Typography>
              {headline.note && <Alert severity="warning" sx={{ mt: 0.5 }} data-testid="headline-note">{headline.note}</Alert>}
            </>
          : <Typography color="text.secondary">No evaluation has been recorded for this case.</Typography>}
        <Stack direction="row" spacing={1} sx={{ mt: 0.5 }} flexWrap="wrap" useFlexGap>
          <Chip size="small" data-testid="model-state" label={
            data.model.state === 'approved'
              ? `approved version ${String(data.model.bundle ?? '').slice(0, 12)} (rev ${data.model.revision})`
              : data.model.state === 'experimental' ? 'experimental: nothing approved'
              : 'approval unknown'} color={data.model.state === 'approved' ? 'success' : 'default'} />
          <Chip size="small" variant="outlined" label={`metrics in ${data.units.metrics}`} />
          {!!data.units.rated && <Chip size="small" variant="outlined" label={`rated ${data.units.rated}`} />}
          {!!data.units.steps && <Chip size="small" variant="outlined" label={`${data.units.steps} horizon steps, ${data.units.resolution}`} />}
        </Stack>
        <Typography variant="caption" sx={{ display: 'block' }}>{data.model.detail}</Typography>
        {data.units.warning && <Alert severity="warning" sx={{ mt: 0.5 }}>{data.units.warning}</Alert>}
        {data.model.state === 'approved' && (
          <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
            approved by {data.model.approved_by} on {when(data.model.approved_at)}
            {data.model.note ? ` — ${data.model.note}` : ''}
          </Typography>
        )}
      </Box>

      <Divider />
      <Box>
        <Typography variant="subtitle2">The forecast colleagues are using</Typography>
        <Alert severity={forecast.tone} data-testid="forecast-line">{forecast.text}</Alert>
        {data.forecast.last_cycle && (
          <Typography variant="caption" color="text.secondary" data-testid="last-cycle">
            Last cycle on disk: issue {data.forecast.last_cycle.issue_time}, {data.forecast.last_cycle.state}
            {data.forecast.last_cycle.rows !== undefined ? `, ${data.forecast.last_cycle.rows} rows` : ''}
            {(data.forecast.last_cycle.warnings ?? []).length ? ` — ${(data.forecast.last_cycle.warnings ?? [])[0]}` : ''}
          </Typography>
        )}
      </Box>

      <Divider />
      <Stack spacing={1.5}>{data.scores.map(block => <Scores key={block.role} block={block} />)}</Stack>

      <Divider />
      <Box data-testid="results-gaps">
        <Typography variant="subtitle2">What is missing</Typography>
        {!data.coverage.recorded
          ? <Alert severity="info">{data.coverage.detail}</Alert>
          : data.coverage.worst.length
            ? <Table size="small">
                <TableHead><TableRow><TableCell>predictor</TableCell><TableCell>slice</TableCell><TableCell>requested</TableCell><TableCell>answered</TableCell><TableCell>missing</TableCell></TableRow></TableHead>
                <TableBody>{data.coverage.worst.map((row, index) => (
                  <TableRow key={index}><TableCell>{row.level}/{row.cluster}/{row.predictor}</TableCell><TableCell>{row.slice}</TableCell>
                    <TableCell>{row.requested}</TableCell><TableCell>{row.covered}</TableCell><TableCell>{row.missing}</TableCell></TableRow>))}
                </TableBody>
              </Table>
            : <Typography color="text.secondary">Every predictor answered every requested timestamp.</Typography>}
        {!!data.incomplete.unsupported_clusters.length && (
          <Typography variant="caption" sx={{ display: 'block' }}>
            {data.incomplete.unsupported_clusters.length} cluster(s) could not be ranked: {data.incomplete.unsupported_clusters[0].reason}
          </Typography>)}
        {!!data.incomplete.missing_artifacts.length && (
          <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
            not written: {data.incomplete.missing_artifacts.join(', ')}
          </Typography>)}
      </Box>

      {!!data.caveats.length && (
        <Alert severity="warning" data-testid="results-caveats">
          <AlertTitle>Known problem with this case's inputs</AlertTitle>
          {data.caveats.map(item => (
            <Box key={item.code}><b>{item.detail}.</b> {item.consequence}. <i>{item.remedy}</i></Box>
          ))}
        </Alert>
      )}

      <Box>
        <Typography variant="caption" color="text.secondary">
          {data.freshness.detail}. Every number above comes from these files:
        </Typography>
        <Box component="ul" sx={{ m: 0, pl: 2 }}>
          {Object.entries(data.sources).filter(([, info]) => info.exists).map(([name, info]) => (
            <li key={name}><Tooltip title={info.path}><Typography variant="caption">{name} — {when(info.modified)}</Typography></Tooltip></li>
          ))}
        </Box>
      </Box>
    </Stack>
  )
}
