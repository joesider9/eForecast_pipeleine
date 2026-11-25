import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Alert, Box, Button, Chip, CircularProgress, Dialog, DialogContent, DialogTitle, Link, Stack, Table, TableBody, TableCell, TableHead, TableRow, Tooltip, Typography } from '@mui/material'
import { api, ApiError, CsvTable, StudyRow } from '../api/client'

const fmt = (v: any) => typeof v === 'number' ? (Math.abs(v) >= 100 ? v.toFixed(1) : v.toFixed(4)) : String(v ?? '')

function Csv({ table, title }: { table: CsvTable | null; title: string }) {
  if (!table) return null
  return (
    <Box>
      <Typography variant="subtitle2">{title} <Typography component="span" variant="caption" color="text.secondary">{table.path}</Typography></Typography>
      <Box sx={{ overflowX: 'auto' }}>
        <Table size="small">
          <TableHead><TableRow>{table.columns.map((c, i) => <TableCell key={i}>{c}</TableCell>)}</TableRow></TableHead>
          <TableBody>{table.rows.map((r, i) => <TableRow key={i}>{r.map((c, j) => <TableCell key={j}>{c}</TableCell>)}</TableRow>)}</TableBody>
        </Table>
      </Box>
    </Box>
  )
}

function TrialsDialog({ caseName, row, onClose }: { caseName: string; row: StudyRow | null; onClose: () => void }) {
  const q = useQuery({ queryKey: ['trials', caseName, row?.cluster, row?.method], queryFn: ({ signal }) => api.trials(caseName, row!.cluster, row!.method, signal), enabled: !!row })
  const data = q.data
  return (
    <Dialog open={!!row} onClose={onClose} maxWidth="lg" fullWidth>
      <DialogTitle>{row?.cluster} / {row?.method} {data && <Typography component="span" variant="caption" color="text.secondary">{data.study}</Typography>}</DialogTitle>
      <DialogContent>
        {q.isLoading && <CircularProgress size={20} />}
        {q.isError && <Alert severity="error">{String((q.error as ApiError)?.message ?? q.error)}</Alert>}
        {data && (
          <Box sx={{ overflowX: 'auto' }}>
            <Table size="small" stickyHeader>
              <TableHead><TableRow><TableCell>#</TableCell><TableCell>state</TableCell><TableCell>value</TableCell><TableCell>duration</TableCell>{data.param_names.map(p => <TableCell key={p}>{p}</TableCell>)}</TableRow></TableHead>
              <TableBody>
                {[...data.trials].sort((a, b) => (a.value ?? Infinity) - (b.value ?? Infinity)).map(t => (
                  <TableRow key={t.number} hover selected={data.summary.best?.number === t.number}>
                    <TableCell>{t.number}</TableCell><TableCell><Chip size="small" label={t.state} color={t.state === 'COMPLETE' ? 'success' : t.state === 'RUNNING' ? 'info' : t.state === 'FAIL' ? 'error' : 'default'} /></TableCell>
                    <TableCell>{t.value === null ? '' : fmt(t.value)}</TableCell><TableCell>{t.duration_s === null ? '' : `${(t.duration_s / 60).toFixed(1)} min`}</TableCell>
                    {data.param_names.map(p => <TableCell key={p}>{fmt(t.params[p])}</TableCell>)}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Box>
        )}
      </DialogContent>
    </Dialog>
  )
}

// What the case has produced: dataset files, clusters, one row per Optuna study with trial counts and the
// best value, the selected trials of the combine stage, and the evaluation tables.
export default function StudiesPanel({ caseName }: { caseName: string }) {
  const q = useQuery({ queryKey: ['studies', caseName], queryFn: ({ signal }) => api.studies(caseName, true, signal), refetchInterval: 30000 })
  const [row, setRow] = useState<StudyRow | null>(null)
  if (q.isLoading) return <Stack direction="row" spacing={1} alignItems="center"><CircularProgress size={20} /><Typography variant="body2">Reading the studies…</Typography></Stack>
  if (q.isError) return <Alert severity="error">{String((q.error as ApiError)?.message ?? q.error)}</Alert>
  const d = q.data!
  return (
    <Stack spacing={2}>
      <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap>
        <Typography variant="subtitle1">{d.project}</Typography>
        <Chip size="small" label={d.model_exists ? 'model folder exists' : 'nothing produced yet'} color={d.model_exists ? 'success' : 'default'} />
        <Typography variant="caption" color="text.secondary">{d.path_model}</Typography>
        <Box sx={{ flex: 1 }} />
        <Link href={d.links.optuna_dashboard} target="_blank" rel="noreferrer" variant="body2">optuna-dashboard</Link>
        <Link href={d.links.grafana} target="_blank" rel="noreferrer" variant="body2">Grafana</Link>
        <Link href={d.links.metrics} target="_blank" rel="noreferrer" variant="body2">metrics</Link>
        <Button size="small" onClick={() => q.refetch()}>Refresh</Button>
      </Stack>

      <Box>
        <Typography variant="subtitle2">Datasets</Typography>
        <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap sx={{ mt: 0.5 }}>
          {Object.entries(d.datasets).map(([name, info]) => (
            <Tooltip key={name} title={info.path}><Chip size="small" label={info.size_mb === null ? `${name}: missing` : `${name}: ${info.size_mb} MB`} color={info.size_mb === null ? 'default' : 'success'} variant={info.size_mb === null ? 'outlined' : 'filled'} /></Tooltip>
          ))}
        </Stack>
      </Box>
      <Box>
        <Typography variant="subtitle2">Clusters</Typography>
        <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap sx={{ mt: 0.5 }}>
          {Object.entries(d.clusters).map(([name, info]) => <Tooltip key={name} title={info.folder}><Chip size="small" label={name} color={info.exists ? 'primary' : 'default'} variant={info.exists ? 'filled' : 'outlined'} /></Tooltip>)}
        </Stack>
        {Object.keys(d.clusters).length === 1 && <Typography variant="caption" color="text.secondary">clusters.pickle not found: run the clustering stage first.</Typography>}
      </Box>

      <Box>
        <Typography variant="subtitle2">Studies</Typography>
        {d.studies.length === 0 ? <Typography variant="body2" color="text.secondary">No study yet: training creates one per cluster and method.</Typography> : (
          <Box sx={{ overflowX: 'auto' }}>
            <Table size="small">
              <TableHead><TableRow><TableCell>cluster</TableCell><TableCell>method</TableCell><TableCell>trials</TableCell><TableCell>complete</TableCell><TableCell>running</TableCell><TableCell>pruned</TableCell><TableCell>failed</TableCell><TableCell>best value</TableCell><TableCell>best trial</TableCell><TableCell>selected</TableCell><TableCell>last activity</TableCell><TableCell /></TableRow></TableHead>
              <TableBody>
                {d.studies.map(s => (
                  <TableRow key={`${s.cluster}/${s.method}`} hover>
                    <TableCell>{s.cluster}</TableCell><TableCell>{s.method}</TableCell>
                    <TableCell>{s.summary?.n_trials ?? s.trial_folders}</TableCell>
                    <TableCell>{s.summary?.counts.COMPLETE ?? ''}</TableCell><TableCell>{s.summary?.counts.RUNNING || ''}</TableCell><TableCell>{s.summary?.counts.PRUNED || ''}</TableCell><TableCell>{s.summary?.counts.FAIL || ''}</TableCell>
                    <TableCell>{s.summary?.best ? fmt(s.summary.best.value) : ''}</TableCell>
                    <TableCell>{s.summary?.best ? <Tooltip title={JSON.stringify(s.summary.best.params)}><span>#{s.summary.best.number}</span></Tooltip> : ''}</TableCell>
                    <TableCell>{s.selected ? (Array.isArray(s.selected) ? s.selected.join(', ') : JSON.stringify(s.selected).slice(0, 40)) : ''}</TableCell>
                    <TableCell>{s.summary?.last_activity ? new Date(s.summary.last_activity).toLocaleString() : ''}</TableCell>
                    <TableCell>{s.error ? <Tooltip title={s.error}><Chip size="small" color="error" label="unreadable" /></Tooltip> : <Button size="small" onClick={() => setRow(s)}>trials</Button>}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Box>
        )}
      </Box>
      <Csv table={d.summary} title="Holdout summary" />
      <Csv table={d.intervals ?? null} title="Conformal intervals (holdout)" />
      <Csv table={d.evaluation} title="Evaluation" />
      <TrialsDialog caseName={caseName} row={row} onClose={() => setRow(null)} />
    </Stack>
  )
}
