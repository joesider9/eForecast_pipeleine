import { useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Alert, Autocomplete, Box, Button, Card, CardActions, CardContent, Checkbox, Chip, Collapse, Dialog, DialogActions, DialogContent, DialogContentText, DialogTitle, FormControl, FormControlLabel, InputLabel, MenuItem, Select, Stack, Table, TableBody, TableCell, TableHead, TableRow, TextField, Typography } from '@mui/material'
import { api, ApiError, isAbort, Document, Run } from '../api/client'
import LogDrawer from './LogDrawer'
import RunInputs from './RunInputs'
import ResourceReport from './ResourceReport'
import StudyControls from './StudyControls'
import { emptyStudyDraft, studyOverrides } from './studyResources'
import RunPlanDialog, { type RunAction } from './RunPlanDialog'
import { RunStages } from './StageTable'
import { describe as runDescribe } from './runStatus'
import { clearSubmissionKey, requestFingerprint, submissionKey } from './submissionKey'
import { useRequestScope } from '../lib/useRequestScope'
import { useIdentity, useSessionConfirmed } from '../session/context'

const fmtTime = (t: number | null) => t ? new Date(t * 1000).toLocaleString() : ''
const fmtDuration = (r: Run) => { const end = r.finished ?? Date.now() / 1000; const s = Math.max(0, end - r.started); return s < 90 ? `${s.toFixed(0)} s` : s < 5400 ? `${(s / 60).toFixed(1)} min` : `${(s / 3600).toFixed(1)} h` }
const statusColour = (s: Run['status']) => s === 'running' ? 'info' : s === 'finished' ? 'success' : s === 'failed' ? 'error' : 'default'
const unresolvedCleanup = (r: Run) => !!r.cleanup && !['not_required', 'complete'].includes(r.cleanup)
const claimsCase = (r: Run) => (r.state !== undefined ? r.state !== 'terminal' : r.status === 'running') || unresolvedCleanup(r)
// Only the server may end a run it cannot confirm; the UI offers it exactly where that is true.
const resolvable = (r: Run) => r.state === 'unknown' || r.cleanup === 'unknown'
// A case stays claimed while cleanup is unresolved or the execution cannot be confirmed, and only
// a person can end the second case. Both have to be visible as something other than "finished".
const heldCase = (r: Run) => r.state === 'unknown' || unresolvedCleanup(r)
const runLabel = (r: Run) => (r.state === 'pending' ? 'accepted' : r.state === 'preparing' || r.state === 'queued' ? r.state : r.status)
  + (r.returncode !== null && r.returncode !== 0 ? ` (${r.returncode})` : '')
  + (unresolvedCleanup(r) ? ` · cleanup ${r.cleanup}` : '')

// The three pipeline scripts as cards with their options, and the table of past and running runs of the case.
// A run reads the saved case files, so the panel refuses to start while the editor has unsaved changes.
/**
 * What this run means and what to do about it -- A14-01.
 *
 * Deliberately narrow. The chip and the outcome detail beside it already say what state the
 * run is in and why it ended that way; repeating any of that here would put the same sentence
 * on the screen twice, which is what A13-07's consequence panel got wrong. So this adds only
 * the two things nothing else said: how much is done, counted rather than estimated, and --
 * when there is one -- the single action to take.
 */
function RunNextAction({ run }: { run: Run }) {
  const view = runDescribe(run)
  const actionable = view.remedy && !view.remedy.startsWith('Nothing to do')
  if (!view.progress && !actionable) return null
  return (
    <Stack spacing={0.25} alignItems="flex-start">
      {view.progress && <Typography variant="caption" color="text.secondary">{view.progress}</Typography>}
      {actionable && view.consequence &&
        <Typography variant="caption" color="text.secondary">{view.consequence}</Typography>}
      {actionable &&
        <Typography variant="caption" data-testid={`next-action-${run.id}`}
                    color={view.kind === 'failed' || view.kind === 'uncertain' ? 'error.main' : 'warning.main'}>
          {view.remedy}
        </Typography>}
    </Stack>
  )
}


export default function RunsPanel({ caseName, document, dirty, readOnly = false, manageable = false, expectedRevision }: { caseName: string; document: Document; dirty: boolean; readOnly?: boolean; manageable?: boolean; expectedRevision?: string }) {
  const identity = useIdentity()
  return <CaseRuns key={JSON.stringify([identity, caseName])} caseName={caseName} document={document} dirty={dirty} readOnly={readOnly} manageable={manageable} expectedRevision={expectedRevision} />
}

// `manageable` is the tenant's owner/admin capability the server requires for recovery and
// resolution. Being able to start a run is not the same permission, so the controls that change
// what artifacts claim to be are only offered to whoever the server would actually accept.
function CaseRuns({ caseName, document, dirty, readOnly, manageable, expectedRevision }: { caseName: string; document: Document; dirty: boolean; readOnly: boolean; manageable: boolean; expectedRevision?: string }) {
  const sessionConfirmed = useSessionConfirmed()
  const qc = useQueryClient()
  const signal = useRequestScope()
  const scriptsQuery = useQuery({ queryKey: ['scripts'], queryFn: ({ signal }) => api.scripts(signal), staleTime: Infinity })
  const scripts = scriptsQuery.data
  // Poll while anything is live or held: a run not started yet, pending cleanup and unknown
  // ownership all change without the legacy "running" label ever being set (R10-04).
  const runsQuery = useQuery({ queryKey: ['runs', caseName], queryFn: ({ signal }) => api.runs(caseName, signal),
    refetchInterval: (q) => (q.state.data ?? []).some(claimsCase) ? 3000 : 15000 })
  const runs = runsQuery.data ?? []
  const running = runs.find(claimsCase)
  const [open, setOpen] = useState<Run | null>(null)
  const [error, setError] = useState<string | null>(null)
  const identity = useIdentity()
  const [action, setAction] = useState<Record<string, RunAction>>({})
  const [planning, setPlanning] = useState<{ script: string; options: Document } | null>(null)
  const [expanded, setExpanded] = useState<string | null>(null)
  const [resolving, setResolving] = useState<Run | null>(null)
  const [resolution, setResolution] = useState('')
  const failed = (action: string, exception: unknown) => {
    if (!isAbort(exception)) {
      setError(`${action} failed: ${exception instanceof Error ? exception.message : String(exception)}`)
      void qc.invalidateQueries({ queryKey: ['runs', caseName] })
    }
  }

  const [datasets, setDatasets] = useState<Document>({ recreate: false, serial: false, dry_run: false })
  const [training, setTraining] = useState<Document>({ cpu: false, refit: false, prepare_only: false, docker: false, halve_batch: false, gpus: '', n_jobs: '', global_method: '', cluster_method: '', cpu_method: '' })
  const [studyDraft, setStudyDraft] = useState(emptyStudyDraft)
  const [combine, setCombine] = useState<Document>({ refit: false })
  const [logLevel, setLogLevel] = useState('INFO')

  const families = useMemo(() => Object.keys((document.methods?.method_params as Document) ?? {}), [document.methods])
  const globalFamilies = families.filter(m => scripts?.training.options.global_method.choices?.includes(m))
  const clusterFamilies = families.filter(m => scripts?.training.options.cluster_method.choices?.includes(m))
  const start = useMutation({
    mutationFn: ({ script, options, chosen, accept }: { script: string; options: Document; chosen: RunAction; accept?: boolean }) => {
      // The key survives this page: a retry after a lost response, a second click or a reload
      // reaches the run that was already accepted instead of starting another one. The answer
      // to what the preview showed is part of the request, so it is part of what names it.
      const fingerprint = requestFingerprint(script, options, chosen, expectedRevision, accept)
      return api.startRun(caseName, script, options, signal(),
        { client_key: submissionKey(identity, caseName, fingerprint), expected_revision: expectedRevision, action: chosen,
          ...(accept ? { accept_changes: true } : {}) })
    },
    onSuccess: (run, variables) => {
      clearSubmissionKey(identity, caseName, requestFingerprint(variables.script, variables.options, variables.chosen, expectedRevision, variables.accept))
      setError(null); setPlanning(null); qc.invalidateQueries({ queryKey: ['runs', caseName] }); setOpen(run)
    },
    onError: (e, variables) => {
      if (isAbort(e)) return
      if (e instanceof ApiError && e.status === 428) {
        // Not a failure. The run would replace datasets or trained models that were built from
        // another configuration, and the server starts no such run unasked: its preview opens,
        // where what goes is read and accepted by name. Nothing was started.
        setError(null); setPlanning({ script: variables.script, options: variables.options })
        return
      }
      setError(e instanceof ApiError ? e.message : String(e)); void qc.invalidateQueries({ queryKey: ['runs', caseName] })
    },
  })
  const resolve = useMutation({
    mutationFn: ({ id, reason }: { id: string; reason: string }) => api.resolveRun(caseName, id, reason, signal()),
    onSuccess: () => { setError(null); setResolving(null); setResolution(''); return qc.invalidateQueries({ queryKey: ['runs', caseName] }) },
    onError: (e) => failed('Resolve', e) })
  const stop = useMutation({ mutationFn: (id: string) => api.stopRun(caseName, id, signal()),
    onSuccess: (run) => { setError(null); setOpen(value => value?.id === run.id ? run : value); return qc.invalidateQueries({ queryKey: ['runs', caseName] }) },
    onError: (e) => failed('Stop', e) })
  const remove = useMutation({ mutationFn: (id: string) => api.deleteRun(caseName, id, signal()),
    onSuccess: (_, id) => { setError(null); setOpen(value => value?.id === id ? null : value); return qc.invalidateQueries({ queryKey: ['runs', caseName] }) },
    onError: (e) => failed('Delete', e) })

  // The dialog may have been open while the draft, the session or the case changed under it.
  const approve = (script: string, options: Document, chosen: RunAction, accept = false) => {
    if (launchBlock) { setError(launchBlock); setPlanning(null); return }
    start.mutate({ script, options, chosen, accept })
  }
  const launch = (script: string, options: Document) => {
    const clean = prepared(script, options)
    if (clean) approve(script, clean, action[script] ?? 'continue')
  }
  const prepared = (script: string, options: Document): Document | null => {
    const clean: Document = { log_level: logLevel }
    const cpuOnly = ['cpu_method', 'cpu_scope', 'cpu_workers', 'cpu_study_threads', 'cpu_study_device']
    const deepOnly = ['gpus', 'n_jobs', 'global_method', 'cluster_method', 'prepare_only', 'docker', 'halve_batch']
    for (const [k, v] of Object.entries(options)) {
      if (v === '' || v === false || v === null || v === undefined) continue
      if (script === 'training' && (options.cpu ? deepOnly : cpuOnly).includes(k)) continue
      if (k === 'gpus') { const ids = String(v).split(/[ ,]+/).filter(Boolean).map(Number); if (!ids.length || ids.some(id => !Number.isInteger(id) || id < 0) || new Set(ids).size !== ids.length) { setError('GPU ids must be unique nonnegative integers'); return null } clean[k] = ids; continue }
      if (k === 'n_jobs') { const count = Number(v); if (!Number.isSafeInteger(count) || count <= 0) { setError('Trials per GPU must be a positive integer'); return null } clean[k] = count; continue }
      clean[k] = v
    }
    if (script === 'training' && options.cpu) {
      try { Object.assign(clean, studyOverrides(studyDraft)) }
      catch (error) { setError(error instanceof Error ? error.message : String(error)); return null }
    }
    setError(null)
    return clean
  }
  // The choice of action belongs to the request, so the preview and the run that follows it
  // carry the same one: "continue" keeps compatible work, "rebuild" replaces it.
  const actions = (script: string, options: Document) => (
    <Stack direction="row" spacing={1} alignItems="center">
      <FormControl size="small" sx={{ minWidth: 150 }}>
        <InputLabel id={`action-${script}`}>action</InputLabel>
        <Select labelId={`action-${script}`} label="action" value={action[script] ?? 'continue'}
          onChange={e => setAction({ ...action, [script]: e.target.value as RunAction })}>
          <MenuItem value="continue">continue</MenuItem>
          <MenuItem value="new">new revision</MenuItem>
          <MenuItem value="rebuild">rebuild</MenuItem>
        </Select>
      </FormControl>
      <Button size="small" disabled={disabled} onClick={() => { const clean = prepared(script, options); if (clean) setPlanning({ script, options: clean }) }}>Preview</Button>
    </Stack>
  )
  // One rule, with the sentence that explains it, so a dialog can refuse for the same reason the
  // card does instead of quietly bypassing the guard (R10-02).
  const launchBlock = !sessionConfirmed ? 'The session is not confirmed again yet; reload before starting a run.'
    : readOnly ? 'This case belongs to another user: runs can be watched, not started.'
    : dirty ? 'Save the case first: a run reads the saved files, not the editor.'
    : running ? `A ${running.script} run is in progress; one run per case at a time.`
    : !scripts || !runsQuery.isSuccess ? 'The run options are still loading.' : null
  const disabled = !!launchBlock || start.isPending
  const flag = (state: Document, set: (d: Document) => void, key: string, label: string, help?: string) => (
    <FormControlLabel key={key} control={<Checkbox size="small" checked={!!state[key]} onChange={e => set({ ...state, [key]: e.target.checked })} />}
      label={<Stack><Typography variant="body2">{label}</Typography>{help && <Typography variant="caption" color="text.secondary">{help}</Typography>}</Stack>} />
  )

  return (
    <Stack spacing={2}>
      {readOnly && <Alert severity="info" sx={{ py: 0 }}>This case belongs to another user: runs can be watched, not started.</Alert>}
      {dirty && <Alert severity="warning" sx={{ py: 0 }}>Save the case first: a run reads the saved files, not the editor.</Alert>}
      {running && <Alert severity="info" sx={{ py: 0 }}>A {running.script} run is in progress; one run per case at a time.</Alert>}
      {error && <Alert severity="error" onClose={() => setError(null)}>{error}</Alert>}
      {scriptsQuery.isError && <Alert severity="error">Could not load the available run options: {String(scriptsQuery.error)}</Alert>}
      <Stack direction="row" spacing={2} alignItems="stretch" flexWrap="wrap" useFlexGap>
        <Card variant="outlined" sx={{ flex: '1 1 300px' }}>
          <CardContent>
            <Typography variant="subtitle1">1 · Datasets and clustering</Typography>
            <Typography variant="caption" color="text.secondary">run_datasets.py: extraction of the NWP files, the nwp / target / row / lstm datasets, then the train/validation/test split, preprocessing, clustering and feature selection. Existing outputs are skipped unless "recreate" is on.</Typography>
            <Autocomplete multiple size="small" options={scripts?.datasets.options.stages.choices ?? []} value={datasets.stages ?? scripts?.datasets.defaults.stages ?? []} onChange={(_, v) => setDatasets({ ...datasets, stages: v })}
              renderInput={p => <TextField {...p} label="stages" helperText="datasets builds the inputs; all also draws the split and runs preprocessing, clustering and feature selection" />} sx={{ mt: 1.5 }} />
            {flag(datasets, setDatasets, 'recreate', 'recreate', 'delete the outputs of the selected stages first')}
            {flag(datasets, setDatasets, 'recreate_extraction', 'recreate extraction', 'also redo the NWP extraction')}
            {flag(datasets, setDatasets, 'serial', 'serial', 'no process pools; for debugging')}
            {flag(datasets, setDatasets, 'dry_run', 'dry run', 'resolve everything, write nothing')}
          </CardContent>
          <CardActions sx={{ flexWrap: 'wrap', gap: 1 }}>
            {actions('datasets', { ...scripts?.datasets.defaults, ...datasets })}
            <Button variant="contained" size="small" disabled={disabled} onClick={() => launch('datasets', { ...scripts?.datasets.defaults, ...datasets })}>Run datasets</Button>
          </CardActions>
        </Card>
        <Card variant="outlined" sx={{ flex: '1 1 300px' }}>
          <CardContent>
            <Typography variant="subtitle1">2 · Training</Typography>
            <Typography variant="caption" color="text.secondary">run_training.py: the deep methods on the GPUs through the orchestrator (Optuna studies per cluster and method), or the shallow CPU methods with "cpu".</Typography>
            <Stack spacing={1} sx={{ mt: 1.5 }}>
              {flag(training, setTraining, 'cpu', 'CPU methods', 'Shallow estimators and foundation methods; independent studies side by side')}
              {training.cpu && <FormControl size="small"><InputLabel>cpu method</InputLabel><Select label="cpu method" value={training.cpu_method} onChange={e => setTraining({ ...training, cpu_method: e.target.value })}><MenuItem value="">all</MenuItem>{(scripts?.training.options.cpu_method.choices ?? []).map(m => <MenuItem key={m} value={m}>{m}</MenuItem>)}</Select></FormControl>}
              {training.cpu && <StudyControls value={studyDraft} onChange={setStudyDraft} saved={document.jobs ?? {}} />}
              {!training.cpu && <>
                <Stack direction="row" spacing={1}>
                  <TextField size="small" label="gpus" placeholder="all" value={training.gpus} onChange={e => setTraining({ ...training, gpus: e.target.value })} helperText="ids, e.g. 0 1" />
                  <TextField size="small" label="trials per GPU" placeholder="n_jobs" value={training.n_jobs} onChange={e => setTraining({ ...training, n_jobs: e.target.value })} />
                </Stack>
                <Stack direction="row" spacing={1}>
                  <FormControl size="small" fullWidth><InputLabel>global method</InputLabel><Select label="global method" value={training.global_method} onChange={e => setTraining({ ...training, global_method: e.target.value })}><MenuItem value="">all</MenuItem>{globalFamilies.map(m => <MenuItem key={m} value={m}>{m}</MenuItem>)}</Select></FormControl>
                  <FormControl size="small" fullWidth><InputLabel>cluster method</InputLabel><Select label="cluster method" value={training.cluster_method} onChange={e => setTraining({ ...training, cluster_method: e.target.value })}><MenuItem value="">all</MenuItem>{clusterFamilies.map(m => <MenuItem key={m} value={m}>{m}</MenuItem>)}</Select></FormControl>
                </Stack>
                {flag(training, setTraining, 'prepare_only', 'prepare only', 'create the studies and the first trial folders, train nothing')}
                {flag(training, setTraining, 'docker', 'Docker workers', 'run every trial from the eforecast-worker image')}
                {flag(training, setTraining, 'halve_batch', 'halve batch on 2nd OOM')}
              </>}
              {flag(training, setTraining, 'refit', 'refit', 'delete the studies and trial folders and search again')}
            </Stack>
          </CardContent>
          <CardActions sx={{ flexWrap: 'wrap', gap: 1 }}>
            {actions('training', training)}
            <Button variant="contained" size="small" disabled={disabled} onClick={() => launch('training', training)}>Run training</Button>
          </CardActions>
        </Card>
        <Card variant="outlined" sx={{ flex: '1 1 300px' }}>
          <CardContent>
            <Typography variant="subtitle1">3 · Combine and evaluate</Typography>
            <Typography variant="caption" color="text.secondary">run_combine.py: the selection slice chooses trials and fits combiners. A separate calibration slice fits prediction intervals when enabled. The final report slice measures performance without selecting or fitting models.</Typography>
            <Autocomplete multiple size="small" options={scripts?.combine.options.stages.choices ?? []} value={combine.stages ?? scripts?.combine.defaults.stages ?? []} onChange={(_, v) => setCombine({ ...combine, stages: v })}
              renderInput={p => <TextField {...p} label="stages" />} sx={{ mt: 1.5 }} />
            {flag(combine, setCombine, 'refit', 'refit', 'recompute the cached predictions and refit the combiners')}
          </CardContent>
          <CardActions sx={{ flexWrap: 'wrap', gap: 1 }}>
            {actions('combine', { ...scripts?.combine.defaults, ...combine })}
            <Button variant="contained" size="small" disabled={disabled} onClick={() => launch('combine', { ...scripts?.combine.defaults, ...combine })}>Run combine</Button>
          </CardActions>
        </Card>
      </Stack>
      <Stack direction="row" spacing={2} alignItems="center">
        <Typography variant="subtitle1">Runs</Typography>
        <FormControl size="small" sx={{ minWidth: 120 }}><InputLabel>log level</InputLabel><Select label="log level" value={logLevel} onChange={e => setLogLevel(e.target.value)}>{['DEBUG', 'INFO', 'WARNING'].map(l => <MenuItem key={l} value={l}>{l}</MenuItem>)}</Select></FormControl>
        <Box sx={{ flex: 1 }} />
        <Button size="small" onClick={() => runsQuery.refetch()}>Refresh</Button>
      </Stack>
      {runsQuery.isPending && <Typography>Loading runs…</Typography>}
      {runsQuery.isError && <Alert severity="error" action={<Button onClick={() => runsQuery.refetch()}>Retry</Button>}>Could not load runs: {String(runsQuery.error)}</Alert>}
      {runsQuery.isSuccess && runs.length === 0 && <Typography variant="body2" color="text.secondary">No run yet for this case.</Typography>}
      {runs.length > 0 && (
        <Box sx={{ overflowX: 'auto' }}>
          <Table size="small">
            <TableHead><TableRow><TableCell>started</TableCell><TableCell>script</TableCell><TableCell>options</TableCell><TableCell>status</TableCell><TableCell>inputs</TableCell><TableCell>resources</TableCell><TableCell>duration</TableCell><TableCell align="right">actions</TableCell></TableRow></TableHead>
            <TableBody>
              {runs.flatMap(r => [
                <TableRow key={r.id} hover>
                  <TableCell>{fmtTime(r.started)}</TableCell>
                  <TableCell>{r.script}</TableCell>
                  <TableCell><Typography variant="caption" sx={{ fontFamily: 'monospace' }}>{r.command.slice(4).join(' ') || '(defaults)'}</Typography></TableCell>
                  <TableCell>
                    <Stack spacing={0.5} alignItems="flex-start">
                      <Chip size="small" label={runLabel(r)} color={unresolvedCleanup(r) ? 'warning' : statusColour(r.status)} />
                      {heldCase(r) && <Typography variant="caption" color="warning.main">
                        this case is held{r.report_error ? `: ${r.report_error}` : r.state === 'unknown' ? ': the execution could not be confirmed' : ''}
                      </Typography>}
                      {r.worker_result?.kind === 'partial' && <Typography variant="caption" color="warning.main">partial: {r.worker_result.detail}</Typography>}
                      {/* why it ended that way: a resolved run carries the person and the reason */}
                      {r.outcome && r.outcome.kind !== 'success' && r.outcome.detail &&
                        <Typography variant="caption" color="text.secondary">{r.outcome.detail}</Typography>}
                      <RunNextAction run={r} />
                    </Stack>
                  </TableCell>
                  <TableCell><RunInputs run={r} /></TableCell>
                  <TableCell>
                    <ResourceReport resources={r.resources} live={claimsCase(r)} />
                  </TableCell>
                  <TableCell>{fmtDuration(r)}</TableCell>
                  <TableCell align="right">
                    {!!r.stages?.length && <Button size="small" onClick={() => setExpanded(expanded === r.id ? null : r.id)}>
                      {expanded === r.id ? 'Hide stages' : `Stages (${r.stages.length})`}</Button>}
                    <Button size="small" onClick={() => setOpen(r)}>Log</Button>
                    {!readOnly && manageable && resolvable(r) && r.status !== 'running' &&
                      <Button size="small" color="warning" disabled={!sessionConfirmed || resolve.isPending}
                        onClick={() => { setResolving(r); setResolution('') }}>Resolve</Button>}
                    {!readOnly && (r.status === 'running' ? <Button size="small" color="warning" disabled={!sessionConfirmed || stop.isPending} onClick={() => stop.mutate(r.id)}>Stop</Button>
                      /* the server refuses history deletion while cleanup holds the case; so does this */
                      : <Button size="small" color="error" title={unresolvedCleanup(r) ? 'cleanup still holds this case' : undefined}
                          disabled={!sessionConfirmed || remove.isPending || unresolvedCleanup(r)}
                          onClick={() => remove.mutate(r.id)}>Delete</Button>)}
                  </TableCell>
                </TableRow>,
                <TableRow key={`${r.id}-stages`}>
                  <TableCell colSpan={8} sx={{ py: 0, borderBottom: expanded === r.id ? undefined : 'none' }}>
                    <Collapse in={expanded === r.id} unmountOnExit>
                      <Box sx={{ py: 1 }}>{expanded === r.id && r.stages && <RunStages stages={r.stages} />}</Box>
                    </Collapse>
                  </TableCell>
                </TableRow>,
              ])}
            </TableBody>
          </Table>
        </Box>
      )}
      {planning && <RunPlanDialog caseName={caseName} script={planning.script} options={planning.options}
        action={action[planning.script] ?? 'continue'} expectedRevision={expectedRevision} tenantScope={identity}
        starting={start.isPending} onClose={() => setPlanning(null)}
        eligible={!launchBlock} ineligible={launchBlock} manageable={manageable}
        onStart={accept => approve(planning.script, planning.options, action[planning.script] ?? 'continue', accept)} />}
      <Dialog open={!!resolving} onClose={() => setResolving(null)} fullWidth maxWidth="sm">
        <DialogTitle>End this run and release the case</DialogTitle>
        <DialogContent>
          <DialogContentText variant="body2">
            {resolving?.outcome && resolving.outcome.kind === 'success'
              ? 'This run\'s observed success is kept as it is. Your decision ends the cleanup that cannot be '
                + 'confirmed and releases the case, with your name and reason recorded beside it.'
              : 'Nothing here is observed: the run stays recorded as a failure whose evidence was lost, with '
                + 'your name and reason on it.'}
            {' '}Use it when the execution cannot be confirmed any more — a machine that is gone, a report
            that cannot be read — and never while the work may still be running.
          </DialogContentText>
          {resolving?.report_error && <Alert severity="warning" sx={{ mt: 1 }}>{resolving.report_error}</Alert>}
          <TextField fullWidth size="small" sx={{ mt: 2 }} label="reason" value={resolution}
            onChange={e => setResolution(e.target.value)} placeholder="the host was rebuilt" />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setResolving(null)}>Cancel</Button>
          <Button color="warning" variant="contained" disabled={!sessionConfirmed || !manageable || readOnly || resolve.isPending || !resolution.trim()}
            onClick={() => sessionConfirmed && manageable && !readOnly && resolving && resolve.mutate({ id: resolving.id, reason: resolution.trim() })}>Resolve</Button>
        </DialogActions>
      </Dialog>
      <LogDrawer caseName={caseName} run={open} onClose={() => setOpen(null)} />
    </Stack>
  )
}
