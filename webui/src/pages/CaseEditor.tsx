import { lazy, Suspense, useContext, useEffect, useMemo, useReducer, useRef, useState } from 'react'
import { useBlocker, useNavigate, useParams } from 'react-router-dom'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Alert, Box, Button, Chip, CircularProgress, Dialog, DialogActions, DialogContent, DialogTitle, Paper, Snackbar, Stack, Tab, Tabs, TextField, Typography } from '@mui/material'
import { api, ApiError, isAbort, type CaseResponse, type Consequences, type Document, type Validation } from '../api/client'
import ErrorList from '../components/ErrorList'
import CaseHeader from '../components/CaseHeader'
import ReadinessPanel from '../components/ReadinessPanel'
import ConsequencesDialog from '../components/ConsequencesDialog'
import ConsequencesPanel from '../components/ConsequencesPanel'
import DiffView from '../components/DiffView'
import { toYaml } from '../lib/yaml'
import { SECTION_LABELS } from '../lib/schema'
import { DEFAULT_PRESETS_PATH, type FormContext, type UiOptions } from '../fields/context'
import PresetsSource from '../components/PresetsSource'
import { cloneDocument, draftDirty, draftReducer, emptyDraft, equalDocuments } from '../lib/caseDraft'
import { useRequestScope } from '../lib/useRequestScope'
import { useIdentity, useSessionConfirmed } from '../session/context'
import { YamlDraftContext, unfinishedYaml } from '../forms/yamlDraftContext'
import GraphBoundary from '../graph/GraphBoundary'
import type { ExperimentLayouts } from '../graph/types'
import { LogoutGuardContext } from '../session/logoutGuard'

const ExperimentsEditor = lazy(() => import('../graph/ExperimentsEditor'))
const RunsPanel = lazy(() => import('../runs/RunsPanel'))
const StudiesPanel = lazy(() => import('../runs/StudiesPanel'))
const ResultSummaryPanel = lazy(() => import('../runs/ResultSummary'))
const SectionForm = lazy(() => import('../forms/SectionForm'))

// Resource policy stays in the complete case document for compatibility, but is
// not an everyday case-editing step. Runs retains explicit advanced overrides.
const SECTIONS = ['project', 'folders', 'data', 'methods'] as const
const TABS = ['header', ...SECTIONS, 'experiments', 'review', 'runs', 'results', 'studies'] as const
const TAB_LABELS: Record<string, string> = { header: 'Everyday settings', experiments: 'Experiments', review: 'Review & save', runs: 'Runs', results: 'Results', studies: 'Studies & results' }
type TabName = typeof TABS[number]

// Reset component-local state when either the case or the identity changes.
export default function CaseEditor() {
  const { name = '' } = useParams()
  const identity = useIdentity()
  return <Editor key={`${identity}:${name}`} name={name} identity={identity} />
}

function Editor({ name, identity }: { name: string; identity: string }) {
  const registerLogoutGuard = useContext(LogoutGuardContext)
  const sessionConfirmed = useSessionConfirmed()
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const signal = useRequestScope()
  const casesQuery = useQuery({ queryKey: ['cases'], queryFn: ({ signal }) => api.cases(signal) })
  const summary = casesQuery.data?.find(c => c.name === name)
  const readOnly = summary?.writable !== true
  // recovery and operator resolution are the tenant owner's/admin's, not every writer's
  const manageable = summary?.manageable === true
  const key = ['case', name]
  const caseQuery = useQuery({ queryKey: key, queryFn: ({ signal }) => api.getCase(name, signal) })
  const schemaQuery = useQuery({ queryKey: ['schema'], queryFn: ({ signal }) => api.schema(signal), staleTime: Infinity })
  const optionsQuery = useQuery({ queryKey: ['options'], queryFn: ({ signal }) => api.options(signal), staleTime: Infinity })
  const [state, dispatch] = useReducer(draftReducer, emptyDraft)
  const current = useRef(state)
  current.current = state
  const document = state.document
  const dirty = draftDirty(state)
  const yamlErrors = unfinishedYaml(state.yaml)
  const yamlInvalid = yamlErrors.length > 0
  const allowNavigation = useRef(false)
  const blocker = useBlocker(() => draftDirty(current.current) && !allowNavigation.current)
  const presetsKey = `eforecast:presets:${location.origin}:${identity}:${name}`
  const [presetsPath, setPresetsPathState] = useState(() => {
    try { return localStorage.getItem(presetsKey) || DEFAULT_PRESETS_PATH } catch { return DEFAULT_PRESETS_PATH }
  })
  const setPresetsPath = (path: string) => {
    setPresetsPathState(path)
    try { localStorage.setItem(presetsKey, path) } catch { /* private window */ }
  }
  // A13-06 adds 'Everyday settings' as the first tab. The editor still opens on Project: the
  // common action is a maintainer editing an existing case, and changing where every case opens
  // is a larger change than the six groups needed -- they are one click away.
  const [tab, setTab] = useState<TabName>('project')
  // Kept outside the tab so arranging the canvas survives tab changes, without
  // entering the saved document, revision hash or unsaved-change guard.
  const graphLayoutState = useState<ExperimentLayouts>({})
  const [validation, setValidation] = useState<{ edit: number; value: Validation } | null>(null)
  const [message, setMessage] = useState<string | null>(null)
  const [conflict, setConflict] = useState(false)
  const [refreshNeeded, setRefreshNeeded] = useState(false)
  // A save that changes what the case's datasets or models were built from is refused until
  // the person has read what a later run then replaces and accepted it: the refused draft
  // waits here with what it would cost, and is saved as it was when that is accepted.
  const [asking, setAsking] = useState<{ report: Consequences; submitted: { document: Document; edit: number; revision?: string } } | null>(null)
  const [copyName, setCopyName] = useState<string | null>(null)
  const [logoutPrompt, setLogoutPrompt] = useState(false)
  const [logoutSaving, setLogoutSaving] = useState(false)
  const [logoutError, setLogoutError] = useState<string | null>(null)
  const logoutDecision = useRef<((allowed: boolean) => void) | null>(null)
  const pendingWrites = useRef(false)

  useEffect(() => {
    if (caseQuery.data) { dispatch({ type: 'server', snapshot: caseQuery.data }); setRefreshNeeded(false) }
  }, [caseQuery.data, caseQuery.dataUpdatedAt])
  useEffect(() => {
    const beforeUnload = (event: BeforeUnloadEvent) => { if (dirty) { event.preventDefault(); event.returnValue = '' } }
    window.addEventListener('beforeunload', beforeUnload)
    return () => window.removeEventListener('beforeunload', beforeUnload)
  }, [dirty])

  const savedYaml = useMemo(() => toYaml(state.baseline?.document ?? {}), [state.baseline])
  const currentYaml = useMemo(() => toYaml(document ?? {}), [document])
  const setDocument = (next: Document) => dispatch({ type: 'edit', document: next })
  const validate = useMutation({
    mutationFn: async () => {
      const edit = current.current.edit
      const value = await api.validate(current.current.document ?? {}, signal())
      return { edit, value }
    },
    onSuccess: setValidation,
    onError: (error) => { if (!isAbort(error)) setMessage(`Validation failed: ${String(error)}`) },
  })
  const save = useMutation({
    mutationFn: async (submitted: { document: Document; edit: number; revision?: string; accepted?: string }) => {
      await queryClient.cancelQueries({ queryKey: key })
      return api.saveCase(name, submitted.document, submitted.revision, signal(), submitted.accepted)
    },
    onSuccess: async (result, submitted) => {
      if (signal().aborted) return
      setAsking(null)
      await queryClient.cancelQueries({ queryKey: key })
      const snapshot: CaseResponse = { name, document: result.document ?? submitted.document,
        revision: result.revision, validation: result.validation ?? state.baseline!.validation }
      dispatch({ type: 'saved', snapshot, submittedEdit: submitted.edit })
      queryClient.setQueryData(key, snapshot)
      void queryClient.invalidateQueries({ queryKey: ['cases'] })
      if (result.validation) setValidation({ edit: submitted.edit, value: result.validation })
      setConflict(false)
      setMessage(current.current.edit === submitted.edit ? 'Case saved.' : 'Case saved. Your newer edits are still unsaved.')
    },
    onError: (error, submitted) => {
      if (isAbort(error)) return
      if (error instanceof ApiError && error.status === 428 && error.payload.consequences) {
        // Not a failure: the server says what this change makes a later run build again, and
        // waits for that to be accepted. An answer given for another list is asked again.
        setAsking({ report: error.payload.consequences as Consequences,
          submitted: { document: submitted.document, edit: submitted.edit, revision: submitted.revision } })
        setMessage('Nothing saved yet: read what this change replaces, and accept it to save.')
        return
      }
      setAsking(null)
      if (error instanceof ApiError && error.status === 409) {
        setConflict(true)
        if (error.payload.document && typeof error.payload.document === 'object') {
          const server = error.payload.document as Document
          dispatch({ type: 'server', snapshot: { name, document: server, revision: error.payload.revision as string,
            validation: error.payload.validation as Validation } })
          if (current.current.document && equalDocuments(server, current.current.document)) {
            setConflict(false); setMessage('The current saved case already matches your draft.'); return
          }
        }
        setMessage('Nothing saved: the saved case changed. Review it or save your draft as a new case.')
      } else if (error instanceof ApiError && error.status === 422) {
        setValidation({ edit: submitted.edit, value: { valid: false, complete: true, missing_sections: [], errors: error.errors } })
        setMessage('Nothing saved: the submitted draft has validation errors. Validate your current draft again if you edited it meanwhile.')
      } else {
        setRefreshNeeded(true)
        setMessage(`Save could not be confirmed: ${error instanceof Error ? error.message : String(error)}. Refresh the saved case before retrying.`)
      }
    },
  })
  const copy = useMutation({
    mutationFn: (submitted: { name: string; document: Document; edit: number }) => api.createCase(submitted.name, { document: submitted.document }, signal()),
    onSuccess: (_, submitted) => {
      void queryClient.invalidateQueries({ queryKey: ['cases'] })
      if (current.current.edit !== submitted.edit) {
        setCopyName(null); setMessage(`Saved ${submitted.name}. Your newer edits remain in this draft.`); return
      }
      allowNavigation.current = true
      navigate(`/cases/${encodeURIComponent(submitted.name)}`)
    },
    onError: (error) => { if (!isAbort(error)) setMessage(`Copy failed: ${String(error)}`) },
  })
  const sync = useMutation({
    mutationFn: async () => {
      const edit = current.current.edit
      return { edit, result: await api.sync(current.current.document ?? {}, signal()) }
    },
    onSuccess: ({ edit, result }) => {
      if (current.current.edit !== edit) { setMessage('The draft changed while syncing. Retry sync to use your latest edits.'); return }
      setDocument(result.document); setMessage(result.note)
    },
    onError: (error) => { if (!isAbort(error)) setMessage(`Sync failed: ${String(error)}`) },
  })

  pendingWrites.current = save.isPending || copy.isPending
  useEffect(() => {
    if (!registerLogoutGuard) return
    const unregister = registerLogoutGuard(() => {
      if (!draftDirty(current.current) && !pendingWrites.current) return Promise.resolve(true)
      if (logoutDecision.current) return Promise.resolve(false)
      return new Promise<boolean>(resolve => {
        logoutDecision.current = resolve; setLogoutError(null); setLogoutPrompt(true)
      })
    })
    return () => {
      unregister()
      logoutDecision.current?.(false); logoutDecision.current = null
    }
  }, [registerLogoutGuard])

  const decideLogout = (allowed: boolean) => {
    const resolve = logoutDecision.current
    logoutDecision.current = null; setLogoutPrompt(false); setLogoutError(null)
    resolve?.(allowed)
  }

  if (caseQuery.isLoading || schemaQuery.isLoading) return <Box sx={{ p: 6, textAlign: 'center' }}><CircularProgress /></Box>
  if (caseQuery.isError && !document) return <Alert severity="error" action={<Button onClick={() => caseQuery.refetch()}>Retry</Button>}>{String(caseQuery.error)}</Alert>
  if (schemaQuery.isError) return <Alert severity="error" action={<Button onClick={() => schemaQuery.refetch()}>Retry</Button>}>Could not load the form schema: {String(schemaQuery.error)}</Alert>
  if (!document || !schemaQuery.data) return null

  // Whether the draft a field's request was built on is still the draft in front of the person. Both
  // facts are read through the ref rather than from this render, because a field can be holding a
  // context from an earlier one: rjsf's Form skips re-rendering when its props are deeply equal, and
  // its comparison treats every function as equal, so a draft edit that leaves this section's values
  // untouched -- a sync that copies nothing, an edit that is undone -- replaces the document object
  // and raises the count without the field ever seeing the new context. So the document is compared
  // by value, which the older context still answers correctly, and the count is the one the field
  // took when it started: the count also moves for an unfinished raw YAML edit, which changes the
  // draft without changing the document.
  const unchangedDraft = (expected: Document, since?: number) =>
    equalDocuments(current.current.document ?? {}, expected)
    && (since === undefined || current.current.edit === since)
  const context: FormContext = { document, options: (optionsQuery.data as UiOptions | undefined) ?? null,
    setDocument, presetsPath, isCurrent: unchangedDraft, stamp: () => current.current.edit }
  const hasConflict = !!state.incoming || conflict
  const cannotSaveForLogout = yamlInvalid || !sessionConfirmed || readOnly || hasConflict || refreshNeeded || save.isPending || copy.isPending || logoutSaving
  const saveAndLogout = async () => {
    const decision = logoutDecision.current
    if (!decision || cannotSaveForLogout) return
    if (!draftDirty(current.current)) { decideLogout(true); return }
    const submitted = { document: cloneDocument(current.current.document!), edit: current.current.edit, revision: current.current.baseline?.revision }
    setLogoutSaving(true); setLogoutError(null)
    try {
      await save.mutateAsync(submitted)
      if (logoutDecision.current !== decision) return
      if (current.current.edit !== submitted.edit || unfinishedYaml(current.current.yaml).length > 0) {
        setLogoutError('New edits arrived while saving. They remain unsaved. Save them or cancel logout.'); return
      }
      decideLogout(true)
    } catch (error) {
      if (logoutDecision.current === decision && error instanceof ApiError && error.status === 428) {
        // The save asks a question that deserves the whole screen: logging out waits for it.
        decideLogout(false)
      } else if (logoutDecision.current === decision && !isAbort(error)) {
        setLogoutError(`Save did not complete. You are still signed in. ${error instanceof Error ? error.message : String(error)}`)
      }
    } finally { setLogoutSaving(false) }
  }
  return (
    <YamlDraftContext.Provider value={{ buffers: state.yaml,
      setBuffer: (key, buffer) => dispatch({ type: 'yaml', key, buffer }),
      discardBuffer: key => dispatch({ type: 'discard-yaml', key }) }}>
    <Stack spacing={2}>
      <Stack direction="row" alignItems="center" spacing={2} useFlexGap flexWrap="wrap">
        <Typography variant="h6">{name}</Typography>
        {document.project?.name && <Chip size="small" label={`project ${document.project.name}`} />}
        {readOnly ? <Chip size="small" label="read-only" /> : <Chip size="small" color={dirty ? 'warning' : 'success'} label={dirty ? 'unsaved changes' : 'saved'} />}
        <Box sx={{ flex: 1 }} />
        <Button variant="outlined" onClick={() => validate.mutate()} disabled={yamlInvalid || validate.isPending}>Validate</Button>
        <Button variant="outlined" onClick={() => caseQuery.refetch()} disabled={caseQuery.isFetching || save.isPending}>Refresh saved case</Button>
        <Button disabled={yamlInvalid || !sessionConfirmed} onClick={() => setCopyName(`${name}_copy`)}>Save as new case</Button>
        <Button variant="contained" onClick={() => save.mutate({ document: cloneDocument(document), edit: state.edit, revision: state.baseline?.revision })}
          disabled={yamlInvalid || !sessionConfirmed || !dirty || save.isPending || readOnly || hasConflict || refreshNeeded}>Save</Button>
      </Stack>
      {yamlInvalid && <Alert severity="error" action={<Button onClick={() => dispatch({ type: 'discard-yaml' })}>Discard unfinished YAML</Button>}>
        Fix or discard unfinished YAML before saving, validating, or starting a run. Your text is retained across tabs.
        <Stack direction="row" useFlexGap flexWrap="wrap">
          {yamlErrors.map(([key, buffer]) => <Button key={key} size="small" onClick={() => setTab(buffer.section as TabName)}>
            {SECTION_LABELS[buffer.section] ?? TAB_LABELS[buffer.section] ?? buffer.section}: {buffer.label}
          </Button>)}
        </Stack>
      </Alert>}
      {caseQuery.isError && <Alert severity="error">Could not refresh the saved case: {String(caseQuery.error)}. Your draft is retained.</Alert>}
      {casesQuery.isError && <Alert severity="error" action={<Button onClick={() => casesQuery.refetch()}>Retry permissions</Button>}>Could not verify write permission: {String(casesQuery.error)}</Alert>}
      {optionsQuery.isError && <Alert severity="warning" action={<Button onClick={() => optionsQuery.refetch()}>Retry options</Button>}>Could not load configuration options: {String(optionsQuery.error)}</Alert>}
      {refreshNeeded && <Alert severity="warning">The save outcome is unknown. Your draft is retained. Refresh the saved case before retrying.</Alert>}
      {!state.baseline?.revision && <Alert severity="warning">This server does not provide save revisions. Concurrent saves cannot be checked.</Alert>}
      {hasConflict && <Alert severity="warning" action={<Stack direction="row">
        <Button onClick={() => setTab('review')}>Review differences</Button>
        <Button disabled={!state.incoming || save.isPending} onClick={() => {
          dispatch({ type: 'reload', snapshot: state.incoming! }); setConflict(false); setValidation(null)
        }}>Use saved version</Button>
      </Stack>}>The saved case changed. Your draft is retained. Review both versions or save it as a new case.</Alert>}
      <Paper>
        <Tabs value={tab} onChange={(_, value) => setTab(value)} variant="scrollable" scrollButtons="auto">
          {TABS.map(value => <Tab key={value} value={value} label={SECTION_LABELS[value] ?? TAB_LABELS[value] ?? value} />)}
        </Tabs>
        <Box sx={{ p: 2 }}>
          <Suspense fallback={<Typography>Loading editor…</Typography>}>
          {tab === 'project' && <PresetsSource value={presetsPath} onChange={setPresetsPath} />}
          {tab === 'methods' && <Stack direction="row" spacing={1} sx={{ mb: 1 }}>
            <Button size="small" variant="outlined" disabled={yamlInvalid || sync.isPending} onClick={() => sync.mutate()}>Sync data options into the families</Button>
            <Typography variant="caption" color="text.secondary" sx={{ alignSelf: 'center' }}>copies merge, compress, scaling and feature-selection options of the Data tab into every family that uses them</Typography>
          </Stack>}
          {tab === 'header' && <CaseHeader document={document} setDocument={setDocument}
            options={optionsQuery.data} readOnly={readOnly} />}
          {SECTIONS.map(section => tab === section && <SectionForm key={section} fullSchema={schemaQuery.data} section={section} context={context}
            value={document[section]} onChange={value => setDocument({ ...document, [section]: value })} />)}
          {tab === 'experiments' && <GraphBoundary><ExperimentsEditor key={name} document={document} setDocument={setDocument} presetsPath={presetsPath} layoutState={graphLayoutState}
            graphFamilies={(optionsQuery.data as UiOptions | undefined)?.flags?.cluster_methods ?? []} /></GraphBoundary>}
          {tab === 'runs' && <RunsPanel caseName={name} document={document} dirty={dirty} readOnly={readOnly}
            manageable={manageable} expectedRevision={state.baseline?.revision} />}
          {tab === 'results' && <ResultSummaryPanel caseName={name} />}
          {tab === 'studies' && <StudiesPanel caseName={name} />}
          {tab === 'review' && <Stack spacing={2}>
            <ErrorList validation={validation?.edit === state.edit ? validation.value : dirty ? null : state.baseline?.validation ?? null} />
            <Typography variant="subtitle2">Is this case ready to run?</Typography>
            <ReadinessPanel document={document} caseName={name} disabled={yamlInvalid} />
            <Typography variant="subtitle2">This configuration and what is already built</Typography>
            <ConsequencesPanel caseName={name} document={document} dirty={dirty} disabled={yamlInvalid} />
            <Typography variant="subtitle2">Your draft against the version you loaded</Typography>
            <DiffView before={savedYaml} after={currentYaml} />
            {state.incoming && <>
              <Typography variant="subtitle2">Current saved version against the version you loaded</Typography>
              <DiffView before={savedYaml} after={toYaml(state.incoming.document)} />
            </>}
          </Stack>}
          </Suspense>
        </Box>
      </Paper>
      <ConsequencesDialog name={name} report={asking?.report ?? null} saving={save.isPending}
        onCancel={() => setAsking(null)}
        onAccept={digest => asking && save.mutate({ ...asking.submitted, accepted: digest })} />
      <Dialog open={copyName !== null} onClose={() => { if (!copy.isPending) setCopyName(null) }}>
        <DialogTitle>Save draft as a new case</DialogTitle>
        <DialogContent><TextField autoFocus sx={{ mt: 1 }} label="New case name" value={copyName ?? ''} onChange={event => setCopyName(event.target.value)} /></DialogContent>
        <DialogActions><Button disabled={copy.isPending} onClick={() => setCopyName(null)}>Cancel</Button><Button disabled={yamlInvalid || !sessionConfirmed || !copyName?.trim() || copy.isPending}
          onClick={() => copy.mutate({ name: copyName!.trim(), document: cloneDocument(document), edit: state.edit })}>Create case</Button></DialogActions>
      </Dialog>
      <Dialog open={blocker.state === 'blocked'} onClose={() => blocker.state === 'blocked' && blocker.reset()}>
        <DialogTitle>Leave unsaved changes?</DialogTitle>
        <DialogContent>Your draft has unsaved changes. Stay to save it, or leave and discard it.</DialogContent>
        <DialogActions><Button onClick={() => blocker.state === 'blocked' && blocker.reset()}>Stay</Button>
          <Button color="warning" onClick={() => blocker.state === 'blocked' && blocker.proceed()}>Leave without saving</Button></DialogActions>
      </Dialog>
      <Dialog open={logoutPrompt} onClose={() => decideLogout(false)} aria-labelledby="logout-draft-title">
        <DialogTitle id="logout-draft-title">Unsaved changes before logging out</DialogTitle>
        <DialogContent>
          <Typography>Your case {name} has unsaved changes. Save them, discard them and log out, or cancel to keep editing.</Typography>
          {yamlInvalid && <Alert severity="warning">Unfinished YAML cannot be saved. Cancel to repair it, or explicitly discard it and log out.</Alert>}
          {!yamlInvalid && cannotSaveForLogout && <Alert severity="info">Saving is currently unavailable. Wait for pending saves, or cancel to resolve case permissions, session checks or save conflicts.</Alert>}
          {logoutError && <Alert severity="error">{logoutError}</Alert>}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => decideLogout(false)}>Cancel</Button>
          <Button color="warning" disabled={save.isPending || copy.isPending || logoutSaving} onClick={() => decideLogout(true)}>Discard and log out</Button>
          <Button variant="contained" disabled={cannotSaveForLogout} onClick={() => { void saveAndLogout() }}>
            {logoutSaving ? 'Saving…' : dirty ? 'Save and log out' : 'Log out'}
          </Button>
        </DialogActions>
      </Dialog>
      <Snackbar open={!!message} autoHideDuration={6000} onClose={() => setMessage(null)} message={message} />
    </Stack>
    </YamlDraftContext.Provider>
  )
}
