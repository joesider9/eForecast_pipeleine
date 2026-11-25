// A thin typed client over the FastAPI routes (eforecast/config/api.py).  Every call returns parsed JSON;
// a non-2xx response throws an ApiError carrying the status, the detail and the validation errors if any.
import type { ResourcePreview, ResourceView } from './resourceContract.generated'

export type Document = Record<string, any>

export interface VarImpEntry {
  entry: string; kind: 'calendar' | 'column' | 'group' | 'nwp_prefix' | 'unknown' | 'empty'
  matched: string[]; count: number; level: 'ok' | 'warn' | 'error'; note: string
}
export interface VarImpResolution {
  source: 'dataset' | 'declared'; column_count: number; note: string
  rule_sets: { index: number; entries: VarImpEntry[]; columns: number; level: 'ok' | 'warn' | 'error' }[]
  suggestions: { calendar: string[]; columns: string[]; nwp: string[]; nwp_series: string[] }
}

// A13-04 readiness: three outcomes, and for each finding the field it belongs to, what was
// checked, what was found, what follows and one thing to do.
export type Severity = 'ok' | 'attention' | 'blocking'
export type Verdict = 'ready' | 'needs attention' | 'cannot start'
export interface ReadinessFinding {
  check: string; field: string | null; severity: Severity
  checked: string; finding: string; consequence: string; remedy: string
}
export interface Readiness {
  verdict: Verdict; checked_at: string; checked_at_epoch: number
  // The document and runtime this verdict describes. A timestamp alone cannot authorise it.
  inputs_digest: string
  findings: ReadinessFinding[]
  counts: Record<Severity, number>
  coverage: { first: string; last: string; rows: number; span_days: number; truncated: boolean } | null
  bounded: { max_index_rows: number }
}

export interface ValidationIssue { path: string; message: string; type?: string }
export interface Validation { valid: boolean; complete: boolean; missing_sections: string[]; errors: ValidationIssue[] }
export interface CaseSummary {
  name: string; project: string | null; type: string | null; horizon_type: string | null
  complete: boolean; parts: string[]; modified: number
  owner: string | null; shared: boolean; writable: boolean
  // tenant capabilities: writing a case is not the same as disposing of its artifacts
  tenant?: string | null; role?: string | null; manageable?: boolean
}
export interface Me { user: string | null; auth: boolean; role: 'admin' | 'user' | null }
export interface User { name: string; role: 'admin' | 'user' }
export interface CaseResponse { name: string; document: Document; validation: Validation; revision?: string }
export interface SaveResult { main: string | null; parts: string[]; note: string; document?: Document; revision?: string; validation?: Validation }
export interface Browse { path: string | null; parent: string | null; dirs: string[]; files: string[]; roots: string[] }

/** One card as the GPU scheduler of a run sees it. `reserved_mb` is that scheduler's own figure
 *  for the trials it has running on the card -- what they were started for. Nothing is set
 *  aside for it anywhere, so the page calls it "planned for running trials". */
export interface GpuCardUse {
  measured_used_mb: number | null; reserved_mb: number; charged_mb: number | null; free_mb: number
  attribution: string; telemetry: string
}
/** The status file a run writes about its own use of the machine, as it wrote it. `compute` is
 *  the server's typed reading of the same file and what the page shows first; the raw fields
 *  beside it are the measurements that reading does not carry. A file written by an older
 *  version may hold sections this one does not have; nothing here reads them. */
export interface RunResources {
  compute?: ResourceView
  run_id?: string; updated?: number; phase?: string | null; error?: string | null
  cpu?: { physical_cores?: number | null; logical_cpus?: number; affinity_cpus: number; cgroup_quota_cpus: number | null; effective_cpus: number }
  running?: { studies?: number; trials?: number; cpu_threads: number; worker_threads?: number; at_most_trials?: number | null }
  observed?: { pids?: number[]; pss_mb: number; rss_mb: number; os_threads: number; confidence: string }
  gpus?: Record<string, GpuCardUse>
  devices?: Record<string, { key?: string | null; host?: string; name?: string | null; total_mb?: number | null; identified?: boolean }>
  /** Written by the CPU studies of a run only, and read through `compute`. */
  dispatch?: unknown; progress?: unknown; studies?: unknown
}
// ---------------------------------------------------------------- stages, plans and repairs
export type StageDecision = 'reuse' | 'resume' | 'verify' | 'rebuild' | 'blocked' | 'skip'
export interface StageTrialUnit { cluster: string; method: string; reuse: number; redo: number; other_revision: number
  budget?: { done: number; wanted: number; missing: number } | null; study_error?: string
  // The study holds finished trials of another configuration of its family: it is trained from
  // the beginning, and `discard` trial folders are removed first.
  retrain?: boolean; discard?: number
  // The trials of the family's global study that this cluster trains again (`test_tl<n>`): how
  // many there are to start from, how many are trained, how many this run trains, and how many
  // the global study may still add. No study wants them, so `budget` does not count them.
  transfer?: { sources: number; done: number; pending: number; later: number } | null }
export interface StageDecisionEntry {
  key: string; scope: string; decision: StageDecision; reason: string; dependencies: string[]
  status?: string | null; receipt?: string | null; affected?: string[]; affected_unresolved?: string[]
  narrowed_from?: StageDecision; units?: { units: StageTrialUnit[]; totals: Record<string, number> }
  // Why a stage is done again. `configuration`: the case no longer says what it was built from.
  cause?: 'configuration' | 'upstream' | 'requested' | 'missing' | 'interrupted' | 'content' | null
  // extract only: the daily weather files that stay as they are while the stage is checked again
  kept?: number
}
// One setting that differs between two case documents, at the path the editor shows it under.
export interface SettingChange { path: string; change: 'changed' | 'added' | 'removed' | 'reordered'; before?: unknown; after?: unknown }
// What a run would replace because the case no longer says what it was built from: the dataset
// stages built again, and the families trained from the beginning (their trials removed first).
export interface ConfigurationChanges {
  stages: { key: string; reason: string; files: number }[]
  families: Record<string, { units: number; trials: number; clusters: string[] }>
}
// What saving a draft would do to what the case has built (eforecast/config/change_consequences.py).
export interface Consequences {
  version: number; case: string | null
  status: 'resolved' | 'unresolved' | 'new' | 'unchanged'; reason: string | null
  // The settings that change as the case is read. `as_written` counts where the two files
  // differ, which includes defaults a draft merely writes down: those change nothing.
  changes: SettingChange[]; as_written?: number
  datasets: { stage: string; files: number; because: string[] }[]
  families: { method: string; studies: number; trials: number; because: string[] }[]
  results: string[]
  extract: { kept: number } | null
  weather: { from: string; to: string; extracted: boolean } | null
  relocated: { from: string; to: string; holds_work: boolean } | null
  pending: { datasets: { stage: string; files: number; because: string[] }[]
             families: { method: string; studies: number; trials: number; because: string[] }[] }
  restored: { datasets: string[]; families: string[] }
  unresolved: string[]
  // true when a later run would replace datasets or trained models that exist: the save then
  // has to carry `digest` back as its acceptance.
  consequential: boolean; digest: string | null
}
export interface InterruptedPublication {
  stage: string; canonical: 'intact' | 'changed' | 'unknown'; reason: string; recoverable: boolean
  scratch: string | null; marker: string; action?: string; detail?: string
}
export interface ResumePlan {
  action: 'continue' | 'new' | 'rebuild'; verified: boolean; plan_id: string; script: string
  stages: StageDecisionEntry[]; interrupted: InterruptedPublication[]
  totals: Record<string, number>; accepted?: Record<string, StageDecision>
  revision?: { training: string | null; note: string; limit?: string }
}
export interface StagePlanSummary {
  identity: string; execution_order: string[]; blocked: { reason: string; node?: string }[]
  unverified?: { product: string; producer: string; reason: string }[]
  resume?: Record<string, StageDecision>
  nodes: { key: string; scope: string; enabled: boolean; selected: boolean; skip_reason: string | null }[]
  compatibility?: { status: string; reason: string }
}
export interface RunPreview {
  case_revision: string; status: 'ready' | 'verify' | 'blocked' | 'unresolved'; reason?: string
  stage_plan: StagePlanSummary | null; resume: ResumePlan | null
  resource_controls?: ResourcePreview
  // What this run replaces for a changed configuration (null when nothing): it starts only
  // with `accept_changes`. `since` names the settings that changed since the model was built,
  // where the model keeps the configuration it was built from.
  changes?: ConfigurationChanges | null
  since?: { record: Document; changes: SettingChange[] } | null
}
export interface RunStage {
  id: string; stage: string; scope: string; state: string; dependencies: string[]; options: Document
  outcome: { kind: string; detail: string; observed_at: number } | null
  blocked: { reason: string }[]; readiness?: { status: string; id?: string }; attempt_id: string | null
}

export interface Run {
  state?: 'pending' | 'preparing' | 'queued' | 'running' | 'unknown' | 'terminal'
  submission_key?: string | null
  accepted_case_revision?: string | null
  preparation_error?: string | null
  id: string; case: string; script: string; options: Document; command: string[]; pid: number | null
  started: number; finished: number | null; status: 'running' | 'finished' | 'failed' | 'stopped' | 'lost'; returncode: number | null; log: string
  resources?: RunResources
  config_frozen?: boolean; config_revision?: string; config_error?: string; normalized_error?: string
  input_snapshot_status?: 'pending' | 'pinned' | 'degraded'; input_snapshot_version?: number
  effective_revision?: string; effective_options?: Document
  cleanup?: 'not_required' | 'pending' | 'running' | 'unknown' | 'complete' | null
  cleanup_detail?: Document | null
  report_error?: string | null
  worker_result?: { kind: string; detail: string; counts?: Record<string, number> } | null
  outcome?: { kind: string; detail: string; observed_at: number; returncode: number | null } | null
  stages?: RunStage[]
  stage_plan?: StagePlanSummary | null
}
export interface RunLog { text: string; offset: number; next: number; size: number; status: Run['status']; truncated: boolean; generation?: string }
export interface StudySummary {
  n_trials: number; counts: Record<string, number>; best: { number: number; value: number; params: Record<string, any> } | null
  last_activity: string | null; user_attrs: Record<string, any>
}
export interface StudyRow { cluster: string; method: string; folder: string; journal: string | null; trial_folders: number; selected: any; summary: StudySummary | null; error?: string }
export interface CsvTable { columns: string[]; rows: string[][]; path: string; total?: number }
export interface Studies {
  project: string; path_model: string; path_data: string; path_logs: string | null; model_exists: boolean
  datasets: Record<string, { path: string; size_mb: number | null }>
  clusters: Record<string, { folder: string; exists: boolean }>
  studies: StudyRow[]; evaluation: CsvTable | null; summary: CsvTable | null; intervals?: CsvTable | null
  links: { optuna_dashboard: string; grafana: string; metrics: string }
}
export interface Trial { number: number; state: string; value: number | null; duration_s: number | null; params: Record<string, any>; user_attrs: Record<string, any>; started: string | null }
export interface Trials { study: string; trials: Trial[]; param_names: string[]; summary: StudySummary }
export interface ScriptSpec {
  file: string
  defaults: { stages?: string[] }
  options: Record<string, { kind: string; choices: string[] | null }>
}
export type ScriptCatalogue = Record<'datasets' | 'training' | 'combine', ScriptSpec>

export class ApiError extends Error {
  status: number
  errors: ValidationIssue[]
  payload: Record<string, unknown>
  constructor(status: number, message: string, errors: ValidationIssue[] = [], payload: Record<string, unknown> = {}) {
    super(message); this.status = status; this.errors = errors; this.payload = payload
  }
}

let session = new AbortController()
let checkedIdentity: string | undefined

// Every identity transition invalidates imperative requests as well as query fetches.
export function advanceSession() {
  session.abort()
  session = new AbortController()
  checkedIdentity = undefined
}

export const isAbort = (error: unknown) => error instanceof Error && error.name === 'AbortError'

// A14-03's support report is served as text, because a report a person pastes into a message
// is text. Same session/abort wiring as `request`; the only difference is that the body is not
// parsed, and a failure still arrives as an ApiError with the server's own detail.
async function requestText(method: string, url: string, signal?: AbortSignal): Promise<string> {
  const identity = session.signal
  const controller = new AbortController()
  const abort = () => controller.abort()
  for (const source of [identity, signal]) {
    if (source?.aborted) abort()
    else source?.addEventListener('abort', abort, { once: true })
  }
  try {
    const response = await fetch(url, { method, signal: controller.signal, credentials: 'same-origin' })
    const text = await response.text()
    if (controller.signal.aborted) throw new DOMException('Request cancelled', 'AbortError')
    if (response.status === 401) window.dispatchEvent(new CustomEvent('eforecast:unauthorized'))
    if (!response.ok) {
      let detail = text || response.statusText
      try { const body = JSON.parse(text); detail = typeof body?.detail === 'string' ? body.detail : detail } catch { /* text it is */ }
      throw new ApiError(response.status, detail)
    }
    return text
  } finally {
    for (const source of [identity, signal]) source?.removeEventListener('abort', abort)
  }
}

async function request<T>(method: string, url: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  const identity = session.signal
  const controller = new AbortController()
  const abort = () => controller.abort()
  for (const source of [identity, signal]) {
    if (source?.aborted) abort()
    else source?.addEventListener('abort', abort, { once: true })
  }
  try {
    const response = await fetch(url, {
      method, signal: controller.signal,
      headers: body === undefined ? {} : { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body), credentials: 'same-origin',
    })
    const text = await response.text()
    // Also rejects delayed mock/proxy transports which do not honor AbortSignal.
    if (controller.signal.aborted) throw new DOMException('Request cancelled', 'AbortError')
    if (response.status === 401 && !url.endsWith('/api/me') && !url.endsWith('/api/login')) {
      window.dispatchEvent(new CustomEvent('eforecast:unauthorized'))
    }
    let data
    try { data = text ? JSON.parse(text) : null } catch {
      throw new ApiError(response.status, response.ok ? 'The server returned invalid JSON.' :
        `Request failed (${response.status} ${response.statusText || 'server error'}).`)
    }
    if (!response.ok) {
      const detail = typeof data?.detail === 'string' ? data.detail : JSON.stringify(data?.detail ?? response.statusText)
      throw new ApiError(response.status, detail, data?.errors ?? [], data ?? {})
    }
    return data as T
  } finally {
    for (const source of [identity, signal]) source?.removeEventListener('abort', abort)
  }
}

// A14-02: the result summary a colleague reads before using a number. Shapes mirror
// eforecast/config/result_summary.py; nothing here is computed in the browser.
export type VerdictState = 'usable' | 'attention' | 'experimental' | 'none'
export type ScoreRole = 'final' | 'selection' | 'bands' | 'tuning'
export interface SourceFile { path: string; exists: boolean; modified?: number; size_bytes?: number; rows?: number; truncated?: boolean; error?: string }
export interface Metrics { level: string | null; cluster: string | null; predictor: string | null; mae: number | null; rmse: number | null; bias: number | null; n: number | null }
export interface ScoreBlock {
  role: ScoreRole; label: string; period: string; estimates_accuracy: boolean; caption: string
  source: string | null; rows: Record<string, unknown>[]; missing: string | null
}
export interface PublicationAnswer { state: string; issue_id?: string; issued_at?: number; age_seconds?: number; detail: string; remedy?: string }
export interface ResultSummary {
  case: { name: string; type: string | null; horizon_type: string | null; path_model: string; model_exists: boolean }
  verdict: { state: VerdictState; sentence: string; actions: { what: string; why: string }[] }
  model: { state: 'approved' | 'experimental' | 'unknown'; detail: string; remedy?: string
           bundle?: string; revision?: number; approved_at?: number; approved_by?: string
           note?: string; runtimes?: string[]; components?: number; root?: string }
  headline: { served: Metrics | null; best: Metrics | null; is_best: boolean; note?: string; compared?: string } | null
  scores: ScoreBlock[]
  coverage: { recorded: boolean; detail?: string; source?: string
              slices: { slice: string; requested: number; covered: number; missing: number; predictors: number }[]
              worst: { level: string; cluster: string; predictor: string; slice: string; requested: number; covered: number; missing: number }[] }
  period: { declared: Record<string, unknown>; slices_present: string[]; rows_per_slice: Record<string, number>
            written_at: number | null; protocol: string; note?: string | null }
  units: { rated: number | null; rated_recorded: number | null; target: string | null; metrics: string
           steps: number | null; resolution: string | null; horizon_type: string | null; warning?: string }
  freshness: { state: 'current' | 'case_changed' | 'none' | 'unknown'; detail: string; evidence: string
               written_at?: number | null; edited_at?: number | null; run_id?: string | null; remedy?: string }
  produced_by: { state: 'current' | 'case_changed' | 'none' | 'unknown'; detail: string
                 run_id?: string; finished?: number | null; accepted_case_revision?: string
                 case_revision?: string; remedy?: string; document_modified?: number }
  forecast: {
    case?: string; state?: 'unknown'; detail?: string
    forecast?: PublicationAnswer; attempt?: PublicationAnswer
    covered?: { first: string | null; last: string | null; rows: number | null; columns: string[] | null }
    bundle?: string; verified?: boolean; approved_history?: boolean; matches_active?: boolean
    files: { latest: SourceFile; status: SourceFile }
    last_cycle?: { issue_time?: string; state?: string; rows?: number; days?: string[]; warnings?: string[]
                   publication?: { eligible?: boolean; reasons?: string[]; detail?: string[] }; duration_s?: number; error?: string }
  }
  incomplete: { unsupported_clusters: { cluster: string; method: string; reason: string }[]
                unfinished_trials: { cluster: string; method: string; trial: string; reason: string }[]
                missing_artifacts: string[] }
  caveats: { code: string; detail: string; consequence: string; remedy: string }[]
  sources: Record<string, SourceFile>
}

/** §4.1: the installation's own health, as the page reads it.
 *
 * The same checks `bin/eforecast doctor` runs, served from a cached reading. `scope` says how
 * much of it this caller is entitled to: a maintainer sees every finding, everyone else sees
 * the verdict and the findings addressed to them, plus how many are not theirs. */
export interface HealthFinding {
  check: string; field?: string | null; severity: Severity
  checked?: string; finding: string; consequence?: string; remedy?: string
  who?: 'colleague' | 'maintainer'
}

export interface Health {
  verdict: Verdict | null
  counts: Partial<Record<Severity, number>>
  findings: HealthFinding[]
  checked_at: number | null
  age_s: number | null
  stale: boolean
  examined: number
  case: string | null
  scope: 'maintainer' | 'colleague'
  maintainer_findings?: number
  unavailable?: string
}

export const api = {
  me: async (signal?: AbortSignal) => {
    const result = await request<Me>('GET', '/api/me', undefined, signal)
    const identity = JSON.stringify([result.user, result.auth, result.role])
    if (checkedIdentity !== undefined && identity !== checkedIdentity) advanceSession()
    checkedIdentity = identity
    return result
  },
  users: (signal?: AbortSignal) => request<User[]>('GET', '/api/users', undefined, signal),
  createUser: (name: string, password: string, role: string) => request<User>('POST', '/api/users', { name, password, role }),
  setRole: (name: string, role: string) => request<User>('PUT', `/api/users/${encodeURIComponent(name)}/role`, { role }),
  setPassword: (name: string, password: string, current?: string) => request<User>('PUT', `/api/users/${encodeURIComponent(name)}/password`, { password, current }),
  deleteUser: (name: string) => request<null>('DELETE', `/api/users/${encodeURIComponent(name)}`),
  shareCase: (name: string, shared: boolean) => request<{ owner: string | null; shared: boolean }>('PATCH', `/api/cases/${encodeURIComponent(name)}/share`, { shared }),
  login: (name: string, password: string) => request<{ user: string }>('POST', '/api/login', { name, password }),
  logout: () => request<{ user: null }>('POST', '/api/logout'),
  schema: (signal?: AbortSignal) => request<any>('GET', '/api/schema', undefined, signal),
  defaults: (signal?: AbortSignal) => request<Document>('GET', '/api/defaults', undefined, signal),
  options: (signal?: AbortSignal) => request<Document>('GET', '/api/options', undefined, signal),
  cases: (signal?: AbortSignal) => request<CaseSummary[]>('GET', '/api/cases', undefined, signal),
  getCase: (name: string, signal?: AbortSignal) => request<CaseResponse>('GET', `/api/cases/${encodeURIComponent(name)}`, undefined, signal),
  createCase: (name: string, options: { document?: Document; copy_from?: string }, signal?: AbortSignal) =>
    request<SaveResult>('POST', '/api/cases', { name, ...options }, signal),
  // `accepted_consequences` is the digest of the consequences the person was shown and accepted
  // for this save. A save that needs one and has none, or another, answers 428 with them.
  saveCase: (name: string, document: Document, expected_revision?: string, signal?: AbortSignal,
    accepted_consequences?: string) =>
    request<SaveResult>('PUT', `/api/cases/${encodeURIComponent(name)}`,
      { document, expected_revision, accepted_consequences }, signal),
  consequences: (name: string, document: Document, signal?: AbortSignal) =>
    request<Consequences>('POST', `/api/cases/${encodeURIComponent(name)}/consequences`, { document }, signal),
  deleteCase: (name: string) => request<null>('DELETE', `/api/cases/${encodeURIComponent(name)}`),
  validate: (document: Document, signal?: AbortSignal) => request<Validation>('POST', '/api/validate', { document }, signal),
  preview: (document: Document) => request<{ static_data: Document; derived: Document }>('POST', '/api/preview', { document }),
  derived: (document: Document, signal?: AbortSignal) => request<Document>('POST', '/api/derived', { document }, signal),
  applyDataTypes: (document: Document, signal?: AbortSignal) => request<{ document: Document; note: string }>('POST', '/api/apply-data-types', { document }, signal),
  sync: (document: Document, signal?: AbortSignal) => request<{ document: Document; note: string }>('POST', '/api/sync', { document }, signal),
  readiness: (document: Document, path?: string, caseName?: string, signal?: AbortSignal) =>
    request<Readiness>('POST', '/api/readiness', { document, path, case: caseName }, signal),
  browse: (path?: string, mode: 'dir' | 'file' = 'dir', signal?: AbortSignal) =>
    request<Browse>('GET', `/api/browse?mode=${mode}${path ? `&path=${encodeURIComponent(path)}` : ''}`, undefined, signal),
  layers: (signal?: AbortSignal) => request<any>('GET', '/api/layers', undefined, signal),
  graphNormalize: (experiments: Document, signal?: AbortSignal) => request<Document>('POST', '/api/graph/normalize', { experiments }, signal),
  graphPresetTemplates: (source: 'repo' | 'default', path?: string, signal?: AbortSignal) => request<Document>('GET', `/api/graph/preset-templates?source=${source}${path ? `&path=${encodeURIComponent(path)}` : ''}`, undefined, signal),
  graphModuleTemplates: (path: string, signal?: AbortSignal) => request<Document>('GET', `/api/graph/module-templates?path=${encodeURIComponent(path)}`, undefined, signal),
  graphShapes: (document: Document, signal?: AbortSignal) => request<Document>('POST', '/api/graph/shapes', { document }, signal),
  graphDryBuild: (document: Document, tag: string, method: string | null, signal?: AbortSignal) => request<any>('POST', '/api/graph/dry-build', { document, tag, method }, signal),
  scripts: (signal?: AbortSignal) => request<ScriptCatalogue>('GET', '/api/scripts', undefined, signal),
  runs: (name: string, signal?: AbortSignal) => request<Run[]>('GET', `/api/cases/${encodeURIComponent(name)}/runs`, undefined, signal),
  startRun: (name: string, script: string, options: Document, signal?: AbortSignal,
    submission?: { client_key: string; expected_revision?: string; action?: string; accept_changes?: boolean }) =>
    request<Run>('POST', `/api/cases/${encodeURIComponent(name)}/runs`, { script, options, ...submission }, signal),
  previewRun: (name: string, script: string, options: Document, action: string, expected_revision?: string, signal?: AbortSignal) =>
    request<RunPreview>('POST', `/api/cases/${encodeURIComponent(name)}/runs/preview`,
      { script, options, action, expected_revision }, signal),
  resolveRun: (name: string, id: string, reason: string, signal?: AbortSignal) =>
    request<Run>('POST', `/api/cases/${encodeURIComponent(name)}/runs/${encodeURIComponent(id)}/resolve`, { reason }, signal),
  recoverStages: (name: string, apply: boolean, signal?: AbortSignal) =>
    request<InterruptedPublication[]>('POST', `/api/cases/${encodeURIComponent(name)}/stages/recover`, { apply }, signal),
  runLog: (name: string, id: string, offset: number, signal?: AbortSignal) => request<RunLog>('GET', `/api/cases/${encodeURIComponent(name)}/runs/${id}/log?offset=${offset}`, undefined, signal),
  // A14-03: the sanitized report a colleague hands a maintainer. Text, because its two uses
  // are pasting it into a message and saving it as a file.
  supportReport: (name: string, id: string, signal?: AbortSignal) =>
    requestText('GET', `/api/cases/${encodeURIComponent(name)}/runs/${encodeURIComponent(id)}/support?text=true`, signal),
  supportReportDownload: (name: string, id: string) =>
    `/api/cases/${encodeURIComponent(name)}/runs/${encodeURIComponent(id)}/support?text=true`,
  runLogDownload: (name: string, id: string) => `/api/cases/${encodeURIComponent(name)}/runs/${encodeURIComponent(id)}/log/download`,
  stopRun: (name: string, id: string, signal?: AbortSignal) => request<Run>('POST', `/api/cases/${encodeURIComponent(name)}/runs/${id}/stop`, undefined, signal),
  deleteRun: (name: string, id: string, signal?: AbortSignal) => request<null>('DELETE', `/api/cases/${encodeURIComponent(name)}/runs/${id}`, undefined, signal),
  results: (name: string, signal?: AbortSignal) => request<ResultSummary>('GET', `/api/cases/${encodeURIComponent(name)}/results`, undefined, signal),
  // The reading is cached, so asking is cheap; `refresh` takes a new one and is a maintainer's
  // to ask for, because it costs seconds of the machine's time.
  health: (options?: { case?: string; refresh?: boolean }, signal?: AbortSignal) => {
    const query = new URLSearchParams()
    if (options?.case) query.set('case', options.case)
    if (options?.refresh) query.set('refresh', 'true')
    const suffix = query.toString()
    return request<Health>('GET', `/api/doctor${suffix ? `?${suffix}` : ''}`, undefined, signal)
  },
  studies: (name: string, load = true, signal?: AbortSignal) => request<Studies>('GET', `/api/cases/${encodeURIComponent(name)}/studies?load=${load}`, undefined, signal),
  trials: (name: string, cluster: string, method: string, signal?: AbortSignal) => request<Trials>('GET', `/api/cases/${encodeURIComponent(name)}/studies/${encodeURIComponent(cluster)}/${encodeURIComponent(method)}`, undefined, signal),
  familyPreset: (name: string, document: Document, signal?: AbortSignal) => request<Document>('POST', '/api/presets/family', { name, document }, signal),
  resolveRbfVariables: (document: Document, rule_sets: unknown, signal?: AbortSignal) =>
    request<VarImpResolution>('POST', '/api/clustering/resolve-variables', { document, rule_sets }, signal),
  presets: (params: Record<string, string | number>, signal?: AbortSignal) =>
    request<Document[]>('GET', '/api/presets/variables?' + new URLSearchParams(params as Record<string, string>).toString(), undefined, signal),
}
