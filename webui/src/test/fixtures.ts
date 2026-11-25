import type { CaseResponse, CaseSummary, Consequences, Document, Run, RunPreview, RunResources, RunStage, ScriptCatalogue, Validation } from '../api/client'

export const valid: Validation = { valid: true, complete: true, missing_sections: [], errors: [] }
export const caseDocument: Document = {
  project: { name: 'baseline', metadata: { note: 'preserve me', enabled: false, interval_coverage: [] } },
  jobs: { n_cpus: 4 }, folders: {}, data: {}, methods: {},
  extension: { legacy_zero: 0, custom_search: { choices: [null, 2, 4] } },
}

// Small schema fixture for state/HTTP tests. The exported production schema has
// its own round-trip checks and is served unchanged in browser workflows.
export const formSchema = {
  type: 'object', properties: {
    project: { $ref: '#/$defs/Project' }, jobs: { $ref: '#/$defs/Jobs' },
    folders: { type: 'object', properties: {} }, data: { type: 'object', properties: {} },
    methods: { type: 'object', properties: {} },
  }, $defs: {
    Project: { type: 'object', properties: { name: { type: 'string', title: 'Project name' },
      metadata: { type: 'object', title: 'Metadata', additionalProperties: true } } },
    Jobs: { type: 'object', properties: { n_cpus: { type: 'integer', title: 'CPU count' } } },
  },
}

export const snapshot = (document: Document = caseDocument, revision = 'v1', name = 'demo'): CaseResponse => ({
  name, document: structuredClone(document), validation: valid, revision,
})
export const caseSummary = (name = 'demo', project = 'baseline'): CaseSummary => ({
  name, project, type: 'pv', horizon_type: 'short_term', complete: true, parts: [], modified: 1,
  owner: 'alice', shared: false, writable: true,
  // the real payload carries the tenant capability as well: the owner of this fixture manages it
  tenant: 'alice', role: 'owner', manageable: true,
})
export const runFixture = (status: Run['status'] = 'running', id = 'run-1'): Run => ({
  id, case: 'demo', script: 'training', options: {}, command: ['python', 'run_training.py', '--config', 'main.yaml'],
  pid: 123, started: 1, finished: status === 'running' ? null : 2, status,
  returncode: status === 'running' ? null : 0, log: '/logs/demo.log',
})

// What the CPU studies of a run report while two of three run and the third waits its turn: the
// status file as written, and beside it the server's reading of it (`compute`). The case asks
// for four CPUs and leaves the rest automatic; the process may use two.
export const resourceFixture = (): RunResources => ({
  run_id: 'run-1', updated: Date.now() / 1000, phase: 'running',
  cpu: { physical_cores: 8, logical_cpus: 16, affinity_cpus: 2, cgroup_quota_cpus: null, effective_cpus: 2 },
  running: { studies: 2, cpu_threads: 2 },
  observed: { pids: [4101, 4102], pss_mb: 1536, rss_mb: 2048, os_threads: 9, confidence: 'measured_pss' },
  gpus: {},
  compute: { version: 2, source: 'cpu_studies', updated: Date.now() / 1000, phase: 'running', error: null,
    running_threads: 2, pss_mb: 1536, pss_confidence: 'measured_pss',
    dispatch: { requested_workers: null, effective_workers: 2, requested_threads: null, threads_per_study: 1,
      requested_total_threads: 4, total_threads: 2, trials_per_study: 1, studies: 3 },
    progress: { verified_complete: 2, pruned: 1, failed: 0, elapsed_seconds: 120, verified_models_per_minute: 1 },
    studies: [
      { id: 'global:global:Chronos2:study', method: 'Chronos2', cluster: 'global', state: 'active', threads: 1,
        requested_device: 'auto', resolved_device: 'cuda', normalization: null, reason: null },
      { id: 'cluster:rule-a:RF:study', method: 'RF', cluster: 'rule-a', state: 'active', threads: 1,
        requested_device: 'cpu', resolved_device: 'cpu',
        normalization: 'Shallow models use CPU; a foundation device preference does not apply.', reason: null },
      { id: 'cluster:rule-b:RF:study', method: 'RF', cluster: 'rule-b', state: 'waiting', threads: null,
        requested_device: null, resolved_device: null, normalization: null,
        reason: 'its turn has not come: the studies that run at once are running' },
    ],
    warnings: [] },
})

const stages = { kind: 'list', choices: ['all', 'row', 'split', 'preprocess', 'clustering', 'features'] }
const methods = { kind: 'choice', choices: ['MLP', 'RF'] }
export const scripts: ScriptCatalogue = {
  datasets: { file: 'run_datasets.py', defaults: { stages: ['all'] }, options: { stages } },
  training: { file: 'run_training.py', defaults: {}, options: { global_method: methods, cluster_method: methods, cpu_method: methods } },
  combine: { file: 'run_combine.py', defaults: { stages: ['all'] }, options: { stages } },
}

export const stageFixture = (stage: string, overrides: Partial<RunStage> = {}): RunStage => ({
  id: `${stage}-task`, stage, scope: 'global', state: 'terminal', dependencies: [], options: {},
  outcome: { kind: 'success', detail: `${stage} observed`, observed_at: 2 },
  blocked: [], readiness: { status: 'verified' }, attempt_id: `${stage}-attempt`, ...overrides,
})

export const previewFixture = (overrides: Partial<RunPreview> = {}): RunPreview => ({
  case_revision: 'saved-v1', status: 'ready',
  stage_plan: { identity: 'plan-identity', execution_order: ['target', 'row'], blocked: [], unverified: [],
    resume: { target: 'reuse', row: 'rebuild' },
    nodes: [{ key: 'target', scope: 'global', enabled: true, selected: true, skip_reason: null },
            { key: 'row', scope: 'global', enabled: true, selected: true, skip_reason: null }] },
  resume: { action: 'continue', verified: true, plan_id: 'plan-identity', script: 'datasets', interrupted: [],
    stages: [
      { key: 'target', scope: 'global', decision: 'reuse', reason: 'content and input identities match the receipt',
        dependencies: [], status: 'verified' },
      { key: 'row', scope: 'global', decision: 'rebuild', reason: 'this stage has produced nothing yet',
        dependencies: ['target'], affected: ['/models/DATA/dataset_row_data.pickle'] }],
    totals: { reuse: 1, resume: 0, verify: 0, rebuild: 1, blocked: 0, skip: 0, artifacts_to_replace: 1 } },
  ...overrides,
})

// What the server answers when a draft changes a variable of a case that has datasets and
// trained families: the datasets are built again, and every family with them.
export const consequencesFixture = (overrides: Partial<Consequences> = {}): Consequences => ({
  version: 1, case: 'demo', status: 'resolved', reason: null,
  changes: [{ path: 'data.variables[Cloud].lags', change: 'changed', before: [0, 1, 2], after: [0, 1, 2, 3] }],
  datasets: [{ stage: 'nwp', files: 1, because: ['variables'] }, { stage: 'row', files: 1, because: ['variables'] }],
  families: [{ method: 'TRANSFORMER', studies: 8, trials: 101, because: ['variables'] }],
  results: ['predict', 'select', 'combine', 'evaluate'],
  extract: { kept: 911 }, weather: null, relocated: null,
  pending: { datasets: [], families: [] }, restored: { datasets: [], families: [] }, unresolved: [],
  consequential: true, digest: 'digest-1', ...overrides,
})
