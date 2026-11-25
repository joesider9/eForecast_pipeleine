import { Box, Stack, Typography } from '@mui/material'
import type { GpuCardUse, RunResources } from '../api/client'
import type { ResourceControls, ResourcePreview, ResourceProgress, ResourceView, StudyDispatch, StudyResources } from '../api/resourceContract.generated'

// What a run is seen to use. Nothing shown here is a limit or something set aside: the numbers
// say what is running and what the machine was seen to hold for it. A snapshot can come from a
// version that wrote other things, or be damaged, so the names are the contract's and every
// value is checked before it is shown; one that is missing is called unknown or left out.
type Loose<T> = { [K in keyof T]?: unknown }
const isObject = (value: unknown) => value !== null && typeof value === 'object' && !Array.isArray(value)
const loose = <T,>(value: unknown): Loose<T> => isObject(value) ? value as Loose<T> : {}
const number = (value: unknown) => typeof value === 'number' && Number.isFinite(value) ? value : undefined
const text = (value: unknown) => typeof value === 'string' && value ? value : undefined
const amount = (value: unknown, unit = '') => { const known = number(value); return known === undefined ? 'unknown' : `${Math.round(known)}${unit}` }
const asked = (value: unknown) => { const known = number(value); return known === undefined ? 'automatic' : String(Math.round(known)) }
const words = (value: string) => value.replace(/_/g, ' ')
const several = (count: number, one: string, many: string) => `${count} ${count === 1 ? one : many}`
const cpus = (count: number) => Number.isInteger(count) ? String(count) : count.toFixed(1)

// Plain words for the terms a snapshot uses; a term that has none is shown as it was written.
type Words = Readonly<Record<string, string | undefined>>
const worded = (table: Words, term: unknown) =>
  typeof term === 'string' && Object.prototype.hasOwnProperty.call(table, term) ? table[term] : undefined
const STUDY_STATE: Words = { active: 'running', cleanup: 'stopping', study_finished: 'finished' }
const CONFIDENCE: Words = { measured_pss: 'measured' }
const ATTRIBUTION: Words = {
  per_pid: 'memory counted per process',
  unknown_conservative: 'memory per process unknown, so every trial counts in full',
}
// Who wrote the snapshot, as the server names it. A name the contract gains is a compile error
// here until it has words. 'legacy' has none of its own: it is every writer the server does not
// name, such as a file an older version wrote.
const WRITER: Record<NonNullable<ResourceView['source']>, string | undefined> = {
  cpu_studies: 'CPU studies', gpu_trials: 'GPU trials', worker: 'CPU study worker', legacy: undefined,
}

export function ResourcePreviewReport({ value }: { value: ResourcePreview }) {
  const controls = loose<ResourceControls>(value.controls)
  return <Stack spacing={0.5} role="group" aria-label="Requested study resources">
    <Typography variant="subtitle2">Requested study resources</Typography>
    <Typography variant="body2">
      Independent studies at a time: {asked(controls.cpu_workers)} · threads per study: {asked(controls.cpu_study_threads)}
      {' · '}CPUs for the studies: {amount(controls.n_cpus)}
    </Typography>
    <Typography variant="body2">Foundation device: {text(controls.cpu_study_device) ?? 'family setting'}</Typography>
    <Typography variant="caption" color="text.secondary">{value.note}</Typography>
  </Stack>
}

// Where the numbers come from. A record that carries no reading from the server is told apart
// by what it wrote: the trials on the GPUs count their trials, a file of an older version does not.
function origin(view: Loose<ResourceView>, raw: Loose<RunResources>) {
  return worded(WRITER, view.source) ??
    (number(loose<NonNullable<RunResources['running']>>(raw.running).trials) === undefined ? 'Older record' : 'GPU trials')
}

// A record without the server's reading is read for the little it says by itself.
function reading(raw: Loose<RunResources>): Loose<ResourceView> {
  const view = loose<ResourceView>(raw.compute)
  if (Object.keys(view).length) return view
  const running = loose<NonNullable<RunResources['running']>>(raw.running)
  const observed = loose<NonNullable<RunResources['observed']>>(raw.observed)
  return { updated: raw.updated, phase: raw.phase, error: raw.error, running_threads: running.cpu_threads,
    pss_mb: observed.pss_mb, pss_confidence: observed.confidence }
}

function StudyRow({ value }: { value: Loose<StudyResources> }) {
  const state = text(value.state) ?? 'unknown'
  const threads = number(value.threads)
  const wanted = text(value.requested_device), used = text(value.resolved_device)
  const reason = state === 'waiting' ? text(value.reason) : undefined
  const note = text(value.normalization)
  return <Stack spacing={0.25}>
    <Typography variant="caption">
      {text(value.cluster) ?? 'unknown'} / {text(value.method) ?? 'unknown'} · {worded(STUDY_STATE, state) ?? words(state)}
      {reason && ` — ${reason}`}
      {threads !== undefined && ` · ${several(Math.round(threads), 'thread', 'threads')}`}
      {(wanted || used) && (wanted === used ? ` · device ${used}` : ` · device asked ${wanted ?? 'unknown'}, used ${used ?? 'unknown'}`)}
    </Typography>
    {note && <Typography variant="caption" color="text.secondary">{note}</Typography>}
  </Stack>
}

function CardRow({ ordinal, value }: { ordinal: string; value: unknown }) {
  if (!isObject(value)) return <Typography variant="caption">GPU {ordinal}: measurement unavailable</Typography>
  const card = loose<GpuCardUse>(value)
  const used = number(card.measured_used_mb), planned = number(card.reserved_mb), free = number(card.free_mb)
  const attribution = text(card.attribution)
  return <Typography variant="caption">
    GPU {ordinal}: {used === undefined ? 'memory in use not measured' : `${Math.round(used)} MiB in use`}
    {planned !== undefined && ` · ${Math.round(planned)} MiB planned for running trials`}
    {free !== undefined && ` · ${Math.max(0, Math.round(free))} MiB free for another trial`}
    {attribution && ` · ${worded(ATTRIBUTION, attribution) ?? words(attribution)}`}
  </Typography>
}

export default function ResourceReport({ resources, live }: { resources?: RunResources; live: boolean }) {
  const raw = loose<RunResources>(resources)
  const view = reading(raw)
  const updated = number(view.updated), phase = text(view.phase), error = text(view.error)
  const threads = number(view.running_threads), memory = number(view.pss_mb), confidence = text(view.pss_confidence)
  const limit = number(view.trial_limit) ?? number(loose<NonNullable<RunResources['running']>>(raw.running).at_most_trials)
  const dispatch = isObject(view.dispatch) ? loose<StudyDispatch>(view.dispatch) : undefined
  const studies = (Array.isArray(view.studies) ? view.studies : []).filter(isObject).map(item => loose<StudyResources>(item))
  const progress = loose<ResourceProgress>(view.progress)
  const counted = number(progress.verified_complete) !== undefined
  const warnings = (Array.isArray(view.warnings) ? view.warnings : []).filter((item): item is string => typeof item === 'string')
  const cards = Object.entries(loose<Record<string, GpuCardUse>>(raw.gpus))
  const cpu = loose<NonNullable<RunResources['cpu']>>(raw.cpu)
  const usable = number(cpu.effective_cpus), logical = number(cpu.logical_cpus), quota = number(cpu.cgroup_quota_cpus)
  // Nothing to show: no snapshot was written, the file could not be read and says so, or it
  // holds only what an older version wrote and this one does not read.
  if (updated === undefined && phase === undefined && threads === undefined && memory === undefined && !dispatch &&
      !studies.length && !counted && !cards.length && usable === undefined) {
    return <Typography variant="caption" color={error ? 'error' : 'text.secondary'}>
      {error ?? (resources ? 'No resource measurements in this record' : 'No resource snapshot')}
    </Typography>
  }
  // A live run rewrites its snapshot as it goes; one that has stopped doing so is said to have.
  const age = updated === undefined ? undefined : Date.now() / 1000 - updated
  const stale = live && (age === undefined || age > 120)
  const total = number(dispatch?.studies)
  return <Stack spacing={0.5} sx={{ minWidth: 230, maxWidth: 440, overflowWrap: 'anywhere' }} role="group" aria-label="Run resources">
    <Typography variant="caption" color={stale ? 'warning.main' : 'text.secondary'}>
      {origin(view, raw)}{phase ? ` · ${words(phase)}` : ''}
      {stale && (age === undefined ? ' · time of the last update unknown' : ' · not updated for over 2 minutes')}
    </Typography>
    {error && <Typography variant="caption" color="error">{error}</Typography>}
    {dispatch && <Typography variant="caption">
      Independent studies at a time: {amount(dispatch.effective_workers)} (asked: {asked(dispatch.requested_workers)})
      {' · '}threads per study: {amount(dispatch.threads_per_study)}{number(dispatch.requested_threads) === undefined ? ' or more' : ''} (asked: {asked(dispatch.requested_threads)})
      {' · '}CPUs for the studies: {amount(dispatch.total_threads)} (asked: {asked(dispatch.requested_total_threads)})
      {total !== undefined && ` · ${several(Math.round(total), 'study', 'studies')} in all`}
    </Typography>}
    {usable !== undefined && <Typography variant="caption">
      CPUs this run may use: {cpus(usable)}{logical !== undefined && logical !== usable ? ` of ${cpus(logical)}` : ''}
      {quota !== undefined && ` · container limit ${cpus(quota)}`}
    </Typography>}
    <Typography variant="caption">Threads in use by running work: {amount(threads)}</Typography>
    {limit !== undefined && <Typography variant="caption" color="warning.main">
      At most {several(Math.round(limit), 'trial', 'trials')} at a time now: the machine ran out of memory with more running
    </Typography>}
    <Typography variant="caption" title="PSS: memory shared between processes is counted once">
      Memory held by this run's processes (PSS): {amount(memory, ' MiB')} · {confidence ? worded(CONFIDENCE, confidence) ?? words(confidence) : 'measurement unavailable'}
    </Typography>
    {counted && <Typography variant="caption">
      Verified models: {amount(progress.verified_complete)} · {number(progress.verified_models_per_minute)?.toFixed(2) ?? 'unknown'} per minute
      {' '}over {amount(progress.elapsed_seconds, ' s')} · {amount(progress.pruned)} pruned · {amount(progress.failed)} failed.
      {' '}Counted as studies finish; models reused from an earlier run and pruned trials are not in the rate.
    </Typography>}
    {studies.length > 0 && <Box component="details" open>
      <Typography component="summary" variant="caption">Studies ({studies.length})</Typography>
      <Stack spacing={0.75} sx={{ mt: 0.5 }}>
        {studies.map((study, index) => <StudyRow key={`${index}:${text(study.id) ?? ''}`} value={study} />)}
      </Stack>
    </Box>}
    {cards.map(([ordinal, card]) => <CardRow key={ordinal} ordinal={ordinal} value={card} />)}
    {warnings.map((warning, index) => <Typography key={index} variant="caption" color="warning.main">{warning}</Typography>)}
  </Stack>
}
