import type { Run, RunStage } from '../api/client'

/**
 * One place that turns a run's durable state into something a colleague can act on -- A14-01.
 *
 * Every field read here is already recorded on the run: its state, its stages and what the
 * cleanup reconciler found. Nothing is inferred and nothing is predicted: there is
 * deliberately no ETA, because an invented one is worse than silence -- it gets planned around.
 *
 * Three distinctions the spec insists on, and each is easy to lose:
 *  - a failed *request* is not "no runs": that belongs to the caller, which must not pass a
 *    run it does not have;
 *  - "the forecast failed" is not "cleanup is still running", and the remedy differs;
 *  - uncertain work is never called finished.
 */

export type RunStatusKind =
  | 'queued' | 'preparing' | 'working' | 'waiting'
  | 'finished' | 'failed' | 'stopped' | 'cleanup' | 'uncertain'

export interface RunStatusView {
  kind: RunStatusKind
  /** What is happening, in one plain sentence. */
  headline: string
  /** The stage being worked on, when one is. */
  step?: string
  /** Work already done, counted rather than estimated. */
  progress?: string
  /** Why it is not moving, when it is not. */
  reason?: string
  /** What a failure means for the result. */
  consequence?: string
  /** One action, or none when waiting is the only correct thing to do. */
  remedy?: string
  /** Technical text for an expandable section; never the headline. */
  detail?: string
  /** True when a retry or Continue would be wrong right now. */
  settling?: boolean
}

const CLEANUP_SETTLING = new Set(['pending', 'running'])

function stageProgress(stages: RunStage[] | undefined) {
  if (!stages?.length) return { progress: undefined, step: undefined, blocked: undefined }
  const done = stages.filter(s => s.outcome && ['success', 'nothing_to_do', 'complete', 'reused'].includes(s.outcome.kind)).length
  const active = stages.find(s => s.state === 'running' || s.state === 'active')
  const blocked = stages.find(s => s.blocked?.length)
  return {
    progress: `${done} of ${stages.length} stages done`,
    step: active?.stage,
    blocked: blocked?.blocked?.[0]?.reason,
  }
}

/** The first error the run recorded, with the field it came from, or null. */
function firstError(run: Run): { field: string; text: string } | null {
  const candidates: [string, string | null | undefined][] = [
    ['preparation', run.preparation_error],
    ['configuration', run.config_error],
    ['options', run.normalized_error],
    ['report', run.report_error],
  ]
  for (const [field, text] of candidates) if (text) return { field, text }
  return null
}

export function describe(run: Run): RunStatusView {
  const { progress, step, blocked } = stageProgress(run.stages)
  const cleanup = run.cleanup ?? null
  const settling = !!cleanup && CLEANUP_SETTLING.has(cleanup)
  const recorded = firstError(run)
  const outcome = run.outcome?.detail || run.worker_result?.detail || null

  if (cleanup === 'unknown') {
    return {
      kind: 'uncertain', headline: 'Cleanup could not be confirmed.',
      step, progress, settling: true,
      consequence: 'A worker may still be using resources. This run must not be treated as fully stopped.',
      remedy: `Ask the lab maintainer to check run ${run.id} before starting it again.`,
    }
  }

  // A failure and an unfinished cleanup are two different facts, and the remedy differs: a
  // retry while the reconciler is still releasing what the attempt held would collide with
  // it. So the cleanup sentence comes first and no retry is offered yet.
  if (run.status === 'failed' || run.status === 'stopped') {
    const stopped = run.status === 'stopped'
    const head = stopped ? 'This run was stopped.' : 'This run failed.'
    return {
      kind: settling ? 'cleanup' : stopped ? 'stopped' : 'failed',
      headline: settling ? `${head} Cleanup is still running.` : head,
      step, progress, settling,
      consequence: settling
        ? 'No result was produced, and what the attempt held is still being released.'
        : stopped
          ? 'No result was produced from this attempt.'
          : 'No result was produced from this attempt; anything it had already finished is kept.',
      reason: recorded ? `the ${recorded.field} was rejected` : blocked ?? undefined,
      remedy: settling
        ? 'Wait for cleanup to finish, then start a new run. Retrying now would collide with it.'
        : recorded
          ? `Correct the ${recorded.field} and run the plan again; the preview will re-check it.`
          : `Start a new run, or send a support report quoting run ${run.id}.`,
      detail: recorded?.text ?? outcome ?? undefined,
    }
  }

  if (run.status === 'lost' || run.state === 'unknown') {
    return {
      kind: 'uncertain',
      headline: 'This run cannot be accounted for.',
      step, progress, settling,
      consequence: 'Whether it finished its work is unknown, so its result must not be '
        + 'treated as complete.',
      remedy: `Ask the maintainer to reconcile it, quoting run ${run.id}.`,
      detail: outcome ?? undefined,
    }
  }

  if (run.status === 'finished') {
    if (settling) {
      return {
        kind: 'cleanup',
        headline: 'The work finished. Cleanup is still running.',
        step, progress, settling: true,
        consequence: 'The result is there; what the run held is still being released.',
        remedy: 'Nothing to do. The run remains in history after cleanup confirms.',
      }
    }
    return { kind: 'finished', headline: 'Finished.', progress, settling: false }
  }

  if (blocked) {
    return {
      kind: 'waiting', headline: 'Not started yet.', step, progress, settling,
      reason: blocked,
      remedy: 'Supply what the reason names, then run the plan again.',
    }
  }

  if (run.state === 'preparing' || run.state === 'pending') {
    return {
      kind: 'preparing', headline: 'Preparing its inputs.', step, progress, settling,
      remedy: 'Nothing to do.',
    }
  }
  if (run.state === 'queued' || (run.pid === null && run.state !== 'running')) {
    return {
      kind: 'queued', headline: 'Queued.', step, progress, settling,
      remedy: 'Nothing to do.',
    }
  }
  return {
    kind: 'working',
    headline: step ? `Working on ${step}.` : 'Working.',
    step, progress, settling,
    remedy: 'Nothing to do. Stop it if you no longer want the result.',
  }
}

/** Colour for a status chip. Kept beside the vocabulary so the two cannot drift. */
export const STATUS_COLOUR: Record<RunStatusKind, 'default' | 'info' | 'success' | 'error' | 'warning'> = {
  queued: 'default', preparing: 'default', working: 'info', waiting: 'warning',
  finished: 'success', failed: 'error', stopped: 'warning', cleanup: 'warning',
  uncertain: 'error',
}
