import { describe as group, expect, it } from 'vitest'
import type { Run } from '../api/client'
import { describe as describeRun, STATUS_COLOUR } from './runStatus'

const base = (over: Partial<Run> = {}): Run => ({
  id: 'run-42', case: 'pv_park', script: 'run_training.py', options: {}, command: ['python'],
  pid: 1234, started: 0, finished: null, status: 'running', returncode: null, log: '/tmp/l',
  ...over,
})

// A run that has not started, in the states a record can show it in.
const notStarted: Partial<Run>[] = [
  { pid: null, state: 'pending' }, { pid: null, state: 'preparing' }, { pid: null, state: 'queued' },
]

group('a run says what its record says, and nothing it would have to guess', () => {
  it('never offers an action when waiting is the correct thing to do', () => {
    for (const state of notStarted) {
      const view = describeRun(base(state))
      expect(['preparing', 'queued']).toContain(view.kind)
      expect(view.remedy).toMatch(/Nothing to do/)
    }
  })

  it('keeps the recorded text in the detail, not in the headline', () => {
    const view = describeRun(base({ status: 'failed', cleanup: 'complete',
      outcome: { kind: 'failure', detail: 'no node matches gpu=2080Ti,cc=7.5', observed_at: 2, returncode: 1 } }))
    expect(view.detail).toContain('2080Ti')          // the executor may show through here
    expect(view.headline).not.toContain('2080Ti')
  })

  it('invents no estimate', () => {
    const stages = [
      { id: '1', stage: 'datasets', scope: 'global', state: 'done', dependencies: [], options: {},
        outcome: { kind: 'complete', detail: '', observed_at: 1 }, blocked: [], attempt_id: null },
      { id: '2', stage: 'training', scope: 'global', state: 'running', dependencies: [], options: {},
        outcome: null, blocked: [], attempt_id: 'a' },
    ]
    const views = [...notStarted, { stages }, { state: 'running' as const }, { status: 'finished' as const, cleanup: 'pending' as const }]
      .map(state => describeRun(base(state)))
    for (const view of views) {
      expect(JSON.stringify(view)).not.toMatch(/\beta\b|estimat|minutes? (left|remaining)|should finish/i)
    }
  })

  // A record written by an older version may still say where the run stood in a queue; this
  // version has no such queue and reads the run's own state.
  it('reads the state of a run, not what an older version recorded about its place in a queue', () => {
    const old: unknown = { ...base({ pid: null, state: 'queued' }), profile_error: 'the profile store could not be read',
      dispatch: { status: 'waiting', reserved_bytes: 0, reservations: [
        { key: 'k', kind: 'run', state: 'waiting', reason: 'queued behind 2', ticket: 1, wait_kind: 'compute', roots: [] }] } }
    const view = describeRun(old as Run)
    expect(view).toEqual(describeRun(base({ pid: null, state: 'queued' })))
    expect(view.kind).toBe('queued')
    // nor does a failure blame a setting that no longer exists
    const failed = describeRun({ ...(old as Run), status: 'failed', cleanup: 'complete' })
    expect(failed.reason).toBeUndefined()
    expect(failed.remedy).toMatch(/support report/)
  })
})

group('a failure and an unfinished cleanup are different facts', () => {
  it('reports the failure alone when cleanup is not required', () => {
    const view = describeRun(base({ status: 'failed', cleanup: 'not_required',
                                    preparation_error: 'target column missing' }))
    expect(view.kind).toBe('failed')
    expect(view.headline).toBe('This run failed.')
    expect(view.consequence).toContain('No result was produced')
    expect(view.remedy).toContain('Correct the preparation')
    expect(view.detail).toBe('target column missing')
    expect(view.settling).toBe(false)
  })

  it('says cleanup is still running and refuses to suggest a retry', () => {
    const view = describeRun(base({ status: 'failed', cleanup: 'running' }))
    expect(view.kind).toBe('cleanup')
    expect(view.headline).toContain('Cleanup is still running')
    expect(view.remedy).toContain('Wait for cleanup')
    expect(view.remedy).toContain('collide')
    expect(view.settling).toBe(true)
  })

  it('quotes the run id when there is nothing to correct', () => {
    const view = describeRun(base({ status: 'failed', cleanup: 'complete' }))
    expect(view.remedy).toContain('run-42')
    expect(view.remedy).toMatch(/support report/)
  })

  it('distinguishes stopped from failed', () => {
    const view = describeRun(base({ status: 'stopped', cleanup: 'complete' }))
    expect(view.kind).toBe('stopped')
    expect(view.headline).toBe('This run was stopped.')
  })
})

group('uncertain work is never called finished', () => {
  it('reports a lost run as unaccounted for', () => {
    const view = describeRun(base({ status: 'lost' }))
    expect(view.kind).toBe('uncertain')
    expect(view.consequence).toContain('must not be')
    expect(view.remedy).toContain('run-42')
    expect(view.headline).not.toMatch(/finish/i)
  })

  it('reports an unknown state the same way', () => {
    expect(describeRun(base({ state: 'unknown' })).kind).toBe('uncertain')
  })

  it('does not call a finished run clean while cleanup is still running', () => {
    const view = describeRun(base({ status: 'finished', finished: 2, cleanup: 'pending' }))
    expect(view.kind).toBe('cleanup')
    expect(view.headline).toContain('Cleanup is still running')
    expect(view.remedy).toContain('Nothing to do')
  })

  it('calls a finished run finished once cleanup has confirmed', () => {
    const view = describeRun(base({ status: 'finished', finished: 2, cleanup: 'complete' }))
    expect(view.kind).toBe('finished')
    expect(view.settling).toBe(false)
  })
})

group('the current step and the work already done', () => {
  const stages = [
    { id: '1', stage: 'datasets', scope: 'global', state: 'done', dependencies: [], options: {},
      outcome: { kind: 'complete', detail: '', observed_at: 1 }, blocked: [], attempt_id: null },
    { id: '2', stage: 'training', scope: 'global', state: 'running', dependencies: [], options: {},
      outcome: null, blocked: [], attempt_id: 'a' },
    { id: '3', stage: 'combine', scope: 'global', state: 'pending', dependencies: [], options: {},
      outcome: null, blocked: [], attempt_id: null },
  ]

  it('counts what is done rather than estimating what is left', () => {
    const view = describeRun(base({ stages }))
    expect(view.progress).toBe('1 of 3 stages done')
    expect(view.step).toBe('training')
    expect(view.headline).toBe('Working on training.')
  })

  it('surfaces a blocked stage as the reason, with an action', () => {
    const blocked = [{ ...stages[0] }, { ...stages[1], state: 'pending',
      blocked: [{ reason: 'the weather files for 2026-05 are missing' }] }]
    const view = describeRun(base({ pid: null, state: 'queued', stages: blocked }))
    expect(view.kind).toBe('waiting')
    expect(view.reason).toContain('weather files')
    expect(view.remedy).toContain('run the plan again')
  })
})

group('the ordinary states', () => {
  it('reports preparing, queued and working', () => {
    expect(describeRun(base({ state: 'preparing' })).kind).toBe('preparing')
    expect(describeRun(base({ state: 'queued', pid: null })).kind).toBe('queued')
    expect(describeRun(base({ state: 'running' })).kind).toBe('working')
  })

  it('gives every kind a colour', () => {
    for (const kind of ['queued', 'preparing', 'working', 'waiting', 'finished', 'failed',
                        'stopped', 'cleanup', 'uncertain'] as const) {
      expect(STATUS_COLOUR[kind]).toBeTruthy()
    }
  })
})

group('review regressions', () => {
  it('does not turn a running remote executor into a queued run', () => {
    expect(describeRun(base({ state: 'running', pid: null })).kind).toBe('working')
  })
  it('does not count failed or cancelled stages as completed work', () => {
    const stages = ['success', 'failure', 'cancelled', 'nothing_to_do'].map((kind, i) => ({
      id: String(i), stage: 'training', scope: 'global', state: 'terminal', dependencies: [],
      options: {}, outcome: { kind, detail: '', observed_at: 1 }, blocked: [], attempt_id: null,
    }))
    expect(describeRun(base({ stages })).progress).toBe('2 of 4 stages done')
  })
  it('keeps unknown cleanup visible after a process finishes', () => {
    const view = describeRun(base({ status: 'finished', finished: 2, cleanup: 'unknown' }))
    expect(view.kind).toBe('uncertain')
    expect(view.settling).toBe(true)
    expect(view.remedy).toContain('run-42')
  })
})
