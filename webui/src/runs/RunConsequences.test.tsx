// A13-07: the summary must say what the plan actually does to what already exists, and a
// consequential run must not start before that summary is accepted.
import { render, screen, within } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import type { ResumePlan, StageDecisionEntry } from '../api/client'
import RunConsequences from './RunConsequences'
import { consequential, grouped, replacements, unresolved } from './runConsequences'

const stage = (over: Partial<StageDecisionEntry> = {}): StageDecisionEntry => ({
  key: 'row', scope: 'global', decision: 'rebuild', reason: 'inputs changed', dependencies: [], ...over,
})

const plan = (over: Partial<ResumePlan> = {}): ResumePlan => ({
  action: 'continue', verified: true, plan_id: 'plan-1', script: 'datasets', interrupted: [],
  stages: [stage()], totals: {}, ...over,
})

describe('what the plan would do', () => {
  it('counts a run consequential only when the plan replaces something that exists', () => {
    expect(consequential(plan({ stages: [stage({ affected: ['/a/b.pickle'] })] }))).toBe(true)
    // a rebuild of a case that has produced nothing yet replaces nothing
    expect(consequential(plan({ action: 'rebuild', stages: [stage({ affected: [] })] }))).toBe(false)
    expect(consequential(plan({ stages: [stage({ decision: 'reuse' })] }))).toBe(false)
    expect(consequential(null)).toBe(false)
  })

  it('counts a Continue whose inputs changed as consequential, not just a Rebuild', () => {
    const continuing = plan({ action: 'continue', stages: [stage({ affected: ['/a/row.pickle'] })] })
    expect(consequential(continuing)).toBe(true)
  })

  it('groups the stages by the decision the plan made', () => {
    const groups = grouped(plan({ stages: [
      stage({ key: 'target', decision: 'reuse' }), stage({ key: 'row', decision: 'rebuild' }),
      stage({ key: 'split', decision: 'skip' })] }))
    expect(groups.reuse.map(s => s.key)).toEqual(['target'])
    expect(groups.rebuild.map(s => s.key)).toEqual(['row'])
    expect(groups.skip.map(s => s.key)).toEqual(['split'])
  })

  it('lists each replacement with the stage that would do it', () => {
    const found = replacements(plan({ stages: [
      stage({ key: 'row', affected: ['/a/row.pickle', '/a/row.csv'] }),
      stage({ key: 'target', decision: 'reuse' })] }))
    expect(found).toEqual([{ stage: 'row', path: '/a/row.pickle' }, { stage: 'row', path: '/a/row.csv' }])
  })

  it('surfaces what the plan could not resolve instead of counting it as nothing', () => {
    expect(unresolved(plan({ stages: [stage({ affected_unresolved: ['lstm: no paths'] })] })))
      .toEqual([{ stage: 'row', item: 'lstm: no paths' }])
  })
})

describe('RunConsequences', () => {
  it('names every file that would be replaced, and by which stage', () => {
    render(<RunConsequences resume={plan({ stages: [
      stage({ key: 'row', affected: ['/models/DATA/dataset_row_data.pickle'] })] })} />)
    const panel = screen.getByTestId('run-consequences')
    expect(panel).toHaveAttribute('data-consequential', 'true')
    expect(panel).toHaveTextContent('replaces 1 file(s) that already exist')
    expect(panel).toHaveTextContent('row: /models/DATA/dataset_row_data.pickle')
  })

  it('says plainly that a replaced file is not kept anywhere', () => {
    render(<RunConsequences resume={plan({ stages: [stage({ affected: ['/a/b.pickle'] })] })} />)
    const recovery = screen.getByTestId('consequences-recovery')
    expect(recovery).toHaveTextContent('overwritten in place')
    expect(recovery).toHaveTextContent('previous content is not kept anywhere')
    expect(recovery).toHaveTextContent('producing it again')
    expect(recovery).toHaveTextContent(/downstream/)
  })

  it('says so when nothing that exists would be replaced', () => {
    render(<RunConsequences resume={plan({ stages: [stage({ decision: 'reuse', reason: 'identities match' })] })} />)
    const panel = screen.getByTestId('run-consequences')
    expect(panel).toHaveAttribute('data-consequential', 'false')
    expect(panel).toHaveTextContent('replaces nothing that already exists')
    expect(screen.queryByTestId('consequences-recovery')).toBeNull()
  })

  it('lists what is kept and reused as it is', () => {
    render(<RunConsequences resume={plan({ stages: [
      stage({ key: 'target', decision: 'reuse', reason: 'content and input identities match the receipt' }),
      stage({ key: 'split', decision: 'skip', reason: 'not selected for this run' })] })} />)
    const panel = screen.getByTestId('run-consequences')
    expect(panel).toHaveTextContent('Kept and reused as it is')
    // names only: the stage table beneath carries each reason, and saying it twice is noise
    expect(panel).toHaveTextContent('target')
    expect(panel).toHaveTextContent('split')
    expect(panel).not.toHaveTextContent('content and input identities match the receipt')
  })

  it('distinguishes a new revision\'s kept training history from its shared datasets', () => {
    render(<RunConsequences resume={plan({ action: 'new', stages: [stage({ affected: ['/a/b.pickle'] })] })} />)
    const note = screen.getByTestId('consequences-revision')
    expect(note).toHaveTextContent('separate training history')
    expect(note).toHaveTextContent('previous models remain')
    expect(note).toHaveTextContent(/Dataset products are.*shared/)
    expect(note).toHaveTextContent('not copied')
  })

  it('does not claim a kept training history when the action is not a new revision', () => {
    render(<RunConsequences resume={plan({ action: 'rebuild', stages: [stage({ affected: ['/a/b.pickle'] })] })} />)
    expect(screen.queryByTestId('consequences-revision')).toBeNull()
  })

  it('warns that an unverified plan may replace something it listed as reused', () => {
    render(<RunConsequences resume={plan({ verified: false, stages: [stage({ decision: 'reuse', reason: 'looks current' })] })} />)
    expect(screen.getByTestId('consequences-unverified'))
      .toHaveTextContent('hashes each artifact before it skips it')
  })

  it('omits the unverified note when the plan was verified', () => {
    render(<RunConsequences resume={plan({ verified: true, stages: [stage({ decision: 'reuse', reason: 'ok' })] })} />)
    expect(screen.queryByTestId('consequences-unverified')).toBeNull()
  })

  it('names what could not be resolved rather than implying it is nothing', () => {
    render(<RunConsequences resume={plan({ stages: [
      stage({ key: 'lstm', affected_unresolved: ['its paths could not be resolved'] })] })} />)
    expect(screen.getByTestId('run-consequences'))
      .toHaveTextContent('lstm: its paths could not be resolved')
  })

  it('samples a long list and says how many more there are', () => {
    const many = Array.from({ length: 9 }, (_, index) => `/a/file${index}.pickle`)
    render(<RunConsequences resume={plan({ stages: [stage({ affected: many })] })} />)
    const panel = screen.getByTestId('run-consequences')
    expect(panel).toHaveTextContent('replaces 9 file(s)')
    expect(panel).toHaveTextContent('… and 4 more')
  })

  it('shows a blocked stage as unable to run rather than as replaced work', () => {
    render(<RunConsequences resume={plan({ stages: [
      stage({ key: 'target', decision: 'blocked', reason: 'required payload is missing' })] })} />)
    const panel = screen.getByTestId('run-consequences')
    expect(panel).toHaveTextContent('Cannot run yet')
    expect(panel).toHaveTextContent('target')
    expect(within(panel).getByTestId('consequences-blocked')).toHaveTextContent('1 blocked')
  })
})
