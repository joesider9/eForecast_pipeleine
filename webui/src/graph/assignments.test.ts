import { describe, expect, it } from 'vitest'
import { addAssignment, assignedTags, assignmentProblem, changeAssignments, renameAssignments } from './assignments'

describe('stored experiment references', () => {
  it.each([
    ['first', ['first', 'second']],
    [['first'], ['first', 'second']],
    [{ choices: ['first'] }, { choices: ['first', 'second'] }],
    [{ values: ['first'] }, { values: ['first', 'second'] }],
    [{ range: ['first'] }, { range: ['first', 'second'] }],
    [{ tag: 'choice', value: ['first'] }, { tag: 'choice', value: ['first', 'second'] }],
    [{ tag: 'fixed', value: 'first' }, { tag: 'choice', value: ['first', 'second'] }],
  ])('preserves existing tags when adding to %j', (stored, expected) => {
    const document = { extension: { keep: 0 }, methods: { method_params: {
      MLP: { experiment_tag: stored, custom: false }, TimesFM: { model: 'unchanged' },
    } } }
    const next = addAssignment(document, 'MLP', 'second')
    expect(next).toEqual({ ...document, methods: { method_params: { ...document.methods.method_params,
      MLP: { experiment_tag: expected, custom: false },
    } } })
    expect(assignedTags(next.methods.method_params.MLP.experiment_tag)).toEqual(['first', 'second'])
    expect(addAssignment(next, 'MLP', 'second')).toBe(next)
    expect(document.methods.method_params.MLP.experiment_tag).toEqual(stored)
  })

  it('does not silently replace unrecognized stored assignment data', () => {
    const document = { methods: { method_params: { MLP: { experiment_tag: { extension: 'keep me' } } } } }
    expect(assignmentProblem(document.methods.method_params.MLP.experiment_tag)).not.toBeNull()
    expect(addAssignment(document, 'MLP', 'second')).toBe(document)
  })

  it('renames supported families only and keeps unrelated references', () => {
    const document = { methods: { method_params: {
      MLP: { experiment_tag: ['first', 'other'] }, CNN: { experiment_tag: 'first' },
      CatBoost: { experiment_tag: 'first', custom: true },
    } } }
    const renamed = renameAssignments(document, ['MLP', 'CNN'], 'first', 'renamed')
    expect(renamed.methods.method_params.MLP.experiment_tag).toEqual(['renamed', 'other'])
    expect(renamed.methods.method_params.CNN.experiment_tag).toBe('renamed')
    expect(renamed.methods.method_params.CatBoost).toEqual(document.methods.method_params.CatBoost)
    const removed = changeAssignments(renamed, 'MLP', tags => tags.filter(tag => tag !== 'renamed'))
    expect(removed.methods.method_params.MLP.experiment_tag).toEqual(['other'])
  })
})
