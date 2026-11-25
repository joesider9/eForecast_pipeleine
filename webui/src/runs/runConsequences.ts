// A13-07: reading the plan. What would run, what would be replaced, what could not be resolved.
//
// Separate from the panel that renders it so that file exports components only (fast refresh),
// and so these four rules can be tested without a DOM. `consequential` is the one with teeth:
// it is not "is the action called rebuild" -- a Continue whose inputs changed can replace files,
// and a Rebuild of a case that has produced nothing yet replaces none. The plan decides.
import type { ResumePlan, StageDecision } from '../api/client'

/** The stages that would run, grouped by what the plan decided for each. */
export function grouped(resume: ResumePlan) {
  const of = (decision: StageDecision) => resume.stages.filter(stage => stage.decision === decision)
  return { reuse: of('reuse'), resume: of('resume'), verify: of('verify'),
           rebuild: of('rebuild'), blocked: of('blocked'), skip: of('skip') }
}

/** Every existing file the plan says a stage would replace, with its stage. */
export function replacements(resume: ResumePlan) {
  return resume.stages.flatMap(stage => (stage.affected ?? []).map(path => ({ stage: stage.key, path })))
}

/** Products the plan could not resolve: named, never counted as nothing. */
export function unresolved(resume: ResumePlan) {
  return resume.stages.flatMap(stage =>
    (stage.affected_unresolved ?? []).map(item => ({ stage: stage.key, item })))
}

/**
 * Is this run consequential -- would it replace something that exists?
 *
 * Not "is the action called rebuild": a Continue whose inputs changed can replace files, and a
 * Rebuild of a case that has produced nothing yet replaces none. The plan decides, not the label.
 */
export function consequential(resume: ResumePlan | null | undefined) {
  return !!resume && replacements(resume).length > 0
}
