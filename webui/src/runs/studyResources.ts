import { resourceControlMetadata, type ResourceControls } from '../api/resourceContract.generated'

export const numeric = ['cpu_workers', 'cpu_study_threads'] as const
type NumericControl = typeof numeric[number]
export type StudyDraft = Record<NumericControl | 'cpu_study_device' | 'cpu_scope', string>
export const emptyStudyDraft: StudyDraft = { cpu_scope: '', cpu_workers: '', cpu_study_threads: '', cpu_study_device: '' }
export const labels: Record<NumericControl, string> = {
  cpu_workers: 'Independent studies', cpu_study_threads: 'Threads per study',
}

// Only explicit overrides travel with a submission. Empty fields inherit the saved
// case; opening the panel never materializes defaults or writes the case.
export function studyOverrides(draft: StudyDraft): ResourceControls & { cpu_scope?: string } {
  const result: ResourceControls & { cpu_scope?: string } = {}
  for (const key of numeric) {
    const raw = draft[key].trim()
    if (!raw) continue
    const value = Number(raw)
    if (!Number.isSafeInteger(value) || value < resourceControlMetadata[key].minimum) throw new Error(`${labels[key]} must be a positive integer`)
    result[key] = value
  }
  if (draft.cpu_study_device.trim()) result.cpu_study_device = draft.cpu_study_device.trim()
  if (draft.cpu_scope) result.cpu_scope = draft.cpu_scope
  return result
}

