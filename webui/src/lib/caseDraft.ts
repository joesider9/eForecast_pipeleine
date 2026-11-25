import type { CaseResponse, Document } from '../api/client'
import { toYaml } from './yaml'
import { unfinishedYaml, type YamlBuffer } from '../forms/yamlDraftContext'

export interface DraftState {
  baseline: CaseResponse | null
  document: Document | null
  incoming: CaseResponse | null
  edit: number
  yaml: Record<string, YamlBuffer>
}
export type DraftAction =
  | { type: 'server'; snapshot: CaseResponse }
  | { type: 'edit'; document: Document }
  | { type: 'saved'; snapshot: CaseResponse; submittedEdit: number }
  | { type: 'reload'; snapshot: CaseResponse }
  | { type: 'yaml'; key: string; buffer: YamlBuffer }
  | { type: 'discard-yaml'; key?: string }

export const emptyDraft: DraftState = { baseline: null, document: null, incoming: null, edit: 0, yaml: {} }
export const cloneDocument = (document: Document): Document => JSON.parse(JSON.stringify(document))
export const equalDocuments = (first: Document, second: Document) => toYaml(first) === toYaml(second)
export const draftDirty = (state: DraftState) => !!state.document && !!state.baseline &&
  (!equalDocuments(state.document, state.baseline.document) || unfinishedYaml(state.yaml).length > 0)

export function draftReducer(state: DraftState, action: DraftAction): DraftState {
  if (action.type === 'edit') return { ...state, document: action.document, edit: state.edit + 1 }
  if (action.type === 'yaml') return { ...state, yaml: { ...state.yaml, [action.key]: action.buffer }, edit: state.edit + 1 }
  if (action.type === 'discard-yaml') {
    const yaml = { ...state.yaml }
    if (action.key !== undefined) delete yaml[action.key]
    else for (const [key] of unfinishedYaml(yaml)) delete yaml[key]
    return { ...state, yaml, edit: state.edit + 1 }
  }
  const snapshot = { ...action.snapshot, document: cloneDocument(action.snapshot.document) }
  if (action.type === 'saved') return {
    ...state, baseline: snapshot, incoming: null,
    document: state.edit === action.submittedEdit ? cloneDocument(snapshot.document) : state.document,
  }
  if (action.type === 'reload' || !state.baseline) return {
    baseline: snapshot, document: cloneDocument(snapshot.document), incoming: null, edit: state.edit + 1, yaml: {},
  }
  if (snapshot.revision === state.baseline.revision && equalDocuments(snapshot.document, state.baseline.document)) return state
  if (state.document && equalDocuments(snapshot.document, state.document)) return { ...state, baseline: snapshot, incoming: null }
  if (draftDirty(state)) return { ...state, incoming: snapshot }
  return { baseline: snapshot, document: cloneDocument(snapshot.document), incoming: null, edit: state.edit + 1, yaml: {} }
}
