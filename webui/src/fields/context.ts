// What every custom field can reach through RJSF's formContext: the whole document (for cross-section
// facts such as the project's horizon), the option lists served by /api/options, and a way to change
// another section (presets write target lags derived from the project).
import { Document } from '../api/client'
import { StructuredSpec } from './StructuredParamField'

export interface UiOptions {
  variable_types: string[]
  variable_sources: string[]
  transformers: string[]
  nwp_models: string[]
  // the shared Google Drive folder of each provider that can be downloaded from, by model
  nwp_gdrive?: Record<string, string>
  data_lists: Record<string, string[]>
  methods_lists: Record<string, string[]>
  flags: Record<string, string[]>
  families: string[]
  set_valued_keys: Record<string, string[]>
  none_label: string
  timezones: string[]
  ts_resolutions: string[]
  nwp_resolutions: number[]
  nwp_levels: number[]
  variable_names: Record<string, string[]>
  variable_source_by_type: Record<string, string>
  kernel_presets: Record<string, number[][]>
  data_type_keys: Record<string, string>
  family_preset_keys: Record<string, string[]>
  // method-family parameters whose value is a structure, each with the editor it needs
  structured_params: Record<string, StructuredSpec>
}

export interface FormContext {
  document: Document
  options: UiOptions | null
  setDocument: (next: Document) => void
  // Has the draft stayed as it was while a request of this field was in flight? It takes the document
  // the request was built on and the edit count the field read from stamp() before starting; a field
  // cannot answer this from its own props, for the reason given where these two are implemented.
  isCurrent?: (document: Document, since?: number) => boolean
  // The edit count of the draft as it is now, which a field must read when it starts a request.
  stamp?: () => number
  yamlSection?: string
  presetsPath?: string          // the case YAML the presets are loaded from (Project tab), default configuration/cases/default.yaml
}

export const DEFAULT_PRESETS_PATH = 'configuration/cases/default.yaml'

export const NONE = 'None'
