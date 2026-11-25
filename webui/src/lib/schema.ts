// Turns the JSON Schema of ExperimentConfig (pydantic) into what react-jsonschema-form renders well.
//
// pydantic writes unions as anyOf; RJSF renders every anyOf as an option selector, which is unreadable for
// `str | None`.  simplify() collapses nullable unions, merges numeric unions, and leaves truly polymorphic or
// untyped values to the YAML field (a CodeMirror box), chosen through the generated uiSchema.

export type JsonSchema = Record<string, any>

const isNull = (s: JsonSchema) => s?.type === 'null'

export function simplify(schema: JsonSchema): JsonSchema {
  if (!schema || typeof schema !== 'object') return schema
  let node: JsonSchema = { ...schema }
  const union = node.anyOf ?? node.oneOf
  if (Array.isArray(union)) {
    const options = union.filter((o: JsonSchema) => !isNull(o))
    const { anyOf: _a, oneOf: _o, ...rest } = node
    if (options.length === 1) {
      node = { ...simplify(options[0]), ...rest }
    } else if (options.length > 1 && options.every((o: JsonSchema) => o.type === 'integer' || o.type === 'number')) {
      node = { ...rest, type: 'number' }
    } else {
      node = { ...rest, 'x-yaml': true }     // polymorphic: edited as YAML
    }
  }
  if (node.properties) {
    node.properties = Object.fromEntries(Object.entries(node.properties).map(([k, v]) => [k, simplify(v as JsonSchema)]))
  }
  if (node.items && typeof node.items === 'object' && !Array.isArray(node.items)) node.items = simplify(node.items)
  if (node.$defs) node.$defs = Object.fromEntries(Object.entries(node.$defs).map(([k, v]) => [k, simplify(v as JsonSchema)]))
  return node
}

// True when a property should be edited as YAML text rather than generated widgets.
export function wantsYaml(schema: JsonSchema): boolean {
  if (!schema) return true
  if (schema['x-yaml']) return true
  if (schema.$ref) return false                                   // a model: RJSF renders its fields
  if (!schema.type && !schema.enum) return true                   // Any
  if (schema.type === 'object') return !schema.properties          // dict[str, ...]
  if (schema.type === 'array') {
    const items = schema.items
    if (!items || Object.keys(items).length === 0) return true    // list[Any]
    if (items.$ref) return false                                   // list[Model]
    if (items['x-yaml'] || (!items.type && !items.enum)) return true
    return items.type === 'object' && !items.properties
  }
  return false
}

// Properties with a dedicated editor (phase 5 widgets), by section.
export const CUSTOM_FIELDS: Record<string, Record<string, any>> = {
  // How pretrained models are fetched is installation policy. Keep the property in the schema
  // and form data so editing an unrelated job setting preserves a stored override.
  jobs: {
    model_cache_mode: { 'ui:widget': 'hidden' },
  },
  project: {
    coordinates: { 'ui:field': 'map' }, area_group: { 'ui:field': 'map' },
    nwp_models: { 'ui:field': 'nwp_models' },
    filename: { 'ui:field': 'path', 'ui:options': { mode: 'file' } },
    local_timezone: { 'ui:field': 'preset_text', 'ui:options': { presets: 'timezones', help: 'timezone of the source timestamps; CET includes summer time, Etc/GMT-1 is fixed UTC+1 all year' } },
    site_timezone: { 'ui:field': 'preset_text', 'ui:options': { presets: 'timezones', help: 'timezone of the site (sun position, calendar)' } },
    ts_resolution: { 'ui:field': 'preset_text', 'ui:options': { presets: 'ts_resolutions', help: '1min, 5min, 15min, H (hourly) or D (daily)' } },
  },
  folders: {
    pycharm_path: { 'ui:field': 'path' }, sys_folder: { 'ui:field': 'path' }, nwp_folder: { 'ui:field': 'path' },
    sat_folder: { 'ui:field': 'path' }, path_image: { 'ui:field': 'path' },
  },
  data: {
    // New cases use the backend as_of default. Preserve old stored policy values
    // without offering legacy_merge as a normal configuration choice.
    nwp_selection_policy: { 'ui:widget': 'hidden' },
    variables: { 'ui:field': 'variables' }, target_variable: { 'ui:field': 'target' },
    global_past_lags: { 'ui:field': 'lags' }, global_future_lags: { 'ui:field': 'lags' },
    nwp_data_merge: { 'ui:field': 'option_list' }, compress_data: { 'ui:field': 'option_list' }, img_data: { 'ui:field': 'option_list' },
    scale_row_method: { 'ui:field': 'option_list' }, scale_img_method: { 'ui:field': 'option_list' }, scale_nwp_method: { 'ui:field': 'option_list' },
  },
  methods: {
    methods_cpu: { 'ui:field': 'flags' }, cluster_methods: { 'ui:field': 'flags' }, global_methods: { 'ui:field': 'flags' },
    feature_selection_methods: { 'ui:field': 'option_list' }, combine_methods: { 'ui:field': 'option_list' },
    method_params: { 'ui:field': 'method_params' },
    env_name: { 'ui:widget': 'hidden' },
    path_env: { 'ui:widget': 'hidden' },
    runtime_ids: { 'ui:field': 'hidden', 'ui:widget': 'hidden' },
    kernels: { 'ui:field': 'kernels' }, base_data_types: { 'ui:field': 'data_types' },
  },
}

// uiSchema for a section: the dedicated editors first, yaml fields where needed, recursing into models
// and arrays of models.
export function uiSchemaFor(schema: JsonSchema, defs: JsonSchema, depth = 0, section?: string): Record<string, any> {
  const resolved = schema?.$ref ? defs[schema.$ref.split('/').pop()!] : schema
  const ui: Record<string, any> = {}
  if (!resolved?.properties) return ui
  for (const [name, prop] of Object.entries<JsonSchema>(resolved.properties)) {
    if (section && CUSTOM_FIELDS[section]?.[name]) {
      ui[name] = CUSTOM_FIELDS[section][name]
    } else if (wantsYaml(prop)) {
      ui[name] = { 'ui:field': 'yaml' }
    } else if (prop.$ref) {
      ui[name] = uiSchemaFor(prop, defs, depth + 1)
    } else if (prop.type === 'array' && prop.items?.$ref) {
      ui[name] = { items: uiSchemaFor(prop.items, defs, depth + 1), 'ui:options': { orderable: true } }
    } else if (prop.type === 'string' && /path|folder|filename/.test(name)) {
      ui[name] = { 'ui:widget': 'text', 'ui:placeholder': '/absolute/path' }
    }
  }
  return ui
}

// The schema of one top-level section (project, folders, jobs, data, methods) with the shared $defs.
export function sectionSchema(full: JsonSchema, section: string): JsonSchema {
  const simplified = simplify(full)
  const prop = simplified.properties?.[section]
  if (!prop) throw new Error(`unknown section ${section}`)
  const defs = simplified.$defs ?? {}
  const body = prop.$ref ? defs[prop.$ref.split('/').pop()!] : prop
  return { ...body, $defs: defs }
}

export const SECTION_LABELS: Record<string, string> = {
  project: 'Project', folders: 'Folders', jobs: 'Jobs', data: 'Data', methods: 'Methods',
}
