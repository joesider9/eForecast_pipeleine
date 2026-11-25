import { useMemo } from 'react'
import Form from '@rjsf/mui'
import validator from '@rjsf/validator-ajv8'
import { IChangeEvent } from '@rjsf/core'
import { Alert } from '@mui/material'
import { Document } from '../api/client'
import { JsonSchema, sectionSchema, uiSchemaFor } from '../lib/schema'
import YamlField from './YamlField'
import FlagsField from '../fields/FlagsField'
import OptionListField from '../fields/OptionListField'
import PathField from '../fields/PathField'
import MapField from '../fields/MapField'
import NwpModelsField from '../fields/NwpModelsField'
import VariablesField from '../fields/VariablesField'
import TargetField from '../fields/TargetField'
import MethodParamsField from '../fields/MethodParamsField'
import PresetTextField from '../fields/PresetTextField'
import LagsField from '../fields/LagsField'
import KernelsField from '../fields/KernelsField'
import DataTypesField from '../fields/DataTypesField'
import { FormContext } from '../fields/context'

const FIELDS = {
  hidden: () => null, // Object fields need a field override; a hidden scalar widget is insufficient.
  yaml: YamlField, flags: FlagsField, option_list: OptionListField, path: PathField, map: MapField,
  nwp_models: NwpModelsField, variables: VariablesField, target: TargetField, method_params: MethodParamsField,
  preset_text: PresetTextField, lags: LagsField, kernels: KernelsField, data_types: DataTypesField,
}

interface Props {
  fullSchema: JsonSchema
  section: string
  value: Document | undefined
  onChange: (value: Document) => void
  context: FormContext
}

// One section of the case (project, folders, ...) rendered from the JSON Schema; the document is the source
// of truth, the form only reports changes.  Client-side validation is off: pydantic (through /api/validate)
// is the authority, and its messages are shown by the Review tab with their paths.
export default function SectionForm({ fullSchema, section, value, onChange, context }: Props) {
  const schema = useMemo(() => sectionSchema(fullSchema, section), [fullSchema, section])
  const uiSchema = useMemo(() => ({
    ...uiSchemaFor(schema, schema.$defs ?? {}, 0, section),
    'ui:submitButtonOptions': { norender: true },
  }), [schema, section])

  return (
    <>
    {section === 'data' && value?.nwp_selection_policy === 'legacy_merge' && (
      <Alert severity="warning" sx={{ mb: 2 }}>
        This saved case uses legacy weather selection. It can include weather forecasts issued later
        than the forecast being evaluated. Ask the maintainer to switch it to as_of and rebuild the
        weather inputs and affected models. Its saved policy has been preserved.
      </Alert>
    )}
    <Form
      schema={schema}
      uiSchema={uiSchema}
      formData={value ?? {}}
      idPrefix={`case_${section}`}
      validator={validator}
      fields={FIELDS}
      formContext={{ ...context, yamlSection: section }}
      noHtml5Validate
      liveValidate={false}
      showErrorList={false}
      omitExtraData={false}
      onChange={(event: IChangeEvent) => onChange(event.formData as Document)}
    />
    </>
  )
}
