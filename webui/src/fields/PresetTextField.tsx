import { FieldProps } from '@rjsf/utils'
import { Autocomplete, Box, TextField } from '@mui/material'
import { FormContext, UiOptions } from './context'

// A text property with preset values to pick from and free entry: ui:options.presets names the list of
// /api/options to use (timezones, ts_resolutions) or gives the values directly.
export default function PresetTextField(props: FieldProps) {
  const { formData, onChange, schema, uiSchema, formContext } = props
  const ctx = formContext as FormContext
  const presets = uiSchema?.['ui:options']?.presets
  const options: string[] = Array.isArray(presets) ? presets.map(String)
    : typeof presets === 'string' ? ((ctx?.options?.[presets as keyof UiOptions] as string[] | undefined) ?? []) : []
  const help = uiSchema?.['ui:options']?.help as string | undefined
  return (
    <Box sx={{ mb: 1.5 }}>
      <Autocomplete freeSolo size="small" options={options} value={formData ?? ''} onChange={(_, v) => onChange(v ?? '')}
        onInputChange={(_, v, reason) => { if (reason === 'input') onChange(v) }}
        renderInput={p => <TextField {...p} label={schema.title} helperText={help ?? schema.description} />} />
    </Box>
  )
}
