import { FieldProps } from '@rjsf/utils'
import { Autocomplete, Chip, TextField } from '@mui/material'
import { FormContext, NONE } from './context'

// The legacy option lists (nwp_data_merge, compress_data, scale_* , feature_selection_methods, ...):
// a multi-select over the known choices; "None" stands for null, custom values are allowed.
export default function OptionListField(props: FieldProps) {
  const { formData, onChange, name, schema, formContext } = props
  const ctx = formContext as FormContext
  const options = ctx?.options
  const choices = options?.data_lists?.[name] ?? options?.methods_lists?.[name] ?? [NONE]
  const emptyIsNoneList = !!options?.data_lists?.[name]      // data lists save [None] when empty, methods lists null
  const items: string[] = Array.isArray(formData) ? formData.map(v => (v === null ? NONE : String(v))) : []
  const commit = (labels: string[]) => {
    const values = labels.map(l => (l === NONE ? null : l))
    onChange(values.length ? values : emptyIsNoneList ? [null] : null)
  }
  return (
    <Autocomplete multiple freeSolo size="small" options={choices} value={items} sx={{ mb: 1.5 }}
      onChange={(_, labels) => commit(labels as string[])}
      renderTags={(tags, getTagProps) => tags.map((tag, i) => <Chip size="small" label={tag} {...getTagProps({ index: i })} />)}
      renderInput={params => <TextField {...params} label={schema.title ?? name} helperText={schema.description} />} />
  )
}
