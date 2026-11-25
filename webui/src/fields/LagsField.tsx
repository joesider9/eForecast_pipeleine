import { useState } from 'react'
import { FieldProps } from '@rjsf/utils'
import { Box, TextField } from '@mui/material'
import { lagsText, parseLags } from './lags'

// A list of lags typed compactly: "0..5" is a range, "0, 3, 5, 8" fixed values, "-24..-1/4" a stepped
// range; stored as a plain list (empty text = null).
export default function LagsField(props: FieldProps) {
  const { formData, onChange, schema } = props
  const [text, setText] = useState(lagsText(formData))
  const [error, setError] = useState<string | null>(null)
  const apply = (t: string) => {
    setText(t)
    try {
      const parsed = parseLags(t)
      onChange(parsed === null ? null : Array.isArray(parsed) ? parsed : [parsed]); setError(null)
    } catch (e: any) { setError(e.message) }
  }
  return (
    <Box sx={{ mb: 1.5 }}>
      <TextField fullWidth size="small" label={schema.title} value={text} onChange={e => apply(e.target.value)} error={!!error}
        helperText={error ?? `${schema.description ? schema.description + ' — ' : ''}a range "0..5" or values "0, 3, 5, 8"; empty = not used`} />
    </Box>
  )
}
