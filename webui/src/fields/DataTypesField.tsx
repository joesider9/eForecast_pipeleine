import { useState } from 'react'
import { FieldProps } from '@rjsf/utils'
import { Alert, Box, Button, MenuItem, Stack, Table, TableBody, TableCell, TableRow, TextField, Typography } from '@mui/material'
import { api, isAbort, Document } from '../api/client'
import { useRequestScope } from '../lib/useRequestScope'
import { FormContext, NONE } from './context'

// The data options used by the clusterer and the combiners (methods.base_data_types): one value per key,
// chosen among the options of the Data tab; "From data tab" takes the first option of every list.
export default function DataTypesField(props: FieldProps) {
  const { formData, onChange, schema, formContext } = props
  const ctx = formContext as FormContext
  const signal = useRequestScope()
  const keys = ctx?.options?.data_type_keys ?? { scale_row_method: 'scale_row_method', scale_nwp_method: 'scale_nwp_method', scale_img_method: 'scale_img_method', merge: 'nwp_data_merge', compress: 'compress_data', image_data_type: 'img_data' }
  const data = (ctx?.document?.data ?? {}) as Document
  const value: Record<string, any> = formData && typeof formData === 'object' ? formData : {}
  const optionsOf = (dataKey: string): any[] => { const v = data[dataKey]; return Array.isArray(v) && v.length ? v : [null] }
  const fromData = () => onChange(Object.fromEntries(Object.entries(keys).map(([k, dataKey]) => [k, optionsOf(dataKey)[0] ?? null])))
  const label = (v: any) => (v === null || v === undefined ? NONE : String(v))
  const [note, setNote] = useState<string | null>(null)
  // pushes the values to every family, the clusterer (data_type and params) and Distributed; the
  // combiners read base_data_types directly.  The current table values are applied, saved or not.
  const applyAll = async () => {
    const since = ctx.stamp?.()
    try {
      const doc = { ...ctx.document, methods: { ...(ctx.document.methods ?? {}), base_data_types: value } }
      const result = await api.applyDataTypes(doc, signal())
      if (ctx.isCurrent?.(ctx.document, since) === false) { setNote('The draft changed while applying options. Retry with the latest draft.'); return }
      ctx.setDocument(result.document); setNote(result.note)
    } catch (e) { if (!isAbort(e)) setNote(`Apply failed: ${e instanceof Error ? e.message : String(e)}`) }
  }
  return (
    <Box sx={{ mb: 1.5 }}>
      <Stack direction="row" alignItems="center" spacing={1}>
        <Typography variant="subtitle2" sx={{ flex: 1 }}>{schema.title ?? 'Base data types'}</Typography>
        <Button size="small" onClick={fromData}>From data tab</Button>
        <Button size="small" variant="outlined" onClick={applyAll}>Apply to families, clusterer, combiners</Button>
      </Stack>
      {note && <Alert severity="info" onClose={() => setNote(null)} sx={{ py: 0, my: 0.5 }}>{note}</Alert>}
      <Typography variant="caption" color="text.secondary">The data options the clusterer and the combiners use; each value comes from the lists of the Data tab.</Typography>
      <Table size="small">
        <TableBody>
          {Object.entries(keys).map(([k, dataKey]) => {
            const opts = optionsOf(dataKey); const current = value[k]
            const all = [...opts, ...(opts.some(o => label(o) === label(current)) ? [] : [current])]
            return (
              <TableRow key={k}>
                <TableCell sx={{ width: 220 }}>{k} <Typography variant="caption" color="text.secondary">← data.{dataKey}</Typography></TableCell>
                <TableCell>
                  <TextField select size="small" value={label(current)} onChange={e => onChange({ ...value, [k]: e.target.value === NONE ? null : e.target.value })} sx={{ minWidth: 200 }}>
                    {all.map(o => <MenuItem key={label(o)} value={label(o)}>{label(o)}</MenuItem>)}
                  </TextField>
                </TableCell>
              </TableRow>
            )
          })}
        </TableBody>
      </Table>
    </Box>
  )
}
