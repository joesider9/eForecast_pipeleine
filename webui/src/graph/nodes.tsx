import { Box, Chip, IconButton, Menu, MenuItem, Stack, Tooltip, Typography } from '@mui/material'
import { Handle, Position, type Node, type NodeProps } from '@xyflow/react'
import { useState } from 'react'
import { BuiltLayer, CatalogueLayer, LayerSpec, fmtParams, fmtShape, otherParamsLabel, paramLabel } from './types'

export interface Selection { branch: string; index: number }

export interface BranchData extends Record<string, unknown> {
  branch: string
  description: string
  layers: LayerSpec[]
  input?: any
  built?: BuiltLayer[]
  error?: string
  errorLayer?: number | null
  selected: Selection | null
  addable: CatalogueLayer[]
  onSelect: (s: Selection) => void
  onAdd: (branch: string, type: string) => void
  onRemoveBranch?: (branch: string) => void
}
export interface HeadData extends Record<string, unknown> { head: { type: string; n_out: number; params: number } | null; steps?: number; total?: number | null }

const KIND_COLOUR: Record<string, string> = { conv: '#dbeafe', attention: '#ede9fe', recurrent: '#dcfce7', dense: '#fef3c7', shape: '#f3f4f6', merge: '#fce7f3', image: '#ffedd5' }

function LayerRow({ layer, index, built, selected, failing, kind, onClick }: {
  layer: LayerSpec; index: number; built?: BuiltLayer; selected: boolean; failing: boolean; kind?: string; onClick: () => void
}) {
  const disabled = layer.enabled === false
  // the size sits next to the type; the remaining parameters go on their own line
  const otherParams = otherParamsLabel(layer)
  return (
    <Box onClick={onClick} role="button" tabIndex={0} aria-label={`${layer.type} layer ${index + 1}`}
      onKeyDown={event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); onClick() } }} className="nodrag" sx={{
      px: 1, py: 0.5, borderRadius: 1, cursor: 'pointer', bgcolor: failing ? '#fee2e2' : KIND_COLOUR[kind ?? ''] ?? '#f3f4f6',
      outline: selected ? '2px solid #2563eb' : '1px solid #e5e7eb', opacity: disabled ? 0.45 : 1, '&:hover': { outline: '2px solid #93c5fd' },
    }}>
      <Stack direction="row" spacing={1} alignItems="baseline">
        <Typography variant="caption" color="text.secondary">{index + 1}</Typography>
        <Typography variant="body2" sx={{ fontWeight: 500 }}>{layer.type}</Typography>
        <Typography variant="body2" color="text.secondary">{paramLabel((layer.params ?? {}).size)}</Typography>
        {disabled && <Typography variant="caption" color="text.secondary">off</Typography>}
      </Stack>
      {otherParams && <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>{otherParams}</Typography>}
      {built && (
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
          → {fmtShape(built.output_shape)}{built.params ? ` · ${fmtParams(built.params)} params` : ''}
          {built.tokens !== undefined ? ` · ${built.tokens} tokens, d ${built.d_model}` : ''}
        </Typography>
      )}
      {built?.query_branches && <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>query {built.query_branches.join(', ')} → keys {built.kv_branches?.join(', ')}</Typography>}
    </Box>
  )
}

function AddButton({ addable, onAdd }: { addable: CatalogueLayer[]; onAdd: (type: string) => void }) {
  const [anchor, setAnchor] = useState<HTMLElement | null>(null)
  return (
    <>
      <Chip className="nodrag" size="small" variant="outlined" label="+ layer" onClick={e => setAnchor(e.currentTarget)} />
      <Menu anchorEl={anchor} open={!!anchor} onClose={() => setAnchor(null)}>
        {addable.map(l => (
          <MenuItem key={l.name} onClick={() => { onAdd(l.name); setAnchor(null) }}>
            <Stack><Typography variant="body2">{l.label}</Typography><Typography variant="caption" color="text.secondary">{l.name}</Typography></Stack>
          </MenuItem>
        ))}
      </Menu>
    </>
  )
}

export function BranchNode({ data }: NodeProps<Node<BranchData>>) {
  const isMerge = data.branch === 'output'
  const kinds = Object.fromEntries(data.addable.map(l => [l.name, l.kind]))
  return (
    <Box sx={{ width: 270, bgcolor: 'background.paper', border: '1px solid', borderColor: data.error ? 'error.main' : '#cbd5e1', borderRadius: 2, boxShadow: 1 }}>
      {!isMerge && <Handle type="source" position={Position.Right} />}
      {isMerge && <Handle type="target" position={Position.Left} />}
      {isMerge && <Handle type="source" position={Position.Right} />}
      <Box sx={{ px: 1.5, py: 1, borderBottom: '1px solid #e5e7eb', bgcolor: isMerge ? '#fdf2f8' : '#f8fafc', borderRadius: '8px 8px 0 0', cursor: 'grab', '&:active': { cursor: 'grabbing' } }}>
        <Stack direction="row" alignItems="center" spacing={1}>
          <Tooltip title={data.description}><Typography variant="subtitle2" sx={{ flex: 1 }}>{isMerge ? 'merge (output)' : data.branch}</Typography></Tooltip>
          {data.onRemoveBranch && <IconButton className="nodrag" size="small" onClick={() => data.onRemoveBranch!(data.branch)} title="remove branch"><span style={{ fontSize: 14 }}>✕</span></IconButton>}
        </Stack>
        {data.input !== undefined && <Typography variant="caption" color="text.secondary">in {fmtShape(data.input)}</Typography>}
      </Box>
      <Stack spacing={0.5} sx={{ p: 1 }}>
        {data.layers.map((layer, i) => (
          <LayerRow key={i} layer={layer} index={i} built={data.built?.[i]} kind={kinds[layer.type]}
            selected={data.selected?.branch === data.branch && data.selected.index === i} failing={data.errorLayer === i}
            onClick={() => data.onSelect({ branch: data.branch, index: i })} />
        ))}
        {data.layers.length === 0 && <Typography variant="caption" color="text.secondary">{isMerge ? 'no merge layer: branches are concatenated' : 'no layer: the input is passed on'}</Typography>}
        <AddButton addable={data.addable} onAdd={t => data.onAdd(data.branch, t)} />
        {data.error && <Typography variant="caption" color="error" sx={{ whiteSpace: 'pre-wrap' }}>{data.error}</Typography>}
      </Stack>
    </Box>
  )
}

export function HeadNode({ data }: NodeProps<Node<HeadData>>) {
  return (
    <Box sx={{ width: 200, bgcolor: '#f0fdf4', border: '1px solid #86efac', borderRadius: 2, p: 1.5, boxShadow: 1 }}>
      <Handle type="target" position={Position.Left} />
      <Typography variant="subtitle2">head</Typography>
      <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
        {data.head ? `${data.head.type}: ${data.head.n_out} outputs · ${fmtParams(data.head.params)} params` : `${data.steps ?? '?'} outputs (one per horizon step)`}
      </Typography>
      {data.total !== null && data.total !== undefined && <Typography variant="caption" sx={{ display: 'block', mt: 0.5 }}>network: {fmtParams(data.total)} parameters</Typography>}
    </Box>
  )
}
