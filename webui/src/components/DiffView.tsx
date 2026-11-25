import { diffLines } from 'diff'
import { Box, Typography } from '@mui/material'

// Line diff between the saved YAML and the edited YAML.  Unchanged blocks are folded to their first and
// last two lines so a long case stays readable.
export default function DiffView({ before, after }: { before: string; after: string }) {
  const parts = diffLines(before, after)
  const changed = parts.some(p => p.added || p.removed)
  if (!changed) return <Typography color="text.secondary">No changes against the saved case.</Typography>
  return (
    <Box sx={{ border: '1px solid #ddd', borderRadius: 1, overflow: 'auto', maxHeight: 520, bgcolor: '#fff' }}>
      {parts.map((part, i) => {
        const lines = part.value.replace(/\n$/, '').split('\n')
        const cls = part.added ? 'diff-add' : part.removed ? 'diff-del' : 'diff-ctx'
        const prefix = part.added ? '+ ' : part.removed ? '- ' : '  '
        const shown = !part.added && !part.removed && lines.length > 6
          ? [...lines.slice(0, 2), `  … ${lines.length - 4} unchanged lines …`, ...lines.slice(-2)]
          : lines
        return shown.map((line, j) => <div key={`${i}-${j}`} className={`diff-line ${cls}`}>{prefix}{line}</div>)
      })}
    </Box>
  )
}
