import { useEffect, useMemo, useState } from 'react'
import { Alert, Autocomplete, Box, Button, Chip, Divider, IconButton, Stack, TextField, Tooltip, Typography } from '@mui/material'
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutline'
import AddIcon from '@mui/icons-material/Add'
import { api, Document, VarImpEntry, VarImpResolution } from '../api/client'

// `rbf_var_imp` is a list of rule sets and the clusterer trains ONE fuzzy network per rule set.
// Each entry names variables of row_all: a calendar name, an exact column, or a pattern that takes
// a whole group of columns. The group is the trap, so every entry is resolved against the case and
// shows what it actually selects.
type RuleSets = string[][]

const asRuleSets = (value: unknown): RuleSets => {
  if (!Array.isArray(value)) return [[]]
  if (!value.length) return [[]]
  if (!value.some(v => Array.isArray(v))) return [value.map(String)]        // the flat Distributed form
  return value.map(v => (Array.isArray(v) ? v.map(String) : [String(v)]))
}

const COLOUR = { ok: 'success', warn: 'warning', error: 'error' } as const

function EntryChip({ entry, resolved, tagProps }: {
  entry: string; resolved?: VarImpEntry; tagProps: Record<string, unknown>
}) {
  const level = resolved?.level ?? 'ok'
  const title = resolved
    ? `${resolved.kind}: ${resolved.note}${resolved.matched.length ? `\n\n${resolved.matched.slice(0, 20).join('\n')}${resolved.matched.length > 20 ? `\n… ${resolved.matched.length - 20} more` : ''}` : ''}`
    : entry
  // the props from getTagProps carry onDelete and the delete icon: they belong on the Chip, not on
  // a wrapper, or the chip cannot be removed
  return (
    <Tooltip title={<span style={{ whiteSpace: 'pre-line' }}>{title}</span>} arrow>
      <Chip size="small" label={resolved && resolved.count > 1 ? `${entry} · ${resolved.count}` : entry}
            color={level === 'ok' ? 'default' : COLOUR[level]} variant={level === 'ok' ? 'outlined' : 'filled'}
            {...tagProps} />
    </Tooltip>
  )
}

export default function RbfVarImpField({ value, document, onChange, flat }: {
  value: unknown; document: Document; onChange: (v: unknown) => void; flat?: boolean
}) {
  const ruleSets = useMemo(() => asRuleSets(value), [value])
  const [resolution, setResolution] = useState<VarImpResolution | null>(null)
  const [error, setError] = useState<string | null>(null)

  // Only the variables and the NWP models decide what an entry can match, so the request is keyed
  // on those rather than on the document's identity: the parent re-renders on every keystroke.
  const caseKey = useMemo(() => JSON.stringify([
    (document.data as Document)?.variables ?? null,
    (document.project as Document)?.nwp_models ?? null,
  ]), [document])
  const entriesKey = useMemo(() => JSON.stringify(ruleSets), [ruleSets])

  // resolve on every edit, debounced: the answer is what makes a pattern safe to type
  useEffect(() => {
    const controller = new AbortController()
    const timer = setTimeout(() => {
      api.resolveRbfVariables(document, ruleSets, controller.signal)
        .then(r => { if (!controller.signal.aborted) { setResolution(r as VarImpResolution); setError(null) } })
        .catch(e => { if (!controller.signal.aborted) setError(String(e?.message ?? e)) })
    }, 300)
    return () => { clearTimeout(timer); controller.abort() }
  }, [entriesKey, caseKey])  // eslint-disable-line react-hooks/exhaustive-deps

  const commit = (next: RuleSets) => onChange(flat ? (next[0] ?? []) : next)
  const setEntries = (i: number, entries: string[]) => commit(ruleSets.map((r, j) => (j === i ? entries : r)))
  // every name worth picking, grouped: a calendar variable, a whole NWP variable, one NWP series,
  // or an exact row_all column once the datasets are built
  const options = useMemo(() => {
    const s = resolution?.suggestions
    const groups: [string, string[]][] = [
      ['calendar variable', s?.calendar ?? []],
      ['NWP variable (every lag)', s?.nwp ?? []],
      ['NWP series', s?.nwp_series ?? []],
      [resolution?.source === 'dataset' ? 'row_all column' : 'row feature', s?.columns ?? []],
    ]
    const seen = new Set<string>()
    const out: { label: string; group: string }[] = []
    for (const [group, names] of groups) {
      for (const label of names) {
        if (seen.has(label)) continue
        seen.add(label)
        out.push({ label, group })
      }
    }
    return out
  }, [resolution])

  const resolvedFor = (i: number, k: number) => resolution?.rule_sets?.[i]?.entries?.[k]

  return (
    <Stack spacing={1} sx={{ width: '100%' }}>
      <Typography variant="caption" color="text.secondary">
        {flat
          ? 'The variables of row_all the distributed model clusters on.'
          : 'One rule set per fuzzy network. Each entry is a calendar name, an exact row_all column, or a pattern that takes every column containing it (write a$b to require both parts).'}
      </Typography>

      {ruleSets.map((entries, i) => {
        const summary = resolution?.rule_sets?.[i]
        return (
          <Box key={i} sx={{ border: '1px solid #e5e7eb', borderRadius: 1, p: 1 }}>
            <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 0.5 }}>
              <Typography variant="body2" sx={{ fontWeight: 600, flexGrow: 1 }}>
                {flat ? 'variables' : `rule set ${i + 1} — one fuzzy network`}
              </Typography>
              {summary && (
                <Chip size="small" variant="outlined" color={summary.level === 'ok' ? 'default' : COLOUR[summary.level]}
                      label={`${summary.columns} column${summary.columns === 1 ? '' : 's'}`} />
              )}
              {!flat && ruleSets.length > 1 && (
                <IconButton size="small" onClick={() => commit(ruleSets.filter((_, j) => j !== i))}>
                  <DeleteOutlineIcon fontSize="small" />
                </IconButton>
              )}
            </Stack>
            <Autocomplete multiple freeSolo size="small" value={entries}
              options={options.map(o => o.label)}
              groupBy={label => options.find(o => o.label === label)?.group ?? 'typed'}
              onChange={(_, items) => setEntries(i, (items as string[]).map(s => String(s).trim()).filter(Boolean))}
              renderTags={(tags, getTagProps) => tags.map((t, k) => {
                const { key, ...rest } = getTagProps({ index: k })
                return <EntryChip key={key} entry={t} resolved={resolvedFor(i, k)} tagProps={rest} />
              })}
              renderInput={p => <TextField {...p} placeholder="pick a variable or column, or type a pattern — Enter adds" />} />
            {summary && summary.entries.some(e => e.level !== 'ok') && (
              <Stack sx={{ mt: 0.5 }}>
                {summary.entries.filter(e => e.level !== 'ok').map((e, k) => (
                  <Typography key={k} variant="caption" color={e.level === 'error' ? 'error.main' : 'warning.main'}>
                    {e.entry}: {e.note}
                  </Typography>
                ))}
              </Stack>
            )}
          </Box>
        )
      })}

      {!flat && (
        <Box>
          <Button size="small" startIcon={<AddIcon />} onClick={() => commit([...ruleSets, []])}>
            Add a rule set
          </Button>
        </Box>
      )}

      <Divider />
      {error && <Alert severity="warning" sx={{ py: 0 }}>cannot resolve the variables: {error}</Alert>}
      {resolution && !error && (
        <Typography variant="caption" color="text.secondary">
          {resolution.note}
          {resolution.source === 'dataset' ? ` (${resolution.column_count} columns)` : ''}
        </Typography>
      )}
    </Stack>
  )
}
