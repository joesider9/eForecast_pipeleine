// A13-06: the six things a colleague changes when a working case becomes a new site.
//
// Not a second editor. It edits the same `document` object the expert tabs edit, through the
// same `setDocument`, so there is no guided/expert synchronisation to keep honest and no second
// draft to lose. Every setting it does not show is preserved untouched -- including fields this
// version of the UI does not know about -- because each change is a spread of what was there.
//
// Six groups, in the order someone actually works through them: site, resolution, horizon,
// timezone, input sources, model scope. The case *name* is not here: that belongs to copying or
// creating the case. The site name is, because it decides which model folder the case owns.
import { ReactNode } from 'react'
import {
  Alert, Box, Checkbox, Chip, Divider, FormControlLabel, MenuItem, Stack, TextField, Typography,
} from '@mui/material'
import { Document } from '../api/client'

const HORIZON_TYPES = ['day_ahead', 'intraday', 'short_term', 'long_term']
const PROBLEM_TYPES = ['pv', 'wind', 'load', 'FA']

type Options = { timezones?: string[]; ts_resolutions?: string[] } | undefined

function at(document: Document, path: string): unknown {
  return path.split('.').reduce<unknown>(
    (node, key) => (node == null ? undefined : (node as Record<string, unknown>)[key]), document)
}

const asText = (value: unknown) => (value === null || value === undefined ? '' : String(value))
const asNumber = (value: unknown) => (typeof value === 'number' ? value : '')
const asList = (value: unknown): unknown[] => (Array.isArray(value) ? value : [])
const asRecord = (value: unknown): Record<string, unknown> =>
  (value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {})

/** A copy of `document` with one path replaced. Everything else is preserved by construction. */
function set(document: Document, path: string, value: unknown): Document {
  const [head, ...rest] = path.split('.')
  const next = rest.length === 0 ? value : set((document[head] ?? {}) as Document, rest.join('.'), value)
  // A list stays a list. Spread like a mapping, `nwp_models` became `{0: {...}}` the moment a
  // source folder was typed here, and a case with that in it does not save.
  if (Array.isArray(document)) {
    return document.map((item, index) => (String(index) === head ? next : item)) as unknown as Document
  }
  return { ...document, [head]: next }
}

function Group({ title, help, children }: { title: string; help: string; children: ReactNode }) {
  return (
    <Box data-testid={`header-group-${title.toLowerCase().replace(/[^a-z]+/g, '-')}`}>
      <Typography variant="subtitle2">{title}</Typography>
      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>{help}</Typography>
      <Stack direction="row" spacing={2} flexWrap="wrap" useFlexGap>{children}</Stack>
      <Divider sx={{ mt: 2 }} />
    </Box>
  )
}

export default function CaseHeader(
  { document, setDocument, options, readOnly }:
  { document: Document; setDocument: (next: Document) => void; options: Options; readOnly?: boolean },
) {
  const field = (path: string) => ({
    value: asText(at(document, path)),
    disabled: !!readOnly,
    onChange: (event: { target: { value: string } }) => setDocument(set(document, path, event.target.value)),
  })
  const numeric = (path: string) => ({
    value: asNumber(at(document, path)),
    disabled: !!readOnly,
    onChange: (event: { target: { value: string } }) => {
      const raw = event.target.value
      setDocument(set(document, path, raw === '' ? null : Number(raw)))
    },
  })
  const coordinate = (index: number) => {
    const current = asList(at(document, 'project.coordinates'))
    return {
      value: asNumber(current[index]),
      disabled: !!readOnly,
      onChange: (event: { target: { value: string } }) => {
        const next = [current[0] ?? null, current[1] ?? null]
        next[index] = event.target.value === '' ? null : Number(event.target.value)
        setDocument(set(document, 'project.coordinates', next))
      },
    }
  }
  const flag = (path: string) => ({
    checked: !!at(document, path),
    disabled: !!readOnly,
    onChange: (event: { target: { checked: boolean } }) => setDocument(set(document, path, event.target.checked)),
  })
  const method = (group: 'methods_cpu' | 'cluster_methods', key: string) => ({
    checked: !!at(document, `methods.${group}.${key}`),
    disabled: !!readOnly,
    onChange: (event: { target: { checked: boolean } }) =>
      setDocument(set(document, `methods.${group}.${key}`, event.target.checked)),
  })

  const horizonType = asText(at(document, 'project.horizon_type'))
  const nwp = asList(at(document, 'project.nwp_models'))
  const variables = asList(at(document, 'data.variables'))
  const cpuMethods = Object.keys(asRecord(at(document, 'methods.methods_cpu')))
  const clusterMethods = Object.keys(asRecord(at(document, 'methods.cluster_methods')))

  return (
    <Stack spacing={2} data-testid="case-header">
      <Alert severity="info" data-testid="header-scope">
        The settings a new site usually needs. Everything else in this case is kept exactly as it
        was — the tabs above still edit all of it, including anything this panel does not show.
      </Alert>

      <Group title="Site" help="Which site this case forecasts, and where its measurements come from. The site name decides which model folder the case owns, so two cases must not share it.">
        <TextField size="small" label="Site name" sx={{ minWidth: 200 }}
                   inputProps={{ 'data-testid': 'header-site-name' }} {...field('project.name')} />
        <TextField size="small" select label="What it forecasts" sx={{ minWidth: 150 }} {...field('project.type')}>
          {PROBLEM_TYPES.map(value => <MenuItem key={value} value={value}>{value}</MenuItem>)}
        </TextField>
        <TextField size="small" label="Rated power" sx={{ minWidth: 130 }} {...numeric('project.rated_power')} />
        <TextField size="small" label="Latitude" sx={{ minWidth: 120 }} {...coordinate(0)} />
        <TextField size="small" label="Longitude" sx={{ minWidth: 120 }} {...coordinate(1)} />
        <TextField size="small" label="Measurements file" sx={{ minWidth: 340 }}
                   helperText="Still the copied case's file until you change it"
                   inputProps={{ 'data-testid': 'header-filename' }} {...field('project.filename')} />
      </Group>

      <Group title="Resolution" help="How often a measurement arrives. Changing it changes the shape of every dataset built afterwards.">
        <TextField size="small" select label="Time step" sx={{ minWidth: 160 }} {...field('project.ts_resolution')}>
          {(options?.ts_resolutions ?? ['15min', 'H', 'D']).map(value =>
            <MenuItem key={value} value={value}>{value}</MenuItem>)}
        </TextField>
      </Group>

      <Group title="Horizon" help="How far ahead, and in what shape. A short-term case predicts one column per step; a long-term case predicts one value.">
        <TextField size="small" select label="Kind of forecast" sx={{ minWidth: 180 }} {...field('project.horizon_type')}>
          {HORIZON_TYPES.map(value => <MenuItem key={value} value={value}>{value}</MenuItem>)}
        </TextField>
        {/* the spread carries the shared `disabled`, so this field's own rule comes after it */}
        <TextField size="small" label="Hours ahead" sx={{ minWidth: 140 }}
                   helperText={horizonType === 'short_term' ? 'steps are derived from this'
                     : 'only used by a short-term case'}
                   inputProps={{ 'data-testid': 'header-horizon' }} {...numeric('project.horizon')}
                   disabled={!!readOnly || horizonType !== 'short_term'} />
      </Group>

      <Group title="Timezone" help="Local timezone describes the timestamps in the source file. Site timezone describes the physical location; the two can be different.">
        <TextField size="small" select label="Local timezone" sx={{ minWidth: 220, maxWidth: 420 }}
          helperText="Used to interpret the time series. CET changes to UTC+2 in summer; Etc/GMT-1 means UTC+1 all year."
          {...field('project.local_timezone')}>
          {[...new Set([...(options?.timezones ?? ['UTC']), asText(document.project?.local_timezone)].filter(Boolean))]
            .map(value => <MenuItem key={value} value={value}>{value === 'Etc/GMT-1' ? 'Etc/GMT-1 — UTC+1 all year' : value}</MenuItem>)}
        </TextField>
        <TextField size="small" select label="Site timezone" sx={{ minWidth: 220, maxWidth: 420 }}
          helperText="Timezone at the site, used for solar calculations. This does not label the source timestamps."
          {...field('project.site_timezone')}>
          {[...new Set([...(options?.timezones ?? ['UTC']), asText(document.project?.site_timezone)].filter(Boolean))]
            .map(value => <MenuItem key={value} value={value}>{value}</MenuItem>)}
        </TextField>
      </Group>

      <Group title="Input sources" help="Where the weather comes from. Add or remove sources in the Project tab; here you set the folder each one is stored in, which is what changes between machines.">
        {nwp.length === 0 && <Typography variant="body2" color="text.secondary"
                                        data-testid="header-no-nwp">No weather source configured.</Typography>}
        {nwp.map((entry, index) => (
          // A source read from Google Drive keeps nothing here: the folder is where a day is held
          // between its download and its conversion. Which of the two it is, is chosen in the Project tab.
          <TextField key={index} size="small" sx={{ minWidth: 380 }}
                     label={`${asText(asRecord(entry).model) || `source ${index + 1}`} ${
                       asRecord(entry).path_nwp_gdrive == null ? 'archive folder' : 'download folder (from Google Drive)'}`}
                     inputProps={{ 'data-testid': `header-nwp-${index}` }}
                     {...field(`project.nwp_models.${index}.path_nwp_source`)} />
        ))}
        <Chip size="small" variant="outlined" data-testid="header-variable-count"
              label={`${variables.length} input variables configured`} />
      </Group>

      <Group title="Model scope" help="Whether one model covers the whole site or clusters do, and which methods are tried. Unticking a method keeps its settings for later.">
        <FormControlLabel control={<Checkbox {...flag('project.is_global')} />} label="One global model" />
        <FormControlLabel control={<Checkbox {...flag('project.is_fuzzy')} />} label="Fuzzy clustering" />
        <Box sx={{ width: '100%' }}>
          <Typography variant="caption" color="text.secondary">CPU methods</Typography>
          <Stack direction="row" flexWrap="wrap" useFlexGap>
            {cpuMethods.length === 0 && <Typography variant="body2" color="text.secondary">none configured</Typography>}
            {cpuMethods.map(key => (
              <FormControlLabel key={key} label={key}
                                control={<Checkbox inputProps={{ 'data-testid': `header-cpu-${key}` } as never}
                                                   {...method('methods_cpu', key)} />} />
            ))}
          </Stack>
        </Box>
        <Box sx={{ width: '100%' }}>
          <Typography variant="caption" color="text.secondary">Cluster models</Typography>
          <Stack direction="row" flexWrap="wrap" useFlexGap>
            {clusterMethods.length === 0 && <Typography variant="body2" color="text.secondary">none configured</Typography>}
            {clusterMethods.map(key => (
              <FormControlLabel key={key} label={key}
                                control={<Checkbox inputProps={{ 'data-testid': `header-cluster-${key}` } as never}
                                                   {...method('cluster_methods', key)} />} />
            ))}
          </Stack>
        </Box>
      </Group>
    </Stack>
  )
}
