// What a change to a case does to what the case has already built.
//
// A case builds one model, and that model is the model of one configuration. A setting changed
// after something was built from it therefore costs something: the datasets made from the old
// variables are made again, a family whose parameters or search space changed is trained from
// the beginning. Every sentence here is read off what the server worked out from the receipts
// and the study journals (`change_consequences`, `stage_resume.changes`); the page decides
// nothing and softens nothing.
//
// Two places show it, and both ask for the same deliberate act before going on -- the name of
// the case, typed: saving the change (`ConsequencesDialog`) and starting the run that replaces
// the work (`RunPlanDialog`). A checkbox is ticked in passing; a name is not.
import { Alert, AlertTitle, Box, Stack, TextField, Typography } from '@mui/material'
import type { ConfigurationChanges, Consequences, SettingChange } from '../api/client'
import { describeChange } from './settingChanges'

const SHOWN = 12

function Items({ label, items, testId }: { label: string; items: string[]; testId?: string }) {
  if (items.length === 0) return null
  return (
    <Box data-testid={testId}>
      <Typography variant="body2" sx={{ fontWeight: 600 }}>{label}</Typography>
      <Box component="ul" sx={{ m: 0, pl: 3 }}>
        {items.slice(0, SHOWN).map((text, index) => <Typography component="li" variant="body2" key={index}>{text}</Typography>)}
        {items.length > SHOWN && <Typography component="li" variant="body2" color="text.secondary">
          … and {items.length - SHOWN} more</Typography>}
      </Box>
    </Box>
  )
}

/** The settings that differ, each at the path the editor shows it under. */
export function SettingChanges({ changes, label }: { changes: SettingChange[]; label: string }) {
  if (changes.length === 0) return null
  return (
    <Box data-testid="setting-changes">
      <Typography variant="body2" sx={{ fontWeight: 600 }}>{label}</Typography>
      <Box component="ul" sx={{ m: 0, pl: 3 }}>
        {changes.slice(0, SHOWN).map(change => (
          <Typography component="li" variant="body2" key={`${change.path}:${change.change}`} sx={{ overflowWrap: 'anywhere' }}>
            <Box component="code" sx={{ fontSize: '0.85em' }}>{change.path}</Box>: {describeChange(change)}
          </Typography>
        ))}
        {changes.length > SHOWN && <Typography component="li" variant="body2" color="text.secondary">
          … and {changes.length - SHOWN} more setting(s)</Typography>}
      </Box>
    </Box>
  )
}

const because = (names: string[]) => names.length ? ` — because of ${names.join(', ')}` : ''
const studies = (count: number) => `${count} stud${count === 1 ? 'y' : 'ies'}`

/** The name of the case, typed: the one act that accepts a change that replaces built work. */
export function TypedConfirmation({ name, value, onChange, disabled, what }: {
  name: string; value: string; onChange: (value: string) => void; disabled?: boolean; what: string
}) {
  return (
    <TextField size="small" fullWidth autoComplete="off" disabled={disabled} value={value}
      label={`Type the name of the case to ${what}`} placeholder={name}
      onChange={event => onChange(event.target.value)}
      error={value !== '' && value !== name}
      helperText={value !== '' && value !== name ? `That is not the name of this case (${name}).` : `The name is ${name}.`}
      inputProps={{ 'data-testid': 'typed-confirmation', spellCheck: false, 'aria-label': `Type the name of the case to ${what}` }} />
  )
}

/** What a run replaces for a changed configuration, as the run's preview reports it. */
export function RunChanges({ changes, since }: { changes: ConfigurationChanges; since?: SettingChange[] | null }) {
  const families = Object.entries(changes.families)
  return (
    <Alert severity="warning" data-testid="configuration-changes">
      <AlertTitle>The configuration is not the one this was built from</AlertTitle>
      <Stack spacing={1.5}>
        {since && <SettingChanges changes={since} label="Changed since it was built" />}
        <Items testId="changes-datasets" label="Datasets built again, replacing what is there"
          items={changes.stages.map(item => `${item.key}: ${item.files} file(s)`)} />
        <Items testId="changes-families" label="Families trained from the beginning — their trials are removed first"
          items={families.map(([method, entry]) => `${method}: ${studies(entry.units)}, ${entry.trials} trial folder(s)`)} />
        <Typography variant="body2">
          <strong>Nothing of this can be had back</strong> except by doing the work again: a replaced dataset
          is overwritten in place, and a removed trial is deleted with its weights. Put the settings back
          instead if that is not what you want — what is built is then this case's again as it is.
        </Typography>
      </Stack>
    </Alert>
  )
}

/** What saving a draft would do to what the case has built. */
export default function ChangeConsequences({ report }: { report: Consequences }) {
  if (report.status === 'unresolved') return (
    <Alert severity="warning" data-testid="consequences-unresolved">
      <AlertTitle>What this change costs could not be worked out</AlertTitle>
      {report.reason}
    </Alert>
  )
  if (report.status === 'new') return <Typography variant="body2" color="text.secondary" data-testid="consequences-none">
    Nothing is saved under this name yet, so nothing was built from it.</Typography>
  const pending = report.pending.datasets.length + report.pending.families.length > 0
  // The saved case itself: is what is built this configuration's, or is something still waiting
  // to be done again for a change that was saved earlier?
  if (report.status === 'unchanged') return (
    <Stack spacing={1}>
      {/* An editor fills in the defaults of the sections it showed. The files then differ and
          the case does not: said, so that "unsaved changes" and "no setting changes" agree. */}
      {(report.as_written ?? 0) > 0 && <Typography variant="body2" color="text.secondary" data-testid="consequences-written-down">
        No setting changes. The draft only writes down {report.as_written} value(s) the case already
        has as defaults; saving it costs nothing.</Typography>}
      {pending ? (
        <Alert severity="warning" data-testid="consequences-waiting">
          <AlertTitle>What is built is not this configuration's yet</AlertTitle>
          <Stack spacing={1}>
            <Items label="Datasets waiting to be built again — the next datasets run"
              items={report.pending.datasets.map(item => `${item.stage}: ${item.files} file(s)`)} />
            <Items label="Families waiting to be trained from the beginning — the next training run"
              items={report.pending.families.map(item => `${item.method}: ${studies(item.studies)}, ${item.trials} trial folder(s)`)} />
            <Typography variant="body2">
              Each of those runs asks before it replaces anything. Until they are done nothing is combined with
              what the configuration before left.
            </Typography>
          </Stack>
        </Alert>
      ) : <Typography variant="body2" color="success.main" data-testid="consequences-whole">
        Everything that is built was built from this configuration.</Typography>}
    </Stack>
  )
  const restored = report.restored.datasets.length + report.restored.families.length > 0
  return (
    <Stack spacing={1.5} data-testid="change-consequences" data-consequential={report.consequential}>
      <SettingChanges changes={report.changes} label={`${report.changes.length} setting(s) change`} />
      {report.relocated && <Alert severity="warning" data-testid="consequences-relocated">
        <AlertTitle>This becomes the case of another model folder</AlertTitle>
        <Typography variant="body2" sx={{ overflowWrap: 'anywhere' }}>
          It will build into <code>{report.relocated.to}</code>
          {report.relocated.holds_work ? ', which already holds work.' : ', where nothing is built yet: every step starts from nothing.'}
          {' '}What is built now stays in <code>{report.relocated.from}</code> and is no longer this case's.
        </Typography>
      </Alert>}
      {report.datasets.length > 0 && <Alert severity="warning" data-testid="consequences-datasets">
        <AlertTitle>Datasets are built again</AlertTitle>
        <Stack spacing={1}>
          <Items label="At the next datasets run, replacing what is there"
            items={report.datasets.map(item => `${item.stage}: ${item.files} file(s)${because(item.because)}`)} />
          {report.weather && <Typography variant="body2" sx={{ overflowWrap: 'anywhere' }}>
            The weather is read from another set of variables, kept in <code>{report.weather.to}</code>
            {report.weather.extracted ? ' (some days are there already)' : ', which holds none of it yet'}:
            every missing day is extracted again — from a Google Drive source that is the download again.
          </Typography>}
          {!report.weather && report.extract && report.extract.kept > 0 && <Typography variant="body2">
            The {report.extract.kept} extracted daily weather file(s) stay as they are.</Typography>}
          <Typography variant="body2">
            Until that run has finished nothing can be trained or combined, and every model trained on the
            datasets as they are is trained again after it.
          </Typography>
        </Stack>
      </Alert>}
      {report.families.length > 0 && <Alert severity="warning" data-testid="consequences-families">
        <AlertTitle>Families are trained from the beginning</AlertTitle>
        <Stack spacing={1}>
          <Items label="At the next training run, after their trials are removed"
            items={report.families.map(item =>
              `${item.method}: ${studies(item.studies)}, ${item.trials} trial folder(s)${because(item.because)}`)} />
          <Typography variant="body2">
            Their search starts over with the whole budget. Until they are trained again the combination step
            does not run: a model is the model of one configuration.
          </Typography>
        </Stack>
      </Alert>}
      {report.results.length > 0 && <Typography variant="body2" data-testid="consequences-results">
        Computed again at the next combine run: {report.results.join(', ')}.</Typography>}
      {pending && <Alert severity="info" data-testid="consequences-pending">
        <AlertTitle>Already waiting before this change</AlertTitle>
        <Items label="Datasets to build again" items={report.pending.datasets.map(item => item.stage)} />
        <Items label="Families to train from the beginning" items={report.pending.families.map(item => item.method)} />
      </Alert>}
      {restored && <Alert severity="success" data-testid="consequences-restored">
        <AlertTitle>This puts back what was built</AlertTitle>
        {[...report.restored.datasets, ...report.restored.families].join(', ')} no longer
        {' '}{report.restored.datasets.length + report.restored.families.length === 1 ? 'has' : 'have'} to be done again.
      </Alert>}
      {report.unresolved.length > 0 && <Items label="Could not be asked, so it is named rather than counted as nothing"
        items={report.unresolved} />}
      {!report.consequential && !pending && <Typography variant="body2" color="success.main" data-testid="consequences-nothing">
        Nothing that is built has to be built or trained again for this change.</Typography>}
    </Stack>
  )
}
