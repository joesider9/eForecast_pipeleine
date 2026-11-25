import { Box, Chip, Stack, Table, TableBody, TableCell, TableHead, TableRow, Tooltip, Typography } from '@mui/material'
import type { ResumePlan, RunStage, StageDecision, StageDecisionEntry, StageTrialUnit } from '../api/client'

// What the plan decided, and what the run observed.  The two tables deliberately look alike:
// the point of a preview is that a person can recognise afterwards what they approved.

const decisionColour: Record<StageDecision, 'default' | 'success' | 'info' | 'warning' | 'error'> = {
  reuse: 'success', resume: 'info', verify: 'warning', rebuild: 'warning', blocked: 'error', skip: 'default',
}
const stateColour = (state: string, kind?: string) =>
  state === 'terminal' ? (kind === 'success' ? 'success' : kind === 'nothing_to_do' ? 'default' :
    kind === 'cancelled' ? 'warning' : kind === 'partial' ? 'warning' : 'error')
    : state === 'running' ? 'info' : state === 'unknown' ? 'error' : 'default'

export function StageDecisions({ resume }: { resume: ResumePlan }) {
  return (
    <Table size="small">
      <TableHead><TableRow>
        <TableCell>stage</TableCell><TableCell>decision</TableCell><TableCell>why</TableCell><TableCell align="right">artifacts</TableCell>
      </TableRow></TableHead>
      <TableBody>
        {resume.stages.map(item => <StageDecisionRow key={item.key} item={item} />)}
      </TableBody>
    </Table>
  )
}

/** The trials of the global study that a cluster trains again; no study's budget counts them. */
function transfersOf(unit: StageTrialUnit): string {
  const transfer = unit.transfer
  if (!transfer || !(transfer.sources || transfer.later)) return ''
  return `, transfers ${transfer.done}/${transfer.sources} trained`
    + (transfer.pending ? `, ${transfer.pending} to train` : '')
    + (transfer.later ? `, up to ${transfer.later} more after the global study` : '')
}

function StageDecisionRow({ item }: { item: StageDecisionEntry }) {
  const affected = item.affected ?? []
  return (
    <TableRow hover>
      <TableCell>{item.key}{item.scope !== 'global' && <Typography variant="caption" color="text.secondary"> · {item.scope}</Typography>}</TableCell>
      <TableCell><Chip size="small" label={item.decision} color={decisionColour[item.decision]} /></TableCell>
      <TableCell>
        <Typography variant="caption">{item.reason}</Typography>
        {item.units && <Typography variant="caption" color="text.secondary" component="div">
          {item.units.units.map(unit => `${unit.cluster}/${unit.method}: ` + (unit.retrain
            // a study of another configuration is not continued: its trials go, its search starts over
            ? `trained from the beginning, ${unit.discard ?? 0} trial folder(s) removed`
            : (unit.budget ? `${unit.budget.done}/${unit.budget.wanted} trials, ` : '') +
              `reuse ${unit.reuse}, redo ${unit.redo}`) + transfersOf(unit)).join(' · ')}
        </Typography>}
      </TableCell>
      <TableCell align="right">
        {affected.length > 0 && <Tooltip title={<Box sx={{ whiteSpace: 'pre-line' }}>{affected.slice(0, 12).join('\n')}
          {affected.length > 12 ? `\n… and ${affected.length - 12} more` : ''}</Box>}>
          <Chip size="small" variant="outlined" color="warning" label={`${affected.length} replaced`} />
        </Tooltip>}
        {!!item.affected_unresolved?.length && <Tooltip title={item.affected_unresolved.join('\n')}>
          <Chip size="small" variant="outlined" color="error" label="unresolved" sx={{ ml: 0.5 }} />
        </Tooltip>}
      </TableCell>
    </TableRow>
  )
}

// A stage of a run that exists: its own state and outcome, never the parent's.
export function RunStages({ stages }: { stages: RunStage[] }) {
  if (!stages.length) return <Typography variant="caption" color="text.secondary">This run has no stage records.</Typography>
  return (
    <Table size="small">
      <TableHead><TableRow>
        <TableCell>stage</TableCell><TableCell>state</TableCell><TableCell>artifact</TableCell><TableCell>detail</TableCell>
      </TableRow></TableHead>
      <TableBody>
        {stages.map(stage => (
          <TableRow key={stage.id} hover>
            <TableCell>{stage.stage}</TableCell>
            <TableCell>
              <Chip size="small" label={stage.outcome ? stage.outcome.kind : stage.state}
                color={stateColour(stage.state, stage.outcome?.kind)} />
            </TableCell>
            <TableCell>
              {/* readiness is the artifact's own verdict: a stage can return and still have published nothing usable */}
              <Typography variant="caption" color={stage.readiness?.status === 'verified' || stage.readiness?.status === 'reused'
                ? 'text.secondary' : 'warning.main'}>{stage.readiness?.status ?? 'unverified'}</Typography>
            </TableCell>
            <TableCell>
              <Stack spacing={0.25}>
                {stage.outcome?.detail && <Typography variant="caption">{stage.outcome.detail}</Typography>}
                {stage.blocked.map((reason, index) => <Typography key={index} variant="caption" color="error">{reason.reason}</Typography>)}
              </Stack>
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  )
}
