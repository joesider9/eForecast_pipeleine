import { useEffect, useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { Alert, AlertTitle, Button, Checkbox, Chip, Dialog, DialogActions, DialogContent, DialogTitle, FormControlLabel, Stack, Typography } from '@mui/material'
import { api, isAbort, type Document, type InterruptedPublication, type RunPreview } from '../api/client'
import { ResourcePreviewReport } from './ResourceReport'
import { StageDecisions } from './StageTable'
import RunConsequences from './RunConsequences'
import { consequential, replacements } from './runConsequences'
import { RunChanges, TypedConfirmation } from '../components/ChangeConsequences'

export type RunAction = 'continue' | 'new' | 'rebuild'

const heading: Record<RunPreview['status'], { severity: 'success' | 'warning' | 'error' | 'info'; title: string }> = {
  ready: { severity: 'success', title: 'Ready to run' },
  verify: { severity: 'warning', title: 'Some artifacts must be verified first' },
  blocked: { severity: 'error', title: 'This run cannot start yet' },
  unresolved: { severity: 'error', title: 'This case could not be resolved' },
}

// What the run would do, before it does it.  The same request is then submitted with the same
// action, so the plan the user approved is the plan the run is accepted with.
export default function RunPlanDialog({ caseName, script, options, action, expectedRevision, tenantScope,
  onClose, onStart, starting, eligible, ineligible, manageable }: {
  caseName: string; script: string; options: Document; action: RunAction; expectedRevision?: string
  tenantScope: string; onClose: () => void; onStart: (acceptChanges: boolean) => void; starting: boolean
  eligible: boolean; ineligible: string | null; manageable: boolean
}) {
  const [recovery, setRecovery] = useState<InterruptedPublication[] | null>(null)
  // What this run replaces because the case no longer says what it was built from is accepted
  // by typing the case's name -- not by a tick, which is given in passing.
  const [typed, setTyped] = useState('')
  // A13-07: a consequential run is approved deliberately. The acceptance belongs to one plan,
  // so it is cleared whenever the plan changes -- a summary accepted for a different set of
  // files is not an approval of this one.
  const [accepted, setAccepted] = useState<string | null>(null)
  const preview = useQuery({
    queryKey: ['run-preview', tenantScope, caseName, script, JSON.stringify(options), action, expectedRevision ?? ''],
    queryFn: ({ signal }) => api.previewRun(caseName, script, options, action, expectedRevision, signal),
    gcTime: 0, retry: false,
  })
  const recover = useMutation({
    mutationFn: (apply: boolean) => api.recoverStages(caseName, apply),
    onSuccess: (report) => { setRecovery(report); void preview.refetch() },
  })
  const report = preview.data
  const resume = report?.resume ?? null
  const interrupted = resume?.interrupted ?? []
  const blockedByPlan = report?.status === 'blocked' || report?.status === 'unresolved'
  // What is being approved: the plan's identity together with the files it would replace, so a
  // re-plan that changes either one invalidates an earlier acceptance.
  const planIdentity = resume
    ? `${resume.plan_id}:${resume.action}:${replacements(resume).map(item => item.path).join('|')}`
    : null
  const changes = report?.changes ?? null
  const needsName = changes !== null
  const needsApproval = consequential(resume)
  const approved = needsName ? typed === caseName
    : !needsApproval || (planIdentity !== null && accepted === planIdentity)
  useEffect(() => {
    if (planIdentity !== null && accepted !== null && accepted !== planIdentity) setAccepted(null)
  }, [planIdentity, accepted])
  // The name was typed for one plan: another plan is another question.
  useEffect(() => { setTyped('') }, [planIdentity])
  // Approval needs a current successful plan and the same eligibility the card requires: a failed
  // preview, a dirty draft or a lost session must not leave this button live (R10-02).
  const refuse = !eligible ? ineligible
    : preview.isError ? 'This request could not be resolved; nothing can be approved from it.'
    : !preview.isSuccess ? 'Waiting for the plan.'
    : blockedByPlan ? 'The plan is blocked; supply what it names first.'
    : !approved ? (needsName ? 'Type the name of the case to accept what this run replaces.'
      : 'Accept the summary of what will be replaced first.') : null

  return (
    <Dialog open fullWidth maxWidth="md" onClose={onClose}>
      <DialogTitle>{script} · {action === 'new' ? 'new revision' : action}</DialogTitle>
      <DialogContent dividers>
        {preview.isPending && <Typography>Resolving the case…</Typography>}
        {preview.isError && !isAbort(preview.error) &&
          <Alert severity="error">Could not resolve this request: {String((preview.error as Error).message)}</Alert>}
        {report && <Stack spacing={2}>
          <Alert severity={(heading[report.status] ?? heading.unresolved).severity}>
            <AlertTitle>{(heading[report.status] ?? heading.unresolved).title}</AlertTitle>
            {report.status === 'unresolved' && (report.reason ?? 'the case did not resolve')}
            {report.status === 'blocked' && (report.stage_plan?.blocked ?? []).map((item, index) =>
              <Typography key={index} variant="body2">{item.reason}</Typography>)}
            {report.status === 'blocked' && report.reason && <Typography>{report.reason}</Typography>}
            {report.status === 'verify' && (report.stage_plan?.unverified ?? []).map((item, index) =>
              <Typography key={index} variant="body2">{item.reason}</Typography>)}
            {report.status === 'ready' && <Typography variant="body2">
              Saved case revision {report.case_revision?.slice(0, 12)}.
            </Typography>}
          </Alert>
          {report.resource_controls && <ResourcePreviewReport value={report.resource_controls} />}
          {resume?.revision?.limit && <Alert severity="warning" sx={{ py: 0 }}>
            {resume.revision.note}. {resume.revision.limit}
          </Alert>}
          {!resume?.verified && resume && <Typography variant="caption" color="text.secondary">
            Planned from recorded sizes and timestamps; the run hashes each artifact before it skips it.
          </Typography>}
          {interrupted.length > 0 && <Alert severity="warning" action={
            <Stack direction="row" spacing={1}>
              <Button size="small" disabled={recover.isPending} onClick={() => recover.mutate(false)}>Inspect</Button>
              {/* restoring a receipt changes what the artifacts claim: the server requires the
                  tenant's owner or admin, so the control is not offered to anyone else */}
              <Button size="small" color="warning" title={manageable ? undefined : 'the tenant owner or an admin restores a publication'}
                disabled={!eligible || recover.isPending || !manageable || !interrupted.some(item => item.recoverable)}
                onClick={() => eligible && manageable && recover.mutate(true)}>Restore</Button>
            </Stack>}>
            <AlertTitle>A publication of this case was interrupted</AlertTitle>
            {interrupted.map(item => <Typography key={item.stage} variant="body2">
              {item.stage}: the bundle on disk is {item.canonical}. {item.reason}
            </Typography>)}
          </Alert>}
          {recovery && <Alert severity={recovery.every(item => item.action === 'restored') ? 'success' : 'info'}>
            {recovery.length === 0 ? 'Nothing is interrupted any more.' : recovery.map(item =>
              <Typography key={item.stage} variant="body2">{item.stage}: {item.action} — {item.detail}</Typography>)}
          </Alert>}
          {changes && <RunChanges changes={changes} since={report.since?.changes ?? null} />}
          {resume && <>
            <RunConsequences resume={resume} />
            <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap>
              {(['reuse', 'resume', 'verify', 'rebuild', 'blocked', 'skip'] as const).map(name => resume.totals[name]
                ? <Chip key={name} size="small" label={`${resume.totals[name]} ${name}`} /> : null)}
              {!!resume.totals.artifacts_to_replace && <Chip size="small" color="warning"
                label={`${resume.totals.artifacts_to_replace} artifact(s) would be replaced`} />}
            </Stack>
            <StageDecisions resume={resume} />
          </>}
          {report && !resume && <Typography variant="body2" color="text.secondary">
            This request has no stage catalogue: it runs as a single operation.
          </Typography>}
        </Stack>}
      </DialogContent>
      <DialogActions sx={{ flexWrap: 'wrap' }}>
        {needsName && !blockedByPlan && (
          <Stack sx={{ mr: 'auto', ml: 1, minWidth: 340, py: 1 }}>
            <TypedConfirmation name={caseName} value={typed} onChange={setTyped} disabled={starting || !eligible}
              what="accept this and start" />
          </Stack>
        )}
        {!needsName && needsApproval && resume && (
          <FormControlLabel sx={{ mr: 'auto', ml: 1 }}
            label={`I have read what will be replaced: ${replacements(resume).length} file(s)`}
            control={<Checkbox inputProps={{ 'data-testid': 'accept-consequences' } as never}
                               checked={approved} disabled={starting || !eligible}
                               onChange={event => setAccepted(event.target.checked ? planIdentity : null)} />} />
        )}
        {refuse && <Typography variant="caption" color="warning.main"
                               data-testid="plan-refusal"
                               sx={{ mr: needsApproval ? 1 : 'auto', ml: 1 }}>{refuse}</Typography>}
        <Button onClick={onClose}>Close</Button>
        <Button variant="contained" disabled={starting || !!refuse} data-testid="plan-start"
                color={needsApproval || needsName ? 'warning' : 'primary'}
                onClick={() => onStart(needsName && approved)}>
          {/* the control is named after what it does, not after the action it was reached by */}
          {needsApproval && resume
            ? `Replace ${replacements(resume).length} file(s) and ${action === 'rebuild' ? 'rebuild' : 'start'}`
            : action === 'rebuild' ? 'Rebuild' : action === 'new' ? 'Start a new revision' : 'Start'}
        </Button>
      </DialogActions>
    </Dialog>
  )
}
