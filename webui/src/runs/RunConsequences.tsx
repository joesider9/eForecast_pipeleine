// A13-07: what starting this run would do to what already exists, before it does it.
//
// Every sentence here is read off the plan the server produced -- stage decisions, the files each
// stage would replace, the revision note -- so nothing is invented and nothing is softened. Two
// facts in particular are stated plainly because a colleague cannot find them out any other way:
//
//   * A replaced file is replaced, not deleted-with-a-copy-kept. Publication promotes with
//     `os.replace`, so the previous content is gone and "recovery" means producing it again.
//   * A new revision keeps a separate *training* history, but dataset products are **shared**.
//     Calling that "a safe copy" would be wrong, so the panel distinguishes the two.
import { Alert, AlertTitle, Box, Chip, Stack, Typography } from '@mui/material'
import type { ResumePlan, StageDecision } from '../api/client'
import { grouped, replacements, unresolved } from './runConsequences'

const SAMPLE = 5

function Lines({ label, items, tone }: { label: string; items: string[]; tone?: 'warning' }) {
  if (items.length === 0) return null
  return (
    <Box>
      <Typography variant="body2" sx={{ fontWeight: 600 }} color={tone === 'warning' ? 'warning.main' : undefined}>
        {label}
      </Typography>
      <Box component="ul" sx={{ m: 0, pl: 3 }}>
        {items.slice(0, SAMPLE).map((text, index) => (
          <Typography component="li" variant="body2" key={index}>{text}</Typography>
        ))}
        {items.length > SAMPLE && (
          <Typography component="li" variant="body2" color="text.secondary">
            … and {items.length - SAMPLE} more
          </Typography>
        )}
      </Box>
    </Box>
  )
}

export default function RunConsequences({ resume }: { resume: ResumePlan }) {
  const groups = grouped(resume)
  const replaced = replacements(resume)
  const cannotResolve = unresolved(resume)
  const newRevision = resume.action === 'new'

  return (
    <Alert severity={replaced.length ? 'warning' : 'info'} data-testid="run-consequences"
           data-consequential={replaced.length > 0}>
      <AlertTitle>
        {replaced.length
          ? `This run replaces ${replaced.length} file(s) that already exist`
          : 'This run replaces nothing that already exists'}
      </AlertTitle>
      <Stack spacing={1.5}>
        <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
          {(Object.keys(groups) as StageDecision[]).map(decision => groups[decision].length
            ? <Chip key={decision} size="small" data-testid={`consequences-${decision}`}
                    color={decision === 'rebuild' ? 'warning' : decision === 'blocked' ? 'error' : 'default'}
                    label={`${groups[decision].length} ${decision}`} />
            : null)}
        </Stack>

        <Lines label="Produced again, replacing what is there" tone="warning"
               items={replaced.map(item => `${item.stage}: ${item.path}`)} />

        {replaced.length > 0 && (
          <Typography variant="body2" data-testid="consequences-recovery">
            <strong>Recovery:</strong> a replaced file is overwritten in place — the previous
            content is not kept anywhere. Getting it back means producing it again, which is this
            same run. Anything downstream that was built from it stops matching until it is rebuilt
            too.
          </Typography>
        )}

        {/* Names only. The stage table underneath already gives each decision's reason, and
            repeating it here made the dialog say the same sentence twice. What this panel owes
            the reader is the part nothing else shows: the files, and what recovery means. */}
        <Lines label="Kept and reused as it is"
               items={[...groups.reuse, ...groups.skip].map(stage => stage.key)} />
        {groups.resume.length > 0 && (
          <Lines label="Continued from where it stopped" items={groups.resume.map(stage => stage.key)} />
        )}
        {groups.verify.length > 0 && (
          <Lines label="Checked before it is trusted" items={groups.verify.map(stage => stage.key)} />
        )}
        {groups.blocked.length > 0 && (
          <Lines label="Cannot run yet" tone="warning" items={groups.blocked.map(stage => stage.key)} />
        )}

        {cannotResolve.length > 0 && (
          <Lines label="Could not be resolved, so it is named rather than counted" tone="warning"
                 items={cannotResolve.map(item => `${item.stage}: ${item.item}`)} />
        )}

        {newRevision && (
          <Typography variant="body2" data-testid="consequences-revision">
            <strong>A new revision</strong> keeps a separate training history, so the previous
            models remain under their own generation. Dataset products are <strong>shared</strong>,
            not copied: they are reused when they verify and replaced when they do not, exactly as
            listed above.
          </Typography>
        )}

        {!resume.verified && (
          <Typography variant="caption" color="text.secondary" data-testid="consequences-unverified">
            Planned from recorded sizes and timestamps. The run hashes each artifact before it
            skips it, so a file that has changed since will be produced again even if it is listed
            as reused above.
          </Typography>
        )}
      </Stack>
    </Alert>
  )
}
