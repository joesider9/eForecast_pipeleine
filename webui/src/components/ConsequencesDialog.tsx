// The question a save asks when it changes what the case's datasets or models were built from.
//
// Saving replaces nothing: the next run does. But it is the save that makes the next run do
// it, so this is where it is said, in full, and where it is accepted -- by typing the name of
// the case. The acceptance is the digest of exactly what is on screen: the server compares it
// with what the change costs when the save arrives, and a list that has moved meanwhile comes
// back here to be read again instead of being saved under an answer given for another one.
import { useEffect, useState } from 'react'
import { Button, Dialog, DialogActions, DialogContent, DialogTitle, Stack, Typography } from '@mui/material'
import type { Consequences } from '../api/client'
import ChangeConsequences, { TypedConfirmation } from './ChangeConsequences'

export default function ConsequencesDialog({ name, report, saving, onCancel, onAccept }: {
  name: string; report: Consequences | null; saving: boolean
  onCancel: () => void; onAccept: (digest: string) => void
}) {
  const [typed, setTyped] = useState('')
  // An answer belongs to one list: when what is shown changes, it has to be typed again.
  useEffect(() => { setTyped('') }, [report?.digest])
  if (!report) return null
  const accepted = typed === name && !!report.digest
  return (
    <Dialog open fullWidth maxWidth="md" onClose={() => { if (!saving) onCancel() }} aria-labelledby="consequences-title">
      <DialogTitle id="consequences-title">Saving this changes what {name} was built from</DialogTitle>
      <DialogContent dividers>
        <Stack spacing={2}>
          <Typography variant="body2">
            Nothing is replaced by saving. The runs after it replace what is listed here, because a case
            builds one model and that model is the model of one configuration. If that is not what you
            want, cancel and put the settings back — or save the draft as a new case, which builds its own.
          </Typography>
          <ChangeConsequences report={report} />
          <TypedConfirmation name={name} value={typed} onChange={setTyped} disabled={saving} what="accept this and save" />
        </Stack>
      </DialogContent>
      <DialogActions>
        <Button onClick={onCancel} disabled={saving}>Cancel, nothing is saved</Button>
        <Button variant="contained" color="warning" disabled={!accepted || saving} data-testid="accept-and-save"
          onClick={() => accepted && onAccept(report.digest!)}>
          {saving ? 'Saving…' : 'Accept and save'}
        </Button>
      </DialogActions>
    </Dialog>
  )
}
