import { Alert, AlertTitle, List, ListItem, ListItemText } from '@mui/material'
import { Validation } from '../api/client'

export default function ErrorList({ validation }: { validation: Validation | null }) {
  if (!validation) return null
  if (validation.valid) return <Alert severity="success">The case is complete and valid.</Alert>
  return (
    <Alert severity={validation.complete ? 'error' : 'warning'}>
      <AlertTitle>{validation.complete ? `${validation.errors.length} validation error(s)` : 'Incomplete case'}</AlertTitle>
      {validation.missing_sections.length > 0 && <div>Missing sections: {validation.missing_sections.join(', ')}</div>}
      <List dense disablePadding>
        {validation.errors.map((e, i) => (
          <ListItem key={i} disableGutters sx={{ py: 0 }}>
            <ListItemText primary={<code>{e.path}</code>} secondary={e.message} />
          </ListItem>
        ))}
      </List>
    </Alert>
  )
}
