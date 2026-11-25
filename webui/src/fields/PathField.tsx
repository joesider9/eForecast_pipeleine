import { useCallback, useEffect, useRef, useState } from 'react'
import { FieldProps } from '@rjsf/utils'
import { Breadcrumbs, Button, Dialog, DialogActions, DialogContent, DialogTitle, IconButton, InputAdornment, Link, List, ListItemButton, ListItemIcon, ListItemText, TextField, Typography } from '@mui/material'
import FolderOpenIcon from '@mui/icons-material/FolderOpen'
import FolderIcon from '@mui/icons-material/Folder'
import InsertDriveFileOutlinedIcon from '@mui/icons-material/InsertDriveFileOutlined'
import { api, isAbort, Browse } from '../api/client'

// A path with a Browse… button: the dialog walks the folders the server allows (/api/browse), so
// the colleague picks folders of the workstation from any machine.
export function BrowseDialog({ open, initial, mode, onClose, onPick }: {
  open: boolean; initial: string; mode: 'dir' | 'file'; onClose: () => void; onPick: (path: string) => void
}) {
  const [listing, setListing] = useState<Browse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const pending = useRef<AbortController | null>(null)
  const load = useCallback(async (path?: string) => {
    pending.current?.abort()
    const controller = new AbortController()
    pending.current = controller
    try { setListing(await api.browse(path, mode, controller.signal)); setError(null) }
    catch (exc) { if (!isAbort(exc)) setError(exc instanceof Error ? exc.message : String(exc)) }
  }, [mode])
  useEffect(() => {
    if (open) { setListing(null); void load(initial || undefined) }
    return () => pending.current?.abort()
  }, [open, initial, load])
  const crumbs = listing?.path ? listing.path.split('/').filter(Boolean) : []
  return (
    <Dialog open={open} onClose={onClose} fullWidth maxWidth="sm">
      <DialogTitle>{mode === 'dir' ? 'Choose a folder' : 'Choose a file'}</DialogTitle>
      <DialogContent dividers sx={{ minHeight: 320 }}>
        <Breadcrumbs sx={{ mb: 1 }}>
          <Link component="button" onClick={() => load()}>roots</Link>
          {crumbs.map((c, i) => <Link key={i} component="button" onClick={() => load('/' + crumbs.slice(0, i + 1).join('/'))}>{c}</Link>)}
        </Breadcrumbs>
        {error && <Typography color="error">{error}</Typography>}
        <List dense>
          {listing?.parent && <ListItemButton onClick={() => load(listing.parent!)}><ListItemIcon><FolderIcon /></ListItemIcon><ListItemText primary=".." /></ListItemButton>}
          {(listing?.dirs ?? []).map(d => (
            <ListItemButton key={d} onClick={() => load(listing?.path ? `${listing.path}/${d}` : d)}>
              <ListItemIcon><FolderIcon color="primary" /></ListItemIcon><ListItemText primary={d} />
            </ListItemButton>
          ))}
          {(listing?.files ?? []).map(f => (
            <ListItemButton key={f} onClick={() => onPick(`${listing?.path}/${f}`)}>
              <ListItemIcon><InsertDriveFileOutlinedIcon /></ListItemIcon><ListItemText primary={f} />
            </ListItemButton>
          ))}
        </List>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        {mode === 'dir' && <Button variant="contained" disabled={!listing?.path} onClick={() => listing?.path && onPick(listing.path)}>Use this folder</Button>}
      </DialogActions>
    </Dialog>
  )
}

export default function PathField(props: FieldProps) {
  const { formData, onChange, name, schema, required, uiSchema } = props
  const mode: 'dir' | 'file' = (uiSchema?.['ui:options'] as any)?.mode === 'file' ? 'file' : 'dir'
  const [open, setOpen] = useState(false)
  return (
    <>
      <TextField fullWidth size="small" sx={{ mb: 1.5 }} label={`${schema.title ?? name}${required ? ' *' : ''}`}
        value={formData ?? ''} onChange={e => onChange(e.target.value || undefined)} helperText={schema.description}
        InputProps={{ endAdornment: <InputAdornment position="end"><IconButton size="small" onClick={() => setOpen(true)} aria-label="browse"><FolderOpenIcon fontSize="small" /></IconButton></InputAdornment> }} />
      <BrowseDialog open={open} initial={mode === 'file' ? String(formData ?? '').replace(/\/[^/]*$/, '') : String(formData ?? '')} mode={mode}
        onClose={() => setOpen(false)} onPick={p => { onChange(p); setOpen(false) }} />
    </>
  )
}
