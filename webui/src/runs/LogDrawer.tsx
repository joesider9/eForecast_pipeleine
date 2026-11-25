import { useEffect, useRef, useState } from 'react'
import { Alert, Box, Button, Chip, Drawer, Stack, Typography } from '@mui/material'
import { api, isAbort, type Run } from '../api/client'
import RunInputs from './RunInputs'

const MAX_CHARS = 512 * 1024
const MAX_LINES = 4000

// Keep only the visible tail; the full log is streamed by the download endpoint.
function appendTail(before: string, incoming: string) {
  let value = before + incoming
  let dropped = value.length > MAX_CHARS
  value = value.slice(-MAX_CHARS)
  const lines = value.split('\n')
  if (lines.length > MAX_LINES) { value = lines.slice(-MAX_LINES).join('\n'); dropped = true }
  return { value, dropped }
}

// A14-03: one action, two outcomes a colleague can act on. Copying may be unavailable --
// clipboard access needs a secure context and the browser's permission -- so a refusal offers
// the download instead of failing silently.
async function copySupportReport(caseName: string, id: string) {
  const text = await api.supportReport(caseName, id)
  if (!navigator.clipboard?.writeText) throw new Error('this browser does not allow copying')
  await navigator.clipboard.writeText(text)
  return text.length
}

export default function LogDrawer({ caseName, run, onClose }: { caseName: string; run: Run | null; onClose: () => void }) {
  return run ? <LogTail key={`${caseName}:${run.id}`} caseName={caseName} run={run} onClose={onClose} /> : null
}

function LogTail({ caseName, run, onClose }: { caseName: string; run: Run; onClose: () => void }) {
  const [support, setSupport] = useState('')
  const [text, setText] = useState('')
  const [status, setStatus] = useState(run.status)
  const [follow, setFollow] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [dropped, setDropped] = useState(false)
  const [notice, setNotice] = useState<string | null>(null)
  const [progress, setProgress] = useState({ next: 0, size: 0, complete: false })
  const retained = useRef('')
  const box = useRef<HTMLPreElement>(null)
  const retry = useRef<() => void>(() => {})

  useEffect(() => {
    const controller = new AbortController()
    let timer: number | undefined
    let offset = 0
    let generation: string | undefined
    let settling = false
    let busy = false
    const schedule = (delay: number) => { timer = window.setTimeout(poll, delay) }
    const poll = async () => {
      if (controller.signal.aborted || busy) return
      if (timer !== undefined) window.clearTimeout(timer)
      busy = true
      try {
        const chunk = await api.runLog(caseName, run.id, offset, controller.signal)
        if (controller.signal.aborted) return
        const replaced = generation !== undefined && chunk.generation !== undefined && generation !== chunk.generation
        generation = chunk.generation
        if (replaced || chunk.offset !== offset) {
          retained.current = ''; setText(''); setDropped(false); setNotice('The log was replaced or truncated; reading it from the beginning.')
          offset = 0; settling = false
          if (chunk.offset !== 0) { schedule(0); return }
        }
        if (chunk.next < chunk.offset || chunk.next > chunk.size || (chunk.next === chunk.offset && chunk.next < chunk.size)) {
          throw new Error('The server returned a log page without valid forward progress.')
        }
        const grew = chunk.next > offset
        offset = chunk.next
        const tail = appendTail(retained.current, chunk.text)
        retained.current = tail.value
        setText(tail.value)
        if (tail.dropped) setDropped(true)
        setError(null); setStatus(chunk.status)
        const drained = offset >= chunk.size
        const complete = drained && chunk.status !== 'running' && settling && !grew
        setProgress({ next: offset, size: chunk.size, complete })
        if (!drained) { settling = false; schedule(0) }
        else if (chunk.status === 'running') { settling = false; schedule(1500) }
        else if (!complete) { settling = true; schedule(500) }
      } catch (exception) {
        if (!controller.signal.aborted && !isAbort(exception)) {
          setError(`Could not read the log: ${exception instanceof Error ? exception.message : String(exception)}`)
          schedule(3000)
        }
      } finally { busy = false }
    }
    retry.current = () => { void poll() }
    void poll()
    return () => { controller.abort(); if (timer !== undefined) window.clearTimeout(timer) }
  }, [caseName, run.id])

  useEffect(() => { if (follow && box.current) box.current.scrollTop = box.current.scrollHeight }, [text, follow])

  return <Drawer anchor="bottom" open onClose={onClose} PaperProps={{ sx: { height: '60vh' } }}>
    <Stack direction="row" spacing={1} alignItems="center" sx={{ px: 2, py: 1, borderBottom: '1px solid #e5e7eb' }}>
      <Typography variant="subtitle2">{run.script} · {run.id}</Typography>
      <Chip size="small" label={status} color={status === 'running' ? 'info' : status === 'finished' ? 'success' : status === 'failed' ? 'error' : 'default'} />
      <Typography variant="caption" sx={{ flex: 1 }}>{progress.next.toLocaleString()} / {progress.size.toLocaleString()} bytes · {progress.complete ? 'complete' : 'reading'}</Typography>
      <Button size="small" data-testid="copy-support-report" onClick={async () => {
        setSupport('collecting the report…')
        try {
          const size = await copySupportReport(caseName, run.id)
          setSupport(`Support report copied (${size.toLocaleString()} characters). Paste it to the maintainer.`)
        } catch (caught) {
          setSupport(`The report could not be copied (${caught instanceof Error ? caught.message : String(caught)}). Use Download support report.`)
        }
      }}>Copy support report</Button>
      <Button size="small" component="a" data-testid="download-support-report"
              href={api.supportReportDownload(caseName, run.id)}>Download support report</Button>
      <Button size="small" component="a" href={api.runLogDownload(caseName, run.id)}>Download full log</Button>
      <Button size="small" onClick={() => setFollow(value => !value)}>{follow ? 'stop following' : 'follow'}</Button>
      <Button size="small" data-testid="close-log" onClick={onClose}>Close</Button>
    </Stack>
    <Box sx={{ px: 2, py: 1, maxHeight: '25%', overflow: 'auto' }}><RunInputs run={run} /></Box>
    {error && <Alert severity="error" action={<Button onClick={() => retry.current()}>Retry now</Button>}>{error}</Alert>}
    {notice && <Alert severity="info">{notice}</Alert>}
    {support && <Alert severity="info" data-testid="support-notice" onClose={() => setSupport('')}>{support}</Alert>}
    {dropped && <Typography variant="caption" sx={{ px: 2 }}>Earlier output is omitted from this view. Showing at most 512 Ki characters / 4,000 lines; download the full log for all output.</Typography>}
    <Box component="pre" ref={box} aria-label="Run log" sx={{ m: 0, p: 2, flex: 1, overflow: 'auto', fontSize: 12, fontFamily: 'ui-monospace, monospace', whiteSpace: 'pre-wrap', bgcolor: '#0f172a', color: '#e2e8f0' }}>
      {text || (error ? 'Log unavailable.' : progress.complete ? '(empty)' : 'Loading log…')}
    </Box>
  </Drawer>
}
