import { act, fireEvent, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import CaseEditor from './CaseEditor'
import { caseDocument, caseSummary, consequencesFixture, formSchema, scripts, snapshot, valid } from '../test/fixtures'
import { deferred, json, mockHttp, type HttpCall } from '../test/http'
import { renderPage } from '../test/render'
import type { CaseResponse, Document } from '../api/client'

vi.mock('@xyflow/react', async () => {
  const { useState } = await import('react')
  return {
    ...await vi.importActual<typeof import('@xyflow/react')>('@xyflow/react'),
    ReactFlow: function CanvasProbe() {
      const [failed, setFailed] = useState(false)
      if (failed) throw new Error('Injected canvas failure')
      return <button onClick={() => setFailed(true)}>Simulate canvas failure</button>
    },
    Background: () => null, Controls: () => null,
  }
})

function serveCase(override: (call: HttpCall) => Response | Promise<Response> | undefined = () => undefined) {
  let saved = snapshot()
  const calls = mockHttp(call => {
    const response = override(call)
    if (response) return response
    if (call.url.pathname === '/api/cases') return json([caseSummary(), caseSummary('other')])
    if (call.url.pathname === '/api/schema') return json(formSchema)
    if (call.url.pathname === '/api/options') return json({ flags: { cluster_methods: ['MLP', 'CNN'] } })
    if (call.url.pathname === '/api/validate') return json(valid)
    if (call.method === 'PUT') {
      saved = snapshot(call.body.document as Document, 'v2')
      return json({ ...saved, main: 'main.yaml', parts: [], note: '' })
    }
    if (call.url.pathname.startsWith('/api/cases/')) return json(saved)
    throw new Error(`Unhandled ${call.method} ${call.url}`)
  })
  return { calls, setSaved: (next: CaseResponse) => { saved = next } }
}

describe('case load/edit/save', () => {
  it('omits the Jobs tab while preserving the complete saved resource policy on save', async () => {
    const jobs = { n_cpus: 8, n_jobs: 4, n_gpus: 0, cpu_workers: 2, cpu_study_threads: null, cpu_study_device: 'cpu',
      n_jobs_mlp: 3, intra_op: 2, model_cache_mode: 'offline' }
    const { calls } = serveCase(call => call.url.pathname === '/api/cases/demo' && call.method === 'GET'
      ? json(snapshot({ ...caseDocument, jobs })) : undefined)
    renderPage(<CaseEditor />)
    const name = await screen.findByRole('textbox', { name: 'Project name' })
    expect(screen.queryByRole('tab', { name: 'Jobs' })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Save' })).toBeDisabled()
    fireEvent.change(name, { target: { value: 'edited site' } })
    await userEvent.click(screen.getByRole('button', { name: 'Save' }))
    await screen.findByText('Case saved.')
    expect((calls.find(call => call.method === 'PUT')!.body.document as Document).jobs).toEqual(jobs)
  })

  it('round trips schema and YAML edits without dropping unknown or legacy values', async () => {
    const { calls } = serveCase()
    renderPage(<CaseEditor />)
    const user = userEvent.setup()
    const name = await screen.findByRole('textbox', { name: 'Project name' })
    await user.clear(name); await user.type(name, 'edited')
    fireEvent.change(screen.getByRole('textbox', { name: 'YAML editor' }), {
      target: { value: 'note: updated\nenabled: false\ninterval_coverage: []\nthreshold: 0\n' },
    })
    await user.click(screen.getByRole('button', { name: 'Save' }))
    await screen.findByText('Case saved.')
    const call = calls.find(item => item.method === 'PUT')!
    const document = call.body.document as Document
    expect(call.body.expected_revision).toBe('v1')
    expect(document.project.name).toBe('edited')
    expect(document.project.metadata).toMatchObject({ enabled: false, interval_coverage: [], threshold: 0 })
    expect(document.extension).toEqual(caseDocument.extension)
    expect(screen.getByRole('button', { name: 'Save' })).toBeDisabled()
  })

  it('preserves edits made while a save response is delayed', async () => {
    const saving = deferred<Response>()
    const { calls } = serveCase(call => call.method === 'PUT' ? saving.promise : undefined)
    renderPage(<CaseEditor />)
    const user = userEvent.setup()
    const input = await screen.findByRole('textbox', { name: 'Project name' })
    await user.clear(input); await user.type(input, 'submitted')
    await user.click(screen.getByRole('button', { name: 'Save' }))
    await waitFor(() => expect(calls.some(call => call.method === 'PUT')).toBe(true))
    await user.clear(input); await user.type(input, 'newer local edit')
    await act(async () => saving.resolve(json({ ...snapshot({ ...caseDocument, project: { name: 'SUBMITTED canonical' } }, 'v2'), main: 'main.yaml', parts: [], note: '' })))
    await screen.findByText('Case saved. Your newer edits are still unsaved.')
    expect(input).toHaveValue('newer local edit')
    expect(screen.getByRole('button', { name: 'Save' })).toBeEnabled()
  })

  it('retains a dirty draft when a saved-case refetch arrives', async () => {
    const server = serveCase()
    renderPage(<CaseEditor />)
    const user = userEvent.setup()
    const input = await screen.findByRole('textbox', { name: 'Project name' })
    await user.clear(input); await user.type(input, 'local')
    server.setSaved(snapshot({ ...caseDocument, project: { name: 'remote change' } }, 'v2'))
    await user.click(screen.getByRole('button', { name: 'Refresh saved case' }))
    await screen.findByText(/The saved case changed. Your draft is retained/)
    expect(input).toHaveValue('local')
    expect(screen.getByRole('button', { name: 'Save' })).toBeDisabled()
    await user.click(screen.getByRole('button', { name: 'Use saved version' }))
    expect(input).toHaveValue('remote change')
  })

  it('shows a concurrent-save conflict and keeps the current draft', async () => {
    serveCase(call => call.method === 'PUT' ? json({ detail: 'Case changed', ...snapshot({ ...caseDocument, project: { name: 'other tab' } }, 'v2') }, 409) : undefined)
    renderPage(<CaseEditor />)
    const user = userEvent.setup()
    const input = await screen.findByRole('textbox', { name: 'Project name' })
    await user.clear(input); await user.type(input, 'my draft')
    await user.click(screen.getByRole('button', { name: 'Save' }))
    await screen.findByText(/Nothing saved: the saved case changed/)
    expect(input).toHaveValue('my draft')
    expect(screen.getByRole('button', { name: 'Save as new case' })).toBeEnabled()
    expect(screen.getByRole('button', { name: 'Save' })).toBeDisabled()
    await user.click(screen.getByRole('button', { name: 'Review differences' }))
    expect(screen.getByText('Current saved version against the version you loaded')).toBeInTheDocument()
  })

  it('keeps the last valid YAML value while showing a parse error', async () => {
    const { calls } = serveCase()
    renderPage(<CaseEditor />)
    await screen.findByRole('textbox', { name: 'Project name' })
    fireEvent.change(screen.getByRole('textbox', { name: 'YAML editor' }), { target: { value: 'broken: [' } })
    expect(await screen.findByText(/unexpected end|flow collection/i)).toBeInTheDocument()
    expect(calls.some(call => call.method === 'PUT')).toBe(false)
  })

  it('retains unfinished YAML across tabs and blocks actions until it is repaired', async () => {
    const { calls } = serveCase()
    const { router } = renderPage(<CaseEditor />, { extraRoutes: [{ path: '/', element: <div>home</div> }] })
    const user = userEvent.setup()
    await screen.findByRole('textbox', { name: 'Project name' })
    fireEvent.change(screen.getByRole('textbox', { name: 'YAML editor' }), { target: { value: 'unfinished: [' } })
    expect(await screen.findByText('unsaved changes', { exact: true })).toBeInTheDocument()
    for (const name of ['Save', 'Validate', 'Save as new case']) expect(screen.getByRole('button', { name })).toBeDisabled()
    const unload = new Event('beforeunload', { cancelable: true })
    window.dispatchEvent(unload)
    expect(unload.defaultPrevented).toBe(true)
    await user.click(screen.getByRole('tab', { name: 'Data' }))
    await user.click(screen.getByRole('tab', { name: 'Project' }))
    expect(screen.getByRole('textbox', { name: 'YAML editor' })).toHaveValue('unfinished: [')
    await act(() => router.navigate('/'))
    await user.click(await screen.findByRole('button', { name: 'Stay' }))
    fireEvent.change(await screen.findByRole('textbox', { name: 'YAML editor' }), { target: { value: '# retained comment\nnote: repaired\nenabled: false\nthreshold: 0\ninterval_coverage: []\n' } })
    await user.click(screen.getByRole('tab', { name: 'Data' }))
    await user.click(screen.getByRole('tab', { name: 'Project' }))
    expect((screen.getByRole('textbox', { name: 'YAML editor' }) as HTMLTextAreaElement).value).toContain('# retained comment')
    await user.click(screen.getByRole('button', { name: 'Save' }))
    await screen.findByText('Case saved.')
    expect((calls.find(call => call.method === 'PUT')!.body.document as Document).project.metadata).toEqual({ note: 'repaired', enabled: false, threshold: 0, interval_coverage: [] })
  })

  it('isolates equal field names by section and explicitly discards buffers without losing parsed edits', async () => {
    const schema: Document = structuredClone(formSchema)
    schema.$defs.Data = { type: 'object', properties: { metadata: { type: 'object', title: 'Metadata', additionalProperties: true } } }
    schema.properties.data = { $ref: '#/$defs/Data' }
    const document = { ...caseDocument, data: { metadata: { note: 'data metadata' } } }
    serveCase(call => call.url.pathname === '/api/schema' ? json(schema) :
      call.url.pathname === '/api/cases/demo' && call.method === 'GET' ? json(snapshot(document)) : undefined)
    renderPage(<CaseEditor />)
    const user = userEvent.setup()
    fireEvent.change(await screen.findByRole('textbox', { name: 'Project name' }), { target: { value: 'keep parsed edit' } })
    fireEvent.change(screen.getByRole('textbox', { name: 'YAML editor' }), { target: { value: 'project: [' } })
    await user.click(screen.getByRole('tab', { name: 'Data' }))
    expect((screen.getByRole('textbox', { name: 'YAML editor' }) as HTMLTextAreaElement).value).toContain('data metadata')
    fireEvent.change(screen.getByRole('textbox', { name: 'YAML editor' }), { target: { value: 'data: [' } })
    await user.click(screen.getByRole('button', { name: 'Discard YAML edits' }))
    expect((screen.getByRole('textbox', { name: 'YAML editor' }) as HTMLTextAreaElement).value).toContain('data metadata')
    expect(screen.getByRole('button', { name: 'Save' })).toBeDisabled()
    await user.click(screen.getByRole('tab', { name: 'Project' }))
    expect(screen.getByRole('textbox', { name: 'YAML editor' })).toHaveValue('project: [')
    await user.click(screen.getByRole('button', { name: 'Discard unfinished YAML' }))
    expect(screen.getByRole('textbox', { name: 'Project name' })).toHaveValue('keep parsed edit')
    expect((screen.getByRole('textbox', { name: 'YAML editor' }) as HTMLTextAreaElement).value).toContain('preserve me')
    expect(screen.getByRole('button', { name: 'Save' })).toBeEnabled()
  })

  it('preserves unfinished-only edits on refetch and clears them when explicitly using the saved version', async () => {
    const server = serveCase()
    renderPage(<CaseEditor />)
    const user = userEvent.setup()
    await screen.findByRole('textbox', { name: 'Project name' })
    fireEvent.change(screen.getByRole('textbox', { name: 'YAML editor' }), { target: { value: 'pending: [' } })
    await user.click(screen.getByRole('button', { name: 'Refresh saved case' }))
    await waitFor(() => expect(screen.getByRole('button', { name: 'Refresh saved case' })).toBeEnabled())
    expect(screen.getByRole('textbox', { name: 'YAML editor' })).toHaveValue('pending: [')
    server.setSaved(snapshot({ ...caseDocument, project: { name: 'remote', metadata: { note: 'server' } } }, 'v2'))
    await user.click(screen.getByRole('button', { name: 'Refresh saved case' }))
    await screen.findByText(/The saved case changed. Your draft is retained/)
    expect(screen.getByRole('textbox', { name: 'YAML editor' })).toHaveValue('pending: [')
    await user.click(screen.getByRole('button', { name: 'Use saved version' }))
    expect(screen.getByRole('textbox', { name: 'Project name' })).toHaveValue('remote')
    expect((screen.getByRole('textbox', { name: 'YAML editor' }) as HTMLTextAreaElement).value).toContain('server')
    expect(screen.queryByText(/Fix or discard unfinished YAML/)).not.toBeInTheDocument()
  })

  it('does not discard unfinished-only edits made after a submitted save', async () => {
    const saving = deferred<Response>()
    const { calls } = serveCase(call => call.method === 'PUT' ? saving.promise : undefined)
    renderPage(<CaseEditor />)
    const user = userEvent.setup()
    fireEvent.change(await screen.findByRole('textbox', { name: 'Project name' }), { target: { value: 'submitted' } })
    await user.click(screen.getByRole('button', { name: 'Save' }))
    await waitFor(() => expect(calls.some(call => call.method === 'PUT')).toBe(true))
    fireEvent.change(screen.getByRole('textbox', { name: 'YAML editor' }), { target: { value: 'new unfinished: [' } })
    const submitted = calls.find(call => call.method === 'PUT')!.body.document as Document
    await act(async () => saving.resolve(json({ ...snapshot(submitted, 'v2'), main: 'main.yaml', parts: [], note: '' })))
    await screen.findByText('Case saved. Your newer edits are still unsaved.')
    await user.click(screen.getByRole('tab', { name: 'Data' }))
    await user.click(screen.getByRole('tab', { name: 'Project' }))
    expect(screen.getByRole('textbox', { name: 'YAML editor' })).toHaveValue('new unfinished: [')
    expect(screen.getByRole('button', { name: 'Save' })).toBeDisabled()
    await user.click(screen.getByRole('button', { name: 'Discard YAML edits' }))
    expect(screen.getByText('saved', { exact: true })).toBeInTheDocument()
    expect(calls.filter(call => call.method === 'PUT')).toHaveLength(1)
  })

  it('invalidates a late validation result when only raw YAML changed and blocks launching from another tab', async () => {
    const checking = deferred<Response>()
    const { calls } = serveCase(call => call.url.pathname === '/api/validate' ? checking.promise :
      call.url.pathname === '/api/scripts' ? json(scripts) :
      call.url.pathname.endsWith('/runs') ? json([]) : undefined)
    renderPage(<CaseEditor />)
    const user = userEvent.setup()
    await screen.findByRole('textbox', { name: 'Project name' })
    await user.click(screen.getByRole('button', { name: 'Validate' }))
    fireEvent.change(screen.getByRole('textbox', { name: 'YAML editor' }), { target: { value: 'unfinished: [' } })
    await act(async () => checking.resolve(json(valid)))
    await user.click(screen.getByRole('tab', { name: 'Review & save' }))
    expect(screen.queryByText('The case is complete and valid.')).not.toBeInTheDocument()
    await user.click(screen.getByRole('tab', { name: 'Runs' }))
    await screen.findByText('No run yet for this case.')
    for (const name of ['Run datasets', 'Run training', 'Run combine']) expect(screen.getByRole('button', { name })).toBeDisabled()
    expect(calls.some(call => call.method === 'POST' && call.url.pathname.endsWith('/runs'))).toBe(false)
    await user.click(screen.getByRole('button', { name: 'Discard unfinished YAML' }))
    expect(screen.getByRole('button', { name: 'Run training' })).toBeEnabled()
  })

  it('rejects a delayed full-document field action after an unfinished-only edit', async () => {
    const applying = deferred<Response>()
    const schema: Document = structuredClone(formSchema)
    schema.properties.methods.properties = { base_data_types: { type: 'object', additionalProperties: true },
      metadata: { type: 'object', title: 'Metadata', additionalProperties: true } }
    const document = { ...caseDocument, methods: { base_data_types: {}, metadata: { note: 'last valid' } } }
    serveCase(call => call.url.pathname === '/api/schema' ? json(schema) :
      call.url.pathname === '/api/cases/demo' && call.method === 'GET' ? json(snapshot(document)) :
      call.url.pathname === '/api/apply-data-types' ? applying.promise : undefined)
    renderPage(<CaseEditor />)
    const user = userEvent.setup()
    await screen.findByRole('textbox', { name: 'Project name' })
    await user.click(screen.getByRole('tab', { name: 'Methods' }))
    await user.click(screen.getByRole('button', { name: 'Apply to families, clusterer, combiners' }))
    fireEvent.change(screen.getByRole('textbox', { name: 'YAML editor' }), { target: { value: 'new raw edit: [' } })
    await act(async () => applying.resolve(json({ document: { ...document, project: { name: 'stale response' } }, note: 'applied' })))
    expect(await screen.findByText(/The draft changed while applying options/)).toBeInTheDocument()
    expect(screen.getByRole('textbox', { name: 'YAML editor' })).toHaveValue('new raw edit: [')
    await user.click(screen.getByRole('tab', { name: 'Project' }))
    expect(screen.getByRole('textbox', { name: 'Project name' })).toHaveValue('baseline')
  })

  it('loads a family preset after a sync that changed nothing in the draft', async () => {
    const schema: Document = structuredClone(formSchema)
    schema.properties.methods.properties = { method_params: { type: 'object', additionalProperties: true } }
    const document = { ...caseDocument, methods: { method_params: { MLP: { epochs: 50 } } } }
    serveCase(call => call.url.pathname === '/api/schema' ? json(schema) :
      call.url.pathname === '/api/options' ? json({ families: ['TIMM'], family_preset_keys: { TIMM: ['epochs'] } }) :
      call.url.pathname === '/api/cases/demo' && call.method === 'GET' ? json(snapshot(document)) :
      call.url.pathname === '/api/sync' ? json({ document: structuredClone(document), note: 'Nothing to copy: the families already hold the data options.' }) :
      call.url.pathname === '/api/presets/family' ? json({ epochs: 100 }) : undefined)
    renderPage(<CaseEditor />)
    const user = userEvent.setup()
    await screen.findByRole('textbox', { name: 'Project name' })
    await user.click(screen.getByRole('tab', { name: 'Methods' }))
    await user.click(screen.getByRole('button', { name: /Sync data options/ }))
    await screen.findByText(/already hold the data options/)
    await user.type(screen.getByPlaceholderText('family'), 'TIMM')
    await user.click(screen.getByRole('button', { name: 'Add with presets' }))
    expect(await screen.findByRole('dialog')).toHaveTextContent('TIMM')
    expect(screen.queryByText(/The draft changed while loading the family preset/)).not.toBeInTheDocument()
  })

  it('refuses a family preset when raw YAML was edited while it loaded', async () => {
    const loading = deferred<Response>()
    const schema: Document = structuredClone(formSchema)
    schema.properties.methods.properties = { method_params: { type: 'object', additionalProperties: true },
      metadata: { type: 'object', title: 'Metadata', additionalProperties: true } }
    const document = { ...caseDocument, methods: { method_params: {}, metadata: { note: 'last valid' } } }
    serveCase(call => call.url.pathname === '/api/schema' ? json(schema) :
      call.url.pathname === '/api/options' ? json({ families: ['TIMM'] }) :
      call.url.pathname === '/api/cases/demo' && call.method === 'GET' ? json(snapshot(document)) :
      call.url.pathname === '/api/presets/family' ? loading.promise : undefined)
    renderPage(<CaseEditor />)
    const user = userEvent.setup()
    await screen.findByRole('textbox', { name: 'Project name' })
    await user.click(screen.getByRole('tab', { name: 'Methods' }))
    await user.type(screen.getByPlaceholderText('family'), 'TIMM')
    await user.click(screen.getByRole('button', { name: 'Add with presets' }))
    fireEvent.change(screen.getByRole('textbox', { name: 'YAML editor' }), { target: { value: 'note: [' } })
    await act(async () => loading.resolve(json({ epochs: 100 })))
    expect(await screen.findByText(/The draft changed while loading the family preset/)).toBeInTheDocument()
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    expect(screen.getByRole('textbox', { name: 'YAML editor' })).toHaveValue('note: [')
  })

  it('retains experiment YAML per tag across tabs and blocks save/build until discarded or repaired', async () => {
    const document = { ...caseDocument, methods: { method_params: { MLP: { experiment_tag: ['first'] } } }, experiments: { first: { row_all: [], output: [] }, second: { row_all: [], output: [] } } }
    serveCase(call => call.url.pathname === '/api/cases/demo' && call.method === 'GET' ? json(snapshot(document)) :
      call.url.pathname === '/api/layers' ? json({ layers: [], branches: [], merges: [], aliases: {} }) :
      call.url.pathname === '/api/graph/shapes' ? json({ shapes: {}, steps: 1 }) :
      call.url.pathname === '/api/graph/normalize' ? json(call.body.experiments) : undefined)
    renderPage(<CaseEditor />)
    const user = userEvent.setup()
    await screen.findByRole('textbox', { name: 'Project name' })
    await user.click(screen.getByRole('tab', { name: 'Experiments' }))
    await user.click(await screen.findByRole('button', { name: 'yaml' }))
    fireEvent.change(screen.getByRole('textbox', { name: 'YAML editor' }), { target: { value: 'row_all: [' } })
    expect(screen.getByRole('button', { name: 'Dry build' })).toBeDisabled()
    await user.click(screen.getByRole('combobox', { name: 'experiment' }))
    await user.click(screen.getByRole('option', { name: 'second' }))
    expect((screen.getByRole('textbox', { name: 'YAML editor' }) as HTMLTextAreaElement).value).toContain('output: []')
    expect(screen.getByRole('button', { name: 'Dry build' })).toBeDisabled()
    await user.click(screen.getByRole('tab', { name: 'Data' }))
    await user.click(screen.getByRole('tab', { name: 'Experiments' }))
    await user.click(await screen.findByRole('button', { name: 'yaml' }))
    expect(screen.getByRole('textbox', { name: 'YAML editor' })).toHaveValue('row_all: [')
    expect(screen.getByRole('button', { name: 'Save' })).toBeDisabled()
    await user.click(screen.getByRole('button', { name: 'Discard YAML edits' }))
    expect(screen.getByRole('button', { name: 'Validate' })).toBeEnabled()
    expect(screen.getByRole('button', { name: 'Dry build' })).toBeEnabled()
  })

  it('keeps the case draft and YAML buffer above a graph rendering failure and recovers locally', async () => {
    const logged = vi.spyOn(console, 'error').mockImplementation(() => {})
    const document = { ...caseDocument, experiments: { first: { row_all: [], output: [] } } }
    serveCase(call => call.url.pathname === '/api/cases/demo' && call.method === 'GET' ? json(snapshot(document)) :
      call.url.pathname === '/api/layers' ? json({ layers: [], branches: [], merges: [], aliases: {} }) :
      call.url.pathname === '/api/graph/shapes' ? json({ shapes: {}, steps: 1 }) :
      call.url.pathname === '/api/graph/normalize' ? json(call.body.experiments) : undefined)
    renderPage(<CaseEditor />)
    const user = userEvent.setup()
    fireEvent.change(await screen.findByRole('textbox', { name: 'Project name' }), { target: { value: 'preserved case edit' } })
    await user.click(screen.getByRole('tab', { name: 'Experiments' }))
    await user.click(await screen.findByRole('button', { name: 'yaml' }))
    fireEvent.change(screen.getByRole('textbox', { name: 'YAML editor' }), { target: { value: 'row_all: null\noutput: []\n' } })
    await user.click(screen.getByRole('button', { name: 'graph' }))
    await user.click(screen.getByRole('button', { name: 'Simulate canvas failure' }))
    expect(await screen.findByText(/The experiment editor could not render. Your case draft and YAML edits are retained/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Save' })).toBeDisabled()
    await user.click(screen.getByRole('button', { name: 'Try editor again' }))
    await user.click(await screen.findByRole('button', { name: 'yaml' }))
    expect(screen.getByRole('textbox', { name: 'YAML editor' })).toHaveValue('row_all: null\noutput: []\n')
    await user.click(screen.getByRole('tab', { name: 'Project' }))
    expect(screen.getByRole('textbox', { name: 'Project name' })).toHaveValue('preserved case edit')
    expect(screen.getByRole('button', { name: 'Save' })).toBeDisabled()
    logged.mockRestore()
  })

  it('disposes of YAML buffers when explicitly leaving a case for another case', async () => {
    serveCase(call => call.url.pathname === '/api/cases/other' ? json(snapshot({ ...caseDocument,
      project: { name: 'other project', metadata: { note: 'other case' } } }, 'o1', 'other')) : undefined)
    const { router } = renderPage(<CaseEditor />)
    const user = userEvent.setup()
    await screen.findByRole('textbox', { name: 'Project name' })
    fireEvent.change(screen.getByRole('textbox', { name: 'YAML editor' }), { target: { value: 'first case: [' } })
    await act(() => router.navigate('/cases/other'))
    await user.click(await screen.findByRole('button', { name: 'Leave without saving' }))
    expect(await screen.findByRole('textbox', { name: 'Project name' })).toHaveValue('other project')
    expect((screen.getByRole('textbox', { name: 'YAML editor' }) as HTMLTextAreaElement).value).toContain('other case')
    expect(screen.queryByText(/Fix or discard unfinished YAML/)).not.toBeInTheDocument()
  })

  it('reconciles an uncertain save without reissuing a write or losing the draft', async () => {
    let committed = snapshot()
    const { calls } = serveCase(call => {
      if (call.method === 'PUT') {
        committed = snapshot(call.body.document as Document, 'v2')
        return Promise.reject(new TypeError('connection lost after commit'))
      }
      if (call.url.pathname === '/api/cases/demo') return json(committed)
      return undefined
    })
    renderPage(<CaseEditor />)
    const user = userEvent.setup()
    const input = await screen.findByRole('textbox', { name: 'Project name' })
    await user.clear(input); await user.type(input, 'committed once')
    await user.click(screen.getByRole('button', { name: 'Save' }))
    await screen.findByText(/The save outcome is unknown/)
    expect(input).toHaveValue('committed once')
    expect(screen.getByRole('button', { name: 'Save' })).toBeDisabled()
    await user.click(screen.getByRole('button', { name: 'Refresh saved case' }))
    await screen.findByText('saved', { exact: true })
    expect(calls.filter(call => call.method === 'PUT')).toHaveLength(1)
  })

  it('keeps newer edits after a delayed save-as-new completes', async () => {
    const response = deferred<Response>()
    serveCase(call => call.method === 'POST' && call.url.pathname === '/api/cases' ? response.promise : undefined)
    const { router } = renderPage(<CaseEditor />)
    const user = userEvent.setup()
    const input = await screen.findByRole('textbox', { name: 'Project name' })
    await user.click(screen.getByRole('button', { name: 'Save as new case' }))
    await user.click(screen.getByRole('button', { name: 'Create case' }))
    // Background editor changes can arrive while the modal's request is pending.
    fireEvent.change(input, { target: { value: 'newer local content' } })
    await act(async () => response.resolve(json({ main: 'copy.yaml', parts: [], note: '' }, 201)))
    await screen.findByText(/Your newer edits remain in this draft/)
    expect(input).toHaveValue('newer local content')
    expect(router.state.location.pathname).toBe('/cases/demo')
  })

  it('blocks navigation while a draft is dirty and supports staying by keyboard', async () => {
    serveCase()
    const { router } = renderPage(<CaseEditor />, { extraRoutes: [{ path: '/', element: <div>home</div> }] })
    const user = userEvent.setup()
    const input = await screen.findByRole('textbox', { name: 'Project name' })
    await user.type(input, '-edit')
    await act(() => router.navigate('/'))
    const stay = await screen.findByRole('button', { name: 'Stay' })
    act(() => stay.focus()); await user.keyboard('{Enter}')
    expect(input).toHaveValue('baseline-edit')
    expect(router.state.location.pathname).toBe('/cases/demo')
  })

  it('isolates local form and preference state when the case changes', async () => {
    serveCase(call => call.url.pathname === '/api/cases/other' ? json(snapshot({ project: { name: 'other project' } }, 'o1', 'other')) : undefined)
    const { router } = renderPage(<CaseEditor />)
    await screen.findByRole('textbox', { name: 'Project name' })
    fireEvent.change(screen.getByRole('textbox', { name: 'Presets file' }), { target: { value: '/private/demo.yaml' } })
    await act(() => router.navigate('/cases/other'))
    expect(await screen.findByRole('textbox', { name: 'Project name' })).toHaveValue('other project')
    expect(screen.getByRole('textbox', { name: 'Presets file' })).not.toHaveValue('/private/demo.yaml')
  })
})

// A case builds one model, the model of one configuration. A save that changes what its
// datasets or models were built from is what makes a later run replace them, so that is
// read and accepted here -- by typing the name of the case -- before anything is saved.
describe('a save that changes what the case was built from', () => {
  // The server as it answers: a save is refused with what it would cost until the digest of
  // exactly that comes back. With two digests, what it costs moves on once after it was shown.
  const asking = (digests = ['digest-1']) => {
    let shown = 0
    return serveCase(call => {
      if (call.method !== 'PUT') return undefined
      const accepted = call.body.accepted_consequences
      if (shown === digests.length - 1 && accepted === digests[shown]) return undefined      // saved
      if (accepted === digests[shown]) shown += 1
      return json({ detail: 'Saving this changes what the case was built from.',
        consequences: consequencesFixture({ digest: digests[shown] }) }, 428)
    })
  }
  const edit = async (user: ReturnType<typeof userEvent.setup>, value = 'edited site') => {
    const input = await screen.findByRole('textbox', { name: 'Project name' })
    await user.clear(input); await user.type(input, value)
    return input
  }

  it('saves nothing until what it replaces was read and accepted by name', async () => {
    const { calls } = asking()
    renderPage(<CaseEditor />)
    const user = userEvent.setup()
    const input = await edit(user)
    await user.click(screen.getByRole('button', { name: 'Save' }))
    const dialog = await screen.findByRole('dialog')
    expect(dialog).toHaveTextContent('Saving this changes what demo was built from')
    expect(dialog).toHaveTextContent('Datasets are built again')
    expect(dialog).toHaveTextContent('TRANSFORMER: 8 studies, 101 trial folder(s)')
    expect(dialog).toHaveTextContent('Nothing is replaced by saving')
    expect(screen.getByText('Nothing saved yet: read what this change replaces, and accept it to save.')).toBeInTheDocument()
    expect(calls.filter(call => call.method === 'PUT')).toHaveLength(1)
    expect(calls.find(call => call.method === 'PUT')!.body).not.toHaveProperty('accepted_consequences')

    const accept = screen.getByTestId('accept-and-save')
    expect(accept).toBeDisabled()
    await user.type(screen.getByTestId('typed-confirmation'), 'demo')
    await user.click(accept)
    await screen.findByText('Case saved.')
    const saves = calls.filter(call => call.method === 'PUT')
    expect(saves).toHaveLength(2)
    expect(saves[1].body.accepted_consequences).toBe('digest-1')
    expect(saves[1].body.expected_revision).toBe('v1')
    expect((saves[1].body.document as Document).project.name).toBe('edited site')
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    expect(input).toHaveValue('edited site')
    expect(screen.getByRole('button', { name: 'Save' })).toBeDisabled()
  })

  it('keeps the draft and saves nothing when the question is declined', async () => {
    const { calls } = asking()
    renderPage(<CaseEditor />)
    const user = userEvent.setup()
    const input = await edit(user)
    await user.click(screen.getByRole('button', { name: 'Save' }))
    await screen.findByRole('dialog')
    await user.click(screen.getByRole('button', { name: 'Cancel, nothing is saved' }))
    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument())
    expect(screen.getByText('unsaved changes')).toBeInTheDocument()
    expect(input).toHaveValue('edited site')
    expect(screen.getByRole('button', { name: 'Save' })).toBeEnabled()
    expect(calls.filter(call => call.method === 'PUT')).toHaveLength(1)
  })

  it('asks again when what the change costs moved between the answer and the save', async () => {
    const { calls } = asking(['digest-1', 'digest-2'])
    renderPage(<CaseEditor />)
    const user = userEvent.setup()
    await edit(user)
    await user.click(screen.getByRole('button', { name: 'Save' }))
    await screen.findByRole('dialog')
    await user.type(screen.getByTestId('typed-confirmation'), 'demo')
    await user.click(screen.getByTestId('accept-and-save'))
    // the server answers with another list: the name typed for the first one accepts nothing
    await waitFor(() => expect(calls.filter(call => call.method === 'PUT')).toHaveLength(2))
    await waitFor(() => expect(screen.getByTestId('typed-confirmation')).toHaveValue(''))
    expect(screen.getByTestId('accept-and-save')).toBeDisabled()
    await user.type(screen.getByTestId('typed-confirmation'), 'demo')
    await user.click(screen.getByTestId('accept-and-save'))
    await screen.findByText('Case saved.')
    expect(calls.filter(call => call.method === 'PUT').map(call => call.body.accepted_consequences))
      .toEqual([undefined, 'digest-1', 'digest-2'])
  })

  it('tells before saving what the draft would cost, and keeps no answer for a draft that moved on', async () => {
    const { calls } = serveCase(call => call.url.pathname.endsWith('/consequences')
      ? json(consequencesFixture()) : undefined)
    renderPage(<CaseEditor />)
    const user = userEvent.setup()
    await screen.findByRole('textbox', { name: 'Project name' })
    await user.click(screen.getByRole('tab', { name: 'Review & save' }))
    expect(screen.getByTestId('consequences-check')).toHaveTextContent('Is what is built this configuration\'s?')
    await user.click(screen.getByRole('tab', { name: /Project/ }))
    await edit(user)
    await user.click(screen.getByRole('tab', { name: 'Review & save' }))
    await user.click(screen.getByTestId('consequences-check'))
    expect(await screen.findByTestId('consequences-datasets')).toHaveTextContent('nwp: 1 file(s)')
    const asked = calls.find(call => call.url.pathname === '/api/cases/demo/consequences')!
    expect((asked.body.document as Document).project.name).toBe('edited site')
    expect(calls.some(call => call.method === 'PUT')).toBe(false)
    expect(screen.queryByTestId('consequences-outdated')).not.toBeInTheDocument()
    // the draft is edited again: what was worked out for the earlier one is not shown as its answer
    await user.click(screen.getByRole('tab', { name: /Project/ }))
    await edit(user, 'edited again')
    await user.click(screen.getByRole('tab', { name: 'Review & save' }))
    expect(screen.queryByTestId('consequences-datasets')).not.toBeInTheDocument()
    expect(screen.getByTestId('consequences-check')).toHaveTextContent('What does saving this draft cost?')
  })
})
