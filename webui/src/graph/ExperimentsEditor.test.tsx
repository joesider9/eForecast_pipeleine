import { act, fireEvent, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useState } from 'react'
import { describe, expect, it, vi } from 'vitest'
import ExperimentsEditor from './ExperimentsEditor'
import { deferred, json, mockHttp, type HttpCall } from '../test/http'
import { renderPage } from '../test/render'
import type { Document } from '../api/client'

// API/data/editing behavior is under test; canvas geometry is browser-tested.
vi.mock('@xyflow/react', async () => ({
  ...await vi.importActual<typeof import('@xyflow/react')>('@xyflow/react'),
  ReactFlow: () => null, Background: () => null, Controls: () => null,
}))

const normalized = { legacy: { row_all: [{ type: 'dense', params: { size: { tag: 'fixed', value: 16 } }, enabled: true }],
  output: [{ type: 'concatenate', params: {}, enabled: true }] } }
const initial = { project: { name: 'original' }, methods: { method_params: { MLP: { experiment_tag: ['legacy'] } } }, experiments: { legacy: { row_all: [['dense', 16]] } } }

function Parent({ starting = initial }: { starting?: Document }) {
  const [document, setDocument] = useState<Document>(starting)
  return <>
    <button onClick={() => setDocument(current => ({ ...current, project: { name: 'newer project' } }))}>Edit project</button>
    <button onClick={() => setDocument({ project: {}, experiments: { replacement: { row_all: [] } } })}>Load other graph</button>
    <ExperimentsEditor document={document} setDocument={setDocument} graphFamilies={['MLP', 'CNN']} />
    <output data-testid="document">{JSON.stringify(document)}</output>
  </>
}

function serveGraph(override: (call: HttpCall) => Response | Promise<Response> | undefined = () => undefined) {
  return mockHttp(call => {
    const result = override(call)
    if (result) return result
    if (call.url.pathname === '/api/layers') return json({ layers: [], branches: [], merges: [], aliases: {} })
    if (call.url.pathname === '/api/graph/shapes') return json({ shapes: {}, steps: 1 })
    if (call.url.pathname === '/api/graph/normalize') return json(normalized)
    throw new Error(`Unexpected ${call.method} ${call.url}`)
  })
}

describe('graph normalization and async edits', () => {
  it('keeps null-branch YAML editable and leaves the last valid graph/document intact', async () => {
    serveGraph()
    renderPage(<Parent />)
    const user = userEvent.setup()
    await user.click(await screen.findByRole('button', { name: 'yaml' }))
    fireEvent.change(screen.getByRole('textbox', { name: 'YAML editor' }), { target: { value: 'row_all: null\noutput: []\n' } })
    expect(await screen.findByText('row_all: expected a list of layers.')).toBeInTheDocument()
    expect(screen.getByRole('textbox', { name: 'YAML editor' })).toHaveValue('row_all: null\noutput: []\n')
    expect(JSON.parse(screen.getByTestId('document').textContent!)).toEqual(initial)
    await user.click(screen.getByRole('button', { name: 'graph' }))
    expect(screen.getByText(/This graph shows the last valid value/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Dry build' })).toBeDisabled()
    await user.click(screen.getByRole('button', { name: 'yaml' }))
    fireEvent.change(screen.getByRole('textbox', { name: 'YAML editor' }), { target: { value: 'row_all: []\noutput: []\n' } })
    expect(screen.getByRole('button', { name: 'Dry build' })).toBeEnabled()
  })

  it.each([
    ['null\n', 'experiment'],
    ['[]\n', 'experiment'],
    ['row_all: []\n', 'output'],
    ['row_all: 7\noutput: []\n', 'row_all'],
    ['row_all: [null]\noutput: []\n', 'row_all[0]'],
    ['row_all: [{}]\noutput: []\n', 'row_all[0].type'],
    ['row_all: [{type: dense, params: wrong}]\noutput: []\n', 'row_all[0].params'],
    ['row_all: [{type: dense, params: {size: null}}]\noutput: []\n', 'row_all[0].params.size'],
    ['row_all: [{type: dense, params: {size: {tag: choice, value: 2}}}]\noutput: []\n', 'row_all[0].params.size.value'],
    ['row_all: [{type: dense, params: {size: {tag: range, value: [2]}}}]\noutput: []\n', 'row_all[0].params.size.value'],
    ['row_all: [{type: dense, params: {size: {tag: range, value: [{low: 2}, 4]}}}]\noutput: []\n', 'row_all[0].params.size.value'],
  ])('retains non-renderable YAML with a field path (%s)', async (text, path) => {
    serveGraph()
    renderPage(<Parent />)
    await userEvent.setup().click(await screen.findByRole('button', { name: 'yaml' }))
    fireEvent.change(screen.getByRole('textbox', { name: 'YAML editor' }), { target: { value: text } })
    await waitFor(() => expect(screen.getAllByRole('alert').some(alert => alert.textContent?.startsWith(`${path}:`))).toBe(true))
    expect(screen.getByRole('textbox', { name: 'YAML editor' })).toHaveValue(text)
    expect(JSON.parse(screen.getByTestId('document').textContent!)).toEqual(initial)
    expect(screen.getByRole('button', { name: 'Dry build' })).toBeDisabled()
  })

  it('normalizes legacy YAML through the backend, preserving zero, false and empty fixed lists', async () => {
    const canonical = { legacy: { row_all: [{ type: 'dense', enabled: false,
      params: { size: { tag: 'fixed', value: 0 }, filters: { tag: 'choice', value: [0, 2] } } }],
      output: [{ type: 'concatenate', params: { size: { tag: 'fixed', value: [] } } }] } }
    let checks = 0
    const calls = serveGraph(call => call.url.pathname === '/api/graph/normalize' && ++checks > 1 ? json(canonical) : undefined)
    renderPage(<Parent />)
    const user = userEvent.setup()
    await user.click(await screen.findByRole('button', { name: 'yaml' }))
    fireEvent.change(screen.getByRole('textbox', { name: 'YAML editor' }), { target: {
      value: 'row_all:\n  - layer: dense\n    size: 0\n    enabled: false\n    filters: {choices: [0, 2]}\noutput:\n  - [concatenate, []]\n',
    } })
    expect(screen.getByRole('button', { name: 'Dry build' })).toBeDisabled()
    expect(JSON.parse(screen.getByTestId('document').textContent!)).toEqual(initial)
    await user.click(screen.getByRole('button', { name: 'Edit project' }))
    await user.click(screen.getByRole('button', { name: 'Normalize YAML' }))
    await screen.findByText('YAML normalized. Review the updated draft before saving.')
    const document = JSON.parse(screen.getByTestId('document').textContent!)
    expect(document.experiments).toEqual(canonical)
    expect(document.project.name).toBe('newer project')
    expect(screen.getByRole('button', { name: 'Dry build' })).toBeEnabled()
    const check = calls.filter(call => call.url.pathname === '/api/graph/normalize')[1]
    expect((check.body.experiments as Document).legacy.row_all[0]).toMatchObject({ layer: 'dense', size: 0, enabled: false })
    expect((screen.getByRole('textbox', { name: 'YAML editor' }) as HTMLTextAreaElement).value).toContain('tag: fixed')
  })

  it.each(['success', 'failure'])('ignores an old normalization %s after the YAML changes', async outcome => {
    const delayed = deferred<Response>()
    let checks = 0
    serveGraph(call => call.url.pathname === '/api/graph/normalize' && ++checks > 1 ? delayed.promise : undefined)
    renderPage(<Parent />)
    const user = userEvent.setup()
    await user.click(await screen.findByRole('button', { name: 'yaml' }))
    fireEvent.change(screen.getByRole('textbox', { name: 'YAML editor' }), { target: { value: 'row_all: [[dense, 32]]\noutput: []\n' } })
    await user.click(screen.getByRole('button', { name: 'Normalize YAML' }))
    const latest = 'row_all: []\noutput: []\n'
    fireEvent.change(screen.getByRole('textbox', { name: 'YAML editor' }), { target: { value: latest } })
    await act(async () => delayed.resolve(outcome === 'success' ? json(normalized) : json({ detail: 'old failure' }, 400)))
    expect(screen.getByRole('textbox', { name: 'YAML editor' })).toHaveValue(latest)
    expect(JSON.parse(screen.getByTestId('document').textContent!).experiments.legacy).toEqual({ row_all: [], output: [] })
    expect(screen.queryByText(/Could not normalize YAML: old failure/)).not.toBeInTheDocument()
  })

  it('makes source YAML repairable when normalization returns a malformed graph', async () => {
    let checks = 0
    serveGraph(call => call.url.pathname === '/api/graph/normalize' ? ++checks === 1 ?
      json({ legacy: { row_all: null, output: [] } }) : json(call.body.experiments) : undefined)
    renderPage(<Parent />)
    expect(await screen.findByText(/Could not load experiment templates: experiments.legacy.row_all/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'New' })).toBeDisabled()
    const user = userEvent.setup()
    await user.click(screen.getByRole('button', { name: 'graph' }))
    expect(screen.getByText('Repair the source YAML to display this graph.')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'yaml' }))
    const editor = screen.getByRole('textbox', { name: 'YAML editor' })
    expect((editor as HTMLTextAreaElement).value).toContain('dense')
    fireEvent.change(editor, { target: { value: 'row_all: []\noutput: []\n' } })
    await waitFor(() => expect(screen.queryByText(/Could not load experiment templates/)).not.toBeInTheDocument())
    expect(await screen.findByRole('button', { name: 'Dry build' })).toBeEnabled()
    expect(JSON.parse(screen.getByTestId('document').textContent!).experiments.legacy).toEqual({ row_all: [], output: [] })
  })

  it('rejects malformed imported templates before changing the draft', async () => {
    serveGraph(call => call.url.pathname === '/api/graph/preset-templates' ? json({ broken: { row_all: [null], output: [] } }) : undefined)
    renderPage(<Parent />)
    const user = userEvent.setup()
    await user.click(await screen.findByRole('button', { name: 'Load presets' }))
    await user.click(screen.getByText('Repository templates'))
    expect(await screen.findByText(/Could not import templates: experiments.broken.row_all\[0\]/)).toBeInTheDocument()
    expect(JSON.parse(screen.getByTestId('document').textContent!)).toEqual(initial)
    expect(screen.getByRole('button', { name: 'Dry build' })).toBeEnabled()
  })

  it('normalizes a legacy graph for display, then saves canonical YAML edits without changing other fields', async () => {
    const calls = serveGraph()
    renderPage(<Parent />)
    const user = userEvent.setup()
    await user.click(await screen.findByRole('button', { name: 'yaml' }))
    const editor = screen.getByRole('textbox', { name: 'YAML editor' })
    expect((editor as HTMLTextAreaElement).value).toContain('tag: fixed')
    fireEvent.change(editor, { target: { value: 'row_all:\n  - type: dense\n    params:\n      size: {tag: fixed, value: 32}\noutput: []\n' } })
    const document = JSON.parse(screen.getByTestId('document').textContent!)
    expect(document.experiments.legacy.row_all[0].params.size.value).toBe(32)
    expect(document.project.name).toBe('original')
    expect(calls.filter(call => call.url.pathname === '/api/graph/normalize')).toHaveLength(1)
  })

  it('does not replace a new graph with a delayed normalization of an older graph', async () => {
    const delayed = deferred<Response>()
    let count = 0
    const calls = serveGraph(call => call.url.pathname === '/api/graph/normalize' ?
      (++count === 1 ? delayed.promise : json({ replacement: { row_all: [], output: [] } })) : undefined)
    renderPage(<Parent />)
    await waitFor(() => expect(count).toBe(1))
    await userEvent.setup().click(screen.getByRole('button', { name: 'Load other graph' }))
    await waitFor(() => expect(screen.getByRole('combobox', { name: 'experiment' })).toHaveTextContent('replacement'))
    await act(async () => delayed.resolve(json(normalized)))
    expect(screen.getByRole('combobox', { name: 'experiment' })).not.toHaveTextContent('legacy')
    expect(calls.find(call => call.url.pathname === '/api/graph/normalize')!.signal?.aborted).toBe(true)
  })

  it('shows normalization failure with a retry instead of an empty graph', async () => {
    let failed = true
    serveGraph(call => call.url.pathname === '/api/graph/normalize' ? failed ? json({ detail: 'cannot normalize' }, 500) : json(normalized) : undefined)
    renderPage(<Parent />)
    await screen.findByText(/Could not load experiment templates: cannot normalize/)
    expect(screen.getByTestId('document')).toHaveTextContent('legacy')
    failed = false
    await userEvent.setup().click(screen.getByRole('button', { name: 'Retry' }))
    expect(await screen.findByRole('button', { name: 'Dry build' })).toBeInTheDocument()
  })

  it('merges a late preset response into the latest draft', async () => {
    const delayed = deferred<Response>()
    serveGraph(call => call.url.pathname === '/api/graph/preset-templates' ? delayed.promise : undefined)
    renderPage(<Parent />)
    const user = userEvent.setup()
    await user.click(await screen.findByRole('button', { name: 'Load presets' }))
    await user.click(screen.getByText('Repository templates'))
    await user.click(screen.getByRole('button', { name: 'Edit project' }))
    await act(async () => delayed.resolve(json({ imported: { row_all: [], output: [] } })))
    await screen.findByText(/1 template\(s\) loaded/)
    const document = JSON.parse(screen.getByTestId('document').textContent!)
    expect(document.project.name).toBe('newer project')
    expect(document.experiments.imported).toBeDefined()
  })

  it('does not display an old dry-build result as validation of a newer graph', async () => {
    const delayed = deferred<Response>()
    serveGraph(call => call.url.pathname === '/api/graph/dry-build' ? delayed.promise : undefined)
    renderPage(<Parent />)
    const user = userEvent.setup()
    await user.click(await screen.findByRole('button', { name: 'Dry build' }))
    await user.click(screen.getByRole('button', { name: 'Edit project' }))
    await act(async () => delayed.resolve(json({ ok: true, data_shapes: { shapes: {}, notes: [], steps: 1 }, branches: {}, merge: null, head: { n_out: 1 }, total_params: 123 })))
    expect(await screen.findByText(/The draft changed during the dry build/)).toBeInTheDocument()
    expect(screen.queryByText(/Builds:/)).not.toBeInTheDocument()
  })

  it('rejects a late build when only unfinished YAML changed', async () => {
    const delayed = deferred<Response>()
    serveGraph(call => call.url.pathname === '/api/graph/dry-build' ? delayed.promise : undefined)
    renderPage(<Parent />)
    const user = userEvent.setup()
    await user.click(await screen.findByRole('button', { name: 'Dry build' }))
    await user.click(screen.getByRole('button', { name: 'yaml' }))
    fireEvent.change(screen.getByRole('textbox', { name: 'YAML editor' }), { target: { value: 'row_all: [' } })
    await act(async () => delayed.resolve(json({ ok: true, data_shapes: { shapes: {}, notes: [], steps: 1 }, branches: {}, merge: null, head: { n_out: 1 }, total_params: 123 })))
    expect(await screen.findByText(/The draft changed during the dry build/)).toBeInTheDocument()
    expect(screen.queryByText(/Builds:/)).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Dry build' })).toBeDisabled()
  })
})

describe('method experiment assignments', () => {
  const starting = {
    project: { name: 'site' },
    methods: { global_methods: { MLP: true, CNN: false }, method_params: {
      MLP: { experiment_tag: { choices: ['legacy'] }, n_trials: 4, custom: { keep: false } },
      CNN: { experiment_tag: 'legacy', filters: 16 },
      CatBoost: { experiment_tag: 'inert', depth: 7 },
      TimesFM: { context_length: 96 }, Chronos2: {}, TiRex: {}, GRAPH: {}, clustering: {},
    } },
    experiments: { ...normalized, spare: { row_all: [], output: [] } },
  }
  const draft = (): Document => JSON.parse(screen.getByTestId('document').textContent!)
  function renderAssignments(value: Document = starting) {
    serveGraph(call => call.url.pathname === '/api/graph/normalize' ? json(call.body.experiments) : undefined)
    renderPage(<Parent starting={value} />)
    return userEvent.setup()
  }

  it('lists only custom deep methods and adds an existing template without replacing assignments or other parameters', async () => {
    const user = renderAssignments()
    await user.click(await screen.findByRole('combobox', { name: 'method' }))
    expect(screen.getAllByRole('option').map(option => option.textContent)).toEqual(['MLP', 'CNN'])
    await user.click(screen.getByRole('option', { name: 'MLP' }))
    expect(draft()).toEqual(starting)
    await user.click(screen.getByRole('combobox', { name: 'experiment' }))
    await user.click(screen.getByRole('option', { name: 'spare' }))
    await user.click(screen.getByRole('button', { name: 'Assign to MLP' }))
    expect(draft().methods).toEqual({ ...starting.methods, method_params: { ...starting.methods.method_params,
      MLP: { ...starting.methods.method_params.MLP, experiment_tag: { choices: ['legacy', 'spare'] } },
    } })
    expect(draft().experiments).toEqual(starting.experiments)
    expect(screen.getByRole('button', { name: 'Assign to MLP' })).toBeDisabled()
    await user.click(screen.getByRole('button', { name: 'Unassign from MLP' }))
    expect(draft()).toEqual(starting)
  })

  it('creates and duplicates directly into the chosen method while preserving training enablement', async () => {
    const user = renderAssignments()
    await user.click(await screen.findByRole('combobox', { name: 'method' }))
    await user.click(screen.getByRole('option', { name: 'CNN' }))
    await user.click(screen.getByRole('button', { name: 'New' }))
    await user.type(screen.getByRole('textbox', { name: 'tag' }), 'new_cnn')
    await user.click(screen.getByRole('button', { name: 'Create & assign' }))
    expect(draft().methods.method_params.CNN.experiment_tag).toEqual(['legacy', 'new_cnn'])
    expect(draft().methods.global_methods.CNN).toBe(false)
    expect(draft().experiments.new_cnn.row_all[0].type).toBe('dense')
    await user.click(await screen.findByRole('button', { name: 'Duplicate' }))
    await user.click(screen.getByRole('button', { name: 'Create & assign' }))
    expect(draft().methods.method_params.CNN.experiment_tag).toEqual(['legacy', 'new_cnn', 'new_cnn_copy'])
    expect(draft().experiments.new_cnn_copy).toEqual(draft().experiments.new_cnn)
    expect(draft().methods.method_params.MLP).toEqual(starting.methods.method_params.MLP)
    expect(draft().methods.method_params.CatBoost).toEqual(starting.methods.method_params.CatBoost)
  })

  it('renames shared assignments together and prevents deleting a referenced experiment', async () => {
    const user = renderAssignments()
    await user.click(await screen.findByRole('button', { name: 'Delete' }))
    expect(await screen.findByText(/Remove this experiment from its assigned methods/)).toBeInTheDocument()
    expect(draft()).toEqual(starting)
    await user.click(screen.getByRole('button', { name: 'Rename' }))
    await user.clear(screen.getByRole('textbox', { name: 'tag' }))
    await user.type(screen.getByRole('textbox', { name: 'tag' }), 'shared')
    const buttons = screen.getAllByRole('button', { name: 'Rename' })
    await user.click(buttons[buttons.length - 1])
    expect(draft().methods.method_params.MLP.experiment_tag).toEqual({ choices: ['shared'] })
    expect(draft().methods.method_params.CNN.experiment_tag).toBe('shared')
    expect(draft().experiments.legacy).toBeUndefined()
    expect(draft().experiments.shared).toEqual(starting.experiments.legacy)
    await user.click(await screen.findByRole('button', { name: 'Unassign from MLP' }))
    await user.click(screen.getByRole('button', { name: 'Delete' }))
    expect(draft().experiments.shared).toBeDefined()
    await user.click(screen.getByRole('combobox', { name: 'method' }))
    await user.click(screen.getByRole('option', { name: 'CNN' }))
    await user.click(screen.getByRole('button', { name: 'Unassign from CNN' }))
    await user.click(screen.getByRole('button', { name: 'Delete' }))
    expect(draft().experiments.shared).toBeUndefined()
    expect(draft().methods.method_params.CatBoost).toEqual(starting.methods.method_params.CatBoost)
  })

  it('keeps unfinished YAML and blocks assignment changes until it is repaired', async () => {
    const user = renderAssignments()
    await user.click(await screen.findByRole('button', { name: 'yaml' }))
    fireEvent.change(screen.getByRole('textbox', { name: 'YAML editor' }), { target: { value: 'row_all: [' } })
    expect(screen.getByRole('button', { name: 'Unassign from MLP' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'New' })).toBeDisabled()
    await user.click(screen.getByRole('combobox', { name: 'experiment' }))
    await user.click(screen.getByRole('option', { name: 'spare' }))
    expect(screen.getByRole('button', { name: 'Assign to MLP' })).toBeDisabled()
    expect(draft()).toEqual(starting)
  })

  it('does not offer CPU or foundation families as a fallback when no custom method exists', async () => {
    renderAssignments({ ...starting, methods: { method_params: { RF: {}, Chronos2: {}, TimesFM: {} } } })
    expect(await screen.findByText(/No custom deep-learning method is configured/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'New' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Dry build' })).toBeDisabled()
  })
})
