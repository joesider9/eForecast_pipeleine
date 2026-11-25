import { useCallback, useEffect, useMemo, useRef, useState, type Dispatch, type SetStateAction } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { Alert, Box, Button, Chip, Dialog, DialogActions, DialogContent, DialogTitle, Divider, Drawer, FormControl, InputLabel, List, ListItemButton, ListItemText, Menu, MenuItem, Select, Stack, TextField, ToggleButton, ToggleButtonGroup, Typography } from '@mui/material'
import { Background, Controls, ReactFlow, type Edge, type Node, type OnNodesChange } from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import CodeMirror from '@uiw/react-codemirror'
import { yaml as yamlLanguage } from '@codemirror/lang-yaml'
import { api, ApiError, isAbort, Document } from '../api/client'
import { useRequestScope } from '../lib/useRequestScope'
import { useYamlBuffer } from '../forms/useYamlBuffer'
import { Catalogue, DryBuild, Experiment, ExperimentLayouts, Experiments, LayerSpec } from './types'
import { BranchData, HeadData, Selection } from './nodes'
import { nodeTypes } from './nodeTypes'
import LayerInspector from './LayerInspector'
import { assertExperiment, assertExperiments } from './structure'
import { fromYaml, toYaml } from '../lib/yaml'
import { addAssignment, assignedTags, assignmentProblem, changeAssignments, renameAssignments } from './assignments'

const BRANCH_ORDER = ['nwp', 'images', 'lstm', 'row_all', 'row_obs', 'row_nwp', 'row_calendar', 'row_obs_calendar', 'row_obs_nwp']
const NODE_GAP = 40
const heightOf = (layers: LayerSpec[], built: boolean) => 70 + Math.max(1, layers.length) * (built ? 50 : 34) + 40

// Which catalogue layers make sense on a branch: NWP tensors take convolutions/attention/vision, the lstm
// dict takes recurrent and time-series layers, rows take dense; the merge takes merge layers and dense.
function addableFor(catalogue: Catalogue | null, branch: string): Catalogue['layers'] {
  if (!catalogue) return []
  const layers = catalogue.layers.filter(l => l.name !== 'clip')
  if (branch === 'output') return layers.filter(l => l.kind === 'merge' || l.name === 'dense' || l.name === 'Dropout')
  const shapeLayers = layers.filter(l => l.kind === 'shape' && l.name !== 'Reshape')
  const dense = layers.filter(l => l.name === 'dense')
  if (branch === 'nwp') return [...layers.filter(l => l.inputs.includes('nwp')), ...dense, ...shapeLayers]
  if (branch === 'lstm') return [...layers.filter(l => l.inputs.includes('lstm')), ...dense, ...shapeLayers]
  if (branch === 'images') return [...layers.filter(l => l.inputs.includes('images')), ...dense, ...shapeLayers]
  return [...dense, ...shapeLayers]
}

const defaultLayer = (catalogue: Catalogue | null, type: string): LayerSpec => {
  const entry = catalogue?.layers.find(l => l.name === type)
  // a layer the catalogue gives no size_type takes none (Flatten, concatenate, the gated merges)
  if (!entry?.size_type) return { type, params: {}, enabled: true }
  const value = entry.name === 'cnn_att' ? 'cbam' : entry.kind === 'conv' ? 3 : 64
  return { type, params: { size: { tag: 'fixed', value } }, enabled: true }
}

export default function ExperimentsEditor({ document, setDocument, presetsPath, layoutState, graphFamilies = [] }: {
  document: Document; setDocument: (d: Document) => void; presetsPath?: string
  layoutState?: [ExperimentLayouts, Dispatch<SetStateAction<ExperimentLayouts>>]
  graphFamilies?: readonly string[]
}) {
  const signal = useRequestScope()
  const catalogueQuery = useQuery({ queryKey: ['layers'], queryFn: ({ signal }) => api.layers(signal), staleTime: Infinity })
  const catalogue = (catalogueQuery.data as Catalogue | undefined) ?? null
  const rawExperiments = document.experiments as Experiments | undefined
  const [experiments, setExperiments] = useState<Experiments | null>(null)
  const [tag, setTag] = useState<string | null>(null)
  const localLayout = useState<ExperimentLayouts>({})
  const [layouts, setLayouts] = layoutState ?? localLayout
  // ReactFlow is controlled. Accept positional changes without making layout a
  // scientific edit or replacing fresh layer callbacks/data with stale nodes.
  const onNodesChange = useCallback<OnNodesChange>(changes => {
    if (!tag) return
    const positions = changes.filter(change => change.type === 'position' && change.position)
    if (!positions.length) return
    setLayouts(current => {
      const next = { ...current[tag] }
      for (const change of positions) {
        if (change.type === 'position' && change.position) next[change.id] = change.position
      }
      return { ...current, [tag]: next }
    })
  }, [tag, setLayouts])
  const [selection, setSelection] = useState<Selection | null>(null)
  const [method, setMethod] = useState<string>('')
  const [build, setBuild] = useState<DryBuild | null>(null)
  const [buildError, setBuildError] = useState<string | null>(null)
  const [view, setView] = useState<'graph' | 'yaml'>('graph')
  // Explicit recentering remounts only the canvas. Model edits, layer selection,
  // YAML drafts and the case-owned positions stay above it.
  const [viewportReset, setViewportReset] = useState(0)
  const [help, setHelp] = useState(false)
  const [tagDialog, setTagDialog] = useState<{ mode: 'new' | 'rename' | 'duplicate'; value: string } | null>(null)
  const [branchMenu, setBranchMenu] = useState<HTMLElement | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [normalizationError, setNormalizationError] = useState<string | null>(null)
  const [normalizationRetry, setNormalizationRetry] = useState(0)
  const ownCommit = useRef<Experiments | null>(null)
  const latestDocument = useRef(document)
  const latestExperiments = useRef(experiments)
  latestDocument.current = document
  latestExperiments.current = experiments

  // Normalise whatever the YAML holds (tuples, legacy sizes) into the editor form once per document load.
  useEffect(() => {
    if (rawExperiments === ownCommit.current) return
    const controller = new AbortController()
    setNormalizationError(null); setBuild(null)
    if (!rawExperiments || !Object.keys(rawExperiments).length) { setExperiments({}); setTag(null); return }
    setExperiments(null)
    void api.graphNormalize(rawExperiments, controller.signal).then(data => {
      if (controller.signal.aborted) return
      assertExperiments(data)
      const exps = data
      setExperiments(exps); setTag(value => value && exps[value] ? value : Object.keys(exps)[0] ?? null)
    }).catch(error => {
      if (!controller.signal.aborted && !isAbort(error)) {
        setNormalizationError(String(error instanceof Error ? error.message : error))
        setTag(value => value && Object.prototype.hasOwnProperty.call(rawExperiments, value) ? value : Object.keys(rawExperiments)[0] ?? null)
        setView('yaml')
      }
    })
    return () => controller.abort()
  }, [rawExperiments, normalizationRetry])

  // The server's custom-deep-method catalogue excludes shallow, foundation and
  // site-GRAPH training even when their stored parameters contain a legacy tag.
  const families = useMemo(() => graphFamilies.filter(family => family in (document.methods?.method_params ?? {})), [document.methods, graphFamilies])
  useEffect(() => { if (!families.includes(method)) setMethod(families[0] ?? '') }, [families, method])
  const selectedMethod = useRef('')
  useEffect(() => {
    if (!method || !experiments || selectedMethod.current === method) return
    selectedMethod.current = method
    const assigned = assignedTags(document.methods?.method_params?.[method]?.experiment_tag)
    setTag(assigned.find(name => Object.prototype.hasOwnProperty.call(experiments, name)) ?? Object.keys(experiments)[0] ?? null)
    setSelection(null); setBuild(null)
  }, [method, experiments, document.methods])
  const methodTags = [...new Set(assignedTags(document.methods?.method_params?.[method]?.experiment_tag))]
  const assignmentError = assignmentProblem(document.methods?.method_params?.[method]?.experiment_tag)
  const assigned = !!tag && methodTags.includes(tag)
  const enabled = Boolean(document.methods?.global_methods?.[method] || document.methods?.cluster_methods?.[method])
  const usedBy = useMemo(() => {
    const map: Record<string, string[]> = {}
    for (const family of families) {
      for (const name of assignedTags(document.methods?.method_params?.[family]?.experiment_tag)) (map[name] ??= []).push(family)
    }
    return map
  }, [document.methods, families])

  const experiment: Experiment | null = tag && experiments ? experiments[tag] : null
  const yamlValue: unknown = experiment ?? (tag ? rawExperiments?.[tag] : null)
  const commit = (next: Experiments, base = latestDocument.current) => {
    try { assertExperiments(next) } catch (error) {
      setNotice(error instanceof Error ? error.message : String(error)); return
    }
    ownCommit.current = next; latestExperiments.current = next
    setExperiments(next); setDocument({ ...base, experiments: next }); setBuild(null)
  }
  const update = (exp: Experiment) => {
    if (!tag) return
    if (experiments) commit({ ...experiments, [tag]: exp })
    else setDocument({ ...latestDocument.current, experiments: { ...rawExperiments, [tag]: exp } })
  }

  const yaml = useYamlBuffer(JSON.stringify(['experiment', tag]), `Experiment ${tag ?? ''}`, 'experiments', yamlValue,
    parsed => update(parsed as Experiment), assertExperiment)
  const yamlRequest = { tag, text: yaml.text, source: rawExperiments }
  const currentYaml = useRef(yamlRequest)
  currentYaml.current = yamlRequest
  const normalizeYaml = useMutation({
    mutationFn: async (input: typeof yamlRequest) => {
      const parsed = fromYaml(input.text)
      return api.graphNormalize({ [input.tag!]: parsed }, signal())
    },
    onSuccess: (data, input) => {
      const current = currentYaml.current
      if (input.tag !== current.tag || input.text !== current.text || input.source !== current.source) {
        setNotice('The experiment changed while checking YAML. Normalize the current text again.'); return
      }
      try { assertExperiments(data) }
      catch (error) { setNotice(`Could not normalize YAML: ${error instanceof Error ? error.message : String(error)}`); return }
      if (!input.tag || !Object.prototype.hasOwnProperty.call(data, input.tag)) { setNotice('Could not normalize YAML: the response omitted this experiment.'); return }
      yaml.change(toYaml(data[input.tag]))
      setNotice('YAML normalized. Review the updated draft before saving.')
    },
    onError: (error, input) => {
      const current = currentYaml.current
      if (!isAbort(error) && input.tag === current.tag && input.text === current.text && input.source === current.source) {
        setNotice(`Could not normalize YAML: ${error instanceof Error ? error.message : String(error)}`)
      }
    },
  })

  // Buffer metadata stays on the client and invalidates a late build even when
  // unfinished YAML has not changed the last valid parsed document.
  const buildInput = { document: { ...document, experiments }, tag: tag!, method: method || null,
    yamlDraft: yaml.text, yamlInvalid: yaml.hasErrors }
  const currentBuild = useRef('')
  currentBuild.current = JSON.stringify(buildInput)
  const buildSignature = currentBuild.current
  useEffect(() => { setBuild(null) }, [buildSignature])
  const dry = useMutation({
    mutationFn: (input: typeof buildInput) => api.graphDryBuild(input.document, input.tag, input.method, signal()),
    onSuccess: (result, input) => {
      if (JSON.stringify(input) !== currentBuild.current) { setNotice('The draft changed during the dry build. Build again to check the current graph.'); return }
      setBuild(result as DryBuild); setBuildError(null)
    },
    onError: (e, input) => { if (!isAbort(e) && JSON.stringify(input) === currentBuild.current) { setBuild(null); setBuildError(e instanceof ApiError ? e.message : String(e)) } },
  })
  const shapes = useQuery({ queryKey: ['graph-shapes', document], queryFn: ({ signal }) => api.graphShapes(document, signal), staleTime: 60_000 })
  // "Load presets": templates of the repository module, of the default case, or of this case's own module;
  // tags already present are kept as they are
  const [presetMenu, setPresetMenu] = useState<HTMLElement | null>(null)
  const importPresets = useMutation({
    mutationFn: (source: 'repo' | 'default' | 'module') => source === 'module' ? api.graphModuleTemplates(String(document.experiments_module), signal()) : api.graphPresetTemplates(source, source === 'default' ? presetsPath : undefined, signal()),
    onSuccess: (data, source) => {
      try { assertExperiments(data) }
      catch (error) { setNotice(`Could not import templates: ${error instanceof Error ? error.message : String(error)}`); return }
      const incoming = data; const next = { ...(latestExperiments.current ?? {}) }; let n = 0
      for (const [k, v] of Object.entries(incoming)) if (!next[k]) { next[k] = v; n += 1 }
      commit(next); setNotice(`${n} template(s) loaded from ${source === 'repo' ? 'configuration/experiments.py' : source === 'default' ? (presetsPath ?? 'the default case') : 'the module of this case'} (${Object.keys(incoming).length - n} already present)`)
      if (!tag) setTag(Object.keys(next)[0] ?? null)
    },
    onError: (e) => { if (!isAbort(e)) setNotice(e instanceof ApiError ? e.message : String(e)) },
  })

  // layer operations
  const setLayer = (sel: Selection, layer: LayerSpec) => { if (!experiment) return; const layers = [...experiment[sel.branch]]; layers[sel.index] = layer; update({ ...experiment, [sel.branch]: layers }) }
  const removeLayer = (sel: Selection) => { if (!experiment) return; const layers = experiment[sel.branch].filter((_, i) => i !== sel.index); update({ ...experiment, [sel.branch]: layers }); setSelection(null) }
  const moveLayer = (sel: Selection, delta: number) => {
    if (!experiment) return; const layers = [...experiment[sel.branch]]; const j = sel.index + delta
    if (j < 0 || j >= layers.length) return; [layers[sel.index], layers[j]] = [layers[j], layers[sel.index]]
    update({ ...experiment, [sel.branch]: layers }); setSelection({ branch: sel.branch, index: j })
  }
  const addLayer = (branch: string, type: string) => { if (!experiment) return; const layers = [...(experiment[branch] ?? []), defaultLayer(catalogue, type)]; update({ ...experiment, [branch]: layers }); setSelection({ branch, index: layers.length - 1 }) }
  const addBranch = (branch: string) => { if (!experiment) return; update({ ...experiment, [branch]: [] }); setBranchMenu(null) }
  const removeBranch = (branch: string) => {
    if (!experiment || !tag) return
    const next = { ...experiment }; delete next[branch]; update(next)
    setLayouts(current => { const positions = { ...current[tag] }; delete positions[branch]; return { ...current, [tag]: positions } })
    if (selection?.branch === branch) setSelection(null)
  }

  // tag operations
  const applyTagDialog = () => {
    if (!tagDialog || !experiments) return
    const name = tagDialog.value.trim(); if (!name) return
    if (experiments[name] && (tagDialog.mode !== 'rename' || name !== tag)) { setNotice(`experiment ${name} already exists`); return }
    if (tagDialog.mode === 'new') {
      const fresh: Experiment = { row_all: [{ type: 'dense', params: { size: { tag: 'fixed', value: 64 } }, enabled: true }],
                                  output: [{ type: 'concatenate', params: {}, enabled: true }] }
      if (!method) return
      commit({ ...experiments, [name]: fresh }, addAssignment(latestDocument.current, method, name))
    }
    if (tagDialog.mode === 'duplicate' && experiment && method) commit({ ...experiments, [name]: JSON.parse(JSON.stringify(experiment)) }, addAssignment(latestDocument.current, method, name))
    if (tagDialog.mode === 'rename' && tag) {
      const next: Experiments = {}; for (const [k, v] of Object.entries(experiments)) next[k === tag ? name : k] = v
      commit(next, renameAssignments(latestDocument.current, families, tag, name))
    }
    const mode = tagDialog.mode
    setLayouts(current => {
      const next = { ...current, [name]: mode !== 'new' && tag ? { ...current[tag] } : {} }
      if (mode === 'rename' && tag && tag !== name) delete next[tag]
      return next
    })
    setTag(name); setTagDialog(null); setSelection(null)
  }
  const deleteTag = () => {
    if (!tag || !experiments) return
    if (usedBy[tag]?.length) { setNotice('Remove this experiment from its assigned methods before deleting it.'); return }
    const next = { ...experiments }; delete next[tag]; commit(next)
    setLayouts(current => { const remaining = { ...current }; delete remaining[tag]; return remaining })
    setTag(Object.keys(next)[0] ?? null); setSelection(null)
  }

  // graph layout: data branches in a column on the left, the merge in the middle, the head on the right
  const built = !!build
  const { nodes, edges } = useMemo(() => {
    const nodes: Node[] = []; const edges: Edge[] = []
    if (!experiment) return { nodes, edges }
    const branches = Object.keys(experiment).filter(b => b !== 'output').sort((a, b) => BRANCH_ORDER.indexOf(a.split('/').pop()!) - BRANCH_ORDER.indexOf(b.split('/').pop()!))
    let y = 0
    const shapeInfo = (build?.data_shapes.shapes ?? (shapes.data as any)?.shapes ?? {}) as Record<string, any>
    for (const b of branches) {
      const info = build?.branches[b]
      const data: BranchData = {
        branch: b, description: catalogue?.branches.find(x => x.name === b.split('/').pop())?.description ?? '', layers: experiment[b],
        input: info?.input ?? shapeInfo[b.split('/').pop()!], built: info?.layers, error: info?.error, errorLayer: info?.error_layer,
        selected: selection, addable: addableFor(catalogue, b.split('/').pop()!), onSelect: setSelection, onAdd: addLayer, onRemoveBranch: removeBranch,
      }
      nodes.push({ id: b, type: 'branch', position: { x: 0, y }, data })
      edges.push({ id: `${b}->merge`, source: b, target: 'output', animated: false })
      y += heightOf(experiment[b], built) + NODE_GAP
    }
    const mergeLayers = experiment.output ?? []
    const mergeData: BranchData = {
      branch: 'output', description: 'How the branches are merged before the head.', layers: mergeLayers, built: build?.merge?.layers, error: build?.merge?.error,
      selected: selection, addable: addableFor(catalogue, 'output'), onSelect: setSelection, onAdd: addLayer,
    }
    const mid = Math.max(0, (y - NODE_GAP) / 2 - heightOf(mergeLayers, built) / 2)
    nodes.push({ id: 'output', type: 'branch', position: { x: 360, y: mid }, data: mergeData })
    const headData: HeadData = { head: build?.head ?? null, steps: (shapes.data as any)?.steps, total: build?.total_params }
    nodes.push({ id: 'head', type: 'head', position: { x: 720, y: mid + 20 }, data: headData })
    edges.push({ id: 'merge->head', source: 'output', target: 'head' })
    return { nodes: nodes.map(node => ({ ...node, position: layouts[tag!]?.[node.id] ?? node.position })), edges }
  }, [experiment, build, selection, catalogue, shapes.data, built, layouts, tag])  // eslint-disable-line react-hooks/exhaustive-deps

  const selectedLayer = selection && experiment ? experiment[selection.branch]?.[selection.index] : undefined
  const missingBranches = (catalogue?.branches.map(b => b.name) ?? BRANCH_ORDER).filter(b => experiment && !(b in experiment))

  if (experiments === null && !normalizationError) return <Typography variant="body2" color="text.secondary">Loading the experiment templates…</Typography>
  const tags = Object.keys(experiments ?? rawExperiments ?? {})
  return (
    <Stack spacing={1}>
      {normalizationError && <Alert severity="error" action={<Button onClick={() => setNormalizationRetry(value => value + 1)}>Retry</Button>}>
        Could not load experiment templates: {normalizationError}. Your YAML is available below for repair.
      </Alert>}
      {catalogueQuery.isError && <Alert severity="error" action={<Button onClick={() => catalogueQuery.refetch()}>Retry</Button>}>Could not load the layer catalogue: {String(catalogueQuery.error)}</Alert>}
      {shapes.isError && <Alert severity="warning">Could not calculate data shapes: {String(shapes.error)}</Alert>}
      <Typography variant="body2" color="text.secondary">
        Experiments define custom deep-learning networks. Choose a method, then assign an existing experiment or create one for it.
        {' '}A method can use several experiments. CPU and foundation models are configured in Methods.
        {' '}Click a layer to edit it, "+ layer" to add one.
        {' '}Drag a branch header to move it. Fit graph brings all branches into view. Layout stays while this case is open; it does not change the model.
        The dry build runs the network on dummy tensors of the shapes this case implies and writes the shape after every layer.
        {document.experiments_module && <> The case also points at the module <code>{String(document.experiments_module)}</code>: templates defined there can be imported into the case.</>}
      </Typography>
      <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap>
        <FormControl size="small" sx={{ minWidth: 180 }} disabled={!families.length}>
          <InputLabel id="experiment-method-label">method</InputLabel>
          <Select labelId="experiment-method-label" label="method" value={families.includes(method) ? method : ''} onChange={e => setMethod(e.target.value)}>
            {families.map(f => <MenuItem key={f} value={f}>{f}</MenuItem>)}
          </Select>
        </FormControl>
        {method && <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap role="group" aria-label={`Experiments assigned to ${method}`}>
          <Typography variant="body2">Assigned to {method}:</Typography>
          {methodTags.length ? methodTags.map(name => <Chip key={name} size="small" label={name} variant={name === tag ? 'filled' : 'outlined'}
            disabled={!experiments?.[name]} onClick={() => { setTag(name); setSelection(null); setBuild(null) }} />)
            : <Typography variant="body2" color="text.secondary">None yet</Typography>}
        </Stack>}
      </Stack>
      {!families.length && <Alert severity="info">No custom deep-learning method is configured. Add a custom deep-learning family in Methods, then return here to assign its experiments.</Alert>}
      {assignmentError && <Alert severity="warning">{assignmentError}</Alert>}
      {method && !enabled && <Alert severity="info" sx={{ py: 0 }}>{method} is not enabled for training. You can prepare its experiments here; enable its global or cluster training in Methods when ready.</Alert>}
      {method && enabled && !methodTags.length && !assignmentError && <Alert severity="warning" sx={{ py: 0 }}>Assign at least one experiment to {method} before running it.</Alert>}
      {methodTags.some(name => !experiments?.[name]) && <Alert severity="info" sx={{ py: 0 }}>Some assigned experiments are not loaded here. Use Load presets to import their definitions before editing them.</Alert>}
      <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap>
        <FormControl size="small" sx={{ minWidth: 200 }}>
          <InputLabel id="experiment-tag-label">experiment</InputLabel>
          <Select labelId="experiment-tag-label" label="experiment" value={tag ?? ''} onChange={e => { setTag(e.target.value); setSelection(null); setBuild(null) }}>
            {tags.map(t => <MenuItem key={t} value={t}>{t}{usedBy[t] ? ` · ${usedBy[t].join(', ')}` : ''}</MenuItem>)}
          </Select>
        </FormControl>
        <Button size="small" variant="outlined" disabled={!method || !experiment || assigned || yaml.hasErrors || !!assignmentError}
          onClick={() => method && tag && setDocument(addAssignment(latestDocument.current, method, tag))}>Assign to {method || 'method'}</Button>
        {assigned && <Button size="small" disabled={yaml.hasErrors || !!assignmentError}
          onClick={() => setDocument(changeAssignments(latestDocument.current, method, names => names.filter(name => name !== tag)))}>Unassign from {method}</Button>}
        <Button size="small" disabled={experiments === null || !method || yaml.hasErrors || !!assignmentError} onClick={() => setTagDialog({ mode: 'new', value: '' })}>New</Button>
        <Button size="small" disabled={!experiment || !method || yaml.hasErrors || !!assignmentError} onClick={() => setTagDialog({ mode: 'duplicate', value: `${tag}_copy` })}>Duplicate</Button>
        <Button size="small" disabled={!experiment || yaml.hasErrors} onClick={() => setTagDialog({ mode: 'rename', value: tag ?? '' })}>Rename</Button>
        <Button size="small" color="error" disabled={!experiment || yaml.hasErrors} onClick={deleteTag}>Delete</Button>
        <Button size="small" onClick={e => setPresetMenu(e.currentTarget)} disabled={experiments === null || importPresets.isPending}>Load presets</Button>
        <Menu anchorEl={presetMenu} open={!!presetMenu} onClose={() => setPresetMenu(null)}>
          <MenuItem onClick={() => { importPresets.mutate('repo'); setPresetMenu(null) }}><ListItemText primary="Repository templates" secondary="configuration/experiments.py: cnn1, lstm1, mlp1…, transformer1…, grid1, lstm_att1" /></MenuItem>
          <MenuItem onClick={() => { importPresets.mutate('default'); setPresetMenu(null) }}><ListItemText primary="Presets file" secondary={`the templates of ${presetsPath ?? 'configuration/cases/default.yaml'} (chosen on the Project tab)`} /></MenuItem>
          {document.experiments_module && <MenuItem onClick={() => { importPresets.mutate('module'); setPresetMenu(null) }}><ListItemText primary="Module of this case" secondary={String(document.experiments_module)} /></MenuItem>}
        </Menu>
      </Stack>
      <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap>
        <Button size="small" disabled={!experiment || !missingBranches.length} onClick={e => setBranchMenu(e.currentTarget)}>+ branch</Button>
        <Menu anchorEl={branchMenu} open={!!branchMenu} onClose={() => setBranchMenu(null)}>
          {missingBranches.map(b => <MenuItem key={b} onClick={() => addBranch(b)}><ListItemText primary={b} secondary={catalogue?.branches.find(x => x.name === b)?.description} secondaryTypographyProps={{ sx: { maxWidth: 420 } }} /></MenuItem>)}
        </Menu>
        <Box sx={{ flex: 1 }} />
        <Button size="small" variant="contained" disabled={!method || !experiment || yaml.hasErrors || dry.isPending} onClick={() => dry.mutate(buildInput)}>{dry.isPending ? 'Building…' : 'Dry build'}</Button>
        <ToggleButtonGroup size="small" exclusive value={view} onChange={(_, v) => v && setView(v)}>
          <ToggleButton value="graph">graph</ToggleButton><ToggleButton value="yaml">yaml</ToggleButton>
        </ToggleButtonGroup>
        <Button size="small" onClick={() => setHelp(true)}>Layer guide</Button>
        {view === 'graph' && <>
          <Button size="small" disabled={!experiment} onClick={() => setViewportReset(value => value + 1)}>Fit graph</Button>
          <Button size="small" disabled={!experiment} onClick={() => {
            if (!tag) return
            setLayouts(current => ({ ...current, [tag]: {} }))
            setViewportReset(value => value + 1)
          }}>Reset layout</Button>
        </>}
      </Stack>
      {tag && !assigned && method && <Alert severity="info" sx={{ py: 0 }}><b>{tag}</b> is not assigned to {method}. Choose “Assign to {method}” to include it.</Alert>}
      {tag && (usedBy[tag]?.length ?? 0) > 1 && <Alert severity="info" sx={{ py: 0 }}>Shared by {usedBy[tag].join(', ')}. Editing this experiment affects all assigned methods; duplicate it to make a separate version.</Alert>}
      {buildError && <Alert severity="error">{buildError}</Alert>}
      {yaml.error && view === 'graph' && <Alert severity="warning">Unfinished YAML is retained. This graph shows the last valid value; return to YAML to repair or discard it.</Alert>}
      {build && (
        <Alert severity={build.ok ? 'success' : 'error'} sx={{ py: 0 }}>
          {build.ok ? <>Builds: {build.head?.n_out} outputs, {build.total_params?.toLocaleString()} parameters.</> : <>The build failed; the red node names the layer.</>}
          {' '}{build.data_shapes.notes.join('. ')}
        </Alert>
      )}
      {notice && <Alert severity="info" onClose={() => setNotice(null)} sx={{ py: 0 }}>{notice}</Alert>}
      {view === 'graph' && (
        <Stack direction="row" sx={{ height: 620, border: '1px solid #e5e7eb', borderRadius: 1 }}>
          <Box sx={{ flex: 1, minWidth: 0 }}>
            {experiment ? (
              <ReactFlow key={JSON.stringify([tag, viewportReset])} nodes={nodes} edges={edges} onNodesChange={onNodesChange} nodeTypes={nodeTypes}
                fitView fitViewOptions={{ padding: 0.2, minZoom: 0.05, maxZoom: 1 }} nodesConnectable={false} autoPanOnNodeDrag={false}
                proOptions={{ hideAttribution: true }} onPaneClick={() => setSelection(null)} minZoom={0.05}>
                <Background /><Controls showInteractive={false} />
              </ReactFlow>
            ) : <Box sx={{ p: 3 }}><Typography color="text.secondary">{normalizationError ?
              'Repair the source YAML to display this graph.' : 'No experiment template yet. Create one with New or start from Load presets.'}</Typography></Box>}
          </Box>
          {selectedLayer && selection && (
            <Stack sx={{ height: '100%' }}>
              <LayerInspector key={`${tag}/${selection.branch}/${selection.index}`} layer={selectedLayer} catalogue={catalogue} branch={selection.branch}
                branches={Object.keys(experiment ?? {}).filter(b => b !== 'output')}
                onChange={l => setLayer(selection, l)} onRemove={() => removeLayer(selection)} />
              <Stack direction="row" spacing={1} sx={{ px: 2, pb: 1, borderLeft: '1px solid #e5e7eb' }}>
                <Button size="small" onClick={() => moveLayer(selection, -1)} disabled={selection.index === 0}>↑ move up</Button>
                <Button size="small" onClick={() => moveLayer(selection, 1)} disabled={selection.index >= (experiment?.[selection.branch].length ?? 0) - 1}>↓ move down</Button>
              </Stack>
            </Stack>
          )}
        </Stack>
      )}
      {view === 'yaml' && tag && (
        <Stack spacing={1}>
          <CodeMirror value={yaml.text} height="520px" extensions={[yamlLanguage()]}
            onCreateEditor={view => view.contentDOM.setAttribute('aria-label', 'Experiment YAML')} onChange={yaml.change} />
          {yaml.error && <Alert severity="error">{yaml.error}</Alert>}
          <Button disabled={normalizeYaml.isPending} onClick={() => normalizeYaml.mutate(yamlRequest)}>
            {normalizeYaml.isPending ? 'Normalizing YAML…' : 'Normalize YAML'}
          </Button>
          {yaml.error && <Button onClick={yaml.discard}>Discard YAML edits</Button>}
        </Stack>
      )}

      <Dialog open={!!tagDialog} onClose={() => setTagDialog(null)}>
        <DialogTitle>{tagDialog?.mode === 'new' ? `New experiment for ${method}` : tagDialog?.mode === 'rename' ? 'Rename experiment' : `Duplicate experiment for ${method}`}</DialogTitle>
        <DialogContent><TextField autoFocus size="small" label="tag" value={tagDialog?.value ?? ''} onChange={e => setTagDialog(d => d && { ...d, value: e.target.value })} onKeyDown={e => e.key === 'Enter' && applyTagDialog()} sx={{ mt: 1 }} /></DialogContent>
        <DialogActions><Button onClick={() => setTagDialog(null)}>Cancel</Button><Button variant="contained" onClick={applyTagDialog}>{tagDialog?.mode === 'rename' ? 'Rename' : 'Create & assign'}</Button></DialogActions>
      </Dialog>

      <Drawer anchor="right" open={help} onClose={() => setHelp(false)}>
        <Box sx={{ width: 460, p: 2 }}>
          <Typography variant="h6">Layer guide</Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>What each branch receives and what each layer does. Every layer parameter sits under "params": a fixed value is used as it stands, while "choice" and "range" are searched by Optuna.</Typography>
          <Typography variant="subtitle2">Branches</Typography>
          <List dense>{catalogue?.branches.map(b => <ListItemButton key={b.name} disableRipple><ListItemText primary={b.name} secondary={b.description} /></ListItemButton>)}</List>
          <Divider sx={{ my: 1 }} />
          {['conv', 'attention', 'recurrent', 'dense', 'shape', 'merge', 'image'].map(kind => (
            <Box key={kind} sx={{ mb: 1 }}>
              <Typography variant="subtitle2" sx={{ textTransform: 'capitalize' }}>{kind}</Typography>
              <List dense>{catalogue?.layers.filter(l => l.kind === kind).map(l => (
                <ListItemButton key={l.name} disableRipple>
                  <ListItemText primary={<><Chip size="small" label={l.name} sx={{ mr: 1 }} />{l.label}</>} secondary={<>{l.description}<br /><i>size:</i> {l.size}<br /><i>parameters:</i> {l.params.map(x => x.name).join(", ") || "none"}</>} />
                </ListItemButton>
              ))}</List>
            </Box>
          ))}
        </Box>
      </Drawer>
    </Stack>
  )
}
