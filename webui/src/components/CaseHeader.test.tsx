// A13-06: the six groups must let a colleague change a site without costing the case anything
// they did not touch.
import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import type { Document } from '../api/client'
import CaseHeader from './CaseHeader'

const options = { timezones: ['UTC', 'Europe/Athens'], ts_resolutions: ['15min', 'H', 'D'] }

// A working case, with values nobody is going to edit through the header.
const working: Document = {
  project: {
    name: 'Aggiti', type: 'pv', rated_power: 100, coordinates: [38.0, 23.0],
    filename: '/data/aggiti/pv_ts_long.csv', ts_resolution: 'H', horizon_type: 'long_term',
    horizon: 0, site_timezone: 'Europe/Athens', local_timezone: 'EET', is_global: false,
    is_fuzzy: true, version_model: 1, country: 'Greece',
    nwp_models: [{ model: 'ecmwf', resolution: 0.1, path_nwp_source: '/media/old/HHD2' }],
  },
  data: { variables: [{ name: 'Flux' }, { name: 'Cloud' }], scale_row_method: ['minmax'] },
  methods: {
    methods_cpu: { CatBoost: true, Chronos2: true },
    cluster_methods: { CNN: true, MLP: false },
    method_params: { CatBoost: { iterations: 1000, spare_knob: 'keep me' } },
  },
  folders: { sys_folder: '/media/models' },
  jobs: { n_cpus: 16 },
  extension: { legacy_zero: 0, custom: { choices: [null, 2] } },
}

function mount(document: Document = working) {
  const setDocument = vi.fn()
  render(<CaseHeader document={document} setDocument={setDocument} options={options} />)
  return setDocument
}

describe('CaseHeader', () => {
  it('shows the six groups a new site needs and says the rest is kept', () => {
    mount()
    for (const group of ['site', 'resolution', 'horizon', 'timezone', 'input-sources', 'model-scope']) {
      expect(screen.getByTestId(`header-group-${group}`)).toBeInTheDocument()
    }
    expect(screen.getByTestId('header-scope')).toHaveTextContent(/kept exactly as it was/)
    expect(screen.getByTestId('header-scope')).toHaveTextContent(/does not show/)
  })

  it('does not offer the case name: that belongs to copying or creating', () => {
    mount()
    expect(screen.queryByLabelText(/case name/i)).toBeNull()
    expect(screen.getByTestId('header-site-name')).toHaveValue('Aggiti')
  })

  it('changing the site name touches only that field', async () => {
    const setDocument = mount()
    await userEvent.clear(screen.getByTestId('header-site-name'))
    await userEvent.type(screen.getByTestId('header-site-name'), 'N')
    const next = setDocument.mock.calls.at(-1)![0] as Document
    expect(next.project.name).not.toBe('Aggiti')
    // everything the header never showed is still there, identical
    expect(next.extension).toEqual(working.extension)
    expect(next.methods.method_params.CatBoost).toEqual({ iterations: 1000, spare_knob: 'keep me' })
    expect(next.jobs).toEqual(working.jobs)
    expect(next.project.country).toBe('Greece')
    expect(next.project.version_model).toBe(1)
    expect(next.project.filename).toBe(working.project.filename)
  })

  it('distinguishes source and site timezones without rewriting stored values', async () => {
    const setDocument = mount()
    expect(screen.getByText(/Local timezone describes the timestamps in the source file/)).toBeInTheDocument()
    expect(screen.getByText(/CET changes to UTC\+2 in summer/)).toBeInTheDocument()
    expect(screen.getByRole('combobox', { name: 'Local timezone' })).toHaveTextContent('EET')
    expect(screen.getByRole('combobox', { name: 'Site timezone' })).toHaveTextContent('Europe/Athens')
    expect(setDocument).not.toHaveBeenCalled()
    await userEvent.click(screen.getByRole('combobox', { name: 'Local timezone' }))
    await userEvent.click(screen.getByRole('option', { name: 'UTC' }))
    const next = setDocument.mock.calls.at(-1)![0] as Document
    expect(next.project.local_timezone).toBe('UTC')
    expect(next.project.site_timezone).toBe('Europe/Athens')
    expect(next.methods).toEqual(working.methods)
  })

  it('keeps the measurements file visible as the copied case\'s own', () => {
    mount()
    const filename = screen.getByTestId('header-filename')
    expect(filename).toHaveValue('/data/aggiti/pv_ts_long.csv')
    expect(screen.getByText(/Still the copied case's file until you change it/)).toBeInTheDocument()
  })

  it('edits the weather archive folder of each configured source', async () => {
    const setDocument = mount()
    await userEvent.type(screen.getByTestId('header-nwp-0'), '/new')
    const next = setDocument.mock.calls.at(-1)![0] as Document
    expect(next.project.nwp_models[0].path_nwp_source).not.toBe('/media/old/HHD2')
    expect(next.project.nwp_models[0].model).toBe('ecmwf')
    expect(next.project.nwp_models[0].resolution).toBe(0.1)
    // still the list the schema asks for: as a mapping keyed "0" the case would not save
    expect(Array.isArray(next.project.nwp_models)).toBe(true)
    expect(next.project.nwp_models).toHaveLength(1)
  })

  it('does not call the folder of a Google Drive source an archive, and keeps the link when it is edited', async () => {
    const link = 'https://drive.google.com/drive/folders/17cTFdd4sN-yNZUpynt3Z-tumo8j90WmD'
    const setDocument = mount({ ...working, project: { ...working.project,
      nwp_models: [{ model: 'ecmwf', resolution: 0.1, path_nwp_source: '/media/lab/ECMWF', path_nwp_gdrive: link }] } })
    expect(screen.getByLabelText('ecmwf download folder (from Google Drive)')).toHaveValue('/media/lab/ECMWF')
    expect(screen.queryByLabelText('ecmwf archive folder')).toBeNull()
    await userEvent.type(screen.getByTestId('header-nwp-0'), '2')
    expect((setDocument.mock.calls.at(-1)![0] as Document).project.nwp_models[0].path_nwp_gdrive).toBe(link)
  })

  it('says so when no weather source is configured', () => {
    mount({ ...working, project: { ...working.project, nwp_models: [] } })
    expect(screen.getByTestId('header-no-nwp')).toBeInTheDocument()
  })

  it('reports how many input variables the case carries without editing them here', () => {
    mount()
    expect(screen.getByTestId('header-variable-count')).toHaveTextContent('2 input variables')
  })

  it('unticking a method keeps its parameters for later', async () => {
    const setDocument = mount()
    await userEvent.click(screen.getByTestId('header-cpu-CatBoost'))
    const next = setDocument.mock.calls.at(-1)![0] as Document
    expect(next.methods.methods_cpu.CatBoost).toBe(false)
    expect(next.methods.method_params.CatBoost).toEqual({ iterations: 1000, spare_knob: 'keep me' })
    expect(next.methods.methods_cpu.Chronos2).toBe(true)
  })

  it('only offers the methods the case already configures', () => {
    mount()
    const scope = screen.getByTestId('header-group-model-scope')
    expect(within(scope).getByTestId('header-cpu-CatBoost')).toBeInTheDocument()
    expect(within(scope).getByTestId('header-cluster-MLP')).toBeInTheDocument()
    expect(within(scope).queryByTestId('header-cpu-TimesFM')).toBeNull()
  })

  it('disables hours-ahead for a case that is not short-term, and explains why', () => {
    mount()
    expect(screen.getByTestId('header-horizon')).toBeDisabled()
    expect(screen.getByText(/only used by a short-term case/)).toBeInTheDocument()
  })

  it('enables hours-ahead for a short-term case', () => {
    mount({ ...working, project: { ...working.project, horizon_type: 'short_term' } })
    expect(screen.getByTestId('header-horizon')).toBeEnabled()
    expect(screen.getByText(/steps are derived from this/)).toBeInTheDocument()
  })

  it('keeps a zero rated power rather than treating it as absent', async () => {
    const setDocument = mount({ ...working, project: { ...working.project, rated_power: 0 } })
    expect(screen.getByLabelText('Rated power')).toHaveValue('0')
    await userEvent.type(screen.getByLabelText('Latitude'), '1')
    const next = setDocument.mock.calls.at(-1)![0] as Document
    expect(next.project.rated_power).toBe(0)
  })

  it('writes both coordinates as a pair, never as one', async () => {
    const setDocument = mount()
    await userEvent.clear(screen.getByLabelText('Longitude'))
    const next = setDocument.mock.calls.at(-1)![0] as Document
    expect(Array.isArray(next.project.coordinates)).toBe(true)
    expect(next.project.coordinates[0]).toBe(38.0)
  })

  it('survives a case that has almost nothing set yet', () => {
    mount({ project: {}, data: {}, methods: {} })
    expect(screen.getByTestId('case-header')).toBeInTheDocument()
    expect(screen.getByTestId('header-no-nwp')).toBeInTheDocument()
    expect(screen.getByTestId('header-variable-count')).toHaveTextContent('0 input variables')
  })

  it('changes nothing while read only', async () => {
    const setDocument = vi.fn()
    render(<CaseHeader document={working} setDocument={setDocument} options={options} readOnly />)
    expect(screen.getByTestId('header-site-name')).toBeDisabled()
    expect(screen.getByTestId('header-cpu-CatBoost')).toBeDisabled()
    expect(screen.getByTestId('header-nwp-0')).toBeDisabled()
    // a disabled control refuses the interaction outright, so nothing can reach setDocument
    await userEvent.click(screen.getByTestId('header-cpu-CatBoost'), { pointerEventsCheck: 0 })
    await userEvent.type(screen.getByTestId('header-site-name'), 'x')
    expect(setDocument).not.toHaveBeenCalled()
  })
})
