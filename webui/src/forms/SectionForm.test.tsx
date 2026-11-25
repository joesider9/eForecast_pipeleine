import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import productionSchema from '../../../schema/experiment_config.schema.json'
import SectionForm from './SectionForm'

describe('exported schema integration', () => {
  it.each([undefined, 'as_of', 'legacy_merge'])(
    'hides NWP policy %s while preserving it and allowing publication-delay edits', async policy => {
      const onChange = vi.fn()
      const value = { nwp_availability_hours: 3, ...(policy ? { nwp_selection_policy: policy } : {}) }
      render(<SectionForm fullSchema={productionSchema} section="data" value={value} onChange={onChange}
        context={{ document: { data: value }, options: null, setDocument: vi.fn() }} />)
      expect(screen.queryByText('Nwp Selection Policy')).not.toBeInTheDocument()
      expect(screen.queryByRole('combobox', { name: /selection policy/i })).not.toBeInTheDocument()
      if (policy === 'legacy_merge') {
        expect(screen.getByText(/This saved case uses legacy weather selection/)).toBeVisible()
      } else {
        expect(screen.queryByText(/legacy weather selection/)).not.toBeInTheDocument()
      }
      fireEvent.change(await screen.findByRole('spinbutton', { name: /Nwp Availability Hours/i }),
        { target: { value: '4' } })
      await waitFor(() => expect(onChange).toHaveBeenCalled())
      expect(onChange.mock.lastCall?.[0]).toMatchObject({
        nwp_availability_hours: 4, nwp_selection_policy: policy ?? 'as_of',
      })
    },
  )

  it.each([{}, { TimesFM: 'foundation' }])('hides runtime settings while preserving routing %j and legacy values on other edits', async runtime_ids => {
    const onChange = vi.fn()
    const value = { env_name: 'stored-env', path_env: '/old/env/bin', backend: 'TORCH',
      runtime_ids }
    render(<SectionForm fullSchema={productionSchema} section="methods" value={value} onChange={onChange}
      context={{ document: { methods: value }, options: null, setDocument: vi.fn() }} />)
    expect(screen.getByDisplayValue('stored-env')).not.toBeVisible()
    expect(screen.getByDisplayValue('/old/env/bin')).not.toBeVisible()
    expect(screen.queryByRole('textbox', { name: /Legacy environment|Legacy interpreter/i })).not.toBeInTheDocument()
    expect(screen.queryByText('Runtime Ids')).not.toBeInTheDocument()
    expect(screen.queryByRole('textbox', { name: /runtime ids/i })).not.toBeInTheDocument()
    fireEvent.change(screen.getByRole('textbox', { name: /^backend/i }), { target: { value: 'TF_2' } })
    await waitFor(() => expect(onChange).toHaveBeenCalled())
    expect(onChange.mock.lastCall?.[0]).toMatchObject({ ...value, backend: 'TF_2' })
  })

  it('gives a case that left the prediction intervals out their default, and not none at all', async () => {
    // An array the schema says nothing about is filled with an empty list, and for the intervals an
    // empty list is a choice: no intervals. The exported schema names the default for that reason.
    expect(productionSchema.$defs.MethodsConfig.properties.interval_coverage.default).toEqual([0.8, 0.9])
    const onChange = vi.fn()
    const value = { backend: 'TORCH' }
    render(<SectionForm fullSchema={productionSchema} section="methods" value={value} onChange={onChange}
      context={{ document: { methods: value }, options: null, setDocument: vi.fn() }} />)
    fireEvent.change(screen.getByRole('textbox', { name: /^backend/i }), { target: { value: 'TF_2' } })
    await waitFor(() => expect(onChange).toHaveBeenCalled())
    expect(onChange.mock.lastCall?.[0]).toMatchObject({ backend: 'TF_2', interval_coverage: [0.8, 0.9] })
  })

  it('leaves a case that chose to have no prediction intervals with none', async () => {
    const onChange = vi.fn()
    const value = { backend: 'TORCH', interval_coverage: [] }
    render(<SectionForm fullSchema={productionSchema} section="methods" value={value} onChange={onChange}
      context={{ document: { methods: value }, options: null, setDocument: vi.fn() }} />)
    fireEvent.change(screen.getByRole('textbox', { name: /^backend/i }), { target: { value: 'TF_2' } })
    await waitFor(() => expect(onChange).toHaveBeenCalled())
    expect(onChange.mock.lastCall?.[0]).toMatchObject({ backend: 'TF_2', interval_coverage: [] })
  })

  it('keeps the model access policy hidden and preserves stored values when another job setting changes', async () => {
    const onChange = vi.fn()
    // `legacy_zero` stands for whatever a stored case holds that the schema does not name.
    const value = { n_cpus: 4, n_jobs: 2, n_gpus: 1, cpu_workers: 2, cpu_study_device: 'cpu', legacy_zero: 0,
      model_cache_mode: 'prewarmed' }
    render(<SectionForm fullSchema={productionSchema} section="jobs" value={value} onChange={onChange}
      context={{ document: { jobs: value }, options: null, setDocument: vi.fn() }} />)
    // The actual exported schema must remain intact while the generated editor hides the policy.
    const properties = productionSchema.$defs.JobsConfig.properties
    expect(screen.queryByText(properties.model_cache_mode.title)).not.toBeInTheDocument()
    expect(screen.queryByText(/legacy[ _]zero/i)).not.toBeInTheDocument()
    // The study settings a case names are ordinary fields of the section.
    for (const key of ['cpu_workers', 'cpu_study_threads', 'cpu_study_device'] as const) {
      expect(screen.getAllByText(properties[key].title).length).toBeGreaterThan(0)
    }
    fireEvent.change(await screen.findByRole('spinbutton', { name: /n cpus/i }), { target: { value: '6' } })
    await waitFor(() => expect(onChange).toHaveBeenCalled())
    expect(onChange.mock.lastCall?.[0]).toMatchObject({ ...value, n_cpus: 6 })
  })
})
