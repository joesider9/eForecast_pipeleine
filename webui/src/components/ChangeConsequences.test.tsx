// What a change costs has to be said as it is, and accepted by an act nobody performs in passing.
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import type { Consequences } from '../api/client'
import ChangeConsequences, { RunChanges, TypedConfirmation } from './ChangeConsequences'
import ConsequencesDialog from './ConsequencesDialog'
import { brief, describeChange } from './settingChanges'
import { consequencesFixture } from '../test/fixtures'

describe('a changed setting, in words', () => {
  it('says what a value was and what it is', () => {
    expect(describeChange({ path: 'data.variables[Cloud].lags', change: 'changed', before: [0, 1], after: [0, 1, 2] }))
      .toBe('[0,1] → [0,1,2]')
    expect(describeChange({ path: 'data.variables[Flux]', change: 'added', after: { name: 'Flux' } }))
      .toBe('added ({"name":"Flux"})')
    expect(describeChange({ path: 'project.country', change: 'removed', before: 'Greece' })).toBe('removed (was "Greece")')
  })
  it('cuts a value where it stops being readable', () => {
    expect(brief('x'.repeat(500))).toHaveLength(90)
    expect(brief('x'.repeat(500)).endsWith('…')).toBe(true)
  })
})

describe('what saving a draft would cost', () => {
  it('names the settings, the datasets built again and the families trained from the beginning', () => {
    render(<ChangeConsequences report={consequencesFixture()} />)
    expect(screen.getByTestId('change-consequences')).toHaveAttribute('data-consequential', 'true')
    expect(screen.getByTestId('setting-changes')).toHaveTextContent('data.variables[Cloud].lags')
    expect(screen.getByTestId('setting-changes')).toHaveTextContent('[0,1,2] → [0,1,2,3]')
    const datasets = screen.getByTestId('consequences-datasets')
    expect(datasets).toHaveTextContent('Datasets are built again')
    expect(datasets).toHaveTextContent('nwp: 1 file(s) — because of variables')
    expect(datasets).toHaveTextContent('The 911 extracted daily weather file(s) stay as they are.')
    expect(datasets).toHaveTextContent('nothing can be trained or combined')
    const families = screen.getByTestId('consequences-families')
    expect(families).toHaveTextContent('TRANSFORMER: 8 studies, 101 trial folder(s) — because of variables')
    expect(families).toHaveTextContent('the combination step does not run')
    expect(screen.getByTestId('consequences-results')).toHaveTextContent('predict, select, combine, evaluate')
    expect(screen.queryByTestId('consequences-nothing')).not.toBeInTheDocument()
  })

  it('says so when nothing that is built has to be done again', () => {
    render(<ChangeConsequences report={consequencesFixture({ datasets: [], families: [], results: [], extract: null,
      consequential: false, changes: [{ path: 'jobs.n_jobs', change: 'changed', before: 20, after: 10 }] })} />)
    expect(screen.getByTestId('consequences-nothing')).toHaveTextContent('Nothing that is built has to be built or trained again')
    expect(screen.queryByTestId('consequences-datasets')).not.toBeInTheDocument()
  })

  it('tells a case that would build into another model folder what it leaves behind', () => {
    render(<ChangeConsequences report={consequencesFixture({ datasets: [], families: [], results: [], extract: null,
      relocated: { from: '/models/pv/park/model_ver0', to: '/models/pv/other/model_ver0', holds_work: false } })} />)
    const relocated = screen.getByTestId('consequences-relocated')
    expect(relocated).toHaveTextContent('/models/pv/other/model_ver0')
    expect(relocated).toHaveTextContent('every step starts from nothing')
    expect(relocated).toHaveTextContent('stays in /models/pv/park/model_ver0')
  })

  it('says when the weather itself has to be extracted again', () => {
    render(<ChangeConsequences report={consequencesFixture({
      weather: { from: '/models/nwp/Cloud+Flux', to: '/models/nwp/Flux+Temperature', extracted: false } })} />)
    expect(screen.getByTestId('consequences-datasets')).toHaveTextContent('/models/nwp/Flux+Temperature, which holds none of it yet')
    expect(screen.getByTestId('consequences-datasets')).toHaveTextContent('that is the download again')
    expect(screen.queryByText(/stay as they are/)).not.toBeInTheDocument()
  })

  it('keeps what was already waiting apart from what this change adds, and names what a change puts back', () => {
    render(<ChangeConsequences report={consequencesFixture({ datasets: [], families: [], results: [], extract: null,
      consequential: false,
      pending: { datasets: [{ stage: 'row', files: 1, because: [] }], families: [{ method: 'MLP', studies: 8, trials: 32, because: [] }] },
      restored: { datasets: ['nwp'], families: [] } })} />)
    expect(screen.getByTestId('consequences-pending')).toHaveTextContent('Already waiting before this change')
    expect(screen.getByTestId('consequences-pending')).toHaveTextContent('row')
    expect(screen.getByTestId('consequences-pending')).toHaveTextContent('MLP')
    expect(screen.getByTestId('consequences-restored')).toHaveTextContent('nwp no longer has to be done again')
    expect(screen.queryByTestId('consequences-nothing')).not.toBeInTheDocument()
  })

  it('does not present a draft it could not judge as a draft that costs nothing', () => {
    const unresolved: Consequences = { ...consequencesFixture(), status: 'unresolved',
      reason: 'the draft does not resolve: ValidationError', consequential: false }
    render(<ChangeConsequences report={unresolved} />)
    expect(screen.getByTestId('consequences-unresolved')).toHaveTextContent('could not be worked out')
    expect(screen.getByTestId('consequences-unresolved')).toHaveTextContent('the draft does not resolve')
    expect(screen.queryByTestId('consequences-nothing')).not.toBeInTheDocument()
  })

  it('says that a draft which only writes down defaults changes no setting', () => {
    // The editor fills in the defaults of the sections it showed: the files differ, the case does not.
    const same = consequencesFixture({ status: 'unchanged', changes: [], as_written: 21, datasets: [], families: [],
      results: [], consequential: false, digest: null })
    const { rerender } = render(<ChangeConsequences report={same} />)
    expect(screen.getByTestId('consequences-written-down')).toHaveTextContent('No setting changes')
    expect(screen.getByTestId('consequences-written-down')).toHaveTextContent('21 value(s)')
    expect(screen.getByTestId('consequences-whole')).toBeInTheDocument()
    // the saved case itself: nothing is written down, so nothing is said about it
    rerender(<ChangeConsequences report={{ ...same, as_written: 0 }} />)
    expect(screen.queryByTestId('consequences-written-down')).not.toBeInTheDocument()
    expect(screen.getByTestId('consequences-whole')).toBeInTheDocument()
  })
})

describe('the name of the case, typed', () => {
  it('accepts nothing but the name itself', async () => {
    const onAccept = vi.fn()
    render(<ConsequencesDialog name="AG_PARK" report={consequencesFixture()} saving={false} onCancel={() => {}} onAccept={onAccept} />)
    const user = userEvent.setup()
    const accept = screen.getByTestId('accept-and-save')
    expect(accept).toBeDisabled()
    const field = screen.getByTestId('typed-confirmation')
    for (const wrong of ['yes', 'ag_park', 'AG_PARK ']) {
      await user.clear(field); await user.type(field, wrong)
      expect(accept).toBeDisabled()
    }
    expect(screen.getByText('That is not the name of this case (AG_PARK).')).toBeInTheDocument()
    await user.clear(field); await user.type(field, 'AG_PARK')
    expect(accept).toBeEnabled()
    await user.click(accept)
    expect(onAccept).toHaveBeenCalledWith('digest-1')
  })

  it('asks again when what it shows is no longer what was accepted', async () => {
    const { rerender } = render(<ConsequencesDialog name="park" report={consequencesFixture()} saving={false}
      onCancel={() => {}} onAccept={() => {}} />)
    const user = userEvent.setup()
    await user.type(screen.getByTestId('typed-confirmation'), 'park')
    expect(screen.getByTestId('accept-and-save')).toBeEnabled()
    rerender(<ConsequencesDialog name="park" report={consequencesFixture({ digest: 'digest-2' })} saving={false}
      onCancel={() => {}} onAccept={() => {}} />)
    expect(screen.getByTestId('typed-confirmation')).toHaveValue('')
    expect(screen.getByTestId('accept-and-save')).toBeDisabled()
  })

  it('cannot be answered or cancelled while the save is on its way', async () => {
    render(<ConsequencesDialog name="park" report={consequencesFixture()} saving onCancel={() => {}} onAccept={() => {}} />)
    expect(screen.getByTestId('typed-confirmation')).toBeDisabled()
    expect(screen.getByTestId('accept-and-save')).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Cancel, nothing is saved' })).toBeDisabled()
  })

  it('renders nothing when there is nothing to ask', () => {
    const { container } = render(<ConsequencesDialog name="park" report={null} saving={false} onCancel={() => {}} onAccept={() => {}} />)
    expect(container).toBeEmptyDOMElement()
  })

  it('says whose name it wants', () => {
    render(<TypedConfirmation name="park" value="" onChange={() => {}} what="accept this and start" />)
    expect(screen.getByTestId('typed-confirmation')).toHaveAccessibleName('Type the name of the case to accept this and start')
    expect(screen.getByText('The name is park.')).toBeInTheDocument()
  })
})

describe('what a run replaces for a changed configuration', () => {
  it('names the settings that changed since the model was built, the datasets and the families', () => {
    render(<RunChanges since={[{ path: 'methods.method_params.TRANSFORMER.model_name.choices', change: 'changed',
        before: ['iTransformer', 'PatchTST'], after: ['iTransformer'] }]}
      changes={{ stages: [{ key: 'row', reason: 'row configuration changed', files: 1 }],
        families: { TRANSFORMER: { units: 8, trials: 101, clusters: ['global'] }, MLP: { units: 1, trials: 4, clusters: ['global'] } } }} />)
    const panel = screen.getByTestId('configuration-changes')
    expect(panel).toHaveTextContent('The configuration is not the one this was built from')
    expect(panel).toHaveTextContent('methods.method_params.TRANSFORMER.model_name.choices')
    expect(screen.getByTestId('changes-datasets')).toHaveTextContent('row: 1 file(s)')
    expect(screen.getByTestId('changes-families')).toHaveTextContent('TRANSFORMER: 8 studies, 101 trial folder(s)')
    expect(screen.getByTestId('changes-families')).toHaveTextContent('MLP: 1 study, 4 trial folder(s)')
    expect(panel).toHaveTextContent('Nothing of this can be had back')
  })

  it('shows only what there is', () => {
    render(<RunChanges changes={{ stages: [], families: { MLP: { units: 2, trials: 6, clusters: ['a', 'b'] } } }} />)
    expect(screen.queryByTestId('changes-datasets')).not.toBeInTheDocument()
    expect(screen.queryByTestId('setting-changes')).not.toBeInTheDocument()
    expect(screen.getByTestId('changes-families')).toHaveTextContent('MLP: 2 studies, 6 trial folder(s)')
  })
})
