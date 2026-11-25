import { fireEvent, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useState } from 'react'
import { describe, expect, it } from 'vitest'
import LayerInspector from './LayerInspector'
import { mismatch, seed } from './paramValues'
import type { Catalogue, CatalogueParam, LayerSpec } from './types'

const param = (name: string, type: string, def: unknown = null, help = ''): CatalogueParam => ({ name, type, default: def, help })
const catalogue: Catalogue = {
  layers: [
    { name: 'cross_attention', label: 'Cross attention (merge)', kind: 'merge', inputs: ['branches'], output: 'flat', size: 'd_model', size_type: 'int',
      description: '', params: [param('size', 'int'), param('heads', 'int', 4), param('attention_dropout', 'float', 0.1),
        param('attention_query', 'list', null, 'branch names used as queries (default: the flat branches)'), param('d_model', 'int')] },
    { name: 'conv', label: 'Convolution', kind: 'conv', inputs: ['nwp'], output: 'nwp', size: 'kernel', size_type: 'int', description: '',
      params: [param('size', 'int'), param('filters', 'int'), param('pool', 'int'), param('residual', 'bool', true), param('act_func', 'str')] },
  ],
  branches: [], merges: ['cross_attention'], aliases: {},
}
const branches = ['nwp', 'row_calendar']

function Mounted({ start }: { start: LayerSpec }) {
  const [layer, setLayer] = useState<LayerSpec>(start)
  return <>
    <LayerInspector layer={layer} catalogue={catalogue} branch="output" branches={branches} onChange={setLayer} onRemove={() => undefined} />
    <output data-testid="layer">{JSON.stringify(layer.params ?? {})}</output>
  </>
}
const stored = () => JSON.parse(screen.getByTestId('layer').textContent ?? '{}')
const add = async (name: string) => {
  fireEvent.mouseDown(screen.getAllByRole('combobox').at(-1)!)
  await userEvent.click(await screen.findByRole('option', { name }))
}

describe('the value a parameter starts at', () => {
  it('makes nothing up for a parameter the catalogue gives no default', () => {
    const [size, heads, , query, dModel] = catalogue.layers[0].params
    expect(seed(query, 'fixed')).toEqual([])                      // it was 64, and 64 names no branch
    expect(seed(dModel, 'fixed')).toBeNull()
    expect(seed(catalogue.layers[1].params[4], 'fixed')).toBeNull()   // text: empty, the method's value stands
    expect(seed(undefined, 'fixed')).toBeNull()                   // a name the catalogue does not know
    expect(seed(heads, 'fixed')).toBe(4)
    expect(seed(size, 'fixed')).toBe(64)                          // a layer that takes a size needs one
    expect(seed(catalogue.layers[1].params[3], 'fixed')).toBe(true)
    expect(seed(dModel, 'choice')).toEqual([])
    expect(seed(heads, 'choice')).toEqual([4])
    expect(seed(dModel, 'range')).toEqual([1, 4])
    expect(seed(heads, 'range', 8)).toEqual([8, 32])
  })

  it('adds attention_query empty, as a list of the branches of the graph', async () => {
    render(<Mounted start={{ type: 'cross_attention', params: { size: { tag: 'fixed', value: 64 } }, enabled: true }} />)
    await add('attention_query')
    expect(stored().attention_query).toEqual({ tag: 'fixed', value: [] })
    expect(screen.queryByRole('alert')).toBeNull()
    // a list of names is the value itself and is never searched
    expect(screen.getAllByRole('button', { name: 'choice' }).at(-1)).toBeDisabled()
    expect(screen.getAllByRole('button', { name: 'range' }).at(-1)).toBeDisabled()
    const names = screen.getByLabelText('attention_query names')
    await userEvent.click(names)
    await userEvent.click(await screen.findByRole('option', { name: 'row_calendar' }))
    expect(stored().attention_query).toEqual({ tag: 'fixed', value: ['row_calendar'] })
    await userEvent.type(names, 'row_all{Enter}')
    expect(stored().attention_query.value).toEqual(['row_calendar', 'row_all'])
    expect(screen.getByRole('alert')).toHaveTextContent('row_all is not a branch of this graph')
  })

  it('adds a number without a default as an empty field', async () => {
    render(<Mounted start={{ type: 'cross_attention', params: {}, enabled: true }} />)
    await add('d_model')
    expect(stored().d_model).toEqual({ tag: 'fixed', value: null })
    expect(screen.getByLabelText('d_model value')).toHaveValue('')
    await add('heads')
    expect(stored().heads).toEqual({ tag: 'fixed', value: 4 })
    await add('size')
    expect(stored().size).toEqual({ tag: 'fixed', value: 64 })
  })
})

describe('a stored value of another kind than the layer takes', () => {
  it('shows the 64 an older editor wrote, says what it is read as, and lets it be removed', async () => {
    render(<Mounted start={{ type: 'cross_attention', enabled: true,
      params: { heads: { tag: 'fixed', value: 4 }, attention_query: { tag: 'fixed', value: 64 }, d_model: { tag: 'fixed', value: 64 } } }} />)
    const alert = screen.getByRole('alert')
    expect(alert).toHaveTextContent('64 names no branch; the layer reads this as not set')
    const row = alert.parentElement as HTMLElement
    expect(within(row).getByText('64')).toBeInTheDocument()                   // the stored value, as a chip
    await userEvent.click(within(row).getByLabelText('attention_query names'))
    await userEvent.keyboard('{Backspace}')
    expect(stored().attention_query).toEqual({ tag: 'fixed', value: [] })
    expect(screen.queryByRole('alert')).toBeNull()
    expect(stored().d_model).toEqual({ tag: 'fixed', value: 64 })             // a number where a number is taken
  })

  it('names text where a number is taken, and leaves size and empty values alone', () => {
    const [size, filters, , residual, act] = catalogue.layers[1].params
    expect(mismatch(filters, { tag: 'fixed', value: 'many' }, branches)).toBe('many is not a number, and the layer takes a number here')
    expect(mismatch(filters, { tag: 'choice', value: [16, 'x'] }, branches)).toBe('x is not a number, and the layer takes a number here')
    expect(mismatch(filters, { tag: 'choice', value: [] }, branches)).toBe('add at least one value, or make it fixed')
    expect(mismatch(act, { tag: 'fixed', value: 7 }, branches)).toBe('7 is not text, and the layer takes text here')
    expect(mismatch(residual, { tag: 'fixed', value: 'no' }, branches)).toBe('the layer takes true or false here')
    for (const value of [null, '', 16, 0]) expect(mismatch(filters, { tag: 'fixed', value }, branches)).toBeNull()
    expect(mismatch(size, { tag: 'fixed', value: 'linear' }, branches)).toBeNull()
    expect(mismatch(undefined, { tag: 'fixed', value: 'anything' }, branches)).toBeNull()
    const query = catalogue.layers[0].params[3]
    expect(mismatch(query, { tag: 'fixed', value: ['row_calendar'] }, branches)).toBeNull()
    expect(mismatch(query, { tag: 'fixed', value: 'nwp' }, branches)).toBeNull()
    expect(mismatch(query, { tag: 'fixed', value: ['stack'] }, ['x/stack'])).toBeNull()      // the last part of a nested name
    expect(mismatch(query, { tag: 'fixed', value: [] }, branches)).toBeNull()
  })
})
