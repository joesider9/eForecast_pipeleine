import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import type { FieldProps } from '@rjsf/utils'
import L from 'leaflet'
import MapField from './MapField'

// rjsf passes a field far more than it reads; these are the four this one reads.
function openMap(formData: unknown, name = 'coordinates') {
  const onChange = vi.fn()
  render(<MapField {...({ formData, onChange, name, schema: { title: 'Coordinates' } } as unknown as FieldProps)} />)
  fireEvent.click(screen.getByRole('button', { name: /map/i }))
  return onChange
}

const type = (label: string, text: string) =>
  fireEvent.change(screen.getByRole('spinbutton', { name: label }), { target: { value: text } })
const press = (label: string | RegExp) => fireEvent.click(screen.getByRole('button', { name: label }))
const use = () => press(/use/i)

describe('typing the corners of an area', () => {
  it('draws a half-typed latitude instead of taking the page down with it', () => {
    const onChange = openMap([38.1, 23.7])        // the case held a single point
    press('area')
    type('lat min', '36.4')
    type('lon min', '20.3')
    type('lat max', '4')                          // the first keystroke of 41.5
    type('lat max', '41.5')
    type('lon max', '26.4')
    use()
    expect(onChange).toHaveBeenCalledWith([36.4, 20.3, 41.5, 26.4])
  })

  it('saves the corners minimum first however they were typed', () => {
    const onChange = openMap(undefined)
    press('area')
    type('lat min', '41.5'); type('lon min', '26.4')
    type('lat max', '36.4'); type('lon max', '20.3')
    use()
    expect(onChange).toHaveBeenCalledWith([36.4, 20.3, 41.5, 26.4])
  })

  it('offers the middle of the area when the shape goes back to a point', () => {
    const onChange = openMap([36.4, 20.3, 41.5, 26.4])
    press('point')
    expect(screen.getByRole('spinbutton', { name: 'lat' })).toHaveValue(38.95)
    use()
    expect(onChange).toHaveBeenCalledWith([38.95, 23.35])
  })

  it('turns every place into an area, because the schema asks them to agree', () => {
    const onChange = openMap({ athens: [38.0, 23.7], patras: [38.2, 21.7] })
    press('area')
    type('lat max', '39')
    use()
    expect(onChange).toHaveBeenCalledWith({
      athens: [38.0, 23.7, 39, 23.7], patras: [38.2, 21.7, 38.2, 21.7],
    })
  })

  it('writes the group as two corners and keeps the resolution warning', () => {
    const onChange = openMap([[36.4, 20.3], [41.5, 26.4]], 'area_group')
    expect(screen.getByText(/multiples of the NWP resolution/)).toBeVisible()
    type('lat max', '4')
    type('lat max', '41.4')
    use()
    expect(onChange).toHaveBeenCalledWith([[36.4, 20.3], [41.4, 26.4]])
  })

  it('ignores a stored coordinate that is not a number', () => {
    const onChange = openMap({ athens: [38.0, 23.7], nowhere: ['36.4', null] })
    use()
    expect(onChange).toHaveBeenCalledWith({ athens: [38.0, 23.7] })
  })
})

describe('the map while the corners are typed', () => {
  const container = () => document.querySelector('.leaflet-container')

  it('stays mounted, instead of reloading every tile per keystroke', () => {
    openMap([38.0, 23.7])
    const mounted = container()
    const setView = vi.spyOn(L.Map.prototype, 'setView')
    type('lat', '38.5')
    type('lon', '23.9')
    expect(container()).toBe(mounted)
    expect(setView).not.toHaveBeenCalled()
  })

  it('follows the place being edited when that is what changed', () => {
    openMap({ athens: [38.0, 23.7], patras: [38.2, 21.7] })
    const mounted = container()
    const setView = vi.spyOn(L.Map.prototype, 'setView')
    press('patras')
    expect(setView).toHaveBeenCalledWith([38.2, 21.7], 8)
    expect(container()).toBe(mounted)
  })

  it('pulls back to the wider view when a point becomes an area', () => {
    openMap([38.0, 23.7])
    const setView = vi.spyOn(L.Map.prototype, 'setView')
    press('area')
    expect(setView).toHaveBeenCalledWith([38.0, 23.7], 6)
  })
})
