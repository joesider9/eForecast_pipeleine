// Where a provider's files come from is chosen per row: the local folder, or the shared Google Drive
// folder its archive is kept in. The choice has to end up in the case file as one optional key, and
// leave no trace when it is undone.
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import type { FieldProps } from '@rjsf/utils'
import NwpModelsField from './NwpModelsField'

const LAB = 'https://drive.google.com/drive/folders/17cTFdd4sN-yNZUpynt3Z-tumo8j90WmD'
const options = { nwp_models: ['ecmwf', 'skiron', 'gfs'], nwp_gdrive: { ecmwf: LAB } }
const local = { model: 'ecmwf', resolution: 0.1, levels: 8, path_nwp_source: '/media/lab/ECMWF' }

// rjsf passes a field far more than it reads; these are the four this one reads.
function mount(formData: unknown, served: unknown = options) {
  const onChange = vi.fn()
  render(<NwpModelsField {...({ formData, onChange, schema: { title: 'NWP models' }, formContext: { options: served } } as unknown as FieldProps)} />)
  return onChange
}

const choose = async (source: string) => {
  await userEvent.click(screen.getByRole('combobox', { name: 'ecmwf source' }))
  await userEvent.click(screen.getByRole('option', { name: source }))
}

describe('the source of an NWP provider', () => {
  it('reads from the local folder until Google Drive is chosen, and then names the lab folder', async () => {
    const onChange = mount([local])
    expect(screen.getByRole('combobox', { name: 'ecmwf source' })).toHaveTextContent('Local folder')
    expect(screen.queryByLabelText('Google Drive folder (link)')).toBeNull()
    await choose('Google Drive')
    expect(onChange).toHaveBeenCalledWith([{ ...local, path_nwp_gdrive: LAB }])
  })

  it('shows the link and says what becomes of the downloaded files', () => {
    mount([{ ...local, path_nwp_gdrive: LAB }])
    expect(screen.getByRole('combobox', { name: 'ecmwf source' })).toHaveTextContent('Google Drive')
    expect(screen.getByLabelText('Google Drive folder (link)')).toHaveValue(LAB)
    expect(screen.getByText(/downloaded from here when extract runs, converted to its daily pickle, and the downloaded files are removed/)).toBeInTheDocument()
    // the folder is still asked for: it is where a day is held between its download and its conversion
    expect(screen.getByRole('textbox', { name: 'ecmwf folder' })).toHaveValue('/media/lab/ECMWF')
  })

  it('going back to the local folder takes the key out of the row instead of leaving a null in the case file', async () => {
    const onChange = mount([{ ...local, path_nwp_gdrive: LAB }])
    await choose('Local folder')
    const [row] = onChange.mock.calls.at(-1)![0]
    expect(row).toEqual(local)
    expect('path_nwp_gdrive' in row).toBe(false)
  })

  it('keeps the link field in place while another folder is typed into it', async () => {
    const onChange = mount([{ ...local, path_nwp_gdrive: LAB }])
    await userEvent.clear(screen.getByLabelText('Google Drive folder (link)'))
    // emptied, not removed: the row still reads from Google Drive, and the save says a link is missing
    expect(onChange.mock.calls.at(-1)![0]).toEqual([{ ...local, path_nwp_gdrive: '' }])
  })

  it('still shows an emptied link as a Google Drive source', () => {
    mount([{ ...local, path_nwp_gdrive: '' }])
    expect(screen.getByRole('combobox', { name: 'ecmwf source' })).toHaveTextContent('Google Drive')
    expect(screen.getByLabelText('Google Drive folder (link)')).toHaveValue('')
  })

  it('does not offer the choice for a provider nothing is downloaded for', () => {
    mount([{ model: 'gfs', path_nwp_source: '/media/lab/GFS' }])
    expect(screen.queryByRole('combobox', { name: 'gfs source' })).toBeNull()
    expect(screen.getByRole('textbox', { name: 'gfs folder' })).toHaveValue('/media/lab/GFS')
  })

  it('a row moved to such a provider does not keep a link nobody can see', async () => {
    const onChange = mount([{ ...local, path_nwp_gdrive: LAB }])
    await userEvent.click(screen.getAllByRole('combobox')[0])
    await userEvent.click(screen.getByRole('option', { name: 'gfs' }))
    expect(onChange.mock.calls.at(-1)![0]).toEqual([{ ...local, model: 'gfs' }])
  })

  it('shows a link that a case file holds for a provider this server offers none for', () => {
    mount([{ ...local, path_nwp_gdrive: LAB }], { nwp_models: ['ecmwf'] })
    expect(screen.getByRole('combobox', { name: 'ecmwf source' })).toHaveTextContent('Google Drive')
    expect(screen.getByLabelText('Google Drive folder (link)')).toHaveValue(LAB)
  })
})
