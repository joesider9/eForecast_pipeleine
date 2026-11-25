import { describe, expect, it } from 'vitest'
import { fromYaml, toYaml } from './yaml'

describe('case YAML round trips', () => {
  it('preserves legacy values, empty intervals, search spaces and unknown extensions', () => {
    const document = {
      project: { name: 'pv park', ts_resolution: 'H', horizon_type: 'short_term' },
      methods: { interval_coverage: [], clustering: { thres_act: 0 },
        method_params: { RF: { max_depth: { choices: [null, 4, 8] } } } },
      target_variable: { lags: 0 }, custom: { enabled: false, note: 'a: b', value: 0 },
    }
    expect(fromYaml(toYaml(document))).toEqual(document)
  })

  it('rejects malformed YAML rather than emitting a replacement document', () => {
    expect(() => fromYaml('methods: [')).toThrow()
  })
})
