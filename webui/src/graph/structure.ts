import type { Experiment, Experiments } from './types'

function mapping(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value) &&
    Object.getPrototypeOf(value) === Object.prototype
}

// Guard the shape the canvas/inspector consumes, not catalogue membership or
// model feasibility. The backend remains the authority for legacy normalization
// and configuration/domain validation, including custom layer/parameter names.
export function assertExperiment(value: unknown, prefix = ''): asserts value is Experiment {
  const path = (key: string) => prefix ? `${prefix}.${key}` : key
  if (!mapping(value)) throw new Error(`${prefix || 'experiment'}: expected a mapping of branches.`)
  if (!Object.prototype.hasOwnProperty.call(value, 'output')) throw new Error(`${path('output')}: expected a list of merge layers (use [] for none).`)
  for (const [branch, layers] of Object.entries(value)) {
    const branchPath = path(branch)
    if (!Array.isArray(layers)) throw new Error(`${branchPath}: expected a list of layers.`)
    layers.forEach((layer: unknown, index: number) => {
      const layerPath = `${branchPath}[${index}]`
      if (!mapping(layer)) throw new Error(`${layerPath}: expected a layer mapping. Use Normalize YAML for legacy [type, size] layers.`)
      if (typeof layer.type !== 'string') throw new Error(`${layerPath}.type: expected a string. Use Normalize YAML for legacy layer aliases.`)
      const extra = Object.keys(layer).find(key => !['type', 'params', 'name', 'enabled'].includes(key))
      if (extra) throw new Error(`${layerPath}.${extra}: use Normalize YAML to move legacy parameters into params.`)
      if (layer.name !== undefined && layer.name !== null && typeof layer.name !== 'string') throw new Error(`${layerPath}.name: expected a string or null.`)
      if (layer.enabled !== undefined && typeof layer.enabled !== 'boolean') throw new Error(`${layerPath}.enabled: expected a boolean. Normalize YAML accepts legacy representations.`)
      if (layer.params === undefined) return
      if (!mapping(layer.params)) throw new Error(`${layerPath}.params: expected a parameter mapping. Use Normalize YAML for legacy representations.`)
      for (const [name, param] of Object.entries(layer.params)) {
        const paramPath = `${layerPath}.params.${name}`
        if (!mapping(param)) throw new Error(`${paramPath}: expected a tagged parameter. Use Normalize YAML for shorthand values.`)
        if (param.tag !== 'fixed' && param.tag !== 'choice' && param.tag !== 'range') throw new Error(`${paramPath}.tag: expected fixed, choice or range. Normalize YAML accepts parameter shorthands.`)
        if (param.tag !== 'fixed' && !Array.isArray(param.value)) throw new Error(`${paramPath}.value: expected a list for ${param.tag}.`)
        if (param.tag === 'range' && (param.value as unknown[]).length !== 2) throw new Error(`${paramPath}.value: expected [low, high].`)
        if (param.tag === 'range' && (param.value as unknown[]).some(bound => typeof bound !== 'number')) throw new Error(`${paramPath}.value: range bounds must be numbers.`)
      }
    })
  }
}

export function assertExperiments(value: unknown): asserts value is Experiments {
  if (!mapping(value)) throw new Error('experiments: expected a mapping of experiment tags.')
  for (const [tag, experiment] of Object.entries(value)) assertExperiment(experiment, `experiments.${tag}`)
}
