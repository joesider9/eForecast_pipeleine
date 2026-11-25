import { dump, load } from 'js-yaml'

export const toYaml = (value: unknown): string =>
  value === undefined || value === null ? '' : dump(value, { noRefs: true, lineWidth: 100, sortKeys: false })

export const fromYaml = (text: string): unknown => {
  const trimmed = text.trim()
  return trimmed ? load(trimmed) : null
}
