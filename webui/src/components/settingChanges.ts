// A changed setting, in words. Apart from the components that show it so that file exports
// components only (fast refresh), and so the wording can be tested without a DOM.
import type { SettingChange } from '../api/client'

export const brief = (value: unknown, limit = 90) => {
  const text = value === undefined ? '' : JSON.stringify(value)
  return text.length <= limit ? text : `${text.slice(0, limit - 1)}…`
}

/** One changed setting as the words after its path: what it was and what it is. */
export function describeChange(change: SettingChange) {
  if (change.change === 'added') return `added (${brief(change.after)})`
  if (change.change === 'removed') return `removed (was ${brief(change.before)})`
  if (change.change === 'reordered') return `reordered: ${brief(change.before)} → ${brief(change.after)}`
  return `${brief(change.before)} → ${brief(change.after)}`
}
