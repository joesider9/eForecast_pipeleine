// Lags are written compactly: "0..15" is a range, "-1,-2,-3" a list, "-24..-1" a negative range,
// "0..15/4" a range with a step (15-minute data), a single number is a single lag.
export function parseLags(text: string): number[] | number | null {
  const t = text.trim()
  if (!t) return null
  const parts = t.split(',').map(s => s.trim()).filter(Boolean)
  const out: number[] = []
  for (const part of parts) {
    const m = part.match(/^(-?\d+(?:\.\d+)?)\s*\.\.\s*(-?\d+(?:\.\d+)?)(?:\/(\d+(?:\.\d+)?))?$/)
    if (m) {
      const a = parseFloat(m[1]), b = parseFloat(m[2]), step = m[3] ? parseFloat(m[3]) : 1
      if (step <= 0) throw new Error('step must be positive')
      const n = Math.floor((b - a) / step + 1e-9)
      if (n < 0 || n > 10000) throw new Error(`bad range ${part}`)
      for (let i = 0; i <= n; i++) out.push(+(a + i * step).toFixed(6))
    } else {
      const v = parseFloat(part)
      if (Number.isNaN(v)) throw new Error(`not a number: ${part}`)
      out.push(v)
    }
  }
  return parts.length === 1 && !parts[0].includes('..') ? out[0] : out
}

export function lagsText(lags: unknown): string {
  if (lags === null || lags === undefined) return ''
  if (typeof lags === 'number') return String(lags)
  if (!Array.isArray(lags) || lags.length === 0) return ''
  const nums = lags.map(Number)
  if (nums.length > 2) {
    const step = nums[1] - nums[0]
    const regular = step !== 0 && nums.every((v, i) => Math.abs(v - (nums[0] + i * step)) < 1e-9)
    if (regular) return `${nums[0]}..${nums[nums.length - 1]}${step !== 1 ? `/${step}` : ''}`
  }
  return nums.join(', ')
}
