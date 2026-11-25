// The wording the health chip and its dialog share. Separate from the component so that file
// exports components only (fast refresh), and so this one rule can be tested on its own.

/** How old a reading is, in the words somebody would use out loud. */
export function ageOf(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return 'never'
  if (seconds < 90) return 'just now'
  if (seconds < 5400) return `${Math.round(seconds / 60)} min ago`
  if (seconds < 172800) return `${Math.round(seconds / 3600)} h ago`
  return `${Math.round(seconds / 86400)} days ago`
}
