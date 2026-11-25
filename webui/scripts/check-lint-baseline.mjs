import { readFile } from 'node:fs/promises'
import path from 'node:path'
import { ESLint } from 'eslint'

const baseline = JSON.parse(await readFile(new URL('../eslint-baseline.json', import.meta.url), 'utf8'))
const eslint = new ESLint()
const results = await eslint.lintFiles(['.'])
let remaining = 0
const failures = []
for (const result of results) {
  const file = path.relative(process.cwd(), result.filePath).replaceAll(path.sep, '/')
  const known = result.messages.filter(message => message.ruleId === '@typescript-eslint/no-explicit-any')
  remaining += known.length
  const unexpected = result.messages.filter(message => message.ruleId !== '@typescript-eslint/no-explicit-any')
  if (known.length > (baseline[file] ?? 0)) failures.push(`${file}: ${known.length} explicit-any findings exceed baseline ${baseline[file] ?? 0}`)
  for (const message of unexpected) failures.push(`${file}:${message.line ?? 0}: ${message.message} (${message.ruleId ?? 'parse error'})`)
}
if (failures.length) {
  console.error(failures.join('\n'))
  process.exitCode = 1
} else console.log(`${remaining} existing explicit-any findings; no new lint findings. The full lint command still reports them.`)
