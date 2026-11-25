import '@testing-library/jest-dom/vitest'
import { cleanup } from '@testing-library/react'
import { afterEach, vi } from 'vitest'

// Test the YAML parser/change contract without pretending jsdom lays out a code
// editor. Playwright exercises the real CodeMirror implementation separately.
vi.mock('@uiw/react-codemirror', async () => {
  const { createElement } = await import('react')
  return { default: function CodeEditor({ value, onChange }: { value: string; onChange: (value: string) => void }) {
    return createElement('textarea', { 'aria-label': 'YAML editor', value,
      onChange: (event: { target: { value: string } }) => onChange(event.target.value) })
  } }
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  localStorage.clear()
  sessionStorage.clear()
})

// Layout is browser-tested separately. Components still use real MUI and React Query.
Object.defineProperty(window, 'matchMedia', { writable: true, value: (query: string) => ({
  matches: false, media: query, onchange: null, addListener() {}, removeListener() {},
  addEventListener() {}, removeEventListener() {}, dispatchEvent: () => false,
}) })
