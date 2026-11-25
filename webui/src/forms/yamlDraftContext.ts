import { createContext } from 'react'

export interface YamlBuffer {
  text: string
  error: string | null
  applied: string
  label: string
  section: string
}

export interface YamlDraftStore {
  buffers: Record<string, YamlBuffer>
  setBuffer: (key: string, buffer: YamlBuffer) => void
  discardBuffer: (key: string) => void
}

export const YamlDraftContext = createContext<YamlDraftStore | null>(null)
export const unfinishedYaml = (buffers: Record<string, YamlBuffer>) =>
  Object.entries(buffers).filter(([, buffer]) => buffer.error !== null)
