import { createContext } from 'react'

export type LogoutGuard = () => Promise<boolean>
export type RegisterLogoutGuard = (guard: LogoutGuard) => () => void

// The mounted editor owns the decision and its saved/draft state. The app must
// obtain that decision before sending logout or replacing the private boundary.
export const LogoutGuardContext = createContext<RegisterLogoutGuard | null>(null)
