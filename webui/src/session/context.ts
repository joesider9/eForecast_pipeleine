import { createContext, useContext } from 'react'

export const IdentityContext = createContext('local')
export const useIdentity = () => useContext(IdentityContext)

// A failed refresh retains the last identity's editor/cache, but server changes
// wait for a successful session check. Local editing and reads remain available.
export const SessionConfirmedContext = createContext(true)
export const useSessionConfirmed = () => useContext(SessionConfirmedContext)
