import React from 'react'
import ReactDOM from 'react-dom/client'
import { createBrowserRouter, RouterProvider } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { CssBaseline, ThemeProvider, createTheme } from '@mui/material'
import App from './App'
import './index.css'

// One QueryClient for the whole app: it caches API responses and drives loading / error states.
const queryClient = new QueryClient({ defaultOptions: { queries: { retry: 1, refetchOnWindowFocus: false } } })
const router = createBrowserRouter([{ path: '*', element: <App /> }])

const theme = createTheme({
  palette: { primary: { main: '#1f5fbf' }, secondary: { main: '#0b7a75' }, background: { default: '#f6f7f9' } },
  typography: { fontFamily: 'Inter, Roboto, system-ui, sans-serif', fontSize: 13.5 },
  components: { MuiTextField: { defaultProps: { size: 'small' } }, MuiButton: { defaultProps: { disableElevation: true } } },
})

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <RouterProvider router={router} />
      </ThemeProvider>
    </QueryClientProvider>
  </React.StrictMode>,
)
