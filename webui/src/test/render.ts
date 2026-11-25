import { createElement, type ReactElement } from 'react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { createMemoryRouter, RouterProvider, type RouteObject } from 'react-router-dom'
import { render } from '@testing-library/react'

export function testClient() {
  return new QueryClient({ defaultOptions: {
    queries: { retry: false, gcTime: 0, refetchOnWindowFocus: false }, mutations: { retry: false },
  } })
}

export function renderPage(element: ReactElement, options: {
  path?: string; route?: string; client?: QueryClient; extraRoutes?: RouteObject[]
} = {}) {
  const client = options.client ?? testClient()
  const router = createMemoryRouter([
    { path: options.route ?? '/cases/:name', element }, ...(options.extraRoutes ?? []),
  ], { initialEntries: [options.path ?? '/cases/demo'] })
  return { ...render(createElement(QueryClientProvider, { client }, createElement(RouterProvider, { router }))), client, router }
}
