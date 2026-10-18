import React, { Suspense, lazy } from 'react'
import { BrowserRouter, Routes, Route, Navigate, useLocation } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { AnimatePresence } from 'framer-motion'
import { Layout } from '@/components/layout/Layout'
import { PageSkeleton } from '@/components/common/PageSkeleton'
import { PageTransition } from '@/components/common/PageTransition'
import { ErrorBoundary } from '@/components/common/ErrorBoundary'

// Route-level code splitting with React.lazy for optimal chunking and Lighthouse score
const LivePage = lazy(() => import('@/pages/LivePage').then((m) => ({ default: m.LivePage })))
const MapPage = lazy(() => import('@/pages/MapPage').then((m) => ({ default: m.MapPage })))
const ReviewQueuePage = lazy(() =>
  import('@/pages/ReviewQueuePage').then((m) => ({ default: m.ReviewQueuePage }))
)
const IncidentsPage = lazy(() =>
  import('@/pages/IncidentsPage').then((m) => ({ default: m.IncidentsPage }))
)
const IncidentDetailPage = lazy(() =>
  import('@/pages/IncidentDetailPage').then((m) => ({ default: m.IncidentDetailPage }))
)
const AnalyticsPage = lazy(() =>
  import('@/pages/AnalyticsPage').then((m) => ({ default: m.AnalyticsPage }))
)
const SettingsPage = lazy(() =>
  import('@/pages/SettingsPage').then((m) => ({ default: m.SettingsPage }))
)
const AboutPage = lazy(() =>
  import('@/pages/AboutPage').then((m) => ({ default: m.AboutPage }))
)

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      staleTime: 5000,
    },
  },
})

// Animated routes container for smooth page transitions
const AnimatedRoutes: React.FC = () => {
  const location = useLocation()

  return (
    <AnimatePresence mode="wait">
      <Routes location={location} key={location.pathname}>
        <Route path="/" element={<Layout />}>
          <Route index element={<Navigate to="/live" replace />} />
          <Route
            path="live"
            element={
              <PageTransition>
                <LivePage />
              </PageTransition>
            }
          />
          <Route
            path="map"
            element={
              <PageTransition>
                <MapPage />
              </PageTransition>
            }
          />
          <Route
            path="review"
            element={
              <PageTransition>
                <ReviewQueuePage />
              </PageTransition>
            }
          />
          <Route
            path="incidents"
            element={
              <PageTransition>
                <IncidentsPage />
              </PageTransition>
            }
          />
          <Route
            path="incidents/:id"
            element={
              <PageTransition>
                <IncidentDetailPage />
              </PageTransition>
            }
          />
          <Route
            path="analytics"
            element={
              <PageTransition>
                <AnalyticsPage />
              </PageTransition>
            }
          />
          <Route
            path="settings"
            element={
              <PageTransition>
                <SettingsPage />
              </PageTransition>
            }
          />
          <Route
            path="about"
            element={
              <PageTransition>
                <AboutPage />
              </PageTransition>
            }
          />
          <Route path="*" element={<Navigate to="/live" replace />} />
        </Route>
      </Routes>
    </AnimatePresence>
  )
}

export const App: React.FC = () => {
  return (
    <ErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <BrowserRouter>
          <Suspense fallback={<PageSkeleton />}>
            <AnimatedRoutes />
          </Suspense>
        </BrowserRouter>
      </QueryClientProvider>
    </ErrorBoundary>
  )
}

export default App
