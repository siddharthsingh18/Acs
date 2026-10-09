import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { AnalyticsPage } from './AnalyticsPage'
import { api } from '@/api/client'

vi.mock('@/api/client', () => ({
  api: {
    getStats: vi.fn(),
    getIncidents: vi.fn(),
  },
}))

describe('AnalyticsPage component', () => {
  let queryClient: QueryClient

  const mockStats = {
    total_incidents: 42,
    incidents_today: 7,
    by_class: {
      car_car_accident: 20,
      car_bike_accident: 12,
      bike_bike_accident: 6,
      car_person_accident: 4,
    },
    by_status: {
      pending: 10,
      approved: 22,
      dispatched: 8,
      rejected: 2,
    },
    average_confidence: 0.874,
    average_review_time_seconds: 34.5,
  }

  const mockIncidents = {
    items: [
      {
        incident_id: 'INC-2026-001',
        start_timestamp: '2026-10-10T08:30:00Z',
        max_confidence: 0.91,
        status: 'approved',
        class_name: 'car_car_accident',
      },
      {
        incident_id: 'INC-2026-002',
        start_timestamp: '2026-10-10T17:45:00Z',
        max_confidence: 0.85,
        status: 'dispatched',
        class_name: 'car_bike_accident',
      },
    ],
    total: 2,
  }

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    })
    vi.clearAllMocks()
    vi.mocked(api.getStats).mockResolvedValue(mockStats as any)
    vi.mocked(api.getIncidents).mockResolvedValue(mockIncidents as any)
  })

  const renderPage = () => {
    return render(
      <QueryClientProvider client={queryClient}>
        <AnalyticsPage />
      </QueryClientProvider>
    )
  }

  it('renders all stat cards and animated numbers', async () => {
    renderPage()

    await waitFor(() => {
      expect(screen.getByText(/Incidents Today/i)).toBeInTheDocument()
      expect(screen.getByText(/Total Recorded/i)).toBeInTheDocument()
      expect(screen.getByText(/Avg Collision Confidence/i)).toBeInTheDocument()
      expect(screen.getByText(/Avg Review Time/i)).toBeInTheDocument()
    })
  })

  it('toggles date range presets', async () => {
    renderPage()

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /7 days/i })).toBeInTheDocument()
    })

    const sevenDaysBtn = screen.getByRole('button', { name: /7 days/i })
    fireEvent.click(sevenDaysBtn)

    await waitFor(() => {
      expect(api.getIncidents).toHaveBeenCalledWith(
        expect.objectContaining({
          start_date: expect.any(String),
          end_date: expect.any(String),
        })
      )
    })
  })

  it('renders Recharts sections and Review Status Funnel', async () => {
    renderPage()

    await waitFor(() => {
      expect(screen.getByText(/Incidents Over Time/i)).toBeInTheDocument()
      expect(screen.getByText(/Hour-of-Day Collision Heatmap/i)).toBeInTheDocument()
      expect(screen.getByText(/Average Confidence Trend/i)).toBeInTheDocument()
      expect(screen.getByText(/Human-in-the-Loop Review Funnel/i)).toBeInTheDocument()
    })

    // Funnel stages
    expect(screen.getByText(/Stage 1: Detected/i)).toBeInTheDocument()
    expect(screen.getByText(/Stage 2: Reviewed/i)).toBeInTheDocument()
    expect(screen.getByText(/Stage 3: Dispatched/i)).toBeInTheDocument()
  })
})
