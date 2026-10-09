import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter } from 'react-router-dom'
import { IncidentsPage } from './IncidentsPage'
import { api } from '@/api/client'

vi.mock('@/api/client', () => ({
  api: {
    getIncidents: vi.fn(),
    getExportCsvUrl: vi.fn(() => '/api/incidents/export.csv'),
    getIncident: vi.fn(),
    getIncidentPdfUrl: vi.fn((id: string) => `/api/incidents/${id}/pdf`),
    getAlertSettings: vi.fn().mockResolvedValue({
      enabled: true,
      dry_run: true,
      channels: ['console'],
    }),
  },
}))

describe('IncidentsPage with TanStack Table', () => {
  let queryClient: QueryClient

  const mockIncidentsData = {
    items: [
      {
        incident_id: 'INC-2026-101',
        start_timestamp: '2026-10-10T02:15:00Z',
        end_timestamp: '2026-10-10T02:15:05Z',
        class_name: 'car_car_accident',
        max_confidence: 0.92,
        duration: 4.8,
        severity: 'high',
        status: 'approved',
        reviewer_name: 'Officer Davis',
        reviewed_at: '2026-10-10T02:20:00Z',
        image_path: 'report_images/inc101.jpg',
        keyframe_url: '/media/keyframes/inc101.jpg',
        created_at: 1770000000,
      },
      {
        incident_id: 'INC-2026-102',
        start_timestamp: '2026-10-10T03:30:00Z',
        end_timestamp: '2026-10-10T03:30:04Z',
        class_name: 'bike_bike_accident',
        max_confidence: 0.78,
        duration: 3.5,
        severity: 'medium',
        status: 'pending',
        reviewer_name: null,
        reviewed_at: null,
        image_path: 'report_images/inc102.jpg',
        keyframe_url: '/media/keyframes/inc102.jpg',
        created_at: 1770003600,
      },
    ],
    total: 2,
    page: 1,
    page_size: 10,
    total_pages: 1,
  }

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    })
    vi.clearAllMocks()
    vi.mocked(api.getIncidents).mockResolvedValue(mockIncidentsData as any)
  })

  const renderPage = () => {
    return render(
      <QueryClientProvider client={queryClient}>
        <MemoryRouter>
          <IncidentsPage />
        </MemoryRouter>
      </QueryClientProvider>
    )
  }

  it('renders TanStack Table headers, rows, and export CSV button', async () => {
    renderPage()

    await waitFor(() => {
      expect(screen.getByText('INC-2026-101')).toBeInTheDocument()
      expect(screen.getByText('INC-2026-102')).toBeInTheDocument()
    })

    // Check CSV Export link
    const exportLink = screen.getByRole('link', { name: /export csv/i })
    expect(exportLink).toHaveAttribute('href', '/api/incidents/export.csv')

    // Check confidence progress rendering
    expect(screen.getByText('92%')).toBeInTheDocument()
    expect(screen.getByText('78%')).toBeInTheDocument()
  })

  it('filters rows with instant search input', async () => {
    renderPage()

    await waitFor(() => {
      expect(screen.getByText('INC-2026-101')).toBeInTheDocument()
    })

    const searchInput = screen.getByPlaceholderText(/search incident id/i)
    fireEvent.change(searchInput, { target: { value: '102' } })

    expect(screen.queryByText('INC-2026-101')).not.toBeInTheDocument()
    expect(screen.getByText('INC-2026-102')).toBeInTheDocument()
  })

  it('opens IncidentDrawer on row click', async () => {
    renderPage()

    await waitFor(() => {
      expect(screen.getByText('INC-2026-101')).toBeInTheDocument()
    })

    // Click on row
    const row = screen.getByText('INC-2026-101').closest('tr')
    fireEvent.click(row!)

    // Drawer should open with dialog role
    await waitFor(() => {
      expect(screen.getByRole('dialog')).toBeInTheDocument()
    })
  })

  it('toggles column sorting on header click', async () => {
    renderPage()

    await waitFor(() => {
      expect(screen.getByText('INC-2026-101')).toBeInTheDocument()
    })

    const idHeaderBtn = screen.getByRole('button', { name: /incident id/i })
    fireEvent.click(idHeaderBtn)

    await waitFor(() => {
      expect(api.getIncidents).toHaveBeenCalledWith(
        expect.objectContaining({
          sort_by: 'incident_id',
        })
      )
    })
  })
})
