import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter } from 'react-router-dom'
import { IncidentDrawer } from './IncidentDrawer'
import { api } from '@/api/client'

vi.mock('@/api/client', () => ({
  api: {
    getIncident: vi.fn(),
    generateDispatchReport: vi.fn(),
    updateIncidentStatus: vi.fn(),
    testAlerts: vi.fn(),
    getIncidentPdfUrl: vi.fn((id: string) => `/api/incidents/${id}/pdf`),
    getAlertSettings: vi.fn().mockResolvedValue({
      enabled: true,
      dry_run: true,
      channels: ['console'],
    }),
  },
}))

describe('IncidentDrawer component', () => {
  let queryClient: QueryClient

  const mockIncident = {
    incident_id: 'INC-2026-001',
    class_name: 'car_car_accident',
    max_confidence: 0.88,
    start_timestamp: '2026-10-10T03:00:00Z',
    end_timestamp: '2026-10-10T03:00:05Z',
    duration: 5.2,
    severity: 'high',
    status: 'pending',
    reviewer_name: null,
    reviewed_at: null,
    keyframe_url: '/media/keyframes/inc1.jpg',
    dispatch_report: {
      vehicles: ['Sedan', 'SUV'],
      collision_point: 'Lane 2 intersection',
      damage: 'Front bumper deformity',
      road_conditions: 'Dry asphalt',
      hazards: 'Fluid spill',
      recommended_units: ['EMS', 'police', 'tow'],
      severity: 'high',
    },
    dispatch_text: '[EMERGENCY DISPATCH REPORT - TRAFFIC COLLISION]\nSeverity: HIGH',
    created_at: 123456789,
    image_path: 'inc1.jpg',
  }

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
      },
    })
    vi.clearAllMocks()
    vi.mocked(api.getIncident).mockResolvedValue(mockIncident as any)
  })

  const renderDrawer = (props: any = {}) => {
    return render(
      <QueryClientProvider client={queryClient}>
        <MemoryRouter>
          <IncidentDrawer
            incidentId="INC-2026-001"
            isOpen={true}
            onClose={vi.fn()}
            initialIncident={mockIncident as any}
            {...props}
          />
        </MemoryRouter>
      </QueryClientProvider>
    )
  }

  it('renders keyframe image and metadata grid correctly', () => {
    renderDrawer()

    expect(screen.getByText('INC-2026-001')).toBeInTheDocument()
    expect(screen.getByText(/88\.0%/)).toBeInTheDocument()
    expect(screen.getByText(/5\.2s/)).toBeInTheDocument()
    expect(screen.getByText(/High Severity/i)).toBeInTheDocument()
  })

  it('toggles interactive zoom on keyframe', () => {
    renderDrawer()

    const zoomBtn = screen.getByRole('button', { name: /Zoom 2x/i })
    fireEvent.click(zoomBtn)
    expect(screen.getByText(/Reset/i)).toBeInTheDocument()
  })

  it('renders structured Gemini report sections', () => {
    renderDrawer()

    // Recommended units chips
    expect(screen.getByText('EMS')).toBeInTheDocument()
    expect(screen.getByText('POLICE')).toBeInTheDocument()
    expect(screen.getByText('TOW')).toBeInTheDocument()

    // Structured fields
    expect(screen.getByText('Sedan • SUV')).toBeInTheDocument()
    expect(screen.getByText('Lane 2 intersection')).toBeInTheDocument()
    expect(screen.getByText('Front bumper deformity')).toBeInTheDocument()
    expect(screen.getByText('Fluid spill')).toBeInTheDocument()
  })

  it('opens confirmation modal before executing human-in-the-loop actions', async () => {
    renderDrawer()

    // Click Dispatch 911 button
    const dispatchBtn = screen.getByRole('button', { name: /Dispatch 911/i })
    fireEvent.click(dispatchBtn)

    // Confirmation dialog should be open
    expect(screen.getByText(/Confirm Incident DISPATCHED/i)).toBeInTheDocument()
    expect(screen.getByText(/Authorize 911 emergency services/i)).toBeInTheDocument()

    // Confirm action
    const confirmBtn = screen.getByRole('button', { name: /Confirm dispatched/i })
    fireEvent.click(confirmBtn)

    await waitFor(() => {
      expect(api.updateIncidentStatus).toHaveBeenCalledWith(
        'INC-2026-001',
        'dispatched',
        expect.any(String)
      )
    })
  })

  it('supports Escape key to close the drawer', () => {
    const onClose = vi.fn()
    renderDrawer({ onClose })

    fireEvent.keyDown(window, { key: 'Escape' })
    expect(onClose).toHaveBeenCalled()
  })

  it('supports arrow keys to navigate between incidents', () => {
    const onSelectIncident = vi.fn()
    const incidentList = [
      { incident_id: 'INC-2026-000' },
      { incident_id: 'INC-2026-001' },
      { incident_id: 'INC-2026-002' },
    ]

    renderDrawer({
      incidentList,
      onSelectIncident,
    })

    // Arrow Right should go to next
    fireEvent.keyDown(window, { key: 'ArrowRight' })
    expect(onSelectIncident).toHaveBeenCalledWith('INC-2026-002')

    // Arrow Left should go to previous
    fireEvent.keyDown(window, { key: 'ArrowLeft' })
    expect(onSelectIncident).toHaveBeenCalledWith('INC-2026-000')
  })
})
