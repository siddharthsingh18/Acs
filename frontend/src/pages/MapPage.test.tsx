import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter } from 'react-router-dom'
import { MapPage } from './MapPage'
import { api } from '@/api/client'
import i18n from '@/i18n'

const { leafletMock } = vi.hoisted(() => {
  const mockMap = {
    setView: vi.fn().mockReturnThis(),
    fitBounds: vi.fn().mockReturnThis(),
    addLayer: vi.fn().mockReturnThis(),
    removeLayer: vi.fn().mockReturnThis(),
    remove: vi.fn().mockReturnThis(),
    invalidateSize: vi.fn().mockReturnThis(),
    on: vi.fn().mockReturnThis(),
  }

  const mockLayerGroup = {
    addTo: vi.fn().mockReturnThis(),
    clearLayers: vi.fn().mockReturnThis(),
    addLayer: vi.fn().mockReturnThis(),
  }

  const mockMarker = {
    addTo: vi.fn().mockReturnThis(),
    bindPopup: vi.fn().mockReturnThis(),
    on: vi.fn().mockReturnThis(),
    setLatLng: vi.fn().mockReturnThis(),
    getLatLng: vi.fn().mockReturnValue({ lat: 24.7136, lng: 46.6753 }),
    remove: vi.fn().mockReturnThis(),
  }

  const leafletMock = {
    map: vi.fn(() => mockMap),
    tileLayer: vi.fn(() => ({ addTo: vi.fn() })),
    control: { zoom: vi.fn(() => ({ addTo: vi.fn() })) },
    layerGroup: vi.fn(() => mockLayerGroup),
    markerClusterGroup: vi.fn(() => mockLayerGroup),
    heatLayer: vi.fn(() => ({ addTo: vi.fn() })),
    divIcon: vi.fn((opts: any) => opts),
    marker: vi.fn(() => mockMarker),
    latLngBounds: vi.fn(() => ({})),
  }

  return { leafletMock }
})

vi.mock('leaflet.markercluster', () => ({}))
vi.mock('leaflet.heat', () => ({}))

vi.mock('leaflet', () => ({
  default: leafletMock,
  ...leafletMock,
}))

vi.mock('@/lib/leafletSetup', () => ({
  default: leafletMock,
  L: leafletMock,
}))

vi.mock('@/api/client', () => ({
  api: {
    getCameras: vi.fn(),
    getIncidents: vi.fn(),
    getIncident: vi.fn(),
    updateIncidentStatus: vi.fn(),
    generateDispatchReport: vi.fn(),
  },
}))

describe('MapPage Component', () => {
  let queryClient: QueryClient

  const mockCameras = [
    {
      id: 'cam-riyadh-1',
      name: 'King Fahd Highway 1',
      source_type: 'rtsp',
      source_url: 'rtsp://127.0.0.1:8554/live',
      location_lat: 24.7136,
      location_lng: 46.6753,
      enabled: true,
      health: {
        connected: true,
        fps: 29.8,
        last_frame_age: 0.1,
        dropped_frames: 0,
        reconnect_count: 0,
        is_incident_active: false,
      },
    },
    {
      id: 'cam-olaya-2',
      name: 'Olaya Street North',
      source_type: 'file',
      source_url: 'acc_ex.mp4',
      location_lat: 24.7003,
      location_lng: 46.6841,
      enabled: true,
      health: {
        connected: true,
        fps: 25.0,
        last_frame_age: 0.2,
        dropped_frames: 2,
        reconnect_count: 0,
        is_incident_active: true,
      },
    },
  ]

  const mockIncidents = {
    items: [
      {
        incident_id: 'INC-2026-101',
        start_timestamp: '2026-10-18T10:00:00Z',
        end_timestamp: '2026-10-18T10:00:15Z',
        class_name: 'vehicle_collision',
        max_confidence: 0.88,
        image_path: 'keyframes/inc_101.jpg',
        keyframe_url: '/media/keyframes/inc_101.jpg',
        severity: 'high',
        severity_score: 85,
        status: 'pending',
        created_at: 1776510000,
        duration: 15.0,
      },
      {
        incident_id: 'INC-2026-102',
        start_timestamp: '2026-10-18T11:30:00Z',
        end_timestamp: '2026-10-18T11:30:10Z',
        class_name: 'car_person_accident',
        max_confidence: 0.76,
        image_path: 'keyframes/inc_102.jpg',
        keyframe_url: '/media/keyframes/inc_102.jpg',
        severity: 'medium',
        severity_score: 65,
        status: 'approved',
        created_at: 1776515400,
        duration: 10.0,
      },
    ],
    total: 2,
    page: 1,
    page_size: 100,
    total_pages: 1,
  }

  beforeEach(async () => {
    queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    })
    vi.clearAllMocks()
    await i18n.changeLanguage('en')
    vi.mocked(api.getCameras).mockResolvedValue(mockCameras as any)
    vi.mocked(api.getIncidents).mockResolvedValue(mockIncidents as any)
  })

  const renderComponent = () => {
    return render(
      <QueryClientProvider client={queryClient}>
        <MemoryRouter>
          <MapPage />
        </MemoryRouter>
      </QueryClientProvider>
    )
  }

  it('renders map page header, OSM badge, and telemetry counters', async () => {
    renderComponent()

    await waitFor(() => {
      expect(screen.getByText(/Interactive CCTV Map/i)).toBeInTheDocument()
      expect(screen.getByText('OSM Tiles')).toBeInTheDocument()
      expect(screen.getByText(/Cameras Online/i)).toBeInTheDocument()
      expect(screen.getByText(/Active Incidents/i)).toBeInTheDocument()
    })
  })

  it('renders all map layer controls and allows toggling them', async () => {
    renderComponent()

    await waitFor(() => {
      expect(screen.getByTitle(/Toggle Collision Hotspot Heatmap/i)).toBeInTheDocument()
      expect(screen.getByTitle(/Toggle Marker Clustering/i)).toBeInTheDocument()
      expect(screen.getByTitle(/Toggle Incidents/i)).toBeInTheDocument()
      expect(screen.getByTitle(/Toggle Tactical Dark Mode Tiles/i)).toBeInTheDocument()
      expect(screen.getByTitle(/Center Network/i)).toBeInTheDocument()
    })

    const heatmapToggle = screen.getByTitle(/Toggle Collision Hotspot Heatmap/i)
    fireEvent.click(heatmapToggle)
    expect(heatmapToggle).toBeInTheDocument()
  })

  it('renders chronological incident replay slider dock with speed options', async () => {
    renderComponent()

    await waitFor(() => {
      expect(screen.getByText(/Incident Chronological Replay/i)).toBeInTheDocument()
      expect(screen.getByText(/Show All Incidents/i)).toBeInTheDocument()
      expect(screen.getByText('1x')).toBeInTheDocument()
      expect(screen.getByText('2x')).toBeInTheDocument()
      expect(screen.getByText('5x')).toBeInTheDocument()
      expect(screen.getByText('10x')).toBeInTheDocument()
    })

    // Toggle Replay Mode
    const replayModeBtn = screen.getByText(/Show All Incidents/i)
    fireEvent.click(replayModeBtn)

    await waitFor(() => {
      expect(screen.getByText(/Replay Mode/i)).toBeInTheDocument()
    })
  })

  it('ensures the Leaflet map container has strict LTR direction even in Arabic RTL mode', async () => {
    await i18n.changeLanguage('ar')
    renderComponent()

    const mapViewport = document.getElementById('tactical-map-viewport')
    expect(mapViewport).not.toBeNull()
    expect(mapViewport?.getAttribute('dir')).toBe('ltr')
    expect(mapViewport?.style.direction).toBe('ltr')
  })
})
