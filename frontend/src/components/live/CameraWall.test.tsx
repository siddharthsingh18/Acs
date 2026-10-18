import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { CameraWall } from './CameraWall'
import { api, type CameraItem } from '@/api/client'

// Mock api methods
vi.mock('@/api/client', () => ({
  api: {
    getCameras: vi.fn(),
    createCamera: vi.fn(),
    deleteCamera: vi.fn(),
    getCameraStreamWsUrl: vi.fn((id: string) => `ws://localhost/api/cameras/${id}/stream`),
  },
}))

// Mock useCameraStream hook to simulate stream states without live WebSocket in tests
vi.mock('@/hooks/useCameraStream', () => ({
  useCameraStream: vi.fn((cameraId: string) => {
    const isIncident = cameraId === 'cam-1'
    return {
      imageRef: { current: null },
      status: 'connected',
      fps: 15.0,
      isIncidentActive: isIncident,
      activeIncident: isIncident
        ? {
            type: 'incident',
            camera_id: cameraId,
            camera_name: 'Highway 101 Northbound',
            incident_id: 'INC-2026-TEST',
            class_name: 'car_car_accident',
            confidence: 0.91,
            start_timestamp: '2026-10-18 12:00:00 UTC',
            end_timestamp: '2026-10-18 12:00:05 UTC',
            keyframe_url: '/media/keyframes/test.jpg',
          }
        : null,
      health: {
        connected: true,
        fps: 15.0,
        last_frame_age: 0.1,
        dropped_frames: 0,
        reconnect_count: 0,
        is_incident_active: isIncident,
      },
      reconnect: vi.fn(),
    }
  }),
}))

const mockCameras: CameraItem[] = [
  {
    id: 'cam-1',
    name: 'Highway 101 Northbound',
    source_type: 'rtsp',
    source_url: 'rtsp://***:***@192.168.1.100:554/live',
    enabled: true,
    created_at: 1700000000,
    updated_at: 1700000000,
    health: {
      connected: true,
      fps: 15.0,
      last_frame_age: 0.1,
      dropped_frames: 0,
      reconnect_count: 0,
      is_incident_active: true,
    },
  },
  {
    id: 'cam-2',
    name: 'Arterial Junction 4',
    source_type: 'file',
    source_url: 'acc_ex.mp4',
    enabled: true,
    created_at: 1700000000,
    updated_at: 1700000000,
    health: {
      connected: true,
      fps: 15.0,
      last_frame_age: 0.1,
      dropped_frames: 0,
      reconnect_count: 0,
      is_incident_active: false,
    },
  },
]

describe('CameraWall Component', () => {
  let queryClient: QueryClient

  beforeEach(() => {
    vi.clearAllMocks()
    queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
      },
    })
    ;(api.getCameras as any).mockResolvedValue(mockCameras)
  })

  it('renders Camera Wall with grid layout buttons and camera tiles', async () => {
    render(
      <QueryClientProvider client={queryClient}>
        <CameraWall />
      </QueryClientProvider>
    )

    // Verify grid layout switcher buttons exist
    expect(screen.getByText('1x1')).toBeInTheDocument()
    expect(screen.getByText('2x2')).toBeInTheDocument()
    expect(screen.getByText('3x3')).toBeInTheDocument()

    // Verify camera tiles load
    await waitFor(() => {
      expect(screen.getByText('Highway 101 Northbound')).toBeInTheDocument()
      expect(screen.getByText('Arterial Junction 4')).toBeInTheDocument()
    })

    // Verify masked source url is displayed
    expect(screen.getByText('rtsp://***:***@192.168.1.100:554/live')).toBeInTheDocument()
  })

  it('highlights active incident with red pulse banner', async () => {
    render(
      <QueryClientProvider client={queryClient}>
        <CameraWall />
      </QueryClientProvider>
    )

    await waitFor(() => {
      expect(screen.getByText('ACCIDENT DETECTED')).toBeInTheDocument()
      expect(screen.getByText(/car_car_accident/)).toBeInTheDocument()
    })
  })

  it('switches to 1x1 focused layout when clicking 1x1 button or tile', async () => {
    render(
      <QueryClientProvider client={queryClient}>
        <CameraWall />
      </QueryClientProvider>
    )

    await waitFor(() => {
      expect(screen.getByText('Highway 101 Northbound')).toBeInTheDocument()
    })

    // Click 1x1 button
    fireEvent.click(screen.getByText('1x1'))

    // Should display "Back to Wall" button
    expect(screen.getByText('Back to Wall')).toBeInTheDocument()

    // Click "Back to Wall"
    fireEvent.click(screen.getByText('Back to Wall'))
    expect(screen.queryByText('Back to Wall')).not.toBeInTheDocument()
  })

  it('opens and closes Add Camera modal dialog', async () => {
    render(
      <QueryClientProvider client={queryClient}>
        <CameraWall />
      </QueryClientProvider>
    )

    // Click Add Camera button
    fireEvent.click(screen.getByText('Add Camera'))

    // Modal dialog opens
    expect(screen.getByText('Register New CCTV Camera')).toBeInTheDocument()
    expect(screen.getByText('Quick Presets')).toBeInTheDocument()

    // Close modal
    fireEvent.click(screen.getByText('Cancel'))
    expect(screen.queryByText('Register New CCTV Camera')).not.toBeInTheDocument()
  })
})
