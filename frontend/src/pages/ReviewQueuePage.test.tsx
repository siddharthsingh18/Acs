import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ReviewQueuePage } from './ReviewQueuePage'
import { api, ReviewQueueResponse, ReviewStatsResponse } from '@/api/client'

vi.mock('@/api/client', () => ({
  api: {
    getReviewQueue: vi.fn(),
    getReviewStats: vi.fn(),
    submitReviewDecision: vi.fn(),
    undoReviewDecision: vi.fn(),
  },
}))

describe('ReviewQueuePage', () => {
  let queryClient: QueryClient

  const mockQueue: ReviewQueueResponse = {
    total: 2,
    items: [
      {
        incident_id: 'INC-BORDER-101',
        class_name: 'car_car_accident',
        max_confidence: 0.45,
        image_path: '/media/keyframes/border_101.jpg',
        keyframe_url: '/media/keyframes/border_101.jpg',
        clip_path: '/media/clips/border_101.mp4',
        clip_url: '/api/incidents/INC-BORDER-101/clip',
        clip_annotated_path: '/media/clips/border_101_ann.mp4',
        clip_annotated_url: '/api/incidents/INC-BORDER-101/clip?annotated=true',
        verification_status: 'needs_review',
        vlm_confidence: 0.15,
        vlm_reason: 'No deformation visible',
        physics_passed: true,
        uncertainty_reason: 'Borderline YOLO confidence (45%) • YOLO/VLM verification flagged for human dispatch review',
        start_timestamp: '2026-10-18 20:00:00 UTC',
        severity: 'medium',
        severity_score: 55,
        duration: 4.2,
      },
      {
        incident_id: 'INC-BORDER-102',
        class_name: 'bike_bike_accident',
        max_confidence: 0.52,
        image_path: '/media/keyframes/border_102.jpg',
        keyframe_url: '/media/keyframes/border_102.jpg',
        uncertainty_reason: 'Borderline YOLO confidence (52%)',
        start_timestamp: '2026-10-18 20:10:00 UTC',
        severity: 'high',
        severity_score: 75,
        duration: 3.5,
      },
    ],
  }

  const mockStats: ReviewStatsResponse = {
    total_reviewed: 10,
    accidents_approved: 6,
    hard_negatives: 3,
    wrong_class_count: 1,
    staging_pending_export: 4,
  }

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    })
    vi.clearAllMocks()
    vi.mocked(api.getReviewQueue).mockResolvedValue(mockQueue)
    vi.mocked(api.getReviewStats).mockResolvedValue(mockStats)
    vi.mocked(api.submitReviewDecision).mockResolvedValue({
      incident_id: 'INC-BORDER-101',
      decision: 'accident',
      original_class: 'car_car_accident',
      confidence: 0.45,
      image_path: '/media/keyframes/border_101.jpg',
      reviewer_name: 'Dispatch Operator',
      reviewed_at: '2026-10-18T20:05:00Z',
      is_exported: false,
      created_at: 1234567890,
    })
  })

  it('renders review queue header and telemetry staging stats', async () => {
    render(
      <QueryClientProvider client={queryClient}>
        <MemoryRouter>
          <ReviewQueuePage />
        </MemoryRouter>
      </QueryClientProvider>
    )

    expect(screen.getByText(/Uncertain Detection Review Queue/i)).toBeInTheDocument()
    await waitFor(() => {
      expect(screen.getByText('Approved Candidates')).toBeInTheDocument()
      expect(screen.getByText('Class Corrections')).toBeInTheDocument()
      expect(screen.getAllByText(/Hard Negative/i).length).toBeGreaterThan(0)
    })
  })

  it('displays the uncertain incident card with action buttons', async () => {
    render(
      <QueryClientProvider client={queryClient}>
        <MemoryRouter>
          <ReviewQueuePage />
        </MemoryRouter>
      </QueryClientProvider>
    )

    await waitFor(() => {
      expect(screen.getByText(/ID: INC-BORDER-101/i)).toBeInTheDocument()
      expect(screen.getByText(/Borderline YOLO confidence \(45%\)/i)).toBeInTheDocument()
      expect(screen.getByRole('button', { name: /^Accident/i })).toBeInTheDocument()
      expect(screen.getByRole('button', { name: /^Not Accident/i })).toBeInTheDocument()
      expect(screen.getByRole('button', { name: /^Wrong Class/i })).toBeInTheDocument()
    })
  })

  it('triggers submitReviewDecision when Accident is clicked', async () => {
    render(
      <QueryClientProvider client={queryClient}>
        <MemoryRouter>
          <ReviewQueuePage />
        </MemoryRouter>
      </QueryClientProvider>
    )

    await waitFor(() => {
      expect(screen.getByText(/ID: INC-BORDER-101/i)).toBeInTheDocument()
    })

    const accidentBtn = screen.getByRole('button', { name: /^Accident/i })
    fireEvent.click(accidentBtn)

    await waitFor(() => {
      expect(api.submitReviewDecision).toHaveBeenCalledWith('INC-BORDER-101', {
        decision: 'accident',
        corrected_class: null,
        reviewer_name: 'Dispatch Operator',
      })
    })
  })
})
