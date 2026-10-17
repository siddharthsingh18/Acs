import { renderHook, act } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { useJobStream } from './useJobStream'
import { api } from '@/api/client'
import { useAppStore } from '@/store/appStore'

vi.mock('@/api/client', () => ({
  api: {
    createJobWithSample: vi.fn(),
    createJobWithUpload: vi.fn(),
    cancelJob: vi.fn(),
  },
}))

describe('useJobStream hook', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    useAppStore.setState({ isDemoMode: false })
  })

  it('initializes with idle state and is muted by default', () => {
    const { result } = renderHook(() => useJobStream())

    expect(result.current.status).toBe('idle')
    expect(result.current.jobId).toBeNull()
    expect(result.current.incidents).toEqual([])
    expect(result.current.isIncidentActive).toBe(false)
    expect(result.current.isMuted).toBe(true) // Muted by default requirement
    expect(result.current.summary).toBeNull()
    expect(result.current.error).toBeNull()
  })

  it('allows toggling mute state', () => {
    const { result } = renderHook(() => useJobStream())

    act(() => {
      result.current.setIsMuted(false)
    })
    expect(result.current.isMuted).toBe(false)

    act(() => {
      result.current.setIsMuted(true)
    })
    expect(result.current.isMuted).toBe(true)
  })

  it('starts sample job and connects WebSocket', async () => {
    vi.mocked(api.createJobWithSample).mockResolvedValueOnce({
      job_id: 'test-job-123',
      status: 'started',
      source: 'sample',
      model_path: 'best.pt',
      confidence_threshold: 0.25,
      fps_limit: 15,
      total_frames: 192,
      created_at: 123456,
      stream_url: '/api/jobs/test-job-123/stream',
    } as any)

    const { result } = renderHook(() => useJobStream())

    await act(async () => {
      await result.current.startJob({
        source: 'sample',
        modelName: 'best.pt',
        confidence: 0.25,
      })
    })

    expect(api.createJobWithSample).toHaveBeenCalledWith({
      model_name: 'best.pt',
      confidence: 0.25,
      use_injury_model: undefined,
    })
    expect(result.current.jobId).toBe('test-job-123')
    expect(result.current.status).toBe('streaming')
  })

  it('handles cancellation gracefully', async () => {
    vi.mocked(api.cancelJob).mockResolvedValueOnce({
      job_id: 'test-job-123',
      status: 'cancelled',
      message: 'Job cancelled',
    })

    const { result } = renderHook(() => useJobStream())

    await act(async () => {
      await result.current.cancelJob()
    })

    expect(result.current.status).toBe('cancelled')
    expect(result.current.isIncidentActive).toBe(false)
  })

  it('runs demo simulation mode when isDemoMode is enabled', async () => {
    useAppStore.setState({ isDemoMode: true })

    const { result } = renderHook(() => useJobStream())

    await act(async () => {
      await result.current.startJob({
        source: 'sample',
      })
    })

    expect(result.current.status).toBe('streaming')
    expect(result.current.jobId).toBe('demo_job_active')

    // Cancel demo job cleanly
    await act(async () => {
      await result.current.cancelJob()
    })

    expect(result.current.status).toBe('cancelled')
  })

  it('resets state to idle cleanly', () => {
    const { result } = renderHook(() => useJobStream())

    act(() => {
      result.current.reset()
    })

    expect(result.current.status).toBe('idle')
    expect(result.current.jobId).toBeNull()
    expect(result.current.incidents).toEqual([])
    expect(result.current.error).toBeNull()
  })
})
