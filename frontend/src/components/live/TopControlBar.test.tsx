import { render, screen, fireEvent } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import { TopControlBar } from './TopControlBar'

describe('TopControlBar component', () => {
  const defaultProps = {
    sourceType: 'sample' as const,
    setSourceType: vi.fn(),
    selectedFile: null,
    setSelectedFile: vi.fn(),
    selectedModel: 'best.pt',
    setSelectedModel: vi.fn(),
    confidence: 0.25,
    setConfidence: vi.fn(),
    useInjuryModel: false,
    setUseInjuryModel: vi.fn(),
    isStreaming: false,
    isStarting: false,
    onStart: vi.fn(),
    onCancel: vi.fn(),
    isMuted: true,
    setIsMuted: vi.fn(),
    configData: {
      available_models: ['best.pt', 'trained_yolo_model.pt'],
      defaults: {
        model_path: 'best.pt',
        injury_model_path: 'injury.pt',
        injury_model_available: false,
        confidence_threshold: 0.25,
      },
      device_info: {
        device: 'mps',
        cuda_available: false,
        mps_available: true,
      },
    } as any,
  }

  it('renders sample video info by default and switches to upload', () => {
    render(<TopControlBar {...defaultProps} />)

    expect(screen.getAllByText(/acc_ex.mp4/i).length).toBeGreaterThan(0)

    const uploadBtn = screen.getByRole('button', { name: /upload/i })
    fireEvent.click(uploadBtn)
    expect(defaultProps.setSourceType).toHaveBeenCalledWith('upload')
  })

  it('disables injury model toggle when unavailable in config', () => {
    render(<TopControlBar {...defaultProps} />)

    const checkbox = screen.getByRole('checkbox')
    expect(checkbox).toBeDisabled()
    expect(screen.getByText(/Unavailable/i)).toBeInTheDocument()
  })

  it('handles start stream click', () => {
    render(<TopControlBar {...defaultProps} />)

    const startBtn = screen.getByRole('button', { name: /start/i })
    fireEvent.click(startBtn)
    expect(defaultProps.onStart).toHaveBeenCalled()
  })

  it('toggles mute alert state', () => {
    render(<TopControlBar {...defaultProps} />)

    const soundBtn = screen.getByTitle(/Unmute Sound Siren Alerts/i)
    fireEvent.click(soundBtn)
    expect(defaultProps.setIsMuted).toHaveBeenCalled()
  })
})
