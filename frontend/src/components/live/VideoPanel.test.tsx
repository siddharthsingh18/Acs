import React from 'react'
import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import { VideoPanel } from './VideoPanel'

describe('VideoPanel component', () => {
  it('renders strictly in LTR mode', () => {
    const ref = React.createRef<HTMLImageElement>()
    const { container } = render(
      <VideoPanel
        imageRef={ref}
        status="idle"
        progress={{ currentFrame: 0, totalFrames: 0, fps: 0 }}
        isIncidentActive={false}
        error={null}
      />
    )

    const videoBox = container.querySelector('[dir="ltr"]')
    expect(videoBox).toBeInTheDocument()
    expect(videoBox).toHaveAttribute('dir', 'ltr')
  })

  it('renders MONITORING badge when streaming without active incident', () => {
    const ref = React.createRef<HTMLImageElement>()
    render(
      <VideoPanel
        imageRef={ref}
        status="streaming"
        progress={{ currentFrame: 50, totalFrames: 200, fps: 14.8 }}
        isIncidentActive={false}
        error={null}
      />
    )

    expect(screen.getByText(/MONITORING/i)).toBeInTheDocument()
    expect(screen.getByText(/14.8 FPS/i)).toBeInTheDocument()
    expect(screen.getByText(/50 \/ 200/i)).toBeInTheDocument()
  })

  it('renders INCIDENT ACTIVE badge and pulsing red border when incident is active', () => {
    const ref = React.createRef<HTMLImageElement>()
    const { container } = render(
      <VideoPanel
        imageRef={ref}
        status="streaming"
        progress={{ currentFrame: 80, totalFrames: 200, fps: 15.0 }}
        isIncidentActive={true}
        error={null}
      />
    )

    expect(screen.getByText(/INCIDENT ACTIVE/i)).toBeInTheDocument()

    const videoBox = container.querySelector('[dir="ltr"]')
    expect(videoBox?.className).toContain('border-rose-500')
    expect(videoBox?.className).toContain('ring-rose-500/20')
  })
})
