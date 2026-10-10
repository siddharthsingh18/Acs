import { describe, it, expect, vi } from 'vitest'
import { runDemoSimulation } from './demoSimulation'

describe('runDemoSimulation Utility', () => {
  it('initializes and executes frames sequentially', async () => {
    const onFrame = vi.fn()
    const onIncident = vi.fn()
    const onComplete = vi.fn()
    const onError = vi.fn()

    const cancel = runDemoSimulation({
      onFrame,
      onIncident,
      onComplete,
      onError,
    })

    // Wait 100ms for ~2-3 frames to execute
    await new Promise((resolve) => setTimeout(resolve, 100))

    expect(onFrame).toHaveBeenCalled()
    expect(onError).not.toHaveBeenCalled()

    cancel()
  })

  it('can be cleanly cancelled before completion', async () => {
    const onFrame = vi.fn()
    const onIncident = vi.fn()
    const onComplete = vi.fn()
    const onError = vi.fn()

    const cancel = runDemoSimulation({
      onFrame,
      onIncident,
      onComplete,
      onError,
    })

    // Cancel immediately
    cancel()

    // Wait 100ms
    await new Promise((resolve) => setTimeout(resolve, 100))

    expect(onComplete).not.toHaveBeenCalled()
  })
})
