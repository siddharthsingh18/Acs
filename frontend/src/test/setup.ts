import '@testing-library/jest-dom/vitest'
import { vi } from 'vitest'
import '@/i18n'

// Mock AudioContext for Web Audio API alerts
class MockAudioContext {
  currentTime = 0
  destination = {}
  createOscillator() {
    return {
      type: 'sine',
      frequency: {
        setValueAtTime: vi.fn(),
        exponentialRampToValueAtTime: vi.fn(),
      },
      connect: vi.fn(),
      start: vi.fn(),
      stop: vi.fn(),
    }
  }
  createGain() {
    return {
      gain: {
        setValueAtTime: vi.fn(),
        linearRampToValueAtTime: vi.fn(),
      },
      connect: vi.fn(),
    }
  }
}

;(window as any).AudioContext = MockAudioContext
;(window as any).webkitAudioContext = MockAudioContext

// Mock WebSocket
class MockWebSocket {
  url: string
  onmessage: ((event: any) => void) | null = null
  onerror: ((event: any) => void) | null = null
  onclose: (() => void) | null = null
  readyState = 1

  constructor(url: string) {
    this.url = url
  }
  send = vi.fn()
  close = vi.fn(() => {
    if (this.onclose) this.onclose()
  })
}

;(window as any).WebSocket = MockWebSocket
