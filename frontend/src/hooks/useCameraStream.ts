/**
 * frontend.src.hooks.useCameraStream
 * Custom hook managing high-performance WebSocket connection for a single CCTV camera feed.
 *
 * Key Design Architecture:
 * - Frame rendering is kept strictly OUT of React state:
 *   Updates imageRef.current.src directly upon frame message receipt.
 *   Zero React component re-renders on high-speed 15-30 FPS frame delivery.
 * - Low-frequency state updates:
 *   Status (connected, reconnecting, offline), active incident alert, and fps counters
 *   are throttled to keep React render cycles near 0.
 */
import { useEffect, useRef, useState, useCallback } from 'react'
import { api, type CameraHealth, type CameraStreamIncidentMessage } from '@/api/client'

export type CameraConnectionStatus = 'connecting' | 'connected' | 'reconnecting' | 'offline'

export interface UseCameraStreamResult {
  imageRef: React.RefObject<HTMLImageElement>
  status: CameraConnectionStatus
  fps: number
  isIncidentActive: boolean
  activeIncident: CameraStreamIncidentMessage | null
  health: CameraHealth | null
  reconnect: () => void
}

export function useCameraStream(cameraId: string, enabled: boolean = true): UseCameraStreamResult {
  const imageRef = useRef<HTMLImageElement | null>(null)
  const [status, setStatus] = useState<CameraConnectionStatus>('connecting')
  const [fps, setFps] = useState<number>(0)
  const [isIncidentActive, setIsIncidentActive] = useState<boolean>(false)
  const [activeIncident, setActiveIncident] = useState<CameraStreamIncidentMessage | null>(null)
  const [health, setHealth] = useState<CameraHealth | null>(null)

  const socketRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<any>(null)
  const reconnectAttemptRef = useRef<number>(0)
  const lastFpsUpdateRef = useRef<number>(0)
  const isMountedRef = useRef<boolean>(true)

  const connect = useCallback(() => {
    if (!enabled || !cameraId) {
      setStatus('offline')
      return
    }

    // Clean up any existing connection
    if (socketRef.current) {
      try {
        socketRef.current.close()
      } catch {}
      socketRef.current = null
    }

    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = null
    }

    const wsUrl = api.getCameraStreamWsUrl(cameraId)
    setStatus(reconnectAttemptRef.current > 0 ? 'reconnecting' : 'connecting')

    try {
      const ws = new WebSocket(wsUrl)
      socketRef.current = ws

      ws.onopen = () => {
        if (!isMountedRef.current) return
        setStatus('connected')
        reconnectAttemptRef.current = 0
      }

      ws.onmessage = (event) => {
        if (!isMountedRef.current) return
        try {
          const data = JSON.parse(event.data)

          if (data.type === 'frame') {
            // High-performance DOM update: bypass React state to achieve maximum throughput
            if (imageRef.current && data.frame) {
              imageRef.current.src = data.frame
            }

            // Throttled FPS state update (max once per second)
            const now = Date.now()
            if (now - lastFpsUpdateRef.current >= 1000) {
              lastFpsUpdateRef.current = now
              setFps(data.fps || 0)
            }

            if (data.incident_active !== undefined) {
              setIsIncidentActive(Boolean(data.incident_active))
            }
          } else if (data.type === 'incident') {
            setIsIncidentActive(true)
            setActiveIncident(data as CameraStreamIncidentMessage)
          } else if (data.type === 'heartbeat' || data.type === 'status') {
            if (data.health) {
              setHealth(data.health)
              if (data.health.connected) {
                setStatus('connected')
              }
              setIsIncidentActive(Boolean(data.health.is_incident_active))
            }
            if (data.connected === false) {
              setStatus('reconnecting')
            }
          }
        } catch (err) {
          console.warn(`[CameraStream] Failed to parse message for camera ${cameraId}:`, err)
        }
      }

      ws.onclose = (event) => {
        if (!isMountedRef.current) return
        // Do not auto-reconnect if closed normally (code 1000) or disabled
        if (event.code === 1000 || !enabled) {
          setStatus('offline')
          return
        }

        setStatus('reconnecting')
        // Exponential backoff reconnect: 1s, 2s, 4s, up to 10s
        const attempt = reconnectAttemptRef.current
        const delay = Math.min(10000, 1000 * Math.pow(1.5, attempt))
        reconnectAttemptRef.current = attempt + 1

        reconnectTimeoutRef.current = setTimeout(() => {
          if (isMountedRef.current && enabled) {
            connect()
          }
        }, delay)
      }

      ws.onerror = () => {
        if (!isMountedRef.current) return
        setStatus('reconnecting')
      }
    } catch (e) {
      console.warn(`[CameraStream] WebSocket error for camera ${cameraId}:`, e)
      setStatus('offline')
    }
  }, [cameraId, enabled])

  useEffect(() => {
    isMountedRef.current = true
    connect()

    return () => {
      isMountedRef.current = false
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }
      if (socketRef.current) {
        try {
          socketRef.current.close(1000, 'Unmounting')
        } catch {}
      }
    }
  }, [connect])

  const reconnect = useCallback(() => {
    reconnectAttemptRef.current = 0
    connect()
  }, [connect])

  return {
    imageRef,
    status,
    fps,
    isIncidentActive,
    activeIncident,
    health,
    reconnect,
  }
}
