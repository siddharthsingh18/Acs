import { useState, useRef, useCallback, useEffect } from 'react'
import {
  api,
  type StreamFrameMessage,
  type StreamIncidentMessage,
  type StreamDoneMessage,
  type StreamErrorMessage,
} from '@/api/client'

export type StreamStatus =
  | 'idle'
  | 'starting'
  | 'streaming'
  | 'completed'
  | 'error'
  | 'cancelled'

export interface JobStreamOptions {
  source: 'sample' | 'upload'
  file?: File | null
  modelName?: string
  confidence?: number
  useInjuryModel?: boolean
}

export interface StreamProgress {
  currentFrame: number
  totalFrames: number
  fps: number
}

// Tactical Web Audio chime for emergency incident alerts
function playTacticalChime() {
  try {
    const AudioCtx = window.AudioContext || (window as any).webkitAudioContext
    if (!AudioCtx) return
    const ctx = new AudioCtx()
    const now = ctx.currentTime

    // Tone 1
    const osc1 = ctx.createOscillator()
    const gain1 = ctx.createGain()
    osc1.type = 'sawtooth'
    osc1.frequency.setValueAtTime(880, now) // A5
    osc1.frequency.exponentialRampToValueAtTime(587.33, now + 0.15) // D5
    gain1.gain.setValueAtTime(0.2, now)
    gain1.gain.linearRampToValueAtTime(0.01, now + 0.25)
    osc1.connect(gain1)
    gain1.connect(ctx.destination)
    osc1.start(now)
    osc1.stop(now + 0.25)

    // Tone 2 (echo/siren)
    const osc2 = ctx.createOscillator()
    const gain2 = ctx.createGain()
    osc2.type = 'sine'
    osc2.frequency.setValueAtTime(1046.5, now + 0.1) // C6
    osc2.frequency.exponentialRampToValueAtTime(440, now + 0.3)
    gain2.gain.setValueAtTime(0.15, now + 0.1)
    gain2.gain.linearRampToValueAtTime(0.001, now + 0.35)
    osc2.connect(gain2)
    gain2.connect(ctx.destination)
    osc2.start(now + 0.1)
    osc2.stop(now + 0.35)
  } catch (e) {
    // AudioContext might be blocked until user gesture
    console.warn('Audio alert could not play:', e)
  }
}

export function useJobStream() {
  const [status, setStatus] = useState<StreamStatus>('idle')
  const [jobId, setJobId] = useState<string | null>(null)
  const [incidents, setIncidents] = useState<StreamIncidentMessage[]>([])
  const [latestIncident, setLatestIncident] = useState<StreamIncidentMessage | null>(null)
  const [isIncidentActive, setIsIncidentActive] = useState(false)
  const [summary, setSummary] = useState<StreamDoneMessage['summary'] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isMuted, setIsMuted] = useState(true) // muted by default as requested

  // HUD Progress state (throttled updates)
  const [progress, setProgress] = useState<StreamProgress>({
    currentFrame: 0,
    totalFrames: 0,
    fps: 0,
  })

  // Ref to the video <img> or <canvas> element to avoid React re-renders on every frame
  const imageRef = useRef<HTMLImageElement | null>(null)

  // Internal references
  const socketRef = useRef<WebSocket | null>(null)
  const activeJobIdRef = useRef<string | null>(null)
  const lastProgressUpdateRef = useRef<number>(0)
  const incidentTimeoutRef = useRef<any>(null)
  const isMutedRef = useRef(isMuted)

  useEffect(() => {
    isMutedRef.current = isMuted
  }, [isMuted])

  // Trigger incident active state and play sound if not muted
  const triggerIncidentAlert = useCallback((incident: StreamIncidentMessage) => {
    setLatestIncident(incident)
    setIsIncidentActive(true)

    if (!isMutedRef.current) {
      playTacticalChime()
    }

    if (incidentTimeoutRef.current) {
      clearTimeout(incidentTimeoutRef.current)
    }
    // Keep pulsing red border active for 4.5 seconds after detection
    incidentTimeoutRef.current = setTimeout(() => {
      setIsIncidentActive(false)
    }, 4500)
  }, [])

  const startJob = useCallback(
    async (options: JobStreamOptions) => {
      setError(null)
      setSummary(null)
      setIncidents([])
      setLatestIncident(null)
      setIsIncidentActive(false)
      setProgress({ currentFrame: 0, totalFrames: 0, fps: 0 })
      setStatus('starting')

      try {
        let jobRes
        if (options.source === 'sample') {
          jobRes = await api.createJobWithSample({
            model_name: options.modelName,
            confidence: options.confidence,
            use_injury_model: options.useInjuryModel,
          })
        } else {
          if (!options.file) {
            throw new Error('Please select a video file before starting analysis.')
          }
          jobRes = await api.createJobWithUpload(options.file, {
            model_name: options.modelName,
            confidence: options.confidence,
            use_injury_model: options.useInjuryModel,
          })
        }

        const newJobId = jobRes.job_id
        setJobId(newJobId)
        activeJobIdRef.current = newJobId
        setStatus('streaming')

        // Establish WebSocket connection
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
        const wsUrl = `${protocol}//${window.location.host}/api/jobs/${newJobId}/stream`
        const ws = new WebSocket(wsUrl)
        socketRef.current = ws

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data)

            if (data.type === 'frame') {
              const frame = data as StreamFrameMessage

              // Direct DOM mutation on the imageRef: NO React state update = ZERO re-render overhead!
              if (imageRef.current) {
                imageRef.current.src = `data:image/jpeg;base64,${frame.jpeg_base64}`
              }

              // Throttle React progress updates to ~10 FPS for counters and progress bar
              const now = performance.now()
              if (now - lastProgressUpdateRef.current > 90 || frame.index === frame.total) {
                lastProgressUpdateRef.current = now
                setProgress({
                  currentFrame: frame.index,
                  totalFrames: frame.total,
                  fps: frame.fps,
                })
              }
            } else if (data.type === 'incident') {
              const inc = data as StreamIncidentMessage
              setIncidents((prev) => [inc, ...prev])
              triggerIncidentAlert(inc)
            } else if (data.type === 'done') {
              const done = data as StreamDoneMessage
              setSummary(done.summary)
              setStatus('completed')
              setIsIncidentActive(false)
              if (socketRef.current) {
                socketRef.current.close()
                socketRef.current = null
              }
            } else if (data.type === 'error') {
              const err = data as StreamErrorMessage
              setError(err.message || 'Error reported during video stream')
              setStatus('error')
              setIsIncidentActive(false)
              if (socketRef.current) {
                socketRef.current.close()
                socketRef.current = null
              }
            }
          } catch (parseErr) {
            console.error('Error handling WebSocket message:', parseErr)
          }
        }

        ws.onerror = (wsErr) => {
          console.error('WebSocket stream error:', wsErr)
          setError('WebSocket stream encountered a communication error.')
          setStatus('error')
          setIsIncidentActive(false)
        }

        ws.onclose = () => {
          setStatus((prev) => (prev === 'streaming' ? 'completed' : prev))
        }
      } catch (err: any) {
        console.error('Failed to start analysis job:', err)
        setError(err.message || 'Failed to start inference job.')
        setStatus('error')
        setIsIncidentActive(false)
      }
    },
    [triggerIncidentAlert]
  )

  const cancelJob = useCallback(async () => {
    if (socketRef.current) {
      socketRef.current.close()
      socketRef.current = null
    }

    const currentId = activeJobIdRef.current
    if (currentId) {
      try {
        await api.cancelJob(currentId)
      } catch (err) {
        console.warn('Job cancellation warning:', err)
      }
    }

    setStatus('cancelled')
    setIsIncidentActive(false)
  }, [])

  const reset = useCallback(() => {
    if (socketRef.current) {
      socketRef.current.close()
      socketRef.current = null
    }
    setStatus('idle')
    setJobId(null)
    setIncidents([])
    setLatestIncident(null)
    setIsIncidentActive(false)
    setSummary(null)
    setError(null)
    setProgress({ currentFrame: 0, totalFrames: 0, fps: 0 })
    if (imageRef.current) {
      imageRef.current.src = ''
    }
  }, [])

  // Clean up socket and timers on unmount
  useEffect(() => {
    return () => {
      if (incidentTimeoutRef.current) {
        clearTimeout(incidentTimeoutRef.current)
      }
      if (socketRef.current) {
        socketRef.current.close()
        socketRef.current = null
      }
    }
  }, [])

  return {
    status,
    jobId,
    progress,
    incidents,
    latestIncident,
    isIncidentActive,
    summary,
    error,
    isMuted,
    setIsMuted,
    imageRef,
    startJob,
    cancelJob,
    reset,
  }
}
