import type { StreamIncidentMessage, StreamDoneMessage } from '@/api/client'

export interface DemoSimulationCallbacks {
  onFrame: (dataUrl: string, index: number, total: number, fps: number) => void
  onIncident: (incident: StreamIncidentMessage) => void
  onComplete: (summary: StreamDoneMessage['summary']) => void
  onError: (msg: string) => void
}

/**
 * Creates an in-browser canvas-based CCTV camera incident replay sequence
 * for presentations without requiring an active backend or GPU.
 */
export function runDemoSimulation(callbacks: DemoSimulationCallbacks): () => void {
  const canvas = document.createElement('canvas')
  canvas.width = 640
  canvas.height = 360
  const ctx = canvas.getContext ? canvas.getContext('2d') : null

  const TOTAL_FRAMES = 180
  const FPS = 28
  const FRAME_INTERVAL_MS = 1000 / FPS
  let currentFrame = 0
  let isCancelled = false
  let timerId: any = null

  let incident1Triggered = false
  let incident2Triggered = false

  const renderFrame = (frame: number): string => {
    if (!ctx) {
      // Fallback SVG data URL for headless or non-canvas test environments
      if (frame >= 42 && !incident1Triggered) {
        incident1Triggered = true
        callbacks.onIncident({
          type: 'incident',
          incident_id: 'INC-DEMO-2026-001',
          class_name: 'car_car_accident',
          max_confidence: 0.96,
          start_timestamp: new Date().toISOString(),
          end_timestamp: new Date(Date.now() + 2800).toISOString(),
          start_time: 1.4,
          end_time: 4.2,
          duration: 2.8,
          keyframe_url: '',
        })
      }
      return `data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="640" height="360"><rect width="100%" height="100%" fill="%230f141c"/><text x="20" y="40" fill="%2306b6d4" font-family="monospace">DEMO FRAME ${frame}</text></svg>`
    }
    // 1. Dark Asphalt Background
    ctx.fillStyle = '#0f141c'
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    // 2. Roadway Markings
    ctx.strokeStyle = '#2d3748'
    ctx.lineWidth = 4
    ctx.strokeRect(40, 40, 560, 280) // Intersection perimeter

    // Lane dashes
    ctx.strokeStyle = '#4a5568'
    ctx.lineWidth = 2
    ctx.setLineDash([12, 10])
    ctx.beginPath()
    ctx.moveTo(320, 40)
    ctx.lineTo(320, 320)
    ctx.moveTo(40, 180)
    ctx.lineTo(600, 180)
    ctx.stroke()
    ctx.setLineDash([])

    // 3. Vehicles
    // Vehicle A (moving left to right)
    const carAX = Math.min(180 + frame * 2.2, 330)
    const carAY = 155
    ctx.fillStyle = '#38bdf8'
    ctx.fillRect(carAX - 25, carAY - 15, 50, 30)
    ctx.fillStyle = '#0284c7'
    ctx.fillRect(carAX - 15, carAY - 10, 30, 20) // Cabin

    // Vehicle B (moving top to bottom)
    const carBY = Math.min(80 + frame * 1.8, 175)
    const carBX = 310
    ctx.fillStyle = '#f59e0b'
    ctx.fillRect(carBX - 15, carBY - 25, 30, 50)
    ctx.fillStyle = '#d97706'
    ctx.fillRect(carBX - 10, carBY - 15, 20, 30) // Cabin

    // Pedestrian / Cyclist in second half
    if (frame >= 80) {
      const bikeX = Math.max(500 - (frame - 80) * 2, 360)
      const bikeY = 240
      ctx.fillStyle = '#10b981'
      ctx.fillRect(bikeX - 8, bikeY - 14, 16, 28)
    }

    // 4. Incident 1: Frame 40-75 (Car-Car Collision)
    if (frame >= 40 && frame <= 75) {
      // Impact Spark FX
      ctx.fillStyle = 'rgba(239, 68, 68, 0.3)'
      ctx.beginPath()
      ctx.arc(315, 165, 30 + Math.sin(frame) * 8, 0, Math.PI * 2)
      ctx.fill()

      // Red Collision Bounding Box
      ctx.strokeStyle = '#ef4444'
      ctx.lineWidth = 3
      ctx.strokeRect(270, 125, 95, 80)

      // Label Badge
      ctx.fillStyle = '#ef4444'
      ctx.fillRect(270, 105, 140, 18)
      ctx.fillStyle = '#ffffff'
      ctx.font = 'bold 10px monospace'
      ctx.fillText('CAR-CAR ACCIDENT 96%', 274, 118)

      // Emit Incident Message once
      if (frame >= 42 && !incident1Triggered) {
        incident1Triggered = true
        const keyframe = canvas.toDataURL('image/jpeg', 0.8)
        const inc1: StreamIncidentMessage = {
          type: 'incident',
          incident_id: 'INC-DEMO-2026-001',
          class_name: 'car_car_accident',
          max_confidence: 0.96,
          start_timestamp: new Date().toISOString(),
          end_timestamp: new Date(Date.now() + 2800).toISOString(),
          start_time: 1.4,
          end_time: 4.2,
          duration: 2.8,
          keyframe_url: keyframe,
        }
        callbacks.onIncident(inc1)
      }
    }

    // 5. Incident 2: Frame 110-145 (Bike Collision)
    if (frame >= 110 && frame <= 145) {
      // Collision Bounding Box
      ctx.strokeStyle = '#f43f5e'
      ctx.lineWidth = 3
      ctx.strokeRect(330, 205, 90, 75)

      // Label Badge
      ctx.fillStyle = '#f43f5e'
      ctx.fillRect(330, 185, 150, 18)
      ctx.fillStyle = '#ffffff'
      ctx.font = 'bold 10px monospace'
      ctx.fillText('CAR-BIKE ACCIDENT 89%', 334, 198)

      // Emit Incident Message once
      if (frame >= 115 && !incident2Triggered) {
        incident2Triggered = true
        const keyframe = canvas.toDataURL('image/jpeg', 0.8)
        const inc2: StreamIncidentMessage = {
          type: 'incident',
          incident_id: 'INC-DEMO-2026-002',
          class_name: 'car_bike_accident',
          max_confidence: 0.89,
          start_timestamp: new Date(Date.now() + 4500).toISOString(),
          end_timestamp: new Date(Date.now() + 6600).toISOString(),
          start_time: 3.9,
          end_time: 6.0,
          duration: 2.1,
          keyframe_url: keyframe,
        }
        callbacks.onIncident(inc2)
      }
    }

    // 6. CCTV OSD Watermark Overlay
    ctx.fillStyle = 'rgba(0, 0, 0, 0.65)'
    ctx.fillRect(0, 0, canvas.width, 24)
    ctx.fillRect(0, canvas.height - 22, canvas.width, 22)

    ctx.fillStyle = '#06b6d4'
    ctx.font = 'bold 10px monospace'
    ctx.fillText('CAM-04 | DOWNTOWN INTERSECTION N-32 | DEMO REPLAY', 12, 16)

    // Timestamp & Frame HUD
    const nowStr = new Date().toISOString().replace('T', ' ').slice(0, 19)
    ctx.fillStyle = '#94a3b8'
    ctx.fillText(`${nowStr}  |  FR: ${frame}/${TOTAL_FRAMES}  |  28 FPS`, 12, canvas.height - 7)

    // Red REC blip
    ctx.fillStyle = frame % 20 < 10 ? '#ef4444' : '#4b5563'
    ctx.beginPath()
    ctx.arc(canvas.width - 24, 12, 5, 0, Math.PI * 2)
    ctx.fill()
    ctx.fillStyle = '#f8fafc'
    ctx.fillText('REC', canvas.width - 50, 16)

    return canvas.toDataURL('image/jpeg', 0.75)
  }

  const step = () => {
    if (isCancelled) return

    currentFrame++
    const dataUrl = renderFrame(currentFrame)
    callbacks.onFrame(dataUrl, currentFrame, TOTAL_FRAMES, FPS)

    if (currentFrame >= TOTAL_FRAMES) {
      callbacks.onComplete({
        status: 'completed',
        frames_processed: TOTAL_FRAMES,
        incidents_detected: 2,
        total_frames: TOTAL_FRAMES,
        fps: 28.4,
        job_id: 'demo_presentation_job_01',
      })
      return
    }

    timerId = setTimeout(step, FRAME_INTERVAL_MS)
  }

  timerId = setTimeout(step, FRAME_INTERVAL_MS)

  // Return cancel function
  return () => {
    isCancelled = true
    if (timerId) clearTimeout(timerId)
  }
}
