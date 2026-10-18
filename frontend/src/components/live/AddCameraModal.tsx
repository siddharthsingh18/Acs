import React, { useState } from 'react'
import { Plus, X, Video, Globe, Camera as CameraIcon } from 'lucide-react'
import type { CameraCreatePayload } from '@/api/client'

interface AddCameraModalProps {
  isOpen: boolean
  onClose: () => void
  onAddCamera: (payload: CameraCreatePayload) => Promise<void>
}

export const AddCameraModal: React.FC<AddCameraModalProps> = ({
  isOpen,
  onClose,
  onAddCamera,
}) => {
  const [name, setName] = useState('')
  const [sourceUrl, setSourceUrl] = useState('')
  const [lat, setLat] = useState('')
  const [lng, setLng] = useState('')
  const [confidence, setConfidence] = useState('0.55')
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  if (!isOpen) return null

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!name.trim() || !sourceUrl.trim()) {
      setError('Camera name and source URL are required.')
      return
    }

    setIsSubmitting(true)
    setError(null)

    try {
      await onAddCamera({
        name: name.trim(),
        source_url: sourceUrl.trim(),
        location_lat: lat ? parseFloat(lat) : null,
        location_lng: lng ? parseFloat(lng) : null,
        confidence_override: confidence ? parseFloat(confidence) : 0.55,
        enabled: true,
      })
      onClose()
      setName('')
      setSourceUrl('')
      setLat('')
      setLng('')
    } catch (err: any) {
      setError(err.message || 'Failed to register camera.')
    } finally {
      setIsSubmitting(false)
    }
  }

  const setSampleRTSP = () => {
    setName('Metropolitan Highway Cam 1')
    setSourceUrl('rtsp://127.0.0.1:8554/mystream')
    setLat('37.7749')
    setLng('-122.4194')
  }

  const setSampleWebcam = () => {
    setName('Traffic Desk USB Webcam')
    setSourceUrl('0')
  }

  const setSampleVideo = () => {
    setName('High-Density Arterial Intersection')
    setSourceUrl('acc_ex.mp4')
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="relative w-full max-w-lg rounded-2xl border border-border bg-card p-6 shadow-2xl space-y-5">
        {/* Modal Header */}
        <div className="flex items-center justify-between border-b border-border/60 pb-3">
          <div className="flex items-center gap-2">
            <CameraIcon className="h-5 w-5 text-primary" />
            <h2 className="text-lg font-bold text-foreground">Register New CCTV Camera</h2>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {error && (
          <div className="p-3 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs font-medium">
            {error}
          </div>
        )}

        {/* Quick Presets */}
        <div className="space-y-1.5">
          <label className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
            Quick Presets
          </label>
          <div className="grid grid-cols-3 gap-2">
            <button
              type="button"
              onClick={setSampleRTSP}
              className="flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-lg border border-border bg-muted/40 hover:bg-muted text-xs font-medium text-foreground transition-colors"
            >
              <Globe className="h-3.5 w-3.5 text-blue-400" />
              <span>RTSP Feed</span>
            </button>
            <button
              type="button"
              onClick={setSampleWebcam}
              className="flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-lg border border-border bg-muted/40 hover:bg-muted text-xs font-medium text-foreground transition-colors"
            >
              <CameraIcon className="h-3.5 w-3.5 text-emerald-400" />
              <span>Webcam 0</span>
            </button>
            <button
              type="button"
              onClick={setSampleVideo}
              className="flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-lg border border-border bg-muted/40 hover:bg-muted text-xs font-medium text-foreground transition-colors"
            >
              <Video className="h-3.5 w-3.5 text-purple-400" />
              <span>Sample File</span>
            </button>
          </div>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-1.5">
            <label className="text-xs font-semibold text-foreground">Camera Name *</label>
            <input
              type="text"
              required
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. Highway 101 Northbound"
              className="w-full px-3 py-2 rounded-lg border border-border bg-background text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
            />
          </div>

          <div className="space-y-1.5">
            <label className="text-xs font-semibold text-foreground">Source Connection String / URL *</label>
            <input
              type="text"
              required
              value={sourceUrl}
              onChange={(e) => setSourceUrl(e.target.value)}
              placeholder="rtsp://user:pass@host:554/live, 0 (webcam), or acc_ex.mp4"
              className="w-full px-3 py-2 rounded-lg border border-border bg-background text-sm font-mono text-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
            />
            <p className="text-[11px] text-muted-foreground font-mono">
              Credentials are encrypted at rest and never exposed in API responses or logs.
            </p>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1.5">
              <label className="text-xs font-semibold text-foreground">Latitude (optional)</label>
              <input
                type="number"
                step="any"
                value={lat}
                onChange={(e) => setLat(e.target.value)}
                placeholder="37.7749"
                className="w-full px-3 py-2 rounded-lg border border-border bg-background text-sm font-mono text-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
              />
            </div>
            <div className="space-y-1.5">
              <label className="text-xs font-semibold text-foreground">Longitude (optional)</label>
              <input
                type="number"
                step="any"
                value={lng}
                onChange={(e) => setLng(e.target.value)}
                placeholder="-122.4194"
                className="w-full px-3 py-2 rounded-lg border border-border bg-background text-sm font-mono text-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
              />
            </div>
          </div>

          <div className="space-y-1.5">
            <label className="text-xs font-semibold text-foreground">Confidence Threshold Override</label>
            <input
              type="number"
              min="0.05"
              max="1.0"
              step="0.05"
              value={confidence}
              onChange={(e) => setConfidence(e.target.value)}
              className="w-full px-3 py-2 rounded-lg border border-border bg-background text-sm font-mono text-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
            />
          </div>

          <div className="flex items-center justify-end gap-2.5 pt-3 border-t border-border/60">
            <button
              type="button"
              onClick={onClose}
              disabled={isSubmitting}
              className="px-4 py-2 rounded-lg border border-border bg-muted/30 hover:bg-muted text-sm font-semibold transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isSubmitting}
              className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-primary hover:bg-primary/90 text-primary-foreground text-sm font-semibold shadow transition-colors disabled:opacity-50"
            >
              <Plus className="h-4 w-4" />
              <span>{isSubmitting ? 'Registering...' : 'Register Camera'}</span>
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
