import React, { useEffect, useRef, useState, useCallback } from 'react'
import L from '@/lib/leafletSetup'
import 'leaflet/dist/leaflet.css'
import { MapPin, Navigation, Crosshair, Trash2, CheckCircle2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'

export interface CameraLocationPickerProps {
  lat?: number | null
  lng?: number | null
  onChange: (coords: { lat: number; lng: number } | null) => void
  readOnly?: boolean
  className?: string
  height?: string
}

// Custom DivIcon so we don't rely on Leaflet's missing PNG assets in Vite
const createCameraPickerIcon = () => {
  return L.divIcon({
    className: 'acs-camera-picker-icon',
    html: `
      <div style="position: relative; width: 36px; height: 36px; display: flex; align-items: center; justify-content: center; cursor: grab;">
        <div style="position: absolute; width: 36px; height: 36px; border-radius: 9999px; background: rgba(6, 182, 212, 0.35); animation: ping 2s cubic-bezier(0, 0, 0.2, 1) infinite;"></div>
        <div style="position: relative; width: 28px; height: 28px; border-radius: 9999px; background: #06b6d4; border: 2.5px solid #ffffff; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 14px rgba(0,0,0,0.5);">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z"/>
            <circle cx="12" cy="13" r="3"/>
          </svg>
        </div>
      </div>
    `,
    iconSize: [36, 36],
    iconAnchor: [18, 18],
    popupAnchor: [0, -18],
  })
}

// Preset locations for fast selection
const PRESETS = [
  { label: 'Riyadh HQ', lat: 24.7136, lng: 46.6753 },
  { label: 'Olaya Corridor', lat: 24.7003, lng: 46.6841 },
  { label: 'SF Highway 101', lat: 37.7749, lng: -122.4194 },
  { label: 'Arterial Junction', lat: 37.7833, lng: -122.4167 },
]

export const CameraLocationPicker: React.FC<CameraLocationPickerProps> = ({
  lat,
  lng,
  onChange,
  readOnly = false,
  className,
  height = 'h-64',
}) => {
  const containerRef = useRef<HTMLDivElement>(null)
  const mapInstanceRef = useRef<L.Map | null>(null)
  const markerRef = useRef<L.Marker | null>(null)
  const [geoLocating, setGeoLocating] = useState(false)

  const hasCoords = typeof lat === 'number' && !isNaN(lat) && typeof lng === 'number' && !isNaN(lng)

  // Initialize Leaflet Map
  useEffect(() => {
    if (!containerRef.current || mapInstanceRef.current) return

    const initialLat = hasCoords ? (lat as number) : 24.7136
    const initialLng = hasCoords ? (lng as number) : 46.6753
    const initialZoom = hasCoords ? 15 : 12

    const map = L.map(containerRef.current, {
      center: [initialLat, initialLng],
      zoom: initialZoom,
      zoomControl: true,
      attributionControl: true,
    })

    // OpenStreetMap standard tiles (no paid keys required)
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      attribution:
        '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap</a>',
    }).addTo(map)

    mapInstanceRef.current = map

    // Click on map to place/move marker
    if (!readOnly) {
      map.on('click', (e: L.LeafletMouseEvent) => {
        const newLat = parseFloat(e.latlng.lat.toFixed(6))
        const newLng = parseFloat(e.latlng.lng.toFixed(6))
        onChange({ lat: newLat, lng: newLng })
      })
    }

    // Invalidate map size after mount for accurate layout
    const timer = setTimeout(() => {
      map.invalidateSize()
    }, 150)

    return () => {
      clearTimeout(timer)
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove()
        mapInstanceRef.current = null
      }
    }
  }, [])

  // Sync marker when lat/lng props change
  useEffect(() => {
    const map = mapInstanceRef.current
    if (!map) return

    if (hasCoords) {
      const targetLatLng: L.LatLngExpression = [lat as number, lng as number]

      if (markerRef.current) {
        markerRef.current.setLatLng(targetLatLng)
      } else {
        const marker = L.marker(targetLatLng, {
          icon: createCameraPickerIcon(),
          draggable: !readOnly,
        }).addTo(map)

        if (!readOnly) {
          marker.on('dragend', () => {
            const pos = marker.getLatLng()
            onChange({
              lat: parseFloat(pos.lat.toFixed(6)),
              lng: parseFloat(pos.lng.toFixed(6)),
            })
          })
        }

        markerRef.current = marker
      }
    } else {
      if (markerRef.current) {
        markerRef.current.remove()
        markerRef.current = null
      }
    }
  }, [lat, lng, hasCoords, readOnly, onChange])

  // Center map on coordinates when updated
  const centerOnMarker = useCallback(() => {
    if (mapInstanceRef.current && hasCoords) {
      mapInstanceRef.current.setView([lat as number, lng as number], 16, { animate: true })
    }
  }, [lat, lng, hasCoords])

  // Browser Geolocation
  const handleUseCurrentLocation = useCallback(() => {
    if (!navigator.geolocation) {
      alert('Geolocation is not supported by your browser.')
      return
    }

    setGeoLocating(true)
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        setGeoLocating(false)
        const newLat = parseFloat(pos.coords.latitude.toFixed(6))
        const newLng = parseFloat(pos.coords.longitude.toFixed(6))
        onChange({ lat: newLat, lng: newLng })
        if (mapInstanceRef.current) {
          mapInstanceRef.current.setView([newLat, newLng], 16, { animate: true })
        }
      },
      (err) => {
        setGeoLocating(false)
        console.warn('Geolocation failed:', err.message)
      },
      { enableHighAccuracy: true, timeout: 8000 }
    )
  }, [onChange])

  return (
    <div className={cn('space-y-2 select-none', className)}>
      {/* Location Picker Header & Presets */}
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
        <div className="flex items-center gap-2">
          <MapPin className="h-4 w-4 text-primary shrink-0" />
          <span className="font-semibold text-foreground">
            {hasCoords ? (
              <span className="font-mono text-cyan-400">
                Lat: {lat?.toFixed(5)}, Lng: {lng?.toFixed(5)}
              </span>
            ) : (
              <span className="text-muted-foreground">Click on map to place camera pin</span>
            )}
          </span>
          {hasCoords && (
            <Badge variant="outline" className="text-[10px] text-emerald-400 border-emerald-500/30 bg-emerald-500/10 gap-1">
              <CheckCircle2 className="h-3 w-3" />
              Pin Active
            </Badge>
          )}
        </div>

        {!readOnly && (
          <div className="flex items-center gap-1.5 flex-wrap">
            {hasCoords && (
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={centerOnMarker}
                className="h-7 px-2 text-[11px] gap-1"
                title="Center on pin"
              >
                <Crosshair className="h-3 w-3 text-primary" />
                <span>Center</span>
              </Button>
            )}

            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={handleUseCurrentLocation}
              disabled={geoLocating}
              className="h-7 px-2 text-[11px] gap-1"
              title="Use current GPS location"
            >
              <Navigation className={cn('h-3 w-3 text-blue-400', geoLocating && 'animate-spin')} />
              <span>{geoLocating ? 'Locating...' : 'My Location'}</span>
            </Button>

            {hasCoords && (
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={() => onChange(null)}
                className="h-7 px-2 text-[11px] text-rose-400 hover:text-rose-300 hover:bg-rose-500/10 gap-1"
                title="Remove marker"
              >
                <Trash2 className="h-3 w-3" />
                <span>Clear</span>
              </Button>
            )}
          </div>
        )}
      </div>

      {/* Quick Presets Bar */}
      {!readOnly && (
        <div className="flex items-center gap-1.5 overflow-x-auto pb-1 text-[11px] text-muted-foreground">
          <span className="text-[10px] uppercase font-mono tracking-wider shrink-0">Presets:</span>
          {PRESETS.map((p) => (
            <button
              key={p.label}
              type="button"
              onClick={() => {
                onChange({ lat: p.lat, lng: p.lng })
                if (mapInstanceRef.current) {
                  mapInstanceRef.current.setView([p.lat, p.lng], 15, { animate: true })
                }
              }}
              className="px-2 py-0.5 rounded border border-border bg-card/60 hover:bg-muted text-[10px] font-mono text-foreground shrink-0 transition-colors cursor-pointer"
            >
              {p.label}
            </button>
          ))}
        </div>
      )}

      {/* Map Container - STRICTLY DIR="LTR" to prevent Leaflet RTL inversion */}
      <div
        className={cn(
          'relative w-full rounded-xl overflow-hidden border border-border shadow-inner bg-slate-950',
          height
        )}
      >
        <div
          ref={containerRef}
          dir="ltr"
          style={{ direction: 'ltr' }}
          className="w-full h-full z-10"
        />

        {/* Guidance Watermark / Overlay */}
        {!hasCoords && (
          <div className="absolute inset-x-0 bottom-2 z-20 flex justify-center pointer-events-none">
            <span className="px-3 py-1 rounded-full bg-black/75 backdrop-blur-md text-[11px] font-medium text-slate-200 border border-white/10 shadow-lg">
              Click anywhere on the map to set camera coordinates
            </span>
          </div>
        )}
      </div>
    </div>
  )
}
export default CameraLocationPicker
