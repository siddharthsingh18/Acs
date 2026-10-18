import React, { useEffect, useRef, useState, useMemo, useCallback } from 'react'
import { useTranslation } from 'react-i18next'
import { useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import L from '@/lib/leafletSetup'
import 'leaflet/dist/leaflet.css'
import 'leaflet.markercluster'
import 'leaflet.markercluster/dist/MarkerCluster.css'
import 'leaflet.markercluster/dist/MarkerCluster.Default.css'
import 'leaflet.heat'

import {
  MapPin,
  Flame,
  Layers,
  Play,
  Pause,
  RotateCcw,
  ShieldAlert,
  Crosshair,
  AlertTriangle,
  Clock,
  Sparkles,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { api } from '@/api/client'
import { IncidentDrawer } from '@/components/incidents/IncidentDrawer'
import { cn } from '@/lib/utils'

// Helper: Custom SVG DivIcon for CCTV Camera Marker
function createCameraMarkerIcon(status: 'connected' | 'disconnected' | 'incident') {
  const isIncident = status === 'incident'
  const isConnected = status === 'connected'

  const bgColor = isIncident ? '#ef4444' : isConnected ? '#10b981' : '#64748b'
  const pulseColor = isIncident ? 'rgba(239, 68, 68, 0.45)' : isConnected ? 'rgba(16, 185, 129, 0.3)' : 'transparent'

  return L.divIcon({
    className: 'acs-camera-map-marker',
    html: `
      <div style="position: relative; width: 40px; height: 40px; display: flex; align-items: center; justify-content: center; cursor: pointer;">
        ${
          isIncident
            ? `<div style="position: absolute; width: 44px; height: 44px; border-radius: 9999px; background: ${pulseColor}; animation: ping 1.5s cubic-bezier(0, 0, 0.2, 1) infinite;"></div>`
            : ''
        }
        ${
          isConnected && !isIncident
            ? `<div style="position: absolute; width: 38px; height: 38px; border-radius: 9999px; background: ${pulseColor};"></div>`
            : ''
        }
        <div style="position: relative; width: 30px; height: 30px; border-radius: 9999px; background: ${bgColor}; border: 2.5px solid #ffffff; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 14px rgba(0,0,0,0.6);">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z"/>
            <circle cx="12" cy="13" r="3"/>
          </svg>
        </div>
      </div>
    `,
    iconSize: [40, 40],
    iconAnchor: [20, 20],
    popupAnchor: [0, -20],
  })
}

// Helper: Custom Pulsing Radar Marker for Incidents
function createIncidentMarkerIcon(severity: string, isPulsing: boolean = true) {
  const isHigh = severity === 'high' || severity === 'critical'
  const isMed = severity === 'medium'
  const color = isHigh ? '#ef4444' : isMed ? '#f59e0b' : '#06b6d4'
  const glow = isHigh ? 'rgba(239, 68, 68, 0.5)' : isMed ? 'rgba(245, 158, 11, 0.45)' : 'rgba(6, 182, 212, 0.4)'

  return L.divIcon({
    className: 'acs-incident-radar-marker',
    html: `
      <div style="position: relative; width: 48px; height: 48px; display: flex; align-items: center; justify-content: center; cursor: pointer;">
        ${
          isPulsing
            ? `<div style="position: absolute; width: 48px; height: 48px; border-radius: 9999px; background: ${glow}; animation: ping 1.2s cubic-bezier(0, 0, 0.2, 1) infinite;"></div>
               <div style="position: absolute; width: 36px; height: 36px; border-radius: 9999px; border: 2px solid ${color}; animation: pulse 2s infinite;"></div>`
            : ''
        }
        <div style="position: relative; width: 28px; height: 28px; border-radius: 9999px; background: ${color}; border: 2.5px solid #ffffff; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 16px rgba(0,0,0,0.7);">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/>
            <line x1="12" y1="9" x2="12" y2="13"/>
            <line x1="12" y1="17" x2="12.01" y2="17"/>
          </svg>
        </div>
      </div>
    `,
    iconSize: [48, 48],
    iconAnchor: [24, 24],
    popupAnchor: [0, -24],
  })
}

// Custom Cluster Icon Formatter
function createClusterIcon(cluster: any) {
  const count = cluster.getChildCount()
  return L.divIcon({
    className: 'acs-cluster-icon',
    html: `
      <div style="position: relative; width: 44px; height: 44px; border-radius: 9999px; background: rgba(15, 23, 42, 0.85); border: 2.5px solid #06b6d4; display: flex; align-items: center; justify-content: center; color: #38bdf8; font-weight: 800; font-family: monospace; font-size: 13px; box-shadow: 0 0 16px rgba(6, 182, 212, 0.4); cursor: pointer;">
        <span>${count}</span>
      </div>
    `,
    iconSize: [44, 44],
    iconAnchor: [22, 22],
  })
}

export const MapPage: React.FC = () => {
  const { t, i18n } = useTranslation()
  const navigate = useNavigate()
  const isArabic = i18n.language === 'ar'

  // Map DOM and Instance references
  const mapContainerRef = useRef<HTMLDivElement>(null)
  const mapRef = useRef<L.Map | null>(null)
  const cameraLayerGroupRef = useRef<any>(null)
  const incidentLayerGroupRef = useRef<L.LayerGroup | null>(null)
  const heatLayerRef = useRef<any>(null)

  // Incident drawer state
  const [selectedIncidentId, setSelectedIncidentId] = useState<string | null>(null)

  // Layer toggles
  const [showHeatmap, setShowHeatmap] = useState<boolean>(true)
  const [enableClustering, setEnableClustering] = useState<boolean>(true)
  const [showIncidents, setShowIncidents] = useState<boolean>(true)
  const [tacticalTheme, setTacticalTheme] = useState<boolean>(true)

  // Timeline / Time-Slider Replay State
  const [replayMode, setReplayMode] = useState<boolean>(false)
  const [isPlaying, setIsPlaying] = useState<boolean>(false)
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1)
  const [timelineIndex, setTimelineIndex] = useState<number>(100) // 0 to 100 percentage
  const playTimerRef = useRef<any>(null)

  // Queries
  const { data: cameras = [] } = useQuery({
    queryKey: ['cameras'],
    queryFn: () => api.getCameras(),
    refetchInterval: 6000,
  })

  const { data: incidentsData } = useQuery({
    queryKey: ['incidents', 'map', 1, 100],
    queryFn: () => api.getIncidents({ page: 1, page_size: 100 }),
    refetchInterval: 8000,
  })

  const incidents = useMemo(() => {
    return incidentsData?.items || []
  }, [incidentsData])

  // Coordinate Mapping: Map cameras with coordinates, or assign deterministic city coordinates
  const mappedCameras = useMemo(() => {
    const baseLat = 24.7136
    const baseLng = 46.6753

    return cameras.map((cam, idx) => {
      let lat = cam.location_lat
      let lng = cam.location_lng

      // If camera doesn't have coordinates set yet, distribute across city corridor for demo/inspection
      if (!lat || !lng) {
        const offset = 0.012 * (idx + 1)
        const angle = (idx * 2 * Math.PI) / Math.max(cameras.length, 1)
        lat = Number((baseLat + offset * Math.cos(angle)).toFixed(6))
        lng = Number((baseLng + offset * Math.sin(angle)).toFixed(6))
      }

      const isIncidentActive = cam.health?.is_incident_active || false
      const isConnected = cam.health?.connected || false

      const status: 'connected' | 'disconnected' | 'incident' = isIncidentActive
        ? 'incident'
        : isConnected
        ? 'connected'
        : 'disconnected'

      return {
        ...cam,
        resolvedLat: lat,
        resolvedLng: lng,
        status,
      }
    })
  }, [cameras])

  // Map incidents to locations: link incident to camera coordinates
  const mappedIncidents = useMemo(() => {
    if (!incidents.length) return []

    // Sort incidents chronologically by created_at
    const sorted = [...incidents].sort((a, b) => a.created_at - b.created_at)

    return sorted.map((inc, idx) => {
      // Find matching camera or default to camera list round-robin
      let lat: number | null = null
      let lng: number | null = null

      if (mappedCameras.length > 0) {
        const camIdx = idx % mappedCameras.length
        lat = mappedCameras[camIdx].resolvedLat
        lng = mappedCameras[camIdx].resolvedLng
      } else {
        lat = 24.7136 + (idx % 5) * 0.008
        lng = 46.6753 + (idx % 5) * 0.008
      }

      return {
        ...inc,
        resolvedLat: lat,
        resolvedLng: lng,
      }
    })
  }, [incidents, mappedCameras])

  // Filtered incidents based on Replay Mode and Timeline Slider
  const visibleIncidents = useMemo(() => {
    if (!mappedIncidents.length) return []
    if (!replayMode) return mappedIncidents

    // Calculate cut-off time based on timelineIndex percentage
    const minTime = mappedIncidents[0].created_at
    const maxTime = mappedIncidents[mappedIncidents.length - 1].created_at
    if (minTime === maxTime) return mappedIncidents

    const currentTime = minTime + ((maxTime - minTime) * timelineIndex) / 100
    return mappedIncidents.filter((inc) => inc.created_at <= currentTime)
  }, [mappedIncidents, replayMode, timelineIndex])

  // Current scrubbed timestamp string
  const scrubbedTimeLabel = useMemo(() => {
    if (!mappedIncidents.length) return ''
    const minTime = mappedIncidents[0].created_at
    const maxTime = mappedIncidents[mappedIncidents.length - 1].created_at
    const currentEpoch = minTime + ((maxTime - minTime) * timelineIndex) / 100
    return new Date(currentEpoch * 1000).toLocaleString(isArabic ? 'ar-SA' : 'en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    })
  }, [mappedIncidents, timelineIndex, isArabic])

  // Initialize Map
  useEffect(() => {
    if (!mapContainerRef.current || mapRef.current) return

    const initialLat = mappedCameras[0]?.resolvedLat || 24.7136
    const initialLng = mappedCameras[0]?.resolvedLng || 46.6753

    const map = L.map(mapContainerRef.current, {
      center: [initialLat, initialLng],
      zoom: 13,
      zoomControl: false, // Custom placed zoom control
      attributionControl: true,
    })

    // OpenStreetMap standard tiles (No paid API keys required)
    const osmLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      attribution:
        '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap</a> contributors',
    })
    osmLayer.addTo(map)

    // Add zoom controls to top-end
    L.control.zoom({ position: 'topright' }).addTo(map)

    // Initialize layer groups
    incidentLayerGroupRef.current = L.layerGroup().addTo(map)

    mapRef.current = map

    const timer = setTimeout(() => {
      map.invalidateSize()
    }, 200)

    return () => {
      clearTimeout(timer)
      if (mapRef.current) {
        mapRef.current.remove()
        mapRef.current = null
      }
    }
  }, [])

  // Manage Camera Markers & Clustering Layer
  useEffect(() => {
    const map = mapRef.current
    if (!map) return

    // Remove existing camera layer
    if (cameraLayerGroupRef.current) {
      map.removeLayer(cameraLayerGroupRef.current)
      cameraLayerGroupRef.current = null
    }

    // Create either cluster group or regular layer group
    const cameraGroup = enableClustering
      ? (L as any).markerClusterGroup({
          showCoverageOnHover: false,
          maxClusterRadius: 45,
          spiderfyOnMaxZoom: true,
          iconCreateFunction: createClusterIcon,
        })
      : L.layerGroup()

    mappedCameras.forEach((cam) => {
      const marker = L.marker([cam.resolvedLat, cam.resolvedLng], {
        icon: createCameraMarkerIcon(cam.status),
      })

      // Popup content
      const popupHtml = `
        <div style="font-family: system-ui, -apple-system, sans-serif; min-width: 210px; padding: 2px;">
          <div style="font-weight: 700; font-size: 13px; color: #0f172a; margin-bottom: 4px; display: flex; align-items: center; justify-content: space-between;">
            <span>${cam.name}</span>
            <span style="font-size: 9px; padding: 2px 6px; border-radius: 9999px; font-weight: 800; background: ${
              cam.status === 'incident' ? '#fee2e2; color: #dc2626' : cam.status === 'connected' ? '#dcfce7; color: #16a34a' : '#f1f5f9; color: #475569'
            };">
              ${cam.status.toUpperCase()}
            </span>
          </div>
          <div style="font-size: 11px; color: #64748b; margin-bottom: 6px;">
            <div>FPS: <strong>${cam.health?.fps?.toFixed(1) || '0.0'}</strong> | Source: <strong>${cam.source_type}</strong></div>
            <div style="font-family: monospace; font-size: 10px; margin-top: 2px;">
              Lat: ${cam.resolvedLat.toFixed(5)}, Lng: ${cam.resolvedLng.toFixed(5)}
            </div>
          </div>
          <button id="btn-focus-${cam.id}" style="width: 100%; cursor: pointer; padding: 5px 8px; border-radius: 6px; background: #06b6d4; color: #ffffff; font-size: 11px; font-weight: 700; border: none; transition: opacity 0.2s;">
            ${t('map.inspect_camera', 'Focus in Camera Wall')}
          </button>
        </div>
      `
      marker.bindPopup(popupHtml)

      marker.on('popupopen', () => {
        const btn = document.getElementById(`btn-focus-${cam.id}`)
        if (btn) {
          btn.onclick = () => {
            navigate('/live')
          }
        }
      })

      cameraGroup.addLayer(marker)
    })

    cameraGroup.addTo(map)
    cameraLayerGroupRef.current = cameraGroup
  }, [mappedCameras, enableClustering, navigate, t])

  // Manage Incident Markers & Incident Drawer Links
  useEffect(() => {
    const map = mapRef.current
    const incidentGroup = incidentLayerGroupRef.current
    if (!map || !incidentGroup) return

    incidentGroup.clearLayers()

    if (!showIncidents) return

    visibleIncidents.forEach((inc, idx) => {
      // The newest visible incident or active incident gets intense pulse
      const isLatest = idx === visibleIncidents.length - 1 && replayMode
      const marker = L.marker([inc.resolvedLat, inc.resolvedLng], {
        icon: createIncidentMarkerIcon(inc.severity, isLatest || inc.status === 'pending'),
      })

      // Clicking opens IncidentDrawer directly
      marker.on('click', () => {
        setSelectedIncidentId(inc.incident_id)
      })

      const popupHtml = `
        <div style="font-family: system-ui, -apple-system, sans-serif; min-width: 220px; padding: 2px;">
          <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 4px;">
            <span style="font-family: monospace; font-size: 11px; font-weight: 800; color: #ef4444;">${inc.incident_id}</span>
            <span style="font-size: 9px; padding: 2px 6px; border-radius: 9999px; font-weight: 800; background: #fee2e2; color: #dc2626;">
              ${inc.severity.toUpperCase()} (${inc.severity_score || 50}/100)
            </span>
          </div>
          <div style="font-weight: 700; font-size: 12px; color: #0f172a; margin-bottom: 2px;">
            ${inc.class_name.replace('_', ' ')}
          </div>
          <div style="font-size: 10px; color: #64748b; margin-bottom: 6px;">
            ${new Date(inc.created_at * 1000).toLocaleString()}
          </div>
          <button id="btn-inspect-${inc.incident_id}" style="width: 100%; cursor: pointer; padding: 5px 8px; border-radius: 6px; background: #dc2626; color: #ffffff; font-size: 11px; font-weight: 700; border: none;">
            ${t('map.inspect_incident', 'Inspect in Incident Drawer')}
          </button>
        </div>
      `
      marker.bindPopup(popupHtml)

      marker.on('popupopen', () => {
        const btn = document.getElementById(`btn-inspect-${inc.incident_id}`)
        if (btn) {
          btn.onclick = () => {
            setSelectedIncidentId(inc.incident_id)
          }
        }
      })

      incidentGroup.addLayer(marker)
    })
  }, [visibleIncidents, showIncidents, replayMode, t])

  // Manage Heatmap Layer for Incident Hotspots
  useEffect(() => {
    const map = mapRef.current
    if (!map) return

    if (heatLayerRef.current) {
      map.removeLayer(heatLayerRef.current)
      heatLayerRef.current = null
    }

    if (showHeatmap && mappedIncidents.length > 0) {
      // Points format: [lat, lng, intensity]
      const heatPoints = mappedIncidents.map((inc) => {
        const severityWeight = inc.severity === 'high' ? 1.0 : inc.severity === 'medium' ? 0.7 : 0.4
        return [inc.resolvedLat, inc.resolvedLng, severityWeight] as [number, number, number]
      })

      const heatLayer = (L as any).heatLayer(heatPoints, {
        radius: 32,
        blur: 20,
        maxZoom: 16,
        minOpacity: 0.35,
        gradient: {
          0.2: '#06b6d4',
          0.4: '#10b981',
          0.6: '#f59e0b',
          0.8: '#ef4444',
          1.0: '#991b1b',
        },
      })

      heatLayer.addTo(map)
      heatLayerRef.current = heatLayer
    }
  }, [mappedIncidents, showHeatmap])

  // Center Map on all cameras and incidents bounds
  const centerNetwork = useCallback(() => {
    if (!mapRef.current || mappedCameras.length === 0) return
    const latLngs = mappedCameras.map((c) => [c.resolvedLat, c.resolvedLng] as [number, number])
    const bounds = L.latLngBounds(latLngs)
    mapRef.current.fitBounds(bounds, { padding: [50, 50], maxZoom: 15, animate: true })
  }, [mappedCameras])

  // Replay Timer Hook
  useEffect(() => {
    if (!isPlaying) {
      if (playTimerRef.current) clearInterval(playTimerRef.current)
      return
    }

    const intervalMs = Math.max(50, Math.floor(600 / playbackSpeed))

    playTimerRef.current = setInterval(() => {
      setTimelineIndex((prev) => {
        if (prev >= 100) {
          setIsPlaying(false)
          return 100
        }
        return Math.min(100, prev + 1)
      })
    }, intervalMs)

    return () => {
      if (playTimerRef.current) clearInterval(playTimerRef.current)
    }
  }, [isPlaying, playbackSpeed])

  const connectedCount = mappedCameras.filter((c) => c.status === 'connected').length
  const activeIncidentsCount = mappedCameras.filter((c) => c.status === 'incident').length

  return (
    <div className="relative w-full h-[calc(100vh-4rem)] flex flex-col overflow-hidden bg-slate-950">
      {/* Top Floating Control Bar */}
      <div className="absolute top-3 inset-x-3 z-30 pointer-events-none flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2.5">
        {/* Title and KPIs */}
        <div className="pointer-events-auto flex items-center gap-2.5 p-2 px-3.5 rounded-xl bg-card/90 backdrop-blur-md border border-border shadow-xl">
          <div className="h-8 w-8 rounded-lg bg-primary/20 border border-primary/40 flex items-center justify-center text-primary shrink-0">
            <MapPin className="h-4 w-4" />
          </div>
          <div>
            <h1 className="text-sm font-extrabold text-foreground tracking-tight flex items-center gap-1.5 leading-tight">
              {t('map.title')}
              <Badge variant="outline" className="text-[9px] uppercase font-mono px-1.5 py-0 border-cyan-500/40 text-cyan-400">
                OSM Tiles
              </Badge>
            </h1>
            <div className="flex items-center gap-3 text-[11px] text-muted-foreground font-mono mt-0.5">
              <span className="flex items-center gap-1">
                <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
                {connectedCount}/{mappedCameras.length} {t('map.cameras_online')}
              </span>
              <span className="text-border">|</span>
              <span className="flex items-center gap-1 text-rose-400 font-bold">
                <AlertTriangle className="h-3 w-3" />
                {activeIncidentsCount} {t('map.active_incidents')}
              </span>
            </div>
          </div>
        </div>

        {/* Layer Toggles Panel */}
        <div className="pointer-events-auto flex items-center gap-1.5 p-1.5 rounded-xl bg-card/90 backdrop-blur-md border border-border shadow-xl">
          <Button
            type="button"
            variant={showHeatmap ? 'default' : 'ghost'}
            size="sm"
            onClick={() => setShowHeatmap(!showHeatmap)}
            className="h-8 px-2.5 text-xs gap-1.5 shadow-none"
            title="Toggle Collision Hotspot Heatmap"
          >
            <Flame className={cn('h-3.5 w-3.5', showHeatmap ? 'text-amber-300' : 'text-muted-foreground')} />
            <span className="hidden md:inline">{t('map.toggle_heatmap')}</span>
          </Button>

          <Button
            type="button"
            variant={enableClustering ? 'default' : 'ghost'}
            size="sm"
            onClick={() => setEnableClustering(!enableClustering)}
            className="h-8 px-2.5 text-xs gap-1.5 shadow-none"
            title="Toggle Marker Clustering"
          >
            <Layers className="h-3.5 w-3.5" />
            <span className="hidden md:inline">{t('map.toggle_clusters')}</span>
          </Button>

          <Button
            type="button"
            variant={showIncidents ? 'default' : 'ghost'}
            size="sm"
            onClick={() => setShowIncidents(!showIncidents)}
            className="h-8 px-2.5 text-xs gap-1.5 shadow-none"
            title="Toggle Incidents"
          >
            <ShieldAlert className="h-3.5 w-3.5" />
            <span className="hidden md:inline">{t('map.toggle_incidents')}</span>
          </Button>

          <Button
            type="button"
            variant={tacticalTheme ? 'secondary' : 'ghost'}
            size="sm"
            onClick={() => setTacticalTheme(!tacticalTheme)}
            className="h-8 px-2.5 text-xs gap-1.5"
            title="Toggle Tactical Dark Mode Tiles"
          >
            <Sparkles className="h-3.5 w-3.5 text-cyan-400" />
            <span className="hidden md:inline">{t('map.toggle_dark')}</span>
          </Button>

          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={centerNetwork}
            className="h-8 px-2 text-xs"
            title="Center Network"
          >
            <Crosshair className="h-4 w-4 text-primary" />
          </Button>
        </div>
      </div>

      {/* Main Map Viewport - STRICTLY DIR="LTR" FOR LEAFLET INTERNAL MATH */}
      <div
        className={cn(
          'relative w-full flex-1 overflow-hidden select-none',
          tacticalTheme && 'acs-tactical-map-filter'
        )}
      >
        <div
          ref={mapContainerRef}
          id="tactical-map-viewport"
          dir="ltr"
          style={{ direction: 'ltr' }}
          className="w-full h-full z-10"
        />
      </div>

      {/* Floating Bottom Chronological Incident Replay Slider */}
      <div className="absolute bottom-4 inset-x-4 z-30 pointer-events-none flex justify-center">
        <div className="pointer-events-auto w-full max-w-3xl p-3.5 rounded-2xl bg-card/95 backdrop-blur-xl border border-border shadow-2xl space-y-2">
          {/* Header Row: Replay Mode Controls and Speed */}
          <div className="flex items-center justify-between gap-3 text-xs">
            <div className="flex items-center gap-2">
              <Clock className="h-4 w-4 text-cyan-400" />
              <span className="font-bold text-foreground">{t('map.time_slider_title')}</span>
              {replayMode && (
                <Badge variant="outline" className="text-[10px] text-cyan-400 border-cyan-500/40 bg-cyan-950/30">
                  {scrubbedTimeLabel}
                </Badge>
              )}
            </div>

            <div className="flex items-center gap-2">
              <span className="text-[11px] font-mono text-muted-foreground">
                {t('map.visible_incidents', {
                  current: visibleIncidents.length,
                  total: mappedIncidents.length,
                })}
              </span>

              {/* Mode Toggle Button */}
              <button
                type="button"
                onClick={() => {
                  setReplayMode(!replayMode)
                  if (!replayMode) {
                    setTimelineIndex(50)
                  }
                }}
                className={cn(
                  'px-2.5 py-1 rounded-lg text-[11px] font-semibold border transition-colors cursor-pointer',
                  replayMode
                    ? 'border-primary bg-primary/20 text-primary'
                    : 'border-border bg-muted/40 hover:bg-muted text-muted-foreground'
                )}
              >
                {replayMode ? t('map.replay_mode') : t('map.show_all')}
              </button>
            </div>
          </div>

          {/* Interactive Scrub Slider */}
          <div className="flex items-center gap-3">
            <Button
              type="button"
              variant={isPlaying ? 'default' : 'secondary'}
              size="sm"
              onClick={() => {
                if (!replayMode) setReplayMode(true)
                setIsPlaying(!isPlaying)
              }}
              className="h-8 w-8 p-0 shrink-0"
              title={isPlaying ? t('map.pause') : t('map.play')}
            >
              {isPlaying ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5 fill-current" />}
            </Button>

            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={() => setTimelineIndex(0)}
              className="h-8 w-8 p-0 shrink-0 text-muted-foreground hover:text-foreground"
              title="Reset Timeline to Start"
            >
              <RotateCcw className="h-3.5 w-3.5" />
            </Button>

            <div className="relative flex-1 flex items-center">
              <input
                type="range"
                min={0}
                max={100}
                step={1}
                value={timelineIndex}
                disabled={!replayMode && !isPlaying}
                onChange={(e) => {
                  if (!replayMode) setReplayMode(true)
                  setTimelineIndex(parseInt(e.target.value))
                }}
                className="w-full h-2 bg-muted rounded-lg appearance-none cursor-pointer accent-cyan-400 focus:outline-none"
              />
            </div>

            {/* Playback Speed Multipliers */}
            <div className="flex items-center gap-1 shrink-0 font-mono text-[10px]">
              {[1, 2, 5, 10].map((speed) => (
                <button
                  key={speed}
                  type="button"
                  onClick={() => setPlaybackSpeed(speed)}
                  className={cn(
                    'px-2 py-1 rounded border transition-colors cursor-pointer',
                    playbackSpeed === speed
                      ? 'border-cyan-500 bg-cyan-950/40 text-cyan-400 font-bold'
                      : 'border-border bg-card/60 text-muted-foreground hover:bg-muted'
                  )}
                >
                  {speed}x
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Incident Detail Drawer Integration */}
      <IncidentDrawer
        incidentId={selectedIncidentId}
        isOpen={!!selectedIncidentId}
        onClose={() => setSelectedIncidentId(null)}
        incidentList={mappedIncidents.map((i) => ({ incident_id: i.incident_id }))}
        onSelectIncident={setSelectedIncidentId}
      />
    </div>
  )
}

export default MapPage
