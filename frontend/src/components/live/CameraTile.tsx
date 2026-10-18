import React from 'react'
import {
  Maximize2,
  Minimize2,
  WifiOff,
  RefreshCw,
  AlertTriangle,
  Radio,
  Trash2,
} from 'lucide-react'
import { useCameraStream } from '@/hooks/useCameraStream'
import type { CameraItem } from '@/api/client'
import { cn } from '@/lib/utils'

interface CameraTileProps {
  camera: CameraItem
  isFocused?: boolean
  onFocus?: (camera: CameraItem) => void
  onDelete?: (camera: CameraItem) => void
}

export const CameraTile: React.FC<CameraTileProps> = ({
  camera,
  isFocused = false,
  onFocus,
  onDelete,
}) => {
  const { imageRef, status, fps, isIncidentActive, activeIncident } = useCameraStream(
    camera.id,
    camera.enabled
  )

  const handleTileClick = (e: React.MouseEvent) => {
    // If clicking a button directly, don't trigger tile focus
    if ((e.target as HTMLElement).closest('button')) {
      return
    }
    if (onFocus) {
      onFocus(camera)
    }
  }

  return (
    <div
      onClick={handleTileClick}
      data-testid={`camera-tile-${camera.id}`}
      className={cn(
        'group relative flex flex-col overflow-hidden rounded-xl border bg-card/90 transition-all duration-300 cursor-pointer select-none',
        // Red pulse and glow on active incident
        isIncidentActive
          ? 'border-rose-500 shadow-[0_0_25px_rgba(244,63,94,0.45)] ring-2 ring-rose-500 animate-pulse'
          : 'border-border/60 hover:border-primary/50 hover:shadow-lg',
        isFocused ? 'ring-2 ring-primary/80 shadow-2xl' : ''
      )}
    >
      {/* Top Telemetry Header Bar */}
      <div className="absolute top-0 inset-x-0 z-20 flex items-center justify-between px-3 py-2 bg-gradient-to-b from-black/80 via-black/40 to-transparent pointer-events-none">
        {/* Left: Camera Name & Source Tag */}
        <div className="flex items-center gap-2 pointer-events-auto">
          <div
            className={cn(
              'h-2.5 w-2.5 rounded-full',
              isIncidentActive
                ? 'bg-rose-500 animate-ping'
                : status === 'connected'
                ? 'bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.8)]'
                : status === 'reconnecting'
                ? 'bg-amber-500 animate-pulse'
                : 'bg-muted-foreground'
            )}
          />
          <span className="font-semibold text-xs sm:text-sm text-white drop-shadow-md truncate max-w-[140px] sm:max-w-[200px]">
            {camera.name}
          </span>
          <span className="px-1.5 py-0.5 rounded text-[10px] font-mono uppercase font-bold tracking-wider bg-white/10 text-white/90 backdrop-blur-sm border border-white/10">
            {camera.source_type}
          </span>
        </div>

        {/* Right: Status Badges & Quick Action Controls */}
        <div className="flex items-center gap-1.5 pointer-events-auto">
          {/* Status Badge */}
          {status === 'connected' && (
            <span className="flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-mono font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 backdrop-blur-sm">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
              {fps.toFixed(1)} FPS
            </span>
          )}

          {status === 'reconnecting' && (
            <span className="flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-mono font-bold bg-amber-500/20 text-amber-300 border border-amber-500/30 backdrop-blur-sm animate-pulse">
              <RefreshCw className="h-3 w-3 animate-spin" />
              RECONNECTING
            </span>
          )}

          {status === 'offline' && (
            <span className="flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-mono font-bold bg-slate-500/20 text-slate-300 border border-slate-500/30 backdrop-blur-sm">
              <WifiOff className="h-3 w-3" />
              OFFLINE
            </span>
          )}

          {status === 'connecting' && (
            <span className="flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-mono font-bold bg-blue-500/20 text-blue-300 border border-blue-500/30 backdrop-blur-sm">
              <Radio className="h-3 w-3 animate-pulse" />
              CONNECTING
            </span>
          )}

          {/* Focus Toggle Button */}
          {onFocus && (
            <button
              onClick={() => onFocus(camera)}
              className="p-1.5 rounded-lg bg-black/40 hover:bg-white/20 text-white transition-colors backdrop-blur-sm"
              title={isFocused ? 'Minimize view' : 'Focus tile'}
            >
              {isFocused ? <Minimize2 className="h-3.5 w-3.5" /> : <Maximize2 className="h-3.5 w-3.5" />}
            </button>
          )}

          {/* Delete Button (Hover Only) */}
          {onDelete && (
            <button
              onClick={() => onDelete(camera)}
              className="opacity-0 group-hover:opacity-100 p-1.5 rounded-lg bg-rose-500/20 hover:bg-rose-500/40 text-rose-300 transition-opacity backdrop-blur-sm"
              title="Delete Camera"
            >
              <Trash2 className="h-3.5 w-3.5" />
            </button>
          )}
        </div>
      </div>

      {/* Active Incident Tactical Siren Banner */}
      {isIncidentActive && (
        <div className="absolute top-10 inset-x-2 z-20 flex items-center justify-between px-3 py-1.5 rounded-lg bg-rose-600/90 text-white font-bold text-xs shadow-lg backdrop-blur-md border border-rose-400 animate-bounce">
          <div className="flex items-center gap-2">
            <AlertTriangle className="h-4 w-4 text-amber-300" />
            <span>ACCIDENT DETECTED</span>
          </div>
          {activeIncident && (
            <span className="font-mono text-[11px] bg-black/30 px-1.5 py-0.5 rounded">
              {activeIncident.class_name} ({(activeIncident.confidence * 100).toFixed(0)}%)
            </span>
          )}
        </div>
      )}

      {/* Video Container (Strictly bypasses React state for maximum rendering FPS) */}
      <div className="relative aspect-video w-full bg-slate-950 flex items-center justify-center overflow-hidden">
        {/* Real-time frame image reference */}
        <img
          ref={imageRef}
          alt={camera.name}
          className="h-full w-full object-contain"
          onError={(e) => {
            // Prevent broken image icon if stream drops
            ;(e.target as HTMLImageElement).style.display = 'none'
          }}
          onLoad={(e) => {
            ;(e.target as HTMLImageElement).style.display = 'block'
          }}
        />

        {/* Offline / Reconnecting Tactical Standby Overlay */}
        {status !== 'connected' && (
          <div className="absolute inset-0 flex flex-col items-center justify-center p-4 bg-slate-950/80 text-center">
            {status === 'reconnecting' ? (
              <>
                <RefreshCw className="h-8 w-8 text-amber-400 animate-spin mb-2" />
                <p className="text-sm font-semibold text-amber-200">Reconnecting Stream...</p>
                <p className="text-xs text-muted-foreground mt-1 font-mono">
                  Auto-retry with exponential backoff
                </p>
              </>
            ) : status === 'connecting' ? (
              <>
                <Radio className="h-8 w-8 text-blue-400 animate-pulse mb-2" />
                <p className="text-sm font-semibold text-blue-200">Initializing Connection...</p>
                <p className="text-xs text-muted-foreground mt-1 font-mono">
                  {camera.source_url}
                </p>
              </>
            ) : (
              <>
                <WifiOff className="h-8 w-8 text-slate-500 mb-2" />
                <p className="text-sm font-semibold text-slate-400">Camera Offline</p>
                <p className="text-xs text-muted-foreground mt-1 font-mono">
                  Feed inactive or disabled
                </p>
              </>
            )}
          </div>
        )}
      </div>

      {/* Bottom Subtitle / Metadata Strip */}
      <div className="flex items-center justify-between px-3 py-1.5 text-[11px] text-muted-foreground bg-card/60 border-t border-border/40 font-mono">
        <span className="truncate max-w-[200px]" title={camera.source_url}>
          {camera.source_url}
        </span>
        {camera.confidence_override && (
          <span className="text-primary font-semibold">
            Conf: {camera.confidence_override.toFixed(2)}
          </span>
        )}
      </div>
    </div>
  )
}
