import React from 'react'
import { useTranslation } from 'react-i18next'
import { Film, Radio, AlertOctagon, CheckCircle2 } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'
import type { StreamProgress, StreamStatus } from '@/hooks/useJobStream'

interface VideoPanelProps {
  imageRef: React.RefObject<HTMLImageElement | null>
  status: StreamStatus
  progress: StreamProgress
  isIncidentActive: boolean
  error: string | null
}

export const VideoPanel: React.FC<VideoPanelProps> = ({
  imageRef,
  status,
  progress,
  isIncidentActive,
  error,
}) => {
  const { t } = useTranslation()

  const { currentFrame, totalFrames, fps } = progress
  const progressPercent =
    totalFrames > 0 ? Math.round((currentFrame / totalFrames) * 100) : 0

  return (
    <div className="space-y-3">
      {/* Video Container - STRICT LTR */}
      <div
        dir="ltr"
        className={cn(
          'relative aspect-video w-full rounded-xl bg-black/95 overflow-hidden shadow-2xl transition-all duration-300',
          isIncidentActive
            ? 'border-2 border-rose-500 shadow-[0_0_35px_rgba(244,63,94,0.65)] ring-4 ring-rose-500/20'
            : 'border border-border/80'
        )}
      >
        {/* The Frame Image element manipulated directly by useJobStream ref */}
        <img
          ref={imageRef}
          alt="ACS Live CCTV Inference Stream"
          className={cn(
            'w-full h-full object-contain select-none transition-opacity duration-200',
            status === 'idle' && 'hidden'
          )}
        />

        {/* Standby Placeholder Screen when Idle */}
        {status === 'idle' && (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-center p-6 space-y-3 select-none">
            <div className="h-16 w-16 rounded-full border border-primary/30 bg-primary/5 flex items-center justify-center text-primary/60">
              <Film className="h-8 w-8" />
            </div>
            <p className="text-sm text-muted-foreground font-mono max-w-sm">
              {t('live.stream_idle')}
            </p>
          </div>
        )}

        {/* Error Overlay */}
        {status === 'error' && error && (
          <div className="absolute inset-0 bg-black/85 flex flex-col items-center justify-center text-center p-6 space-y-2">
            <AlertOctagon className="h-10 w-10 text-[var(--status-alert)] animate-pulse" />
            <h4 className="text-sm font-bold text-[var(--status-alert)]">
              Stream Interrupted
            </h4>
            <p className="text-xs text-muted-foreground max-w-md font-mono">
              {error}
            </p>
          </div>
        )}

        {/* Completed Overlay Banner */}
        {status === 'completed' && (
          <div className="absolute bottom-4 left-4 right-4 bg-black/80 backdrop-blur-md border border-[var(--status-ok)]/40 p-3 rounded-lg flex items-center justify-between pointer-events-none">
            <div className="flex items-center gap-2 text-xs font-mono text-[var(--status-ok)]">
              <CheckCircle2 className="h-4 w-4" />
              <span>Inference Complete • {currentFrame} frames processed</span>
            </div>
          </div>
        )}

        {/* TOP-LEFT HUD: Camera Info & Active State Badge */}
        <div className="absolute top-3 left-3 flex items-center gap-2 pointer-events-none z-10">
          <span className="px-2.5 py-1 rounded bg-black/80 backdrop-blur-xs text-cyan-400 font-mono text-[11px] font-bold border border-cyan-500/30 flex items-center gap-1.5">
            <Radio className="h-3 w-3 text-cyan-400 animate-pulse" />
            CCTV-FEED #01
          </span>

          {/* State Overlay Badge (Monitoring / Incident Active) */}
          {status === 'streaming' && (
            <Badge
              variant={isIncidentActive ? 'destructive' : 'default'}
              className={cn(
                'text-[10px] font-mono uppercase font-bold tracking-wider px-2 py-0.5 shadow-md flex items-center gap-1',
                isIncidentActive
                  ? 'bg-rose-600 text-white animate-bounce shadow-rose-600/50'
                  : 'bg-emerald-600/90 text-white'
              )}
            >
              <span
                className={cn(
                  'h-1.5 w-1.5 rounded-full',
                  isIncidentActive ? 'bg-white animate-ping' : 'bg-emerald-300'
                )}
              />
              {isIncidentActive ? 'INCIDENT ACTIVE' : 'MONITORING'}
            </Badge>
          )}
        </div>

        {/* TOP-RIGHT HUD: Live FPS Counter */}
        <div className="absolute top-3 right-3 flex items-center gap-2 pointer-events-none z-10">
          {fps > 0 && status === 'streaming' && (
            <span className="px-2 py-0.5 rounded bg-black/80 backdrop-blur-xs text-slate-300 font-mono text-[11px] border border-white/10 font-bold">
              {fps.toFixed(1)} FPS
            </span>
          )}

          {status === 'streaming' && (
            <span className="px-1.5 py-0.5 rounded bg-rose-600 text-white font-mono text-[10px] font-bold animate-pulse">
              LIVE
            </span>
          )}
        </div>

        {/* BOTTOM-RIGHT HUD: Frame Progress Counter */}
        {totalFrames > 0 && status === 'streaming' && (
          <div className="absolute bottom-3 right-3 px-2.5 py-1 rounded bg-black/80 backdrop-blur-xs text-slate-200 font-mono text-xs border border-white/10 pointer-events-none font-semibold">
            {currentFrame} / {totalFrames} ({progressPercent}%)
          </div>
        )}
      </div>

      {/* Progress Bar under video */}
      {totalFrames > 0 && (
        <div className="w-full bg-muted/60 h-2 rounded-full overflow-hidden border border-border/50">
          <div
            className="bg-primary h-full transition-all duration-150 rounded-full"
            style={{ width: `${progressPercent}%` }}
          />
        </div>
      )}
    </div>
  )
}
