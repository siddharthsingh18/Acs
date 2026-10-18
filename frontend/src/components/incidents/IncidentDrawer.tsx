import React, { useState, useEffect, useCallback, useRef } from 'react'
import { Link } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { motion, AnimatePresence } from 'framer-motion'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  X,
  ChevronLeft,
  ChevronRight,
  ZoomIn,
  ZoomOut,
  Maximize2,
  Copy,
  Check,
  Download,
  Send,
  ShieldCheck,
  ShieldX,
  Siren,
  Sparkles,
  ExternalLink,
  AlertTriangle,
  Flame,
  Truck,
  Car,
  Road,
  UserCheck,
  Radio,
  AlertOctagon,
  Video,
  Film,
  Activity,
  Gauge,
  Archive,
  Trash2,
  Lock,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { api, type IncidentResponse } from '@/api/client'
import { cn } from '@/lib/utils'
import { useAuthStore } from '@/store/authStore'


export interface IncidentDrawerProps {
  incidentId: string | null
  isOpen: boolean
  onClose: () => void
  incidentList?: { incident_id: string }[]
  onSelectIncident?: (id: string) => void
  initialIncident?: Partial<IncidentResponse> | null
}

export const IncidentDrawer: React.FC<IncidentDrawerProps> = ({
  incidentId,
  isOpen,
  onClose,
  incidentList = [],
  onSelectIncident,
  initialIncident,
}) => {
  const { t, i18n } = useTranslation()
  const isArabic = i18n.language === 'ar'
  const queryClient = useQueryClient()

  // State: Zoom and Lightbox
  const [isZoomed, setIsZoomed] = useState(false)
  const [isLightboxOpen, setIsLightboxOpen] = useState(false)
  const [copiedText, setCopiedText] = useState(false)

  // State: Confirmation Dialog
  const [confirmAction, setConfirmAction] = useState<
    'approved' | 'rejected' | 'dispatched' | null
  >(null)
  const { user } = useAuthStore()
  const reviewerName = user?.full_name || user?.username || 'Traffic Desk Officer'
  const isOperatorOrAdmin = user ? (user.role === 'operator' || user.role === 'admin') : true

  // State: Alert dispatch message / feedback
  const [alertFeedback, setAlertFeedback] = useState<string | null>(null)
  const [isSendingAlert, setIsSendingAlert] = useState(false)


  // Streaming-style loading steps for Gemini report generation
  const [streamingStep, setStreamingStep] = useState<string | null>(null)
  const streamingTimerRef = useRef<any>(null)

  // Media view mode & Video timeline state
  const [mediaTab, setMediaTab] = useState<'annotated' | 'raw' | 'keyframe' | 'unredacted'>('annotated')
  const [videoDuration, setVideoDuration] = useState(0)
  const [videoCurrentTime, setVideoCurrentTime] = useState(0)
  const videoRef = useRef<HTMLVideoElement | null>(null)

  // Export and Deletion State
  const [isExporting, setIsExporting] = useState(false)
  const [isDeleting, setIsDeleting] = useState(false)
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false)

  const handleExportBundle = async () => {
    if (!incidentId) return
    try {
      setIsExporting(true)
      const blob = await api.exportIncidentBundle(incidentId)
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${incidentId}_export_bundle.zip`
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)
    } catch (err: any) {
      console.error('Failed to export incident bundle:', err)
    } finally {
      setIsExporting(false)
    }
  }

  const handleDeleteIncident = async () => {
    if (!incidentId) return
    try {
      setIsDeleting(true)
      await api.deleteIncident(incidentId)
      queryClient.invalidateQueries({ queryKey: ['incidents'] })
      queryClient.invalidateQueries({ queryKey: ['stats'] })
      setShowDeleteConfirm(false)
      onClose()
    } catch (err: any) {
      console.error('Failed to delete incident:', err)
    } finally {
      setIsDeleting(false)
    }
  }

  const formatSeconds = (sec: number) => {
    if (!Number.isFinite(sec) || sec < 0) return '00:00.0'
    const m = Math.floor(sec / 60)
    const s = Math.floor(sec % 60)
    const ms = Math.floor((sec % 1) * 10)
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}.${ms}`
  }

  // Fetch full incident data
  const { data: incident, isLoading } = useQuery({
    queryKey: ['incident', incidentId],
    queryFn: () => (incidentId ? api.getIncident(incidentId) : Promise.reject('No ID')),
    enabled: Boolean(isOpen && incidentId),
    initialData: initialIncident as IncidentResponse | undefined,
  })

  // Automatically select best available media mode when incident changes
  useEffect(() => {
    if (incident) {
      if (incident.clip_annotated_url || incident.clip_annotated_path) {
        setMediaTab('annotated')
      } else if (incident.clip_url || incident.clip_path) {
        setMediaTab('raw')
      } else {
        setMediaTab('keyframe')
      }
    }
  }, [incident?.incident_id, incident?.clip_annotated_url, incident?.clip_url])

  const jumpToImpact = useCallback(() => {
    const impactSec = incident?.impact_offset_seconds ?? 8.0
    if (videoRef.current) {
      videoRef.current.currentTime = impactSec
      videoRef.current.play().catch(() => {})
    }
  }, [incident?.impact_offset_seconds])

  // Query Alert Settings to display dry-run status
  const { data: alertSettings } = useQuery({
    queryKey: ['alertSettings'],
    queryFn: () => api.getAlertSettings(),
    enabled: isOpen,
  })

  // Navigation between incidents
  const currentIndex = incidentList.findIndex(
    (item) => item.incident_id === incidentId
  )
  const hasPrevious = currentIndex > 0
  const hasNext = currentIndex >= 0 && currentIndex < incidentList.length - 1

  const goToPrevious = useCallback(() => {
    if (hasPrevious && onSelectIncident) {
      onSelectIncident(incidentList[currentIndex - 1].incident_id)
    }
  }, [hasPrevious, currentIndex, incidentList, onSelectIncident])

  const goToNext = useCallback(() => {
    if (hasNext && onSelectIncident) {
      onSelectIncident(incidentList[currentIndex + 1].incident_id)
    }
  }, [hasNext, currentIndex, incidentList, onSelectIncident])

  // Keyboard accessibility: Escape to close, Arrow keys to navigate
  useEffect(() => {
    if (!isOpen) return

    const handleKeyDown = (e: KeyboardEvent) => {
      if (confirmAction !== null || isLightboxOpen || showDeleteConfirm) {
        if (e.key === 'Escape') {
          setConfirmAction(null)
          setIsLightboxOpen(false)
          setShowDeleteConfirm(false)
        }
        return
      }

      if (e.key === 'Escape') {
        onClose()
      } else if (e.key === 'ArrowLeft') {
        if (isArabic) goToNext()
        else goToPrevious()
      } else if (e.key === 'ArrowRight') {
        if (isArabic) goToPrevious()
        else goToNext()
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [isOpen, confirmAction, isLightboxOpen, showDeleteConfirm, isArabic, goToPrevious, goToNext, onClose])

  // Optimistic status mutation with rollback
  const statusMutation = useMutation({
    mutationFn: async ({
      status,
      reviewer,
    }: {
      status: 'approved' | 'rejected' | 'dispatched'
      reviewer: string
    }) => {
      if (!incidentId) throw new Error('No incident ID')
      return api.updateIncidentStatus(incidentId, status, reviewer)
    },
    onMutate: async ({ status, reviewer }) => {
      await queryClient.cancelQueries({ queryKey: ['incident', incidentId] })
      const previousIncident = queryClient.getQueryData(['incident', incidentId])

      // Optimistically update incident cache
      queryClient.setQueryData(['incident', incidentId], (old: any) => {
        if (!old) return old
        return {
          ...old,
          status,
          reviewer_name: reviewer,
          reviewed_at: new Date().toISOString(),
        }
      })

      // Optimistically update list cache if present
      queryClient.setQueriesData(
        { queryKey: ['incidents'] },
        (oldList: any) => {
          if (!oldList || !oldList.items) return oldList
          return {
            ...oldList,
            items: oldList.items.map((item: any) =>
              item.incident_id === incidentId
                ? {
                    ...item,
                    status,
                    reviewer_name: reviewer,
                    reviewed_at: new Date().toISOString(),
                  }
                : item
            ),
          }
        }
      )

      return { previousIncident }
    },
    onError: (_err, _vars, context) => {
      if (context?.previousIncident) {
        queryClient.setQueryData(
          ['incident', incidentId],
          context.previousIncident
        )
      }
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ['incident', incidentId] })
      queryClient.invalidateQueries({ queryKey: ['incidents'] })
      queryClient.invalidateQueries({ queryKey: ['stats'] })
      setConfirmAction(null)
    },
  })

  // Dispatch report generation with streaming-style loading
  const dispatchMutation = useMutation({
    mutationFn: async () => {
      if (!incidentId) throw new Error('No incident ID')

      // Streaming-style progressive feedback
      setStreamingStep('Analyzing collision geometry and involved vehicles...')
      streamingTimerRef.current = setTimeout(() => {
        setStreamingStep('Assessing vehicle impact severity & road hazards...')
        streamingTimerRef.current = setTimeout(() => {
          setStreamingStep('Synthesizing emergency 911 dispatch response...')
        }, 1200)
      }, 1000)

      return api.generateDispatchReport(incidentId, i18n.language || 'en')
    },
    onSuccess: (data) => {
      setStreamingStep(null)
      if (streamingTimerRef.current) clearTimeout(streamingTimerRef.current)
      queryClient.setQueryData(['incident', incidentId], (old: any) => ({
        ...old,
        dispatch_report: data,
        dispatch_text: data?.rendered_text || old?.dispatch_text,
      }))
      queryClient.invalidateQueries({ queryKey: ['incident', incidentId] })
      queryClient.invalidateQueries({ queryKey: ['incidents'] })
    },
    onError: () => {
      setStreamingStep(null)
      if (streamingTimerRef.current) clearTimeout(streamingTimerRef.current)
    },
  })

  // Handle Copy Report Text
  const handleCopyReport = () => {
    const textToCopy =
      incident?.dispatch_text ||
      (incident?.dispatch_report as any)?.rendered_text ||
      `Incident ${incident?.incident_id}: ${incident?.class_name} detected at ${incident?.start_timestamp}`

    navigator.clipboard.writeText(textToCopy)
    setCopiedText(true)
    setTimeout(() => setCopiedText(false), 2000)
  }

  // Handle Send Alert (Respecting Dry-Run)
  const handleSendAlert = async () => {
    if (!incidentId) return
    setIsSendingAlert(true)
    setAlertFeedback(null)
    try {
      const isDryRun = alertSettings?.dry_run ?? true
      const res = await api.testAlerts(
        isDryRun,
        i18n.language || 'en',
        alertSettings?.channels || ['console']
      )
      const resultsText = Object.entries(res.results || {})
        .map(([k, v]) => `${k}: ${v ? 'OK' : 'Skipped'}`)
        .join(', ')

      setAlertFeedback(
        isDryRun
          ? `Alert dispatched in DRY-RUN mode (${resultsText})`
          : `Alert dispatched to live channels (${resultsText})`
      )
      setTimeout(() => setAlertFeedback(null), 4500)
    } catch (err: any) {
      setAlertFeedback('Failed to send alert: ' + (err.message || 'Network error'))
      setTimeout(() => setAlertFeedback(null), 4500)
    } finally {
      setIsSendingAlert(false)
    }
  }

  // Execute confirmed review action
  const executeConfirmedAction = () => {
    if (!confirmAction) return
    statusMutation.mutate({
      status: confirmAction,
      reviewer: reviewerName,
    })
  }

  // Helper: Severity Badge variant & colors
  const getSeverityBadge = (sev: string, score?: number | null) => {
    const s = sev?.toLowerCase()
    const scoreText = score !== undefined && score !== null ? ` • ${score}/100` : ''
    if (s === 'high') {
      return (
        <Badge variant="destructive" className="font-mono text-xs uppercase px-2.5 py-0.5">
          High{scoreText}
        </Badge>
      )
    }
    if (s === 'medium') {
      return (
        <Badge variant="warning" className="font-mono text-xs uppercase px-2.5 py-0.5">
          Medium{scoreText}
        </Badge>
      )
    }
    return (
      <Badge variant="outline" className="font-mono text-xs uppercase text-cyan-400 border-cyan-500/40 px-2.5 py-0.5">
        Low{scoreText}
      </Badge>
    )
  }

  // Helper: Unit Tag icon
  const getUnitIcon = (unit: string) => {
    const u = unit.toLowerCase()
    if (u.includes('fire')) return <Flame className="h-3 w-3 text-amber-400" />
    if (u.includes('tow')) return <Truck className="h-3 w-3 text-cyan-400" />
    if (u.includes('ems') || u.includes('med')) return <Siren className="h-3 w-3 text-rose-400" />
    return <ShieldCheck className="h-3 w-3 text-primary" />
  }

  const dispatchReport = incident?.dispatch_report as any

  return (
    <AnimatePresence>
      {isOpen && incidentId && (
        <div
          role="dialog"
          aria-modal="true"
          aria-label={t('incidents.details_title')}
          className="fixed inset-0 z-50 overflow-hidden"
        >
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-black/70 backdrop-blur-sm transition-opacity"
          />

          {/* Drawer Panel */}
          <div className="fixed inset-y-0 end-0 flex max-w-full ps-10 pointer-events-none">
            <motion.div
              initial={{ x: isArabic ? '-100%' : '100%' }}
              animate={{ x: 0 }}
              exit={{ x: isArabic ? '-100%' : '100%' }}
              transition={{ type: 'spring', damping: 28, stiffness: 280 }}
              className="w-screen max-w-2xl bg-card border-s border-border shadow-2xl flex flex-col pointer-events-auto h-full overflow-hidden"
            >
              {/* Header */}
              <div className="h-16 px-6 border-b border-border flex items-center justify-between bg-muted/30 shrink-0">
                <div className="flex items-center gap-3">
                  <div className="h-9 w-9 rounded-lg bg-[var(--status-alert)]/15 border border-[var(--status-alert)]/30 flex items-center justify-center text-[var(--status-alert)]">
                    <Siren className="h-5 w-5 animate-pulse" />
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <h3 className="font-bold text-base font-mono text-foreground">
                        {incident?.incident_id || incidentId}
                      </h3>
                      {incident?.severity && getSeverityBadge(incident.severity, incident?.severity_score)}
                      {incident?.is_privacy_blurred && (
                        <Badge variant="outline" className="gap-1 border-emerald-500/30 text-emerald-400 bg-emerald-500/10 text-[10px] font-mono">
                          <ShieldCheck className="h-3 w-3 text-emerald-400" />
                          Privacy Blurred
                        </Badge>
                      )}
                    </div>
                    <span className="text-xs text-muted-foreground font-mono">
                      {incident?.start_timestamp?.replace('T', ' ') || 'Live Event'}
                    </span>
                  </div>
                </div>

                {/* Right controls: Prev/Next & Close */}
                <div className="flex items-center gap-1.5">
                  {incidentList.length > 1 && (
                    <div className="flex items-center rounded-lg border border-border p-0.5 bg-background/50 me-2">
                      <Button
                        variant="ghost"
                        size="icon"
                        disabled={!hasPrevious}
                        onClick={goToPrevious}
                        title="Previous incident (Arrow Left)"
                        className="h-7 w-7 text-muted-foreground hover:text-foreground disabled:opacity-30"
                      >
                        {isArabic ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
                      </Button>
                      <span className="px-2 text-[10px] font-mono text-muted-foreground">
                        {currentIndex + 1}/{incidentList.length}
                      </span>
                      <Button
                        variant="ghost"
                        size="icon"
                        disabled={!hasNext}
                        onClick={goToNext}
                        title="Next incident (Arrow Right)"
                        className="h-7 w-7 text-muted-foreground hover:text-foreground disabled:opacity-30"
                      >
                        {isArabic ? <ChevronLeft className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
                      </Button>
                    </div>
                  )}

                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={onClose}
                    aria-label="Close drawer"
                    className="h-8 w-8 text-muted-foreground hover:text-foreground"
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </div>
              </div>

              {/* Scrollable Body */}
              <div className="flex-1 overflow-y-auto p-6 space-y-6">
                {isLoading && !incident ? (
                  <div className="py-20 text-center text-xs font-mono text-muted-foreground">
                    Loading incident records...
                  </div>
                ) : (
                  <>
                    {/* SECTION 1: Video Player & Keyframe with Timeline Marker */}
                    {(() => {
                      const rawClipUrl = incident?.clip_url || (incident?.clip_path ? `/api/incidents/${incident.incident_id}/clip` : null)
                      const annClipUrl = incident?.clip_annotated_url || (incident?.clip_annotated_path ? `/api/incidents/${incident.incident_id}/clip?annotated=true` : null)
                      const hasAnnotated = Boolean(annClipUrl)
                      const hasRaw = Boolean(rawClipUrl)
                      const isVideo = mediaTab !== 'keyframe' && (hasAnnotated || hasRaw)
                      const activeVideoSrc = mediaTab === 'raw' ? (rawClipUrl || annClipUrl) : (annClipUrl || rawClipUrl)
                      const impactOffset = incident?.impact_offset_seconds ?? 8.0
                      const duration = videoDuration > 0 ? videoDuration : 18.0
                      const impactPct = Math.min(100, Math.max(0, (impactOffset / duration) * 100))
                      const playheadPct = Math.min(100, Math.max(0, (videoCurrentTime / duration) * 100))

                      return (
                        <div className="space-y-2">
                          <div className="flex items-center justify-between flex-wrap gap-2">
                            {/* Media View Selector Tabs */}
                            <div className="flex items-center gap-1 bg-muted/60 p-0.5 rounded-lg border border-border/60">
                              {hasAnnotated && (
                                <button
                                  type="button"
                                  onClick={() => setMediaTab('annotated')}
                                  className={cn(
                                    'px-2.5 py-1 text-xs font-mono font-medium rounded-md transition-all flex items-center gap-1.5',
                                    mediaTab === 'annotated'
                                      ? 'bg-background text-foreground shadow-sm'
                                      : 'text-muted-foreground hover:text-foreground'
                                  )}
                                >
                                  <Film className="h-3.5 w-3.5 text-primary" />
                                  Annotated Clip
                                </button>
                              )}
                              {hasRaw && (
                                <button
                                  type="button"
                                  onClick={() => setMediaTab('raw')}
                                  className={cn(
                                    'px-2.5 py-1 text-xs font-mono font-medium rounded-md transition-all flex items-center gap-1.5',
                                    mediaTab === 'raw'
                                      ? 'bg-background text-foreground shadow-sm'
                                      : 'text-muted-foreground hover:text-foreground'
                                  )}
                                >
                                  <Video className="h-3.5 w-3.5 text-blue-400" />
                                  Raw Clip
                                </button>
                              )}
                              <button
                                type="button"
                                onClick={() => setMediaTab('keyframe')}
                                className={cn(
                                  'px-2.5 py-1 text-xs font-mono font-medium rounded-md transition-all flex items-center gap-1.5',
                                  mediaTab === 'keyframe'
                                    ? 'bg-background text-foreground shadow-sm'
                                    : 'text-muted-foreground hover:text-foreground'
                                )}
                              >
                                <Radio className="h-3.5 w-3.5 text-amber-400" />
                                Peak Keyframe
                              </button>
                              {incident?.has_unredacted_original && user?.role === 'admin' && (
                                <button
                                  type="button"
                                  onClick={() => setMediaTab('unredacted')}
                                  className={cn(
                                    'px-2.5 py-1 text-xs font-mono font-medium rounded-md transition-all flex items-center gap-1.5',
                                    mediaTab === 'unredacted'
                                      ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40 shadow-sm'
                                      : 'text-amber-400/80 hover:text-amber-300'
                                  )}
                                  title="Access-controlled unblurred original (Admin Review Only)"
                                >
                                  <Lock className="h-3.5 w-3.5 text-amber-400" />
                                  Vault Original
                                </button>
                              )}
                            </div>

                            {/* Controls: Jump to Impact, Zoom, Lightbox */}
                            <div className="flex items-center gap-1">
                              {isVideo && (
                                <Button
                                  variant="outline"
                                  size="sm"
                                  onClick={jumpToImpact}
                                  className="h-7 px-2.5 text-xs gap-1.5 border-red-500/30 text-red-400 hover:text-red-300 hover:bg-red-950/30 font-mono"
                                  title="Jump directly to impact moment"
                                >
                                  <Flame className="h-3.5 w-3.5 fill-red-500/20 text-red-500" />
                                  <span>Jump to Impact</span>
                                </Button>
                              )}
                              {(mediaTab === 'keyframe' || mediaTab === 'unredacted') && (
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  onClick={() => setIsZoomed((z) => !z)}
                                  className="h-7 px-2 text-xs gap-1 text-muted-foreground hover:text-foreground"
                                >
                                  {isZoomed ? <ZoomOut className="h-3.5 w-3.5" /> : <ZoomIn className="h-3.5 w-3.5" />}
                                  <span>{isZoomed ? 'Reset' : 'Zoom 2x'}</span>
                                </Button>
                              )}
                              <Button
                                variant="ghost"
                                size="icon"
                                onClick={() => setIsLightboxOpen(true)}
                                title="Fullscreen Lightbox"
                                className="h-7 w-7 text-muted-foreground hover:text-foreground"
                              >
                                <Maximize2 className="h-3.5 w-3.5" />
                              </Button>
                            </div>
                          </div>

                          {/* Video / Keyframe Box */}
                          <div
                            dir="ltr"
                            className={cn(
                              'relative aspect-video w-full rounded-xl bg-black/95 overflow-hidden border border-border/80 shadow-md transition-all select-none',
                              (mediaTab === 'keyframe' || mediaTab === 'unredacted') && (isZoomed ? 'cursor-zoom-out' : 'cursor-zoom-in')
                            )}
                            onClick={(mediaTab === 'keyframe' || mediaTab === 'unredacted') ? () => setIsZoomed((z) => !z) : undefined}
                          >
                            {mediaTab === 'unredacted' && incident ? (
                              <div className="w-full h-full relative flex items-center justify-center">
                                <img
                                  src={api.getUnredactedKeyframeUrl(incident.incident_id)}
                                  alt={`${incident.incident_id} unredacted original`}
                                  className={cn(
                                    'w-full h-full object-contain transition-transform duration-300',
                                    isZoomed && 'scale-175 origin-center'
                                  )}
                                />
                                <div className="absolute top-2.5 start-2.5 px-2.5 py-1 rounded bg-amber-950/85 border border-amber-500/50 text-amber-300 text-[10px] font-mono flex items-center gap-1.5 shadow-md">
                                  <Lock className="h-3.5 w-3.5 text-amber-400" />
                                  <span>VAULT UNREDACTED (ADMIN REVIEW ONLY)</span>
                                </div>
                              </div>
                            ) : isVideo && activeVideoSrc ? (
                              <video
                                ref={videoRef}
                                key={activeVideoSrc}
                                src={activeVideoSrc}
                                controls
                                playsInline
                                preload="metadata"
                                className="w-full h-full object-contain"
                                onTimeUpdate={(e) => setVideoCurrentTime(e.currentTarget.currentTime)}
                                onLoadedMetadata={(e) => setVideoDuration(e.currentTarget.duration)}
                              />
                            ) : incident?.keyframe_url ? (
                              <img
                                src={incident.keyframe_url}
                                alt={incident.incident_id}
                                className={cn(
                                  'w-full h-full object-contain transition-transform duration-300',
                                  isZoomed && 'scale-175 origin-center'
                                )}
                              />
                            ) : (
                              <div className="w-full h-full flex flex-col items-center justify-center text-xs text-muted-foreground font-mono">
                                <Radio className="h-6 w-6 text-muted-foreground/40 mb-2" />
                                No media available for this incident
                              </div>
                            )}

                            <div className="absolute bottom-2.5 start-2.5 px-2 py-0.5 rounded bg-black/75 text-cyan-400 font-mono text-[10px] border border-cyan-500/30 pointer-events-none">
                              {incident?.class_name?.replace(/_/g, ' ') || 'Incident'}
                              {mediaTab === 'unredacted' ? (
                                <span className="ms-1.5 text-amber-400 font-semibold">[Vault Unredacted]</span>
                              ) : isVideo && (
                                <span className="ms-1.5 text-zinc-400">
                                  [{mediaTab === 'annotated' ? 'H.264 Annotated' : 'H.264 Raw'}]
                                </span>
                              )}
                            </div>
                          </div>

                          {/* Timeline Impact Marker Track (Active during video mode) */}
                          {isVideo && (
                            <div className="bg-card/70 border border-border/70 rounded-lg p-2.5 space-y-1.5 font-mono text-xs">
                              <div className="flex items-center justify-between text-[11px] text-muted-foreground">
                                <span>00:00.0</span>
                                <button
                                  type="button"
                                  onClick={jumpToImpact}
                                  className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-red-950/60 border border-red-500/40 text-red-400 hover:bg-red-900/50 transition-colors"
                                >
                                  <Flame className="h-3 w-3 text-red-400" />
                                  Impact Moment: {formatSeconds(impactOffset)}
                                </button>
                                <span>{formatSeconds(duration)}</span>
                              </div>

                              {/* Interactive Timeline Bar with Impact Pin */}
                              <div
                                className="relative h-3 w-full bg-muted/60 rounded-full overflow-hidden cursor-pointer border border-border/40"
                                onClick={(e) => {
                                  const rect = e.currentTarget.getBoundingClientRect()
                                  const clickPct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width))
                                  const seekTime = clickPct * duration
                                  if (videoRef.current) {
                                    videoRef.current.currentTime = seekTime
                                  }
                                }}
                              >
                                {/* Played Progress Track */}
                                <div
                                  className="absolute top-0 bottom-0 start-0 bg-primary/40 transition-all"
                                  style={{ width: `${playheadPct}%` }}
                                />

                                {/* Impact Marker Pin */}
                                <div
                                  className="absolute top-0 bottom-0 w-1.5 bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.8)] z-10 -translate-x-1/2"
                                  style={{ left: `${impactPct}%` }}
                                  title={`Impact moment at ${formatSeconds(impactOffset)}`}
                                />
                              </div>

                              <div className="flex items-center justify-between text-[10px] text-muted-foreground/80">
                                <span>Pre-impact: 8s</span>
                                <span>Current: {formatSeconds(videoCurrentTime)}</span>
                                <span>Post-impact: 10s</span>
                              </div>
                            </div>
                          )}
                        </div>
                      )
                    })()}

                    {/* SECTION 2: Metadata Grid */}
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono">
                      <div className="p-3 rounded-lg border border-border bg-card/60 space-y-1">
                        <span className="text-[10px] text-muted-foreground uppercase block">
                          Collision Class
                        </span>
                        <span className="font-bold text-foreground truncate block">
                          {t(`classes.${incident?.class_name}`, incident?.class_name || '—')}
                        </span>
                      </div>

                      <div className="p-3 rounded-lg border border-border bg-card/60 space-y-1">
                        <span className="text-[10px] text-muted-foreground uppercase block">
                          Peak Confidence
                        </span>
                        <span className="font-bold text-cyan-400 block">
                          {incident?.max_confidence
                            ? `${(incident.max_confidence * 100).toFixed(1)}%`
                            : '—'}
                        </span>
                      </div>

                      <div className="p-3 rounded-lg border border-border bg-card/60 space-y-1">
                        <span className="text-[10px] text-muted-foreground uppercase block">
                          Event Duration
                        </span>
                        <span className="font-bold text-foreground block">
                          {incident?.duration !== undefined
                            ? `${incident.duration.toFixed(1)}s`
                            : '—'}
                        </span>
                      </div>

                      <div className="p-3 rounded-lg border border-border bg-card/60 space-y-1">
                        <span className="text-[10px] text-muted-foreground uppercase block">
                          Review Status
                        </span>
                        <Badge
                          variant={
                            incident?.status === 'dispatched'
                              ? 'destructive'
                              : incident?.status === 'approved'
                              ? 'success'
                              : incident?.status === 'rejected'
                              ? 'secondary'
                              : 'outline'
                          }
                          className="font-mono text-[10px] uppercase font-bold"
                        >
                          {incident?.status || 'Pending'}
                        </Badge>
                      </div>
                    </div>

                    {/* Review History Banner if reviewed */}
                    {incident?.reviewer_name && (
                      <div className="p-3 rounded-lg border border-border bg-muted/30 flex items-center justify-between text-xs font-mono">
                        <div className="flex items-center gap-2 text-foreground font-semibold">
                          <UserCheck className="h-4 w-4 text-[var(--status-ok)]" />
                          <span>Reviewed by: {incident.reviewer_name}</span>
                        </div>
                        <span className="text-muted-foreground text-[11px]">
                          {incident.reviewed_at
                            ? new Date(incident.reviewed_at).toLocaleString()
                            : 'Logged'}
                        </span>
                      </div>
                    )}

                    {/* SECTION: Collision Severity Score & Transparent Breakdown */}
                    {(() => {
                      const breakdown = (incident as any)?.severity_breakdown
                      const score = incident?.severity_score ?? breakdown?.score ?? (incident?.severity === 'high' ? 80 : incident?.severity === 'low' ? 25 : 50)
                      const level = incident?.severity || breakdown?.level || (score > 70 ? 'high' : score > 35 ? 'medium' : 'low')
                      const units: string[] = incident?.recommended_units || breakdown?.recommended_units || (dispatchReport?.recommended_units) || ['police', 'tow']
                      const components = Array.isArray(breakdown?.components) ? breakdown.components : []
                      const factors = Array.isArray(breakdown?.factors) ? breakdown.factors : []
                      const scorerVersion = breakdown?.scorer_version || 'rule_based_v1'

                      const scoreColor =
                        level === 'high'
                          ? 'text-rose-400 border-rose-500/40 bg-rose-950/30'
                          : level === 'medium'
                          ? 'text-amber-400 border-amber-500/40 bg-amber-950/30'
                          : 'text-emerald-400 border-emerald-500/40 bg-emerald-950/30'

                      const barColor =
                        level === 'high' ? 'bg-rose-500' : level === 'medium' ? 'bg-amber-500' : 'bg-emerald-500'

                      return (
                        <div className="rounded-xl border border-border bg-card/80 overflow-hidden shadow-sm space-y-0">
                          {/* Header */}
                          <div className="p-4 border-b border-border bg-muted/40 flex items-center justify-between">
                            <div className="flex items-center gap-2">
                              <Activity className="h-4 w-4 text-primary" />
                              <div>
                                <h4 className="font-bold text-sm text-foreground">
                                  Collision Severity & Unit Routing
                                </h4>
                                <span className="text-[10px] text-muted-foreground font-mono block">
                                  Multi-factor telemetry analysis • {scorerVersion}
                                </span>
                              </div>
                            </div>
                            <div className="flex items-center gap-2">
                              <span className="text-xs font-mono text-muted-foreground">Score:</span>
                              <div
                                className={cn(
                                  'px-2.5 py-1 rounded-md border font-mono font-bold text-sm flex items-center gap-1.5 shadow-2xs',
                                  scoreColor
                                )}
                              >
                                <Gauge className="h-3.5 w-3.5" />
                                <span>{score}</span>
                                <span className="text-[10px] opacity-70">/ 100</span>
                              </div>
                            </div>
                          </div>

                          <div className="p-5 space-y-4">
                            {/* Score Meter & Routing Summary */}
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 items-center">
                              {/* Overall Score Progress Bar */}
                              <div className="p-3.5 rounded-lg border border-border bg-background/50 space-y-2">
                                <div className="flex items-center justify-between text-xs font-mono">
                                  <span className="text-muted-foreground">Calculated Severity Tier</span>
                                  <Badge
                                    variant={level === 'high' ? 'destructive' : level === 'medium' ? 'warning' : 'outline'}
                                    className="uppercase font-mono text-[10px] font-bold"
                                  >
                                    {level} Severity ({score}/100)
                                  </Badge>
                                </div>
                                <div className="h-2.5 w-full bg-muted/80 rounded-full overflow-hidden border border-border/40">
                                  <div
                                    className={cn('h-full transition-all duration-500 rounded-full', barColor)}
                                    style={{ width: `${Math.min(100, Math.max(5, score))}%` }}
                                  />
                                </div>
                                <div className="flex items-center justify-between text-[10px] font-mono text-muted-foreground/80">
                                  <span>0 (Low &le; 35)</span>
                                  <span>Medium &le; 70</span>
                                  <span>100 (High &gt; 70)</span>
                                </div>
                              </div>

                              {/* Recommended Units Routed */}
                              <div className="p-3.5 rounded-lg border border-border bg-background/50 space-y-2">
                                <span className="text-[11px] font-mono text-muted-foreground uppercase font-semibold block">
                                  Automated Unit Routing
                                </span>
                                <div className="flex flex-wrap items-center gap-1.5">
                                  {units.map((unit) => (
                                    <Badge
                                      key={unit}
                                      variant="outline"
                                      className="px-2.5 py-1 text-xs font-mono font-bold border-border bg-background flex items-center gap-1.5 shadow-2xs"
                                    >
                                      {getUnitIcon(unit)}
                                      <span className="uppercase">{unit.toUpperCase()}</span>
                                    </Badge>
                                  ))}
                                </div>
                              </div>
                            </div>

                            {/* Transparent Sub-Factor Breakdown */}
                            {components.length > 0 && (
                              <div className="space-y-2.5">
                                <span className="text-[11px] font-mono text-muted-foreground uppercase font-semibold block">
                                  Transparent Factor Breakdown (Explainable Scoring)
                                </span>
                                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                                  {components.map((comp: any) => {
                                    const compScore = comp.score ?? 0
                                    const maxScore = comp.max_score ?? 25
                                    const pct = Math.min(100, Math.max(0, (compScore / maxScore) * 100))
                                    return (
                                      <div
                                        key={comp.key}
                                        className="p-3 rounded-lg border border-border/80 bg-background/40 space-y-1.5"
                                      >
                                        <div className="flex items-center justify-between text-xs font-mono">
                                          <span className="font-semibold text-foreground truncate">{comp.label}</span>
                                          <span className="text-cyan-400 font-bold">
                                            {compScore.toFixed(1)} <span className="text-[10px] text-muted-foreground">/ {maxScore}</span>
                                          </span>
                                        </div>
                                        <div className="h-1.5 w-full bg-muted/80 rounded-full overflow-hidden">
                                          <div
                                            className="h-full bg-cyan-500 rounded-full transition-all"
                                            style={{ width: `${pct}%` }}
                                          />
                                        </div>
                                        <p className="text-[11px] text-muted-foreground leading-tight font-sans">
                                          {comp.explanation}
                                        </p>
                                      </div>
                                    )
                                  })}
                                </div>
                              </div>
                            )}

                            {/* Contributing Factor Explanations */}
                            {factors.length > 0 && (
                              <div className="p-3 rounded-lg border border-border/60 bg-muted/20 space-y-1.5 text-xs">
                                <span className="text-[10px] font-mono uppercase text-muted-foreground font-semibold block">
                                  Telemetry & Visual Findings:
                                </span>
                                <ul className="space-y-1 text-muted-foreground font-sans">
                                  {factors.map((factor: string, idx: number) => (
                                    <li key={idx} className="flex items-start gap-1.5">
                                      <span className="text-primary select-none">•</span>
                                      <span>{factor}</span>
                                    </li>
                                  ))}
                                </ul>
                              </div>
                            )}
                          </div>
                        </div>
                      )
                    })()}

                    {/* SECTION 3: Multimodal Gemini AI Dispatch Report */}
                    <div className="rounded-xl border border-border bg-card/80 overflow-hidden shadow-sm space-y-0">
                      <div className="p-4 border-b border-border bg-muted/40 flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <Sparkles className="h-4 w-4 text-cyan-400" />
                          <h4 className="font-bold text-sm text-foreground">
                            Gemini AI Dispatch Assessment
                          </h4>
                        </div>

                        {!dispatchReport && (
                          <Button
                            size="sm"
                            disabled={dispatchMutation.isPending}
                            onClick={() => dispatchMutation.mutate()}
                            className="gap-1.5 text-xs font-bold bg-cyan-600 hover:bg-cyan-500 text-white"
                          >
                            <Sparkles className="h-3.5 w-3.5" />
                            <span>{t('incidents.generate_ai_report')}</span>
                          </Button>
                        )}
                      </div>

                      {/* Streaming Loading State */}
                      {dispatchMutation.isPending && (
                        <div className="p-6 text-center space-y-3">
                          <div className="h-2 w-full bg-muted rounded-full overflow-hidden">
                            <div className="h-full bg-cyan-400 rounded-full animate-pulse w-3/4" />
                          </div>
                          <div className="flex items-center justify-center gap-2 text-xs font-mono text-cyan-400">
                            <Radio className="h-3.5 w-3.5 animate-pulse" />
                            <span>{streamingStep || 'Querying Gemini Multimodal AI...'}</span>
                          </div>
                        </div>
                      )}

                      {/* Rendered Structured Sections */}
                      {dispatchReport ? (
                        <div className="p-5 space-y-4">
                          {/* Recommended Units Chips */}
                          <div className="space-y-1.5">
                            <span className="text-[11px] font-mono text-muted-foreground uppercase font-semibold block">
                              Required Emergency Units:
                            </span>
                            <div className="flex flex-wrap items-center gap-2">
                              {Array.isArray(dispatchReport.recommended_units) ? (
                                dispatchReport.recommended_units.map((unit: string) => (
                                  <Badge
                                    key={unit}
                                    variant="outline"
                                    className="px-2.5 py-1 text-xs font-mono font-bold border-border bg-background flex items-center gap-1.5 shadow-2xs"
                                  >
                                    {getUnitIcon(unit)}
                                    <span className="uppercase">{unit.toUpperCase()}</span>
                                  </Badge>
                                ))
                              ) : (
                                <Badge variant="outline">EMS, POLICE</Badge>
                              )}
                            </div>
                          </div>

                          {/* Vehicles Involved */}
                          <div className="p-3.5 rounded-lg border border-border bg-background/50 space-y-1">
                            <div className="flex items-center gap-1.5 text-xs font-bold text-foreground">
                              <Car className="h-3.5 w-3.5 text-primary" />
                              <span>Vehicles Involved</span>
                            </div>
                            <div className="text-xs text-muted-foreground ps-5 font-mono">
                              {Array.isArray(dispatchReport.vehicles)
                                ? dispatchReport.vehicles.join(' • ')
                                : dispatchReport.vehicles || 'Unspecified vehicles'}
                            </div>
                          </div>

                          {/* Collision Point & Damage */}
                          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                            <div className="p-3 rounded-lg border border-border bg-background/50 space-y-1">
                              <span className="text-[10px] text-muted-foreground uppercase font-mono block">
                                Point of Collision
                              </span>
                              <span className="text-foreground font-medium">
                                {dispatchReport.collision_point || 'Roadway intersection'}
                              </span>
                            </div>

                            <div className="p-3 rounded-lg border border-border bg-background/50 space-y-1">
                              <span className="text-[10px] text-muted-foreground uppercase font-mono block">
                                Observed Damage
                              </span>
                              <span className="text-foreground font-medium">
                                {dispatchReport.damage || 'Impact damage observed'}
                              </span>
                            </div>
                          </div>

                          {/* Road Conditions & Active Hazards */}
                          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                            <div className="p-3 rounded-lg border border-border bg-background/50 space-y-1">
                              <div className="flex items-center gap-1 text-[10px] text-muted-foreground uppercase font-mono">
                                <Road className="h-3 w-3" />
                                <span>Road Conditions</span>
                              </div>
                              <span className="text-foreground font-medium">
                                {dispatchReport.road_conditions || 'Standard dry asphalt'}
                              </span>
                            </div>

                            <div className="p-3 rounded-lg border border-border bg-background/50 space-y-1">
                              <div className="flex items-center gap-1 text-[10px] text-[var(--status-alert)] uppercase font-mono font-bold">
                                <AlertTriangle className="h-3 w-3" />
                                <span>Active Hazards</span>
                              </div>
                              <span className="text-foreground font-medium">
                                {dispatchReport.hazards || 'Traffic obstruction'}
                              </span>
                            </div>
                          </div>

                          {/* Rendered Dispatch Text */}
                          {incident?.dispatch_text && (
                            <div className="p-3.5 rounded-lg border border-border bg-muted/20 space-y-1.5">
                              <span className="text-[10px] font-mono uppercase text-muted-foreground block font-bold">
                                Formatted Dispatch Broadcast:
                              </span>
                              <pre className="text-[11px] font-mono text-foreground whitespace-pre-wrap leading-relaxed">
                                {incident.dispatch_text}
                              </pre>
                            </div>
                          )}
                        </div>
                      ) : (
                        !dispatchMutation.isPending && (
                          <div className="p-6 text-center text-xs text-muted-foreground space-y-1">
                            <p>No AI multimodal report generated yet.</p>
                            <p className="text-[11px] text-muted-foreground/70">
                              Generate a report to analyze vehicle collision vectors with Gemini.
                            </p>
                          </div>
                        )
                      )}
                    </div>

                    {/* SECTION 4: Quick Action Toolbar (PDF, Export, Alert, Delete) */}
                    <div className="flex flex-wrap items-center gap-2 pt-2">
                      {/* Copy Text */}
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={handleCopyReport}
                        className="gap-1.5 text-xs border-border"
                      >
                        {copiedText ? <Check className="h-3.5 w-3.5 text-[var(--status-ok)]" /> : <Copy className="h-3.5 w-3.5" />}
                        <span>{copiedText ? 'Copied!' : 'Copy Report'}</span>
                      </Button>

                      {/* Download PDF */}
                      <a
                        href={api.getIncidentPdfUrl(incidentId)}
                        target="_blank"
                        rel="noreferrer"
                        className="inline-flex"
                      >
                        <Button variant="outline" size="sm" className="gap-1.5 text-xs border-border">
                          <Download className="h-3.5 w-3.5 text-primary" />
                          <span>{t('incidents.download_pdf')}</span>
                        </Button>
                      </a>

                      {/* Export ZIP Bundle */}
                      <Button
                        variant="outline"
                        size="sm"
                        disabled={isExporting}
                        onClick={handleExportBundle}
                        className="gap-1.5 text-xs border-indigo-500/30 text-indigo-400 hover:border-indigo-500 hover:text-indigo-300 hover:bg-indigo-950/20"
                        title="Export bundle (.zip) with metadata, media, PDF report, and audit trail"
                      >
                        <Archive className="h-3.5 w-3.5 text-indigo-400" />
                        <span>{isExporting ? 'Exporting...' : 'Export Bundle (.zip)'}</span>
                      </Button>

                      {/* Send Alert */}
                      <Button
                        variant="outline"
                        size="sm"
                        disabled={isSendingAlert}
                        onClick={handleSendAlert}
                        className="gap-1.5 text-xs border-border hover:border-cyan-500 hover:text-cyan-400"
                      >
                        <Send className="h-3.5 w-3.5 text-cyan-400" />
                        <span>Send Alert</span>
                        {alertSettings?.dry_run && (
                          <span className="text-[9px] px-1 py-0.2 rounded bg-amber-500/20 text-amber-400 uppercase font-mono">
                            Dry-Run
                          </span>
                        )}
                      </Button>

                      {/* Admin Delete Incident */}
                      {user?.role === 'admin' && (
                        <Button
                          variant="outline"
                          size="sm"
                          disabled={isDeleting}
                          onClick={() => setShowDeleteConfirm(true)}
                          className="gap-1.5 text-xs border-red-500/30 text-red-400 hover:border-red-500 hover:text-red-300 hover:bg-red-950/20"
                          title="Permanently delete incident and erase all disk media"
                        >
                          <Trash2 className="h-3.5 w-3.5 text-red-400" />
                          <span>Delete</span>
                        </Button>
                      )}

                      <Link to={`/incidents/${incidentId}`} className="ms-auto inline-flex">
                        <Button variant="ghost" size="sm" className="gap-1 text-xs text-primary hover:text-primary">
                          <span>Full Dossier</span>
                          <ExternalLink className="h-3.5 w-3.5" />
                        </Button>
                      </Link>
                    </div>

                    {alertFeedback && (
                      <div className="p-3 rounded-lg bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 text-xs font-mono flex items-center gap-2">
                        <Radio className="h-4 w-4 animate-pulse shrink-0" />
                        <span>{alertFeedback}</span>
                      </div>
                    )}
                  </>
                )}
              </div>

              {/* FOOTER: Human-in-the-Loop Review Buttons */}
              <div className="p-4 px-6 border-t border-border bg-card/95 flex flex-col sm:flex-row items-center justify-between gap-3 shrink-0">
                <div className="w-full sm:w-auto flex items-center gap-2">
                  <span className="text-xs text-muted-foreground whitespace-nowrap font-mono">
                    Officer:
                  </span>
                  <div className="flex items-center gap-1.5 px-2.5 py-1 rounded border border-border bg-background text-xs font-mono text-foreground shadow-2xs">
                    <UserCheck className="h-3.5 w-3.5 text-primary" />
                    <span className="font-semibold">{reviewerName}</span>
                    {user?.role && (
                      <span className="text-[10px] uppercase font-bold px-1.5 py-0.2 rounded bg-primary/10 text-primary border border-primary/20">
                        {user.role}
                      </span>
                    )}
                  </div>
                  {!isOperatorOrAdmin && (
                    <span className="text-[11px] text-amber-500 font-mono hidden sm:inline">
                      (Read-only)
                    </span>
                  )}
                </div>

                <div className="flex items-center gap-2 w-full sm:w-auto justify-end">
                  {/* Reject */}
                  <Button
                    variant="outline"
                    size="sm"
                    disabled={statusMutation.isPending || incident?.status === 'rejected' || !isOperatorOrAdmin}
                    onClick={() => setConfirmAction('rejected')}
                    className="gap-1.5 text-xs border-border hover:border-slate-500 text-muted-foreground"
                  >
                    <ShieldX className="h-3.5 w-3.5 text-slate-400" />
                    <span>{t('incidents.reject')}</span>
                  </Button>

                  {/* Approve */}
                  <Button
                    variant="outline"
                    size="sm"
                    disabled={statusMutation.isPending || incident?.status === 'approved' || !isOperatorOrAdmin}
                    onClick={() => setConfirmAction('approved')}
                    className="gap-1.5 text-xs border-border hover:border-[var(--status-ok)] text-[var(--status-ok)] hover:bg-[var(--status-ok)]/10"
                  >
                    <ShieldCheck className="h-3.5 w-3.5" />
                    <span>{t('incidents.approve')}</span>
                  </Button>

                  {/* Dispatch 911 */}
                  <Button
                    size="sm"
                    disabled={statusMutation.isPending || incident?.status === 'dispatched' || !isOperatorOrAdmin}
                    onClick={() => setConfirmAction('dispatched')}
                    className="gap-1.5 text-xs font-bold bg-[var(--status-alert)] hover:bg-[var(--status-alert)]/90 text-white shadow-sm"
                  >
                    <Siren className="h-3.5 w-3.5 fill-current" />
                    <span>{t('incidents.dispatch')}</span>
                  </Button>
                </div>

              </div>
            </motion.div>
          </div>

          {/* Confirmation Dialog Modal */}
          {confirmAction && (
            <div className="fixed inset-0 z-60 flex items-center justify-center p-4">
              <div
                className="fixed inset-0 bg-black/60 backdrop-blur-xs"
                onClick={() => setConfirmAction(null)}
              />
              <div className="relative bg-card border border-border rounded-xl p-6 max-w-sm w-full shadow-2xl space-y-4">
                <div className="flex items-center gap-3">
                  <div
                    className={cn(
                      'h-10 w-10 rounded-full flex items-center justify-center shrink-0',
                      confirmAction === 'dispatched'
                        ? 'bg-rose-500/20 text-rose-500'
                        : confirmAction === 'approved'
                        ? 'bg-emerald-500/20 text-emerald-500'
                        : 'bg-slate-500/20 text-slate-400'
                    )}
                  >
                    <AlertOctagon className="h-5 w-5" />
                  </div>
                  <div>
                    <h4 className="text-base font-bold text-foreground">
                      Confirm Incident {confirmAction.toUpperCase()}
                    </h4>
                    <p className="text-xs text-muted-foreground mt-0.5">
                      Incident {incident?.incident_id || incidentId}
                    </p>
                  </div>
                </div>

                <p className="text-xs text-muted-foreground leading-relaxed">
                  {confirmAction === 'dispatched'
                    ? 'Are you sure you want to authorize 911 emergency services dispatch for this collision? Emergency notifications will be transmitted.'
                    : confirmAction === 'approved'
                    ? 'Confirm verification of this incident record in the master log as an authenticated collision event.'
                    : 'Mark this incident detection as a false alarm / rejected classification.'}
                </p>

                <div className="p-3 rounded bg-muted/40 text-xs font-mono">
                  <span className="text-muted-foreground block text-[10px]">Reviewer:</span>
                  <span className="font-bold text-foreground">{reviewerName}</span>
                </div>

                <div className="flex items-center justify-end gap-2 pt-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setConfirmAction(null)}
                    className="text-xs border-border"
                  >
                    Cancel
                  </Button>
                  <Button
                    size="sm"
                    onClick={executeConfirmedAction}
                    className={cn(
                      'text-xs font-bold',
                      confirmAction === 'dispatched'
                        ? 'bg-rose-600 hover:bg-rose-500 text-white'
                        : confirmAction === 'approved'
                        ? 'bg-emerald-600 hover:bg-emerald-500 text-white'
                        : 'bg-slate-600 hover:bg-slate-500 text-white'
                    )}
                  >
                    Confirm {confirmAction}
                  </Button>
                </div>
              </div>
            </div>
          )}

          {/* Permanent Incident Deletion Confirmation Modal */}
          {showDeleteConfirm && (
            <div className="fixed inset-0 z-60 flex items-center justify-center p-4">
              <div
                className="fixed inset-0 bg-black/60 backdrop-blur-xs"
                onClick={() => setShowDeleteConfirm(false)}
              />
              <div className="relative bg-card border border-red-500/30 rounded-xl p-6 max-w-sm w-full shadow-2xl space-y-4">
                <div className="flex items-center gap-3">
                  <div className="h-10 w-10 rounded-full bg-red-500/20 text-red-500 flex items-center justify-center shrink-0">
                    <Trash2 className="h-5 w-5" />
                  </div>
                  <div>
                    <h4 className="text-base font-bold text-foreground">
                      Delete Incident Record
                    </h4>
                    <p className="text-xs text-muted-foreground mt-0.5">
                      Incident {incident?.incident_id || incidentId}
                    </p>
                  </div>
                </div>

                <p className="text-xs text-muted-foreground leading-relaxed">
                  Are you sure you want to permanently delete this incident? All stored keyframes, video clips, and audit links will be erased from disk. This action is irreversible.
                </p>

                <div className="flex items-center justify-end gap-2 pt-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setShowDeleteConfirm(false)}
                    className="text-xs border-border"
                  >
                    Cancel
                  </Button>
                  <Button
                    size="sm"
                    disabled={isDeleting}
                    onClick={handleDeleteIncident}
                    className="text-xs font-bold bg-red-600 hover:bg-red-500 text-white"
                  >
                    {isDeleting ? 'Deleting...' : 'Permanently Delete'}
                  </Button>
                </div>
              </div>
            </div>
          )}

          {/* Fullscreen Lightbox Modal */}
          {isLightboxOpen && (incident?.keyframe_url || mediaTab === 'unredacted') && (
            <div
              className="fixed inset-0 z-70 bg-black/95 flex flex-col items-center justify-center p-4 cursor-pointer"
              onClick={() => setIsLightboxOpen(false)}
            >
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setIsLightboxOpen(false)}
                className="absolute top-4 end-4 text-white hover:bg-white/20"
              >
                <X className="h-6 w-6" />
              </Button>
              <img
                src={mediaTab === 'unredacted' && incident ? api.getUnredactedKeyframeUrl(incident.incident_id) : incident?.keyframe_url}
                alt="Enlarged Keyframe"
                className="max-h-[90vh] max-w-[90vw] object-contain rounded-lg shadow-2xl"
              />
              <span className="text-xs text-slate-400 font-mono mt-3">
                {incident?.incident_id} • {mediaTab === 'unredacted' ? 'Unredacted Original' : 'Privacy Redacted'} • Click anywhere to exit fullscreen
              </span>
            </div>
          )}
        </div>
      )}
    </AnimatePresence>
  )
}
