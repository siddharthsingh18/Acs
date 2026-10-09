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
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { api, type IncidentResponse } from '@/api/client'
import { cn } from '@/lib/utils'

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
  const [reviewerName, setReviewerName] = useState(() => {
    return localStorage.getItem('acs_reviewer_name') || 'Traffic Desk Officer'
  })

  // State: Alert dispatch message / feedback
  const [alertFeedback, setAlertFeedback] = useState<string | null>(null)
  const [isSendingAlert, setIsSendingAlert] = useState(false)

  // Streaming-style loading steps for Gemini report generation
  const [streamingStep, setStreamingStep] = useState<string | null>(null)
  const streamingTimerRef = useRef<any>(null)

  // Fetch full incident data
  const { data: incident, isLoading } = useQuery({
    queryKey: ['incident', incidentId],
    queryFn: () => (incidentId ? api.getIncident(incidentId) : Promise.reject('No ID')),
    enabled: Boolean(isOpen && incidentId),
    initialData: initialIncident as IncidentResponse | undefined,
  })

  // Query Alert Settings to display dry-run status
  const { data: alertSettings } = useQuery({
    queryKey: ['alertSettings'],
    queryFn: () => api.getAlertSettings(),
    enabled: isOpen,
  })

  // Save reviewer name on change
  const handleReviewerChange = (name: string) => {
    setReviewerName(name)
    localStorage.setItem('acs_reviewer_name', name)
  }

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
      if (confirmAction !== null || isLightboxOpen) {
        if (e.key === 'Escape') {
          setConfirmAction(null)
          setIsLightboxOpen(false)
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
  }, [isOpen, confirmAction, isLightboxOpen, isArabic, goToPrevious, goToNext, onClose])

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
  const getSeverityBadge = (sev: string) => {
    const s = sev?.toLowerCase()
    if (s === 'high') {
      return (
        <Badge variant="destructive" className="font-mono text-xs uppercase px-2.5 py-0.5">
          High Severity
        </Badge>
      )
    }
    if (s === 'medium') {
      return (
        <Badge variant="warning" className="font-mono text-xs uppercase px-2.5 py-0.5">
          Medium Severity
        </Badge>
      )
    }
    return (
      <Badge variant="outline" className="font-mono text-xs uppercase text-cyan-400 border-cyan-500/40 px-2.5 py-0.5">
        Low Severity
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
                      {incident?.severity && getSeverityBadge(incident.severity)}
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
                    {/* SECTION 1: Large Keyframe with Interactive Zoom */}
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-semibold text-muted-foreground uppercase font-mono flex items-center gap-1.5">
                          <Radio className="h-3.5 w-3.5 text-primary" />
                          Peak Keyframe Capture
                        </span>
                        <div className="flex items-center gap-1">
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => setIsZoomed((z) => !z)}
                            className="h-7 px-2 text-xs gap-1 text-muted-foreground hover:text-foreground"
                          >
                            {isZoomed ? <ZoomOut className="h-3.5 w-3.5" /> : <ZoomIn className="h-3.5 w-3.5" />}
                            <span>{isZoomed ? 'Reset' : 'Zoom 2x'}</span>
                          </Button>
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

                      {/* Video / Keyframe Frame Box */}
                      <div
                        dir="ltr"
                        className={cn(
                          'relative aspect-video w-full rounded-xl bg-black/95 overflow-hidden border border-border/80 shadow-md transition-all select-none',
                          isZoomed ? 'cursor-zoom-out' : 'cursor-zoom-in'
                        )}
                        onClick={() => setIsZoomed((z) => !z)}
                      >
                        {incident?.keyframe_url ? (
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
                            No keyframe image available
                          </div>
                        )}

                        <div className="absolute bottom-2.5 start-2.5 px-2 py-0.5 rounded bg-black/75 text-cyan-400 font-mono text-[10px] border border-cyan-500/30">
                          {incident?.class_name?.replace(/_/g, ' ') || 'Incident'}
                        </div>
                      </div>
                    </div>

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

                    {/* SECTION 4: Quick Action Toolbar (PDF, Copy, Alert) */}
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
                  <input
                    type="text"
                    value={reviewerName}
                    onChange={(e) => handleReviewerChange(e.target.value)}
                    className="px-2.5 py-1 rounded border border-border bg-background text-xs font-mono text-foreground focus:ring-1 focus:ring-primary outline-none w-44"
                  />
                </div>

                <div className="flex items-center gap-2 w-full sm:w-auto justify-end">
                  {/* Reject */}
                  <Button
                    variant="outline"
                    size="sm"
                    disabled={statusMutation.isPending || incident?.status === 'rejected'}
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
                    disabled={statusMutation.isPending || incident?.status === 'approved'}
                    onClick={() => setConfirmAction('approved')}
                    className="gap-1.5 text-xs border-border hover:border-[var(--status-ok)] text-[var(--status-ok)] hover:bg-[var(--status-ok)]/10"
                  >
                    <ShieldCheck className="h-3.5 w-3.5" />
                    <span>{t('incidents.approve')}</span>
                  </Button>

                  {/* Dispatch 911 */}
                  <Button
                    size="sm"
                    disabled={statusMutation.isPending || incident?.status === 'dispatched'}
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

          {/* Fullscreen Lightbox Modal */}
          {isLightboxOpen && incident?.keyframe_url && (
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
                src={incident.keyframe_url}
                alt="Enlarged Keyframe"
                className="max-h-[90vh] max-w-[90vw] object-contain rounded-lg shadow-2xl"
              />
              <span className="text-xs text-slate-400 font-mono mt-3">
                {incident.incident_id} • Click anywhere to exit fullscreen
              </span>
            </div>
          )}
        </div>
      )}
    </AnimatePresence>
  )
}
