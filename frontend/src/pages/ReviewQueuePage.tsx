import React, { useState, useEffect, useCallback, useMemo } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useTranslation } from 'react-i18next'
import { motion, AnimatePresence } from 'framer-motion'
import {
  CheckCircle2,
  XCircle,
  AlertTriangle,
  RotateCcw,
  SkipForward,
  Film,
  Image as ImageIcon,
  Check,
  Flame,
  Layers,
  RefreshCw,
} from 'lucide-react'
import { api, ReviewQueueItem, ReviewStatsResponse } from '@/api/client'
import { cn } from '@/lib/utils'

const CANONICAL_CLASSES = [
  'car_car_accident',
  'car_bike_accident',
  'car_person_accident',
  'car_object_accident',
  'bike_bike_accident',
  'bike_person_accident',
  'bike_object_accident',
  'car',
  'bike',
  'person',
]

export const ReviewQueuePage: React.FC = () => {
  const { t } = useTranslation()
  const queryClient = useQueryClient()

  const [currentIndex, setCurrentIndex] = useState(0)
  const [includeReviewed, setIncludeReviewed] = useState(false)
  const [showVideo, setShowVideo] = useState(false)
  const [useAnnotatedVideo, setUseAnnotatedVideo] = useState(false)
  const [selectedWrongClass, setSelectedWrongClass] = useState<string | null>(null)
  const [isWrongClassModalOpen, setIsWrongClassModalOpen] = useState(false)
  const [lastAction, setLastAction] = useState<{
    incidentId: string
    decision: string
    timestamp: number
  } | null>(null)

  // Fetch Review Queue Items
  const {
    data: queueData,
    isLoading: isQueueLoading,
    refetch: refetchQueue,
    isRefetching,
  } = useQuery({
    queryKey: ['review-queue', includeReviewed],
    queryFn: () => api.getReviewQueue(includeReviewed),
  })

  // Fetch Review Analytics & Staging Telemetry
  const { data: statsData, refetch: refetchStats } = useQuery<ReviewStatsResponse>({
    queryKey: ['review-queue-stats'],
    queryFn: () => api.getReviewStats(),
  })

  const items = useMemo(() => queueData?.items || [], [queueData])
  const currentItem = items[currentIndex] as ReviewQueueItem | undefined
  const remainingCount = Math.max(0, items.length - currentIndex)

  // Decision Mutation
  const decisionMutation = useMutation({
    mutationFn: async ({
      incidentId,
      decision,
      correctedClass,
    }: {
      incidentId: string
      decision: 'accident' | 'not_accident' | 'wrong_class'
      correctedClass?: string
    }) => {
      return api.submitReviewDecision(incidentId, {
        decision,
        corrected_class: correctedClass || null,
        reviewer_name: 'Dispatch Operator',
      })
    },
    onSuccess: (_, variables) => {
      setLastAction({
        incidentId: variables.incidentId,
        decision: variables.decision,
        timestamp: Date.now(),
      })
      queryClient.invalidateQueries({ queryKey: ['review-queue-stats'] })
      // Auto-advance to next card
      if (currentIndex < items.length - 1) {
        setCurrentIndex((prev) => prev + 1)
        setShowVideo(false)
      } else {
        // Refetch if reached end
        refetchQueue()
        refetchStats()
      }
    },
  })

  // Undo Mutation
  const undoMutation = useMutation({
    mutationFn: async (incidentId: string) => {
      return api.undoReviewDecision(incidentId)
    },
    onSuccess: () => {
      setLastAction(null)
      queryClient.invalidateQueries({ queryKey: ['review-queue'] })
      queryClient.invalidateQueries({ queryKey: ['review-queue-stats'] })
      setCurrentIndex((prev) => Math.max(0, prev - 1))
    },
  })

  // Decision Handlers
  const handleApproveAccident = useCallback(() => {
    if (!currentItem || decisionMutation.isPending) return
    decisionMutation.mutate({
      incidentId: currentItem.incident_id,
      decision: 'accident',
    })
  }, [currentItem, decisionMutation])

  const handleRejectNotAccident = useCallback(() => {
    if (!currentItem || decisionMutation.isPending) return
    decisionMutation.mutate({
      incidentId: currentItem.incident_id,
      decision: 'not_accident',
    })
  }, [currentItem, decisionMutation])

  const handleConfirmWrongClass = useCallback(
    (chosenClass: string) => {
      if (!currentItem || decisionMutation.isPending) return
      decisionMutation.mutate({
        incidentId: currentItem.incident_id,
        decision: 'wrong_class',
        correctedClass: chosenClass,
      })
      setIsWrongClassModalOpen(false)
      setSelectedWrongClass(null)
    },
    [currentItem, decisionMutation]
  )

  const handleSkip = useCallback(() => {
    if (currentIndex < items.length - 1) {
      setCurrentIndex((prev) => prev + 1)
      setShowVideo(false)
    }
  }, [currentIndex, items.length])

  const handleUndo = useCallback(() => {
    if (!lastAction || undoMutation.isPending) return
    undoMutation.mutate(lastAction.incidentId)
  }, [lastAction, undoMutation])

  // Global Keyboard Shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (isWrongClassModalOpen) return
      // Ignore if typing in an input
      if (['INPUT', 'TEXTAREA'].includes((e.target as HTMLElement).tagName)) return

      switch (e.key.toLowerCase()) {
        case 'a':
        case '1':
          e.preventDefault()
          handleApproveAccident()
          break
        case 'n':
        case '2':
          e.preventDefault()
          handleRejectNotAccident()
          break
        case 'w':
        case '3':
          e.preventDefault()
          setIsWrongClassModalOpen(true)
          break
        case 's':
        case ' ':
          e.preventDefault()
          handleSkip()
          break
        case 'z':
        case 'u':
          e.preventDefault()
          handleUndo()
          break
        case 'v':
          e.preventDefault()
          setShowVideo((prev) => !prev)
          break
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [
    handleApproveAccident,
    handleRejectNotAccident,
    handleSkip,
    handleUndo,
    isWrongClassModalOpen,
  ])

  return (
    <div className="space-y-6 max-w-6xl mx-auto pb-16">
      {/* Page Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-2 border-b border-border/50">
        <div>
          <div className="flex items-center gap-2.5">
            <h1 className="text-2xl font-black tracking-tight text-foreground flex items-center gap-2">
              <Layers className="h-6 w-6 text-primary" />
              {t('review_queue.title', 'Uncertain Detection Review Queue')}
            </h1>
            <span className="px-2 py-0.5 text-xs font-semibold rounded-full bg-primary/10 text-primary border border-primary/20">
              ACTIVE LEARNING
            </span>
          </div>
          <p className="text-sm text-muted-foreground mt-1 max-w-2xl">
            {t(
              'review_queue.subtitle',
              'Fast operator swipe triage: label borderline detections (confidence 0.30–0.60 or YOLO/VLM disagreement) into training candidates and hard negatives.'
            )}
          </p>
        </div>

        {/* Telemetry Actions */}
        <div className="flex items-center gap-3">
          <label className="flex items-center gap-2 text-xs text-muted-foreground cursor-pointer select-none bg-card/60 px-3 py-1.5 rounded-lg border border-border">
            <input
              type="checkbox"
              checked={includeReviewed}
              onChange={(e) => {
                setIncludeReviewed(e.target.checked)
                setCurrentIndex(0)
              }}
              className="rounded border-border text-primary focus:ring-primary"
            />
            {t('review_queue.view_all', 'Include already reviewed')}
          </label>

          <button
            onClick={() => {
              refetchQueue()
              refetchStats()
            }}
            disabled={isRefetching}
            className="p-2 rounded-lg border border-border bg-card/60 text-muted-foreground hover:text-foreground hover:border-primary/50 transition-colors"
            title="Refresh queue"
          >
            <RefreshCw className={cn('h-4 w-4', isRefetching && 'animate-spin')} />
          </button>
        </div>
      </div>

      {/* Dataset Staging Telemetry Bar */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="bg-card/70 border border-border/70 rounded-xl p-3 flex items-center gap-3">
          <div className="h-10 w-10 rounded-lg bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center shrink-0 text-emerald-400">
            <CheckCircle2 className="h-5 w-5" />
          </div>
          <div>
            <div className="text-xl font-bold font-mono text-foreground">
              {statsData?.accidents_approved ?? 0}
            </div>
            <div className="text-xs text-muted-foreground truncate">
              {t('review_queue.candidates_staged', 'Approved Candidates')}
            </div>
          </div>
        </div>

        <div className="bg-card/70 border border-border/70 rounded-xl p-3 flex items-center gap-3">
          <div className="h-10 w-10 rounded-lg bg-rose-500/10 border border-rose-500/30 flex items-center justify-center shrink-0 text-rose-400">
            <XCircle className="h-5 w-5" />
          </div>
          <div>
            <div className="text-xl font-bold font-mono text-foreground">
              {statsData?.hard_negatives ?? 0}
            </div>
            <div className="text-xs text-muted-foreground truncate">
              {t('review_queue.hard_negatives', 'Hard Negatives (0-box)')}
            </div>
          </div>
        </div>

        <div className="bg-card/70 border border-border/70 rounded-xl p-3 flex items-center gap-3">
          <div className="h-10 w-10 rounded-lg bg-amber-500/10 border border-amber-500/30 flex items-center justify-center shrink-0 text-amber-400">
            <AlertTriangle className="h-5 w-5" />
          </div>
          <div>
            <div className="text-xl font-bold font-mono text-foreground">
              {statsData?.wrong_class_count ?? 0}
            </div>
            <div className="text-xs text-muted-foreground truncate">
              {t('review_queue.wrong_class_count', 'Class Corrections')}
            </div>
          </div>
        </div>

        <div className="bg-card/70 border border-border/70 rounded-xl p-3 flex items-center gap-3">
          <div className="h-10 w-10 rounded-lg bg-primary/10 border border-primary/30 flex items-center justify-center shrink-0 text-primary">
            <Flame className="h-5 w-5" />
          </div>
          <div>
            <div className="text-xl font-bold font-mono text-foreground">
              {statsData?.staging_pending_export ?? 0}
            </div>
            <div className="text-xs text-muted-foreground truncate">
              {t('review_queue.pending_export', 'Staging Pending Export')}
            </div>
          </div>
        </div>
      </div>

      {/* Main Review Card Interface */}
      {isQueueLoading ? (
        <div className="h-96 rounded-2xl border border-border/60 bg-card/40 flex flex-col items-center justify-center gap-3 animate-pulse">
          <RefreshCw className="h-8 w-8 text-primary animate-spin" />
          <p className="text-sm text-muted-foreground">Loading uncertain detections...</p>
        </div>
      ) : items.length === 0 || !currentItem ? (
        <div className="h-96 rounded-2xl border border-dashed border-border/80 bg-card/30 flex flex-col items-center justify-center text-center p-8">
          <div className="h-16 w-16 rounded-2xl bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center text-emerald-400 mb-4 shadow-[0_0_20px_rgba(16,185,129,0.2)]">
            <CheckCircle2 className="h-8 w-8" />
          </div>
          <h3 className="text-xl font-bold text-foreground">
            {t('review_queue.all_caught_up', 'Review Queue All Caught Up!')}
          </h3>
          <p className="text-sm text-muted-foreground mt-2 max-w-md">
            {t(
              'review_queue.all_caught_up_desc',
              'No uncertain detections currently need review. All borderline cases have been triaged into candidates and hard negatives.'
            )}
          </p>
          <div className="mt-6 flex items-center gap-3">
            <button
              onClick={() => refetchQueue()}
              className="px-4 py-2 rounded-lg bg-card border border-border hover:border-primary/50 text-foreground text-sm font-medium flex items-center gap-2"
            >
              <RefreshCw className="h-4 w-4" />
              Re-check Queue
            </button>
            <button
              onClick={() => setIncludeReviewed(true)}
              className="px-4 py-2 rounded-lg bg-primary/10 border border-primary/30 text-primary text-sm font-medium hover:bg-primary/20 transition-colors"
            >
              Inspect Past Reviewed Items
            </button>
          </div>
        </div>
      ) : (
        <div className="space-y-4">
          {/* Card Progress Indicator */}
          <div className="flex items-center justify-between text-xs font-mono text-muted-foreground px-1">
            <span className="flex items-center gap-1.5 font-medium">
              <span className="h-2 w-2 rounded-full bg-primary animate-pulse" />
              {t('review_queue.card_counter', {
                current: currentIndex + 1,
                total: items.length,
              })}
            </span>
            <span>
              {t('review_queue.remaining', { count: remainingCount })}{' '}
              <span className="text-foreground font-bold">({remainingCount} left)</span>
            </span>
          </div>

          {/* Swipe-Style Card Container */}
          <AnimatePresence mode="wait">
            <motion.div
              key={currentItem.incident_id}
              initial={{ opacity: 0, scale: 0.96, y: 12 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.94, y: -12 }}
              transition={{ duration: 0.2 }}
              className="rounded-2xl border border-border/80 bg-card/80 backdrop-blur-md overflow-hidden shadow-2xl relative"
            >
              {/* Uncertainty Banner */}
              <div className="bg-amber-500/10 border-b border-amber-500/20 px-4 py-2.5 flex items-center justify-between gap-3 text-xs">
                <div className="flex items-center gap-2 text-amber-300 font-medium truncate">
                  <AlertTriangle className="h-4 w-4 shrink-0 text-amber-400" />
                  <span className="font-semibold">Reason for Review:</span>
                  <span className="truncate">{currentItem.uncertainty_reason}</span>
                </div>

                <div className="flex items-center gap-2 shrink-0">
                  <span className="font-mono text-muted-foreground">ID: {currentItem.incident_id}</span>
                </div>
              </div>

              {/* Media Viewport */}
              <div className="relative aspect-video max-h-[460px] w-full bg-black/90 flex items-center justify-center overflow-hidden">
                {showVideo && (currentItem.clip_url || currentItem.clip_annotated_url) ? (
                  <video
                    src={useAnnotatedVideo ? currentItem.clip_annotated_url || currentItem.clip_url || '' : currentItem.clip_url || ''}
                    controls
                    autoPlay
                    loop
                    className="w-full h-full object-contain"
                  />
                ) : currentItem.keyframe_url ? (
                  <img
                    src={currentItem.keyframe_url}
                    alt={`Incident ${currentItem.incident_id}`}
                    className="w-full h-full object-contain"
                    onError={(e) => {
                      // Fallback placeholder image
                      ;(e.target as HTMLImageElement).src =
                        'https://placehold.co/1280x720/1e293b/ffffff?text=Keyframe+Image'
                    }}
                  />
                ) : (
                  <div className="flex flex-col items-center justify-center text-muted-foreground gap-2">
                    <ImageIcon className="h-10 w-10 stroke-1" />
                    <span className="text-xs">No media preview available</span>
                  </div>
                )}

                {/* Media Switcher Floating Controls */}
                <div className="absolute top-3 right-3 flex items-center gap-2 z-10 bg-black/75 backdrop-blur-md p-1.5 rounded-xl border border-white/10 shadow-lg">
                  <button
                    onClick={() => setShowVideo(false)}
                    className={cn(
                      'px-2.5 py-1 text-xs rounded-lg font-medium transition-colors flex items-center gap-1.5',
                      !showVideo
                        ? 'bg-primary text-primary-foreground shadow'
                        : 'text-muted-foreground hover:text-white'
                    )}
                  >
                    <ImageIcon className="h-3.5 w-3.5" />
                    Keyframe
                  </button>

                  {(currentItem.clip_url || currentItem.clip_annotated_url) && (
                    <button
                      onClick={() => setShowVideo(true)}
                      className={cn(
                        'px-2.5 py-1 text-xs rounded-lg font-medium transition-colors flex items-center gap-1.5',
                        showVideo
                          ? 'bg-primary text-primary-foreground shadow'
                          : 'text-muted-foreground hover:text-white'
                      )}
                    >
                      <Film className="h-3.5 w-3.5" />
                      Clip (MP4)
                    </button>
                  )}

                  {showVideo && currentItem.clip_annotated_url && (
                    <button
                      onClick={() => setUseAnnotatedVideo((prev) => !prev)}
                      className={cn(
                        'px-2.5 py-1 text-xs rounded-lg font-medium transition-colors border',
                        useAnnotatedVideo
                          ? 'bg-amber-500/20 border-amber-500/50 text-amber-300'
                          : 'border-white/10 text-muted-foreground hover:text-white'
                      )}
                    >
                      {useAnnotatedVideo ? 'Annotated Box burned' : 'Raw stream'}
                    </button>
                  )}
                </div>

                {/* Metadata Pill Floating Bottom-Left */}
                <div className="absolute bottom-3 left-3 flex flex-wrap items-center gap-2 z-10">
                  <span className="px-2.5 py-1 rounded-lg bg-black/80 backdrop-blur-md text-xs font-mono font-bold text-white border border-white/15">
                    {currentItem.class_name}
                  </span>
                  <span className="px-2.5 py-1 rounded-lg bg-black/80 backdrop-blur-md text-xs font-mono text-cyan-400 border border-cyan-500/30">
                    YOLO Conf: {(currentItem.max_confidence * 100).toFixed(1)}%
                  </span>
                  {currentItem.vlm_confidence !== null && currentItem.vlm_confidence !== undefined && (
                    <span className="px-2.5 py-1 rounded-lg bg-black/80 backdrop-blur-md text-xs font-mono text-purple-400 border border-purple-500/30">
                      VLM Conf: {(currentItem.vlm_confidence * 100).toFixed(1)}%
                    </span>
                  )}
                  {currentItem.verification_status && (
                    <span
                      className={cn(
                        'px-2.5 py-1 rounded-lg text-xs font-mono uppercase border',
                        currentItem.verification_status === 'auto_confirmed'
                          ? 'bg-emerald-950/80 border-emerald-500/40 text-emerald-400'
                          : currentItem.verification_status === 'needs_review'
                          ? 'bg-amber-950/80 border-amber-500/40 text-amber-400'
                          : 'bg-slate-900/80 border-slate-700 text-slate-300'
                      )}
                    >
                      {currentItem.verification_status}
                    </span>
                  )}
                </div>
              </div>

              {/* Card Meta & Detail Attributes */}
              <div className="p-4 bg-card border-t border-border/60 grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs">
                <div>
                  <span className="text-muted-foreground block text-[11px]">Timestamp</span>
                  <span className="font-mono font-medium text-foreground">
                    {currentItem.start_timestamp}
                  </span>
                </div>

                <div>
                  <span className="text-muted-foreground block text-[11px]">Motion / Physics Pre-Check</span>
                  <span
                    className={cn(
                      'font-medium inline-flex items-center gap-1',
                      currentItem.physics_passed === false
                        ? 'text-rose-400'
                        : 'text-emerald-400'
                    )}
                  >
                    {currentItem.physics_passed === false ? (
                      <>
                        <XCircle className="h-3.5 w-3.5" /> Flagged / Inconclusive
                      </>
                    ) : (
                      <>
                        <Check className="h-3.5 w-3.5" /> Passed Heuristic
                      </>
                    )}
                  </span>
                </div>

                <div>
                  <span className="text-muted-foreground block text-[11px]">Severity Assessment</span>
                  <span className="font-medium text-foreground uppercase">
                    {currentItem.severity || 'Medium'}{' '}
                    {currentItem.severity_score !== null && (
                      <span className="font-mono text-muted-foreground">
                        ({currentItem.severity_score}/100)
                      </span>
                    )}
                  </span>
                </div>

                <div>
                  <span className="text-muted-foreground block text-[11px]">VLM Verifier Reason</span>
                  <span className="font-medium text-foreground truncate block" title={currentItem.vlm_reason || ''}>
                    {currentItem.vlm_reason || '—'}
                  </span>
                </div>
              </div>

              {/* Rapid Action Buttons Bar */}
              <div className="p-4 bg-muted/20 border-t border-border flex flex-wrap items-center justify-between gap-3">
                {/* Keyboard Shortcut Hints */}
                <div className="hidden lg:flex items-center gap-2 text-[11px] text-muted-foreground">
                  <span className="font-semibold text-foreground">Shortcuts:</span>
                  <kbd className="px-1.5 py-0.5 rounded bg-muted border border-border font-mono">A</kbd> Accident
                  <kbd className="px-1.5 py-0.5 rounded bg-muted border border-border font-mono">N</kbd> Not Accident
                  <kbd className="px-1.5 py-0.5 rounded bg-muted border border-border font-mono">W</kbd> Wrong Class
                  <kbd className="px-1.5 py-0.5 rounded bg-muted border border-border font-mono">S</kbd> Skip
                  <kbd className="px-1.5 py-0.5 rounded bg-muted border border-border font-mono">V</kbd> Toggle Video
                </div>

                {/* Primary Action Buttons */}
                <div className="flex items-center gap-2.5 w-full sm:w-auto ml-auto">
                  {/* Skip */}
                  <button
                    onClick={handleSkip}
                    disabled={decisionMutation.isPending}
                    className="px-3.5 py-2.5 rounded-xl border border-border bg-card hover:bg-muted text-muted-foreground hover:text-foreground text-xs font-semibold flex items-center gap-1.5 transition-colors"
                  >
                    <SkipForward className="h-4 w-4" />
                    {t('review_queue.skip', 'Skip')}
                  </button>

                  {/* Wrong Class */}
                  <button
                    onClick={() => setIsWrongClassModalOpen(true)}
                    disabled={decisionMutation.isPending}
                    className="px-4 py-2.5 rounded-xl border border-amber-500/40 bg-amber-500/10 hover:bg-amber-500/20 text-amber-300 text-xs font-bold flex items-center gap-2 transition-colors shadow-sm"
                  >
                    <AlertTriangle className="h-4 w-4" />
                    {t('review_queue.wrong_class', 'Wrong Class')}
                    <kbd className="hidden sm:inline-block px-1 py-0.2 rounded bg-amber-500/20 text-[10px] font-mono">
                      W
                    </kbd>
                  </button>

                  {/* Not Accident (Hard Negative) */}
                  <button
                    onClick={handleRejectNotAccident}
                    disabled={decisionMutation.isPending}
                    className="px-4 py-2.5 rounded-xl border border-rose-500/40 bg-rose-500/10 hover:bg-rose-500/20 text-rose-300 text-xs font-bold flex items-center gap-2 transition-colors shadow-sm"
                  >
                    <XCircle className="h-4 w-4" />
                    {t('review_queue.not_accident', 'Not Accident')}
                    <span className="text-[10px] text-rose-400 font-normal hidden sm:inline">(Hard Negative)</span>
                    <kbd className="hidden sm:inline-block px-1 py-0.2 rounded bg-rose-500/20 text-[10px] font-mono">
                      N
                    </kbd>
                  </button>

                  {/* Accident (Approve Candidate) */}
                  <button
                    onClick={handleApproveAccident}
                    disabled={decisionMutation.isPending}
                    className="px-5 py-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-bold flex items-center gap-2 transition-all shadow-[0_0_16px_rgba(16,185,129,0.3)] hover:scale-[1.02] active:scale-[0.98]"
                  >
                    <CheckCircle2 className="h-4 w-4" />
                    {t('review_queue.accident', 'Accident')}
                    <span className="text-[10px] text-emerald-200 font-normal hidden sm:inline">(Candidate)</span>
                    <kbd className="hidden sm:inline-block px-1 py-0.2 rounded bg-emerald-700/60 text-[10px] font-mono">
                      A
                    </kbd>
                  </button>
                </div>
              </div>
            </motion.div>
          </AnimatePresence>

          {/* Undo Action Pill */}
          {lastAction && (
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex items-center justify-between p-3 rounded-xl bg-card border border-border/80 text-xs shadow-md"
            >
              <div className="flex items-center gap-2">
                <span className="text-muted-foreground">Labeled incident</span>
                <span className="font-mono font-bold text-foreground">{lastAction.incidentId}</span>
                <span className="text-muted-foreground">as</span>
                <span
                  className={cn(
                    'font-bold px-1.5 py-0.5 rounded text-[11px]',
                    lastAction.decision === 'accident'
                      ? 'bg-emerald-500/20 text-emerald-400'
                      : lastAction.decision === 'not_accident'
                      ? 'bg-rose-500/20 text-rose-400'
                      : 'bg-amber-500/20 text-amber-400'
                  )}
                >
                  {lastAction.decision.toUpperCase()}
                </span>
              </div>

              <button
                onClick={handleUndo}
                disabled={undoMutation.isPending}
                className="px-3 py-1 rounded-lg bg-muted hover:bg-accent text-foreground font-semibold flex items-center gap-1.5 transition-colors border border-border"
              >
                <RotateCcw className="h-3.5 w-3.5" />
                {t('review_queue.undo', 'Undo')} (Z)
              </button>
            </motion.div>
          )}
        </div>
      )}

      {/* Wrong Class Picker Modal */}
      {isWrongClassModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-sm">
          <div className="bg-card border border-border rounded-2xl p-6 max-w-md w-full shadow-2xl space-y-4 animate-in fade-in zoom-in-95 duration-150">
            <div className="flex items-center justify-between pb-2 border-b border-border">
              <h3 className="font-bold text-base text-foreground flex items-center gap-2">
                <AlertTriangle className="h-5 w-5 text-amber-400" />
                {t('review_queue.select_class_prompt', 'Select Correct Collision Class')}
              </h3>
              <button
                onClick={() => setIsWrongClassModalOpen(false)}
                className="text-muted-foreground hover:text-foreground p-1 rounded-lg"
              >
                ✕
              </button>
            </div>

            <p className="text-xs text-muted-foreground">
              Current flagged class is{' '}
              <span className="font-bold font-mono text-foreground">{currentItem?.class_name}</span>.
              Choose the verified ground-truth class to export into candidate training pool:
            </p>

            <div className="grid grid-cols-1 gap-2 max-h-64 overflow-y-auto pr-1">
              {CANONICAL_CLASSES.map((cls) => (
                <button
                  key={cls}
                  onClick={() => setSelectedWrongClass(cls)}
                  className={cn(
                    'w-full text-left px-3 py-2.5 rounded-xl border text-xs font-mono font-medium transition-all flex items-center justify-between',
                    selectedWrongClass === cls
                      ? 'border-primary bg-primary/10 text-primary shadow-[0_0_12px_rgba(6,182,212,0.2)]'
                      : 'border-border bg-card hover:bg-muted/60 text-foreground'
                  )}
                >
                  <span>{cls}</span>
                  {selectedWrongClass === cls && <Check className="h-4 w-4 text-primary" />}
                </button>
              ))}
            </div>

            <div className="flex items-center justify-end gap-2 pt-3 border-t border-border">
              <button
                onClick={() => setIsWrongClassModalOpen(false)}
                className="px-4 py-2 rounded-xl text-xs font-medium text-muted-foreground hover:text-foreground"
              >
                Cancel
              </button>
              <button
                disabled={!selectedWrongClass}
                onClick={() => selectedWrongClass && handleConfirmWrongClass(selectedWrongClass)}
                className="px-4 py-2 rounded-xl text-xs font-bold bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50 transition-colors shadow-md"
              >
                Confirm Correction
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
