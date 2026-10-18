import React, { useState, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { useQuery } from '@tanstack/react-query'
import { Radio } from 'lucide-react'
import { TopControlBar } from '@/components/live/TopControlBar'
import { VideoPanel } from '@/components/live/VideoPanel'
import { LiveIncidentFeed } from '@/components/live/LiveIncidentFeed'
import { IncidentDrawer } from '@/components/incidents/IncidentDrawer'
import { TacticalToast } from '@/components/live/TacticalToast'
import { RunSummaryCards } from '@/components/live/RunSummaryCards'
import { CameraWall } from '@/components/live/CameraWall'
import { useJobStream } from '@/hooks/useJobStream'
import { api, type StreamIncidentMessage } from '@/api/client'
import { LayoutGrid, PlaySquare } from 'lucide-react'
import { cn } from '@/lib/utils'

export const LivePage: React.FC = () => {
  const { t } = useTranslation()

  // View Mode: 'wall' (multi-camera live grid) or 'analyzer' (single file batch inspector)
  const [viewMode, setViewMode] = useState<'wall' | 'analyzer'>(() => {
    if (typeof window !== 'undefined' && localStorage.getItem('acs_demo_mode') === 'true') {
      return 'analyzer'
    }
    return 'wall'
  })

  // Control Form States
  const [sourceType, setSourceType] = useState<'sample' | 'upload'>('sample')
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [selectedModel, setSelectedModel] = useState<string>('best.pt')
  const [confidence, setConfidence] = useState<number>(0.25)
  const [useInjuryModel, setUseInjuryModel] = useState<boolean>(false)

  // Drawer & Toast inspection state
  const [drawerIncident, setDrawerIncident] = useState<StreamIncidentMessage | null>(null)
  const [toastDismissedId, setToastDismissedId] = useState<string | null>(null)

  // System Config Query
  const { data: configData } = useQuery({
    queryKey: ['config'],
    queryFn: () => api.getConfig(),
  })

  // Stream Hook
  const {
    status,
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
  } = useJobStream()

  const handleStart = () => {
    startJob({
      source: sourceType,
      file: selectedFile,
      modelName: selectedModel,
      confidence,
      useInjuryModel,
    })
  }

  // Keyboard shortcuts on /live: Space to start/stop, I to open latest incident
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Do not trigger if user is typing in an input/select/textarea
      const target = e.target as HTMLElement
      if (
        target.tagName === 'INPUT' ||
        target.tagName === 'TEXTAREA' ||
        target.tagName === 'SELECT' ||
        target.isContentEditable
      ) {
        return
      }

      // Space to Start/Stop stream
      if (e.code === 'Space') {
        e.preventDefault()
        if (status === 'streaming') {
          cancelJob()
        } else if (status !== 'starting') {
          handleStart()
        }
      }

      // 'I' or 'i' to open latest incident
      if (e.key.toLowerCase() === 'i' && !e.metaKey && !e.ctrlKey) {
        e.preventDefault()
        const targetInc = latestIncident || (incidents.length > 0 ? incidents[0] : null)
        if (targetInc) {
          setDrawerIncident(targetInc)
        }
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [status, latestIncident, incidents, cancelJob, sourceType, selectedFile, selectedModel, confidence, useInjuryModel])

  // Active toast object (suppressed if user explicitly dismissed it)
  const activeToast =
    latestIncident && latestIncident.incident_id !== toastDismissedId
      ? latestIncident
      : null

  return (
    <div className="space-y-6">
      {/* Page Title & Subtitle + Shortcut Legend */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight flex items-center gap-2.5">
            <Radio className="h-7 w-7 text-primary animate-pulse" />
            {t('live.header')}
          </h1>
          <p className="text-sm text-muted-foreground mt-1">
            {t('live.subtitle')}
          </p>
        </div>

        {/* Tactical Keyboard Shortcuts Badge */}
        <div className="hidden lg:flex items-center gap-2 px-3 py-1.5 rounded-lg border border-border bg-card/60 text-xs font-mono text-muted-foreground">
          <span className="text-[10px] uppercase font-bold text-primary">Shortcuts:</span>
          <div className="flex items-center gap-1.5">
            <kbd className="px-1.5 py-0.5 rounded bg-muted text-[10px] border border-border text-foreground font-bold">
              Space
            </kbd>
            <span>Start/Stop</span>
          </div>
          <span className="text-border">|</span>
          <div className="flex items-center gap-1.5">
            <kbd className="px-1.5 py-0.5 rounded bg-muted text-[10px] border border-border text-foreground font-bold">
              I
            </kbd>
            <span>Latest Incident</span>
          </div>
        </div>
      </div>

      {/* View Switcher Tabs: Live Camera Wall vs File Analyzer */}
      <div className="flex items-center gap-1 p-1 rounded-xl border border-border bg-card/60 backdrop-blur-md shadow-sm w-fit">
        <button
          onClick={() => setViewMode('wall')}
          className={cn(
            'flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold transition-all',
            viewMode === 'wall'
              ? 'bg-primary text-primary-foreground shadow-sm'
              : 'text-muted-foreground hover:text-foreground'
          )}
        >
          <LayoutGrid className="h-4 w-4" />
          <span>Live Camera Wall</span>
        </button>
        <button
          onClick={() => setViewMode('analyzer')}
          className={cn(
            'flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold transition-all',
            viewMode === 'analyzer'
              ? 'bg-primary text-primary-foreground shadow-sm'
              : 'text-muted-foreground hover:text-foreground'
          )}
        >
          <PlaySquare className="h-4 w-4" />
          <span>File Video Analyzer</span>
        </button>
      </div>

      {viewMode === 'wall' ? (
        /* Multi-Camera Live Wall Grid */
        <CameraWall />
      ) : (
        /* Single Video Stream Analyzer */
        <>
          {/* Top Control Bar */}
          <TopControlBar
            sourceType={sourceType}
            setSourceType={setSourceType}
            selectedFile={selectedFile}
            setSelectedFile={setSelectedFile}
            selectedModel={selectedModel}
            setSelectedModel={setSelectedModel}
            confidence={confidence}
            setConfidence={setConfidence}
            useInjuryModel={useInjuryModel}
            setUseInjuryModel={setUseInjuryModel}
            isStreaming={status === 'streaming'}
            isStarting={status === 'starting'}
            onStart={handleStart}
            onCancel={cancelJob}
            isMuted={isMuted}
            setIsMuted={setIsMuted}
            configData={configData}
          />

          {/* Main Analysis Grid (Left: Large Video, Right: Live Incident Feed) */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
            {/* Left Column (8 cols): Video Panel + End-of-Run Summary */}
            <div className="lg:col-span-8 space-y-6">
              <VideoPanel
                imageRef={imageRef}
                status={status}
                progress={progress}
                isIncidentActive={isIncidentActive}
                error={error}
              />

              {/* End-of-Run Summary Cards */}
              {summary && (
                <RunSummaryCards summary={summary} onReset={reset} />
              )}
            </div>

            {/* Right Column (4 cols): Live Incident Feed */}
            <div className="lg:col-span-4">
              <LiveIncidentFeed
                incidents={incidents}
                onSelectIncident={(inc) => setDrawerIncident(inc)}
              />
            </div>
          </div>
        </>
      )}

      {/* Floating Tactical Siren Toast on Incident */}
      <TacticalToast
        incident={activeToast}
        onDismiss={() => {
          if (latestIncident) setToastDismissedId(latestIncident.incident_id)
        }}
        onSelect={(inc) => {
          setDrawerIncident(inc)
          setToastDismissedId(inc.incident_id)
        }}
      />

      {/* Detail Slide-Over Drawer */}
      <IncidentDrawer
        incidentId={drawerIncident?.incident_id || null}
        isOpen={Boolean(drawerIncident)}
        onClose={() => setDrawerIncident(null)}
        incidentList={incidents}
        onSelectIncident={(id) => {
          const found = incidents.find((i) => i.incident_id === id)
          if (found) setDrawerIncident(found)
        }}
        initialIncident={drawerIncident}
      />
    </div>
  )
}
