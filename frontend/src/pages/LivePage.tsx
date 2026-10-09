import React, { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useQuery } from '@tanstack/react-query'
import { Radio } from 'lucide-react'
import { TopControlBar } from '@/components/live/TopControlBar'
import { VideoPanel } from '@/components/live/VideoPanel'
import { LiveIncidentFeed } from '@/components/live/LiveIncidentFeed'
import { IncidentDrawer } from '@/components/incidents/IncidentDrawer'
import { TacticalToast } from '@/components/live/TacticalToast'
import { RunSummaryCards } from '@/components/live/RunSummaryCards'
import { useJobStream } from '@/hooks/useJobStream'
import { api, type StreamIncidentMessage } from '@/api/client'

export const LivePage: React.FC = () => {
  const { t } = useTranslation()

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

  // Active toast object (suppressed if user explicitly dismissed it)
  const activeToast =
    latestIncident && latestIncident.incident_id !== toastDismissedId
      ? latestIncident
      : null

  return (
    <div className="space-y-6">
      {/* Page Title & Subtitle */}
      <div>
        <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight flex items-center gap-2.5">
          <Radio className="h-7 w-7 text-primary animate-pulse" />
          {t('live.header')}
        </h1>
        <p className="text-sm text-muted-foreground mt-1">
          {t('live.subtitle')}
        </p>
      </div>

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
