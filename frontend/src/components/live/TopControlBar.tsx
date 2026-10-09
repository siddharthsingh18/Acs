import React, { useState, useRef } from 'react'
import { useTranslation } from 'react-i18next'
import {
  Play,
  Square,
  Upload,
  Film,
  Volume2,
  VolumeX,
  AlertCircle,
  X,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'
import type { ConfigResponse } from '@/api/client'

interface TopControlBarProps {
  sourceType: 'sample' | 'upload'
  setSourceType: (type: 'sample' | 'upload') => void
  selectedFile: File | null
  setSelectedFile: (file: File | null) => void
  selectedModel: string
  setSelectedModel: (model: string) => void
  confidence: number
  setConfidence: (val: number) => void
  useInjuryModel: boolean
  setUseInjuryModel: (val: boolean) => void
  isStreaming: boolean
  isStarting: boolean
  onStart: () => void
  onCancel: () => void
  isMuted: boolean
  setIsMuted: (val: boolean | ((prev: boolean) => boolean)) => void
  configData?: ConfigResponse
}

const MAX_FILE_SIZE_MB = 250
const ALLOWED_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv']

export const TopControlBar: React.FC<TopControlBarProps> = ({
  sourceType,
  setSourceType,
  selectedFile,
  setSelectedFile,
  selectedModel,
  setSelectedModel,
  confidence,
  setConfidence,
  useInjuryModel,
  setUseInjuryModel,
  isStreaming,
  isStarting,
  onStart,
  onCancel,
  isMuted,
  setIsMuted,
  configData,
}) => {
  const { t } = useTranslation()
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const [isDragOver, setIsDragOver] = useState(false)
  const [validationError, setValidationError] = useState<string | null>(null)

  const injuryModelAvailable = Boolean(
    configData?.defaults?.injury_model_available
  )

  const validateAndSetFile = (file: File) => {
    setValidationError(null)
    const ext = '.' + (file.name.split('.').pop() || '').toLowerCase()
    if (!ALLOWED_EXTENSIONS.includes(ext)) {
      setValidationError(`Invalid file format. Allowed: ${ALLOWED_EXTENSIONS.join(', ')}`)
      return
    }
    const fileSizeMB = file.size / (1024 * 1024)
    if (fileSizeMB > MAX_FILE_SIZE_MB) {
      setValidationError(`File is too large (${fileSizeMB.toFixed(1)}MB). Limit is ${MAX_FILE_SIZE_MB}MB.`)
      return
    }
    setSelectedFile(file)
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(true)
  }

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      validateAndSetFile(e.dataTransfer.files[0])
    }
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      validateAndSetFile(e.target.files[0])
    }
  }

  return (
    <div className="rounded-xl border border-border bg-card/80 backdrop-blur-md p-4 sm:p-5 shadow-sm space-y-4">
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 items-center">
        {/* Source Picker (Cols 1-4) */}
        <div className="lg:col-span-5 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-muted-foreground uppercase font-mono">
              {t('live.source_select')}
            </span>

            {/* Source Segmented Control */}
            <div className="flex rounded-lg border border-border p-0.5 bg-muted/40">
              <button
                type="button"
                disabled={isStreaming || isStarting}
                onClick={() => setSourceType('sample')}
                className={cn(
                  'px-2.5 py-1 text-xs font-medium rounded-md transition-all flex items-center gap-1.5',
                  sourceType === 'sample'
                    ? 'bg-primary text-primary-foreground shadow-xs'
                    : 'text-muted-foreground hover:text-foreground'
                )}
              >
                <Film className="h-3 w-3" />
                <span>{t('live.source_sample')}</span>
              </button>

              <button
                type="button"
                disabled={isStreaming || isStarting}
                onClick={() => setSourceType('upload')}
                className={cn(
                  'px-2.5 py-1 text-xs font-medium rounded-md transition-all flex items-center gap-1.5',
                  sourceType === 'upload'
                    ? 'bg-primary text-primary-foreground shadow-xs'
                    : 'text-muted-foreground hover:text-foreground'
                )}
              >
                <Upload className="h-3 w-3" />
                <span>{t('live.source_upload')}</span>
              </button>
            </div>
          </div>

          {/* Upload Dropzone or Sample Badge */}
          {sourceType === 'upload' ? (
            <div>
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                accept=".mp4,.avi,.mov,.mkv"
                className="hidden"
                disabled={isStreaming || isStarting}
              />
              <div
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => !isStreaming && !isStarting && fileInputRef.current?.click()}
                className={cn(
                  'border-2 border-dashed rounded-lg p-2.5 text-center transition-colors cursor-pointer flex items-center justify-between gap-3 text-xs',
                  isDragOver
                    ? 'border-primary bg-primary/10'
                    : 'border-border hover:border-primary/50 bg-background/50',
                  (isStreaming || isStarting) && 'opacity-60 cursor-not-allowed'
                )}
              >
                <div className="flex items-center gap-2 truncate">
                  <Upload className="h-4 w-4 text-primary shrink-0" />
                  {selectedFile ? (
                    <span className="truncate font-mono font-medium text-foreground">
                      {selectedFile.name} ({(selectedFile.size / (1024 * 1024)).toFixed(1)}MB)
                    </span>
                  ) : (
                    <span className="text-muted-foreground truncate">
                      Drag & drop footage (max {MAX_FILE_SIZE_MB}MB)
                    </span>
                  )}
                </div>

                {selectedFile && (
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation()
                      setSelectedFile(null)
                    }}
                    className="text-muted-foreground hover:text-foreground p-1"
                  >
                    <X className="h-3.5 w-3.5" />
                  </button>
                )}
              </div>
            </div>
          ) : (
            <div className="flex items-center gap-2 px-3 py-2 rounded-lg border border-border bg-background/50 text-xs font-mono text-muted-foreground">
              <Film className="h-4 w-4 text-cyan-400" />
              <span>acc_ex.mp4</span>
              <Badge variant="outline" className="ms-auto text-[10px] font-mono">
                192 Frames • 29.97 FPS
              </Badge>
            </div>
          )}

          {validationError && (
            <div className="text-[11px] text-[var(--status-alert)] flex items-center gap-1 font-medium">
              <AlertCircle className="h-3.5 w-3.5" />
              <span>{validationError}</span>
            </div>
          )}
        </div>

        {/* Model Selector & Parameters (Cols 5-8) */}
        <div className="lg:col-span-4 grid grid-cols-1 sm:grid-cols-2 gap-3">
          {/* Model Select */}
          <div className="space-y-1">
            <span className="text-xs font-semibold text-muted-foreground uppercase font-mono block">
              {t('settings.model_select')}
            </span>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              disabled={isStreaming || isStarting}
              className="w-full px-2.5 py-1.5 rounded-lg border border-border bg-background text-xs font-mono text-foreground focus:ring-1 focus:ring-primary outline-none disabled:opacity-60"
            >
              {configData?.available_models && configData.available_models.length > 0 ? (
                configData.available_models.map((m: string) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))
              ) : (
                <option value="best.pt">best.pt (yolo11n)</option>
              )}
            </select>
          </div>

          {/* Confidence Slider */}
          <div className="space-y-1">
            <div className="flex items-center justify-between text-xs font-mono">
              <span className="text-muted-foreground uppercase font-semibold">
                Confidence
              </span>
              <span className="font-bold text-cyan-400">
                {(confidence * 100).toFixed(0)}%
              </span>
            </div>
            <input
              type="range"
              min="0.10"
              max="0.90"
              step="0.05"
              value={confidence}
              onChange={(e) => setConfidence(parseFloat(e.target.value))}
              disabled={isStreaming || isStarting}
              className="w-full accent-cyan-400 h-1.5 rounded-lg bg-muted cursor-pointer disabled:opacity-60"
            />
          </div>

          {/* Secondary Injury Toggle (with tooltip/note if disabled) */}
          <div className="sm:col-span-2 flex items-center justify-between pt-1">
            <label
              className={cn(
                'flex items-center gap-2 select-none',
                injuryModelAvailable ? 'cursor-pointer' : 'cursor-not-allowed opacity-60'
              )}
              title={
                !injuryModelAvailable
                  ? 'injury.pt weights file not found on disk — disabled'
                  : 'Secondary injury detection'
              }
            >
              <input
                type="checkbox"
                checked={useInjuryModel && injuryModelAvailable}
                onChange={(e) => setUseInjuryModel(e.target.checked)}
                disabled={!injuryModelAvailable || isStreaming || isStarting}
                className="rounded border-border text-primary focus:ring-primary h-3.5 w-3.5"
              />
              <span className="text-xs font-medium text-foreground">
                Secondary Injury Model
              </span>
              {!injuryModelAvailable && (
                <span className="text-[10px] font-mono text-muted-foreground">
                  (Unavailable)
                </span>
              )}
            </label>
          </div>
        </div>

        {/* Start / Cancel Action Buttons & Sound Alert Toggle (Cols 9-12) */}
        <div className="lg:col-span-3 flex items-center justify-end gap-2.5 pt-2 lg:pt-0">
          {/* Sound Alert Toggle */}
          <Button
            variant="outline"
            size="icon"
            onClick={() => setIsMuted((prev) => !prev)}
            title={isMuted ? 'Unmute Sound Siren Alerts' : 'Mute Sound Siren Alerts'}
            className={cn(
              'h-9 w-9 border-border',
              !isMuted && 'text-cyan-400 border-cyan-500/40 bg-cyan-500/10'
            )}
          >
            {isMuted ? (
              <VolumeX className="h-4 w-4 text-muted-foreground" />
            ) : (
              <Volume2 className="h-4 w-4 text-cyan-400 animate-pulse" />
            )}
          </Button>

          {!isStreaming && !isStarting ? (
            <Button
              onClick={onStart}
              className="flex-1 sm:flex-initial gap-2 bg-primary hover:bg-primary/90 text-primary-foreground font-bold shadow-md shadow-primary/20 text-xs px-5 h-9"
            >
              <Play className="h-3.5 w-3.5 fill-current" />
              {t('live.start_stream')}
            </Button>
          ) : (
            <Button
              onClick={onCancel}
              variant="destructive"
              className="flex-1 sm:flex-initial gap-2 font-bold shadow-md text-xs px-5 h-9"
            >
              <Square className="h-3.5 w-3.5 fill-current" />
              {t('live.cancel_stream')}
            </Button>
          )}
        </div>
      </div>
    </div>
  )
}
