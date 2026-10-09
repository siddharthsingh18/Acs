import React from 'react'
import { Link } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { motion } from 'framer-motion'
import {
  CheckCircle2,
  ArrowRight,
  RotateCcw,
} from 'lucide-react'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import type { StreamDoneMessage } from '@/api/client'

interface RunSummaryCardsProps {
  summary: StreamDoneMessage['summary']
  onReset: () => void
}

export const RunSummaryCards: React.FC<RunSummaryCardsProps> = ({
  summary,
  onReset,
}) => {
  const { t } = useTranslation()

  return (
    <motion.div
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="space-y-4"
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <CheckCircle2 className="h-5 w-5 text-[var(--status-ok)]" />
          <h3 className="font-bold text-base text-foreground font-mono">
            Analysis Run Summary
          </h3>
        </div>

        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={onReset}
            className="gap-1.5 text-xs border-border"
          >
            <RotateCcw className="h-3.5 w-3.5" />
            New Run
          </Button>

          <Link to="/incidents">
            <Button
              size="sm"
              className="gap-1.5 text-xs bg-primary hover:bg-primary/90 text-primary-foreground font-semibold"
            >
              <span>View In Master Log</span>
              <ArrowRight className="h-3.5 w-3.5" />
            </Button>
          </Link>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        {/* Frames Processed */}
        <Card className="border-border bg-card/70 p-4 flex flex-col justify-between">
          <span className="text-xs font-semibold text-muted-foreground uppercase font-mono">
            {t('live.stats_frames')}
          </span>
          <span className="text-2xl font-black font-mono text-foreground mt-2">
            {summary.frames_processed}
            <span className="text-xs font-normal text-muted-foreground ms-1">
              / {summary.total_frames}
            </span>
          </span>
        </Card>

        {/* Inference FPS */}
        <Card className="border-border bg-card/70 p-4 flex flex-col justify-between">
          <span className="text-xs font-semibold text-muted-foreground uppercase font-mono">
            {t('live.stats_fps')}
          </span>
          <span className="text-2xl font-black font-mono text-cyan-400 mt-2">
            {summary.fps ? summary.fps.toFixed(1) : '—'}
          </span>
        </Card>

        {/* Incidents Detected */}
        <Card className="border-border bg-card/70 p-4 flex flex-col justify-between">
          <span className="text-xs font-semibold text-muted-foreground uppercase font-mono">
            {t('live.stats_incidents')}
          </span>
          <span className="text-2xl font-black font-mono text-[var(--status-alert)] mt-2">
            {summary.incidents_detected}
          </span>
        </Card>
      </div>
    </motion.div>
  )
}
