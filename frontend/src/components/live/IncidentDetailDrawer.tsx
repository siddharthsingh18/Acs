import React from 'react'
import { Link } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { motion, AnimatePresence } from 'framer-motion'
import {
  X,
  ExternalLink,
  Clock,
  Calendar,
  AlertTriangle,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import type { StreamIncidentMessage } from '@/api/client'

interface IncidentDetailDrawerProps {
  incident: StreamIncidentMessage | null
  onClose: () => void
}

export const IncidentDetailDrawer: React.FC<IncidentDetailDrawerProps> = ({
  incident,
  onClose,
}) => {
  const { t } = useTranslation()

  return (
    <AnimatePresence>
      {incident && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 cursor-pointer"
          />

          {/* Drawer Panel */}
          <motion.div
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ type: 'spring', damping: 25, stiffness: 300 }}
            className="fixed top-0 end-0 bottom-0 w-full max-w-md bg-card border-s border-border shadow-2xl z-50 flex flex-col overflow-hidden"
          >
            {/* Header */}
            <div className="p-4 border-b border-border flex items-center justify-between bg-muted/30">
              <div className="flex items-center gap-2">
                <AlertTriangle className="h-5 w-5 text-[var(--status-alert)]" />
                <h3 className="font-bold text-base font-mono">
                  {incident.incident_id}
                </h3>
              </div>
              <Button
                variant="ghost"
                size="icon"
                onClick={onClose}
                className="h-8 w-8 text-muted-foreground hover:text-foreground"
              >
                <X className="h-4 w-4" />
              </Button>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto p-5 space-y-5">
              {/* Keyframe Preview */}
              <div className="space-y-1.5">
                <span className="text-xs font-semibold text-muted-foreground uppercase font-mono">
                  {t('live.keyframe')}
                </span>
                <div className="rounded-lg overflow-hidden border border-border aspect-video bg-black/80 shadow-md">
                  {incident.keyframe_url ? (
                    <img
                      src={incident.keyframe_url}
                      alt={incident.incident_id}
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <div className="h-full flex items-center justify-center text-xs text-muted-foreground">
                      No keyframe image available
                    </div>
                  )}
                </div>
              </div>

              {/* Classification & Confidence */}
              <div className="p-4 rounded-lg border border-border bg-background/50 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-muted-foreground">
                    Collision Class:
                  </span>
                  <Badge variant="destructive" className="font-bold text-xs uppercase">
                    {t(`classes.${incident.class_name}`, incident.class_name)}
                  </Badge>
                </div>

                <div className="space-y-1">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-muted-foreground">Detection Confidence:</span>
                    <span className="font-mono font-bold text-cyan-400">
                      {(incident.max_confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="h-2 w-full bg-muted rounded-full overflow-hidden">
                    <div
                      className="h-full bg-cyan-400 rounded-full"
                      style={{ width: `${Math.round(incident.max_confidence * 100)}%` }}
                    />
                  </div>
                </div>
              </div>

              {/* Temporal Metrics */}
              <div className="grid grid-cols-2 gap-3 text-xs font-mono">
                <div className="p-3 rounded-lg border border-border bg-muted/20">
                  <div className="flex items-center gap-1.5 text-muted-foreground mb-1">
                    <Clock className="h-3.5 w-3.5 text-primary" />
                    <span>Duration</span>
                  </div>
                  <span className="text-sm font-bold text-foreground">
                    {incident.duration.toFixed(2)}s
                  </span>
                </div>

                <div className="p-3 rounded-lg border border-border bg-muted/20">
                  <div className="flex items-center gap-1.5 text-muted-foreground mb-1">
                    <Calendar className="h-3.5 w-3.5 text-cyan-400" />
                    <span>Timestamp</span>
                  </div>
                  <span className="text-xs font-bold text-foreground truncate block">
                    {incident.start_timestamp.split('T')[1]?.slice(0, 8) || incident.start_timestamp}
                  </span>
                </div>
              </div>

              {/* System Note */}
              <div className="p-3.5 rounded-lg border border-primary/20 bg-primary/5 text-xs text-muted-foreground space-y-1">
                <span className="font-bold text-primary block">
                  Automated Event Manager
                </span>
                <p className="text-[11px] leading-relaxed">
                  Temporal state machine verified 5 consecutive collision frames with peak keyframe selection and cooldown locking.
                </p>
              </div>
            </div>

            {/* Footer with Links */}
            <div className="p-4 border-t border-border bg-card flex items-center justify-between gap-3">
              <Button
                variant="outline"
                size="sm"
                onClick={onClose}
                className="text-xs border-border"
              >
                Close
              </Button>

              <Link to={`/incidents/${incident.incident_id}`} className="flex-1">
                <Button
                  size="sm"
                  className="w-full gap-2 text-xs font-semibold bg-primary hover:bg-primary/90 text-primary-foreground"
                >
                  <ExternalLink className="h-3.5 w-3.5" />
                  View Full Dossier
                </Button>
              </Link>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  )
}
