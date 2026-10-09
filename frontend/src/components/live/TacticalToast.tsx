import React from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Siren, X, ChevronRight } from 'lucide-react'
import type { StreamIncidentMessage } from '@/api/client'

interface TacticalToastProps {
  incident: StreamIncidentMessage | null
  onDismiss: () => void
  onSelect: (incident: StreamIncidentMessage) => void
}

export const TacticalToast: React.FC<TacticalToastProps> = ({
  incident,
  onDismiss,
  onSelect,
}) => {
  return (
    <AnimatePresence>
      {incident && (
        <motion.div
          initial={{ opacity: 0, y: -20, scale: 0.95 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: -20, scale: 0.95 }}
          transition={{ duration: 0.2 }}
          className="fixed top-20 end-6 z-50 max-w-sm w-full p-4 rounded-xl border-2 border-[var(--status-alert)] bg-card/95 backdrop-blur-md shadow-[0_0_30px_rgba(244,63,94,0.35)] flex items-start gap-3 select-none"
        >
          <div className="h-9 w-9 rounded-lg bg-[var(--status-alert)]/20 border border-[var(--status-alert)]/40 flex items-center justify-center shrink-0 text-[var(--status-alert)] animate-bounce">
            <Siren className="h-5 w-5" />
          </div>

          <div
            className="flex-1 min-w-0 cursor-pointer"
            onClick={() => onSelect(incident)}
          >
            <div className="flex items-center gap-2">
              <span className="text-[10px] font-mono uppercase font-bold text-[var(--status-alert)] tracking-wider">
                ALERT DETECTED
              </span>
              <span className="text-xs font-mono text-muted-foreground">
                {incident.incident_id}
              </span>
            </div>
            <h4 className="text-sm font-bold text-foreground truncate mt-0.5">
              {incident.class_name.replace(/_/g, ' ')}
            </h4>
            <div className="flex items-center gap-2 mt-1 text-[11px] text-muted-foreground font-mono">
              <span className="text-cyan-400 font-bold">
                {(incident.max_confidence * 100).toFixed(0)}% Conf
              </span>
              <span>•</span>
              <span>{incident.duration.toFixed(1)}s duration</span>
            </div>
          </div>

          <div className="flex flex-col items-center gap-1">
            <button
              type="button"
              onClick={onDismiss}
              className="text-muted-foreground hover:text-foreground p-1 rounded transition-colors"
            >
              <X className="h-3.5 w-3.5" />
            </button>
            <button
              type="button"
              onClick={() => onSelect(incident)}
              className="text-primary hover:text-primary/80 p-1 rounded transition-colors"
            >
              <ChevronRight className="h-3.5 w-3.5" />
            </button>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
