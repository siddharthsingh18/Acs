import React from 'react'
import { useTranslation } from 'react-i18next'
import { motion, AnimatePresence } from 'framer-motion'
import {
  AlertTriangle,
  Clock,
  ChevronRight,
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { EmptyState } from '@/components/common/EmptyState'
import type { StreamIncidentMessage } from '@/api/client'

interface LiveIncidentFeedProps {
  incidents: StreamIncidentMessage[]
  onSelectIncident: (incident: StreamIncidentMessage) => void
}

export const LiveIncidentFeed: React.FC<LiveIncidentFeedProps> = ({
  incidents,
  onSelectIncident,
}) => {
  const { t } = useTranslation()

  return (
    <Card className="border-border h-full flex flex-col shadow-sm bg-card/90">
      <CardHeader className="py-3 px-4 border-b border-border bg-muted/40 flex flex-row items-center justify-between">
        <div className="flex items-center gap-2">
          <AlertTriangle className="h-4 w-4 text-[var(--status-alert)] animate-pulse" />
          <CardTitle className="text-sm font-semibold tracking-wide">
            {t('live.incidents_sidebar')}
          </CardTitle>
        </div>
        <Badge
          variant={incidents.length > 0 ? 'destructive' : 'outline'}
          className="font-mono text-xs px-2 py-0"
        >
          {incidents.length}
        </Badge>
      </CardHeader>

      <CardContent className="p-3 flex-1 overflow-y-auto max-h-[640px] space-y-3">
        {incidents.length === 0 ? (
          <EmptyState
            variant="shield"
            title="Stream Surveillance Clear"
            description={t('live.no_incidents_yet')}
            className="py-10 border-0 bg-transparent"
          />
        ) : (
          <AnimatePresence initial={false}>
            {incidents.map((inc) => (
              <motion.div
                key={inc.incident_id}
                initial={{ opacity: 0, x: 25, scale: 0.96 }}
                animate={{ opacity: 1, x: 0, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                transition={{ duration: 0.25, ease: 'easeOut' }}
                onClick={() => onSelectIncident(inc)}
                className="group p-3 rounded-lg border border-border bg-card/60 hover:bg-muted/40 hover:border-primary/50 transition-all cursor-pointer shadow-sm relative overflow-hidden"
              >
                {/* Visual active indicator stripe */}
                <div className="absolute start-0 top-0 bottom-0 w-1 bg-[var(--status-alert)]" />

                <div className="flex items-start justify-between gap-2 ps-2">
                  <div className="min-w-0 flex-1">
                    <span className="text-[11px] font-mono font-bold text-muted-foreground block">
                      {inc.incident_id}
                    </span>
                    <h4 className="text-xs font-bold text-foreground group-hover:text-primary transition-colors truncate mt-0.5">
                      {t(`classes.${inc.class_name}`, inc.class_name)}
                    </h4>
                  </div>
                  <Badge
                    variant="destructive"
                    className="text-[10px] font-mono font-bold shrink-0"
                  >
                    {(inc.max_confidence * 100).toFixed(0)}%
                  </Badge>
                </div>

                {/* Keyframe thumbnail */}
                {inc.keyframe_url && (
                  <div className="mt-2.5 ms-2 rounded-md overflow-hidden border border-border aspect-video bg-black/70 relative">
                    <img
                      src={inc.keyframe_url}
                      alt={inc.class_name}
                      className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent opacity-0 group-hover:opacity-100 transition-opacity flex items-end p-2">
                      <span className="text-[10px] text-white font-mono flex items-center gap-1">
                        Click to inspect <ChevronRight className="h-3 w-3" />
                      </span>
                    </div>
                  </div>
                )}

                {/* Confidence bar */}
                <div className="mt-2.5 ms-2 space-y-1">
                  <div className="h-1.5 w-full bg-muted rounded-full overflow-hidden">
                    <div
                      className="h-full bg-cyan-400 rounded-full transition-all duration-300"
                      style={{ width: `${Math.round(inc.max_confidence * 100)}%` }}
                    />
                  </div>
                </div>

                {/* Footer metadata */}
                <div className="flex items-center justify-between mt-2.5 ms-2 text-[11px] text-muted-foreground font-mono">
                  <span className="flex items-center gap-1">
                    <Clock className="h-3 w-3" />
                    {inc.duration.toFixed(1)}s
                  </span>
                  <span>
                    {inc.start_timestamp.split('T')[1]?.slice(0, 8) || inc.start_timestamp}
                  </span>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
        )}
      </CardContent>
    </Card>
  )
}
