import React from 'react'
import { useTranslation } from 'react-i18next'
import { useQuery } from '@tanstack/react-query'
import {
  Info,
  ShieldCheck,
  Layers,
  Database,
  Eye,
  Zap,
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { api, type VisionClassInfo } from '@/api/client'

export const AboutPage: React.FC = () => {
  const { t } = useTranslation()

  const { data: config } = useQuery({
    queryKey: ['config'],
    queryFn: () => api.getConfig(),
  })

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight flex items-center gap-2.5">
          <Info className="h-7 w-7 text-primary" />
          {t('about.header')}
        </h1>
        <p className="text-sm text-muted-foreground mt-1">
          {t('about.subtitle')}
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Core Architecture */}
        <div className="lg:col-span-2 space-y-6">
          <Card className="border-border">
            <CardHeader className="py-4 px-6 border-b border-border">
              <CardTitle className="text-base font-bold flex items-center gap-2">
                <Layers className="h-4 w-4 text-primary" />
                {t('about.tech_stack')}
              </CardTitle>
            </CardHeader>
            <CardContent className="p-6 space-y-4 text-xs text-foreground leading-relaxed">
              <div className="p-3.5 rounded-lg border border-border bg-muted/20 space-y-1">
                <h4 className="font-bold text-sm text-primary flex items-center gap-2">
                  <Zap className="h-3.5 w-3.5" />
                  YOLO Vision & ByteTrack Tracking
                </h4>
                <p className="text-muted-foreground">
                  Processes live CCTV streams with deep learning bounding box detection and multi-object trajectory tracking (<code className="font-mono text-cyan-400">persist=True</code>), preventing duplicate collision counting across frames.
                </p>
              </div>

              <div className="p-3.5 rounded-lg border border-border bg-muted/20 space-y-1">
                <h4 className="font-bold text-sm text-primary flex items-center gap-2">
                  <ShieldCheck className="h-3.5 w-3.5" />
                  Temporal Accident State Machine
                </h4>
                <p className="text-muted-foreground">
                  Enforces rigorous spatio-temporal validation: requires 5 consecutive frames before declaring an incident, selects the peak confidence keyframe, applies a 3.0-second inactivity timeout, and enforces a 10.0-second post-event cooldown.
                </p>
              </div>

              <div className="p-3.5 rounded-lg border border-border bg-muted/20 space-y-1">
                <h4 className="font-bold text-sm text-primary flex items-center gap-2">
                  <Database className="h-3.5 w-3.5" />
                  SQLite SQLModel Persistence & Gemini Dispatch
                </h4>
                <p className="text-muted-foreground">
                  Stores complete incident history with pagination, multi-field filtering, and human review states in SQLite. Integrates Google Gemini Multimodal AI for structured 911 dispatch dossier assessments with automated caching.
                </p>
              </div>
            </CardContent>
          </Card>

          {/* Vision Classes Directory */}
          <Card className="border-border">
            <CardHeader className="py-4 px-6 border-b border-border">
              <CardTitle className="text-base font-bold flex items-center gap-2">
                <Eye className="h-4 w-4 text-cyan-400" />
                {t('about.classes_supported')}
              </CardTitle>
              <CardDescription className="text-xs">
                Object classes recognized by the neural network with accident vs normal categorization.
              </CardDescription>
            </CardHeader>
            <CardContent className="p-6">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                {config?.classes ? (
                  config.classes.map((c: VisionClassInfo) => (
                    <div
                      key={c.name}
                      className="flex items-center justify-between p-2.5 rounded-lg border border-border bg-card/60 text-xs"
                    >
                      <div className="flex items-center gap-2">
                        <span
                          className={`h-2 w-2 rounded-full ${
                            c.is_accident ? 'bg-[var(--status-alert)]' : 'bg-[var(--status-ok)]'
                          }`}
                        />
                        <span className="font-mono font-medium text-foreground">
                          {t(`classes.${c.name}`, c.name)}
                        </span>
                      </div>
                      <Badge
                        variant={c.is_accident ? 'destructive' : 'outline'}
                        className="text-[10px] font-mono"
                      >
                        {c.is_accident ? 'Collision' : 'Normal'}
                      </Badge>
                    </div>
                  ))
                ) : (
                  <div className="col-span-2 text-center text-xs text-muted-foreground font-mono py-4">
                    Loading vision classes...
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* System Specs Column */}
        <div className="space-y-6">
          <Card className="border-border">
            <CardHeader className="py-4 px-6 border-b border-border">
              <CardTitle className="text-sm font-semibold tracking-wide">
                System Specifications
              </CardTitle>
            </CardHeader>
            <CardContent className="p-6 space-y-4 text-xs font-mono">
              <div className="flex justify-between py-1 border-b border-border">
                <span className="text-muted-foreground">{t('about.specs_device')}:</span>
                <span className="font-bold text-foreground uppercase">
                  {config?.device_info?.device || 'MPS (Metal)'}
                </span>
              </div>

              <div className="flex justify-between py-1 border-b border-border">
                <span className="text-muted-foreground">{t('about.specs_storage')}:</span>
                <span className="font-bold text-foreground">
                  SQLite (SQLModel)
                </span>
              </div>

              <div className="flex justify-between py-1 border-b border-border">
                <span className="text-muted-foreground">{t('about.version')}:</span>
                <span className="font-bold text-cyan-400">
                  ACS v2.0 Enterprise
                </span>
              </div>

              <div className="flex justify-between py-1 border-b border-border">
                <span className="text-muted-foreground">Frontend Stack:</span>
                <span className="text-foreground">
                  Vite + React 19 + Tailwind v4
                </span>
              </div>

              <div className="flex justify-between py-1 border-b border-border">
                <span className="text-muted-foreground">Backend API:</span>
                <span className="text-foreground">
                  FastAPI + Uvicorn
                </span>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
