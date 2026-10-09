import React, { useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  ArrowLeft,
  ArrowRight,
  ShieldCheck,
  ShieldX,
  Send,
  Sparkles,
  UserCheck,
  AlertTriangle,
  Download,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { api } from '@/api/client'

export const IncidentDetailPage: React.FC = () => {
  const { id } = useParams<{ id: string }>()
  const { t, i18n } = useTranslation()
  const isArabic = i18n.language === 'ar'
  const queryClient = useQueryClient()

  const [reviewerName, setReviewerName] = useState('Traffic Desk Officer')

  // Fetch Incident Details
  const { data: incident, isLoading, error } = useQuery({
    queryKey: ['incident', id],
    queryFn: () => (id ? api.getIncident(id) : Promise.reject('No ID')),
    enabled: Boolean(id),
  })

  // Status Mutation
  const statusMutation = useMutation({
    mutationFn: (status: 'approved' | 'rejected' | 'dispatched') =>
      api.updateIncidentStatus(id!, status, reviewerName),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['incident', id] })
      queryClient.invalidateQueries({ queryKey: ['incidents'] })
      queryClient.invalidateQueries({ queryKey: ['stats'] })
    },
  })

  // Dispatch Report Mutation
  const dispatchMutation = useMutation({
    mutationFn: () => api.generateDispatchReport(id!, i18n.language || 'en'),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['incident', id] })
      queryClient.invalidateQueries({ queryKey: ['incidents'] })
    },
  })

  if (isLoading) {
    return (
      <div className="py-12 text-center text-muted-foreground text-sm font-mono">
        Loading incident dossier {id}...
      </div>
    )
  }

  if (error || !incident) {
    return (
      <div className="space-y-4">
        <Link to="/incidents">
          <Button variant="ghost" size="sm" className="gap-2 text-xs">
            {isArabic ? <ArrowRight className="h-4 w-4" /> : <ArrowLeft className="h-4 w-4" />}
            {t('incidents.back_to_list')}
          </Button>
        </Link>
        <div className="p-8 text-center rounded-xl border border-border bg-card">
          <AlertTriangle className="h-10 w-10 text-[var(--status-alert)] mx-auto mb-3" />
          <h3 className="text-base font-bold">Incident Not Found</h3>
          <p className="text-xs text-muted-foreground mt-1">
            The requested incident ID could not be loaded from the database.
          </p>
        </div>
      </div>
    )
  }

  const dispatchReport = incident.dispatch_report as any

  return (
    <div className="space-y-6">
      {/* Top Navigation */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <Link to="/incidents">
            <Button variant="outline" size="icon" className="h-9 w-9 border-border">
              {isArabic ? <ArrowRight className="h-4 w-4" /> : <ArrowLeft className="h-4 w-4" />}
            </Button>
          </Link>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-2xl font-extrabold tracking-tight font-mono">
                {incident.incident_id}
              </h1>
              <Badge
                variant={
                  incident.status === 'dispatched'
                    ? 'destructive'
                    : incident.status === 'approved'
                    ? 'success'
                    : 'outline'
                }
                className="text-xs uppercase font-mono"
              >
                {t(`incidents.status_${incident.status}`, incident.status)}
              </Badge>
            </div>
            <p className="text-xs text-muted-foreground font-mono mt-0.5">
              {incident.start_timestamp.replace('T', ' ')}
            </p>
          </div>
        </div>

        {/* PDF Export Link */}
        <div className="flex items-center gap-2">
          <a
            href={api.getIncidentPdfUrl(incident.incident_id)}
            target="_blank"
            rel="noreferrer"
            className="inline-flex"
          >
            <Button variant="outline" className="gap-2 text-xs border-border">
              <Download className="h-4 w-4 text-primary" />
              {t('incidents.download_pdf')}
            </Button>
          </a>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Keyframe & Core Details */}
        <div className="lg:col-span-2 space-y-6">
          <Card className="border-border overflow-hidden">
            <CardHeader className="py-3 px-4 border-b border-border bg-muted/30">
              <CardTitle className="text-sm font-semibold tracking-wide flex items-center justify-between">
                <span>{t('live.keyframe')}</span>
                <span className="font-mono text-xs text-cyan-400">
                  {((incident.max_confidence || 0) * 100).toFixed(0)}% Confidence
                </span>
              </CardTitle>
            </CardHeader>
            <div className="aspect-video w-full bg-black/90 flex items-center justify-center overflow-hidden">
              {incident.keyframe_url ? (
                <img
                  src={incident.keyframe_url}
                  alt={incident.incident_id}
                  className="w-full h-full object-contain"
                />
              ) : (
                <span className="text-xs text-muted-foreground font-mono">
                  No Keyframe Image Available
                </span>
              )}
            </div>
          </Card>

          {/* AI Multimodal Dispatch Report */}
          <Card className="border-border">
            <CardHeader className="border-b border-border py-4 px-6 flex flex-row items-center justify-between">
              <div>
                <CardTitle className="text-base font-bold flex items-center gap-2">
                  <Sparkles className="h-4 w-4 text-cyan-400" />
                  Gemini AI Dispatch Assessment
                </CardTitle>
                <CardDescription className="text-xs">
                  Multimodal incident reasoning, severity assessment, and required dispatch units.
                </CardDescription>
              </div>

              {!dispatchReport && (
                <Button
                  size="sm"
                  onClick={() => dispatchMutation.mutate()}
                  disabled={dispatchMutation.isPending}
                  className="gap-2 text-xs font-semibold bg-cyan-600 hover:bg-cyan-500 text-white"
                >
                  <Sparkles className="h-3.5 w-3.5" />
                  {dispatchMutation.isPending
                    ? t('incidents.generating_ai')
                    : t('incidents.generate_ai_report')}
                </Button>
              )}
            </CardHeader>

            <CardContent className="p-6">
              {dispatchReport ? (
                <div className="space-y-4">
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                    <div className="p-3 rounded-lg border border-border bg-background/50">
                      <span className="text-muted-foreground block text-[10px] uppercase font-mono">
                        Severity
                      </span>
                      <span className="font-bold text-sm text-[var(--status-alert)]">
                        {dispatchReport.severity || incident.severity}
                      </span>
                    </div>

                    <div className="p-3 rounded-lg border border-border bg-background/50">
                      <span className="text-muted-foreground block text-[10px] uppercase font-mono">
                        Vehicles Involved
                      </span>
                      <span className="font-bold text-sm text-foreground">
                        {dispatchReport.vehicles_involved ?? '2'}
                      </span>
                    </div>

                    <div className="p-3 rounded-lg border border-border bg-background/50">
                      <span className="text-muted-foreground block text-[10px] uppercase font-mono">
                        Injuries Likely
                      </span>
                      <span className="font-bold text-sm text-foreground">
                        {dispatchReport.injuries_likely ? 'Yes' : 'No'}
                      </span>
                    </div>

                    <div className="p-3 rounded-lg border border-border bg-background/50">
                      <span className="text-muted-foreground block text-[10px] uppercase font-mono">
                        Dispatch Units
                      </span>
                      <span className="font-bold text-xs text-cyan-400 font-mono">
                        {Array.isArray(dispatchReport.required_services)
                          ? dispatchReport.required_services.join(', ')
                          : 'EMS, Police'}
                      </span>
                    </div>
                  </div>

                  {dispatchReport.description && (
                    <div className="p-4 rounded-lg border border-border bg-muted/30 text-xs text-foreground leading-relaxed">
                      <p className="font-semibold text-xs text-muted-foreground mb-1 uppercase font-mono">
                        Incident Summary:
                      </p>
                      {dispatchReport.description}
                    </div>
                  )}

                  {incident.dispatch_text && !dispatchReport.description && (
                    <div className="p-4 rounded-lg border border-border bg-muted/30 text-xs text-foreground whitespace-pre-wrap leading-relaxed">
                      {incident.dispatch_text}
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-center py-6 text-xs text-muted-foreground">
                  <p>No AI dispatch report generated yet.</p>
                  <p className="text-[11px] text-muted-foreground/70 mt-1">
                    Click "Generate AI Dispatch Report" to run Gemini multimodal inference on the keyframe.
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Right Column: Incident Dossier & Human Review */}
        <div className="space-y-6">
          {/* Dossier Card */}
          <Card className="border-border">
            <CardHeader className="py-4 px-6 border-b border-border">
              <CardTitle className="text-sm font-semibold tracking-wide">
                {t('incidents.details_title')}
              </CardTitle>
            </CardHeader>
            <CardContent className="p-6 space-y-4 text-xs font-mono">
              <div className="flex justify-between py-1 border-b border-border">
                <span className="text-muted-foreground">Class:</span>
                <span className="font-bold text-foreground">
                  {t(`classes.${incident.class_name}`, incident.class_name)}
                </span>
              </div>

              <div className="flex justify-between py-1 border-b border-border">
                <span className="text-muted-foreground">Confidence:</span>
                <span className="font-bold text-cyan-400">
                  {((incident.max_confidence || 0) * 100).toFixed(1)}%
                </span>
              </div>

              <div className="flex justify-between py-1 border-b border-border">
                <span className="text-muted-foreground">Duration:</span>
                <span className="font-bold text-foreground">
                  {(incident.duration || 0).toFixed(1)}s
                </span>
              </div>

              <div className="flex justify-between py-1 border-b border-border">
                <span className="text-muted-foreground">Severity:</span>
                <span className="font-bold text-[var(--status-alert)] uppercase">
                  {incident.severity}
                </span>
              </div>

              <div className="flex justify-between py-1 border-b border-border">
                <span className="text-muted-foreground">Created:</span>
                <span className="text-foreground">
                  {new Date(incident.created_at * 1000).toLocaleString()}
                </span>
              </div>

              {incident.reviewer_name && (
                <div className="p-3 rounded-lg bg-muted/40 border border-border mt-3 space-y-1">
                  <div className="flex items-center gap-1.5 text-foreground font-semibold">
                    <UserCheck className="h-3.5 w-3.5 text-primary" />
                    <span>{incident.reviewer_name}</span>
                  </div>
                  <div className="text-[11px] text-muted-foreground">
                    Reviewed: {incident.reviewed_at || 'Recently'}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Human-in-the-Loop Review Card */}
          <Card className="border-border">
            <CardHeader className="py-4 px-6 border-b border-border">
              <CardTitle className="text-sm font-semibold tracking-wide flex items-center gap-2">
                <UserCheck className="h-4 w-4 text-primary" />
                Human-in-the-Loop Review
              </CardTitle>
              <CardDescription className="text-xs">
                Authorize emergency dispatch, approve, or reject false alarm.
              </CardDescription>
            </CardHeader>
            <CardContent className="p-6 space-y-4">
              <div>
                <label className="text-xs font-semibold text-muted-foreground block mb-1">
                  Reviewing Officer Name / Badge:
                </label>
                <input
                  type="text"
                  value={reviewerName}
                  onChange={(e) => setReviewerName(e.target.value)}
                  className="w-full px-3 py-1.5 rounded-md border border-border bg-background text-xs font-mono text-foreground focus:ring-1 focus:ring-primary outline-none"
                />
              </div>

              <div className="flex flex-col gap-2 pt-2">
                <Button
                  onClick={() => statusMutation.mutate('dispatched')}
                  disabled={statusMutation.isPending}
                  className="w-full gap-2 text-xs font-bold bg-[var(--status-alert)] hover:bg-[var(--status-alert)]/90 text-white shadow-sm"
                >
                  <Send className="h-3.5 w-3.5" />
                  {t('incidents.dispatch')}
                </Button>

                <div className="grid grid-cols-2 gap-2">
                  <Button
                    variant="outline"
                    onClick={() => statusMutation.mutate('approved')}
                    disabled={statusMutation.isPending}
                    className="gap-1.5 text-xs font-semibold border-border hover:border-[var(--status-ok)] text-[var(--status-ok)] hover:bg-[var(--status-ok)]/10"
                  >
                    <ShieldCheck className="h-3.5 w-3.5" />
                    {t('incidents.approve')}
                  </Button>

                  <Button
                    variant="outline"
                    onClick={() => statusMutation.mutate('rejected')}
                    disabled={statusMutation.isPending}
                    className="gap-1.5 text-xs font-semibold border-border hover:border-slate-500 text-muted-foreground hover:bg-muted"
                  >
                    <ShieldX className="h-3.5 w-3.5" />
                    {t('incidents.reject')}
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
