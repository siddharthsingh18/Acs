import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useQuery } from '@tanstack/react-query'
import {
  FileSpreadsheet,
  Download,
  Filter,
  Eye,
  ChevronLeft,
  ChevronRight,
  ShieldAlert,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { api, type IncidentResponse } from '@/api/client'
import { IncidentDrawer } from '@/components/incidents/IncidentDrawer'

export const IncidentsPage: React.FC = () => {
  const { t, i18n } = useTranslation()
  const isArabic = i18n.language === 'ar'

  // Filter & Pagination States
  const [page, setPage] = useState<number>(1)
  const [pageSize] = useState<number>(10)
  const [filterClass, setFilterClass] = useState<string>('')
  const [filterSeverity, setFilterSeverity] = useState<string>('')
  const [filterStatus, setFilterStatus] = useState<string>('')

  // Drawer inspection state
  const [selectedIncidentId, setSelectedIncidentId] = useState<string | null>(null)

  // Query Incidents
  const { data, isLoading } = useQuery({
    queryKey: ['incidents', page, pageSize, filterClass, filterSeverity, filterStatus],
    queryFn: () =>
      api.getIncidents({
        page,
        page_size: pageSize,
        class: filterClass || null,
        severity: filterSeverity || null,
        status: filterStatus || null,
        sort_by: 'created_at',
        order: 'desc',
      }),
  })

  const getSeverityBadgeVariant = (severity: string) => {
    switch (severity?.toLowerCase()) {
      case 'high':
        return 'destructive'
      case 'medium':
        return 'warning'
      default:
        return 'outline'
    }
  }

  const getStatusBadgeVariant = (status: string) => {
    switch (status?.toLowerCase()) {
      case 'dispatched':
        return 'destructive'
      case 'approved':
        return 'success'
      case 'rejected':
        return 'secondary'
      default:
        return 'outline'
    }
  }

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight flex items-center gap-2.5">
            <FileSpreadsheet className="h-7 w-7 text-primary" />
            {t('incidents.header')}
          </h1>
          <p className="text-sm text-muted-foreground mt-1">
            {t('incidents.subtitle')}
          </p>
        </div>

        {/* CSV Export Button */}
        <div className="flex items-center gap-2">
          <a
            href={api.getExportCsvUrl()}
            download="incidents.csv"
            className="inline-flex"
          >
            <Button variant="outline" className="gap-2 text-xs font-semibold border-border shadow-sm">
              <Download className="h-4 w-4" />
              {t('incidents.export_csv')}
            </Button>
          </a>
        </div>
      </div>

      {/* Filter Bar */}
      <Card className="border-border bg-card/60">
        <CardContent className="p-4">
          <div className="flex flex-wrap items-center gap-4">
            <div className="flex items-center gap-2 text-xs font-semibold text-muted-foreground">
              <Filter className="h-3.5 w-3.5" />
              <span>{t('incidents.filter_class')}:</span>
            </div>

            {/* Class Filter */}
            <select
              value={filterClass}
              onChange={(e) => {
                setFilterClass(e.target.value)
                setPage(1)
              }}
              className="px-3 py-1.5 rounded-md border border-border bg-background text-xs text-foreground focus:ring-1 focus:ring-primary outline-none"
            >
              <option value="">{t('incidents.all_classes')}</option>
              <option value="car_car_accident">{t('classes.car_car_accident')}</option>
              <option value="bike_bike_accident">{t('classes.bike_bike_accident')}</option>
              <option value="car_bike_accident">{t('classes.car_bike_accident')}</option>
              <option value="car_person_accident">{t('classes.car_person_accident')}</option>
              <option value="bike_person_accident">{t('classes.bike_person_accident')}</option>
              <option value="car_object_accident">{t('classes.car_object_accident')}</option>
              <option value="accident">{t('classes.accident')}</option>
            </select>

            {/* Severity Filter */}
            <select
              value={filterSeverity}
              onChange={(e) => {
                setFilterSeverity(e.target.value)
                setPage(1)
              }}
              className="px-3 py-1.5 rounded-md border border-border bg-background text-xs text-foreground focus:ring-1 focus:ring-primary outline-none"
            >
              <option value="">{t('incidents.all_severities')}</option>
              <option value="high">{t('incidents.severity_high')}</option>
              <option value="medium">{t('incidents.severity_medium')}</option>
              <option value="low">{t('incidents.severity_low')}</option>
            </select>

            {/* Status Filter */}
            <select
              value={filterStatus}
              onChange={(e) => {
                setFilterStatus(e.target.value)
                setPage(1)
              }}
              className="px-3 py-1.5 rounded-md border border-border bg-background text-xs text-foreground focus:ring-1 focus:ring-primary outline-none"
            >
              <option value="">{t('incidents.all_statuses')}</option>
              <option value="pending">{t('incidents.status_pending')}</option>
              <option value="approved">{t('incidents.status_approved')}</option>
              <option value="dispatched">{t('incidents.status_dispatched')}</option>
              <option value="rejected">{t('incidents.status_rejected')}</option>
            </select>
          </div>
        </CardContent>
      </Card>

      {/* Incidents Table */}
      <Card className="border-border overflow-hidden shadow-sm">
        <div className="overflow-x-auto">
          <table className="w-full text-sm text-start">
            <thead className="bg-muted/50 border-b border-border text-xs uppercase text-muted-foreground font-mono">
              <tr>
                <th className="px-4 py-3 text-start">{t('live.keyframe')}</th>
                <th className="px-4 py-3 text-start">{t('incidents.table_id')}</th>
                <th className="px-4 py-3 text-start">{t('incidents.table_time')}</th>
                <th className="px-4 py-3 text-start">{t('incidents.table_class')}</th>
                <th className="px-4 py-3 text-start">{t('incidents.table_conf')}</th>
                <th className="px-4 py-3 text-start">{t('incidents.table_severity')}</th>
                <th className="px-4 py-3 text-start">{t('incidents.table_status')}</th>
                <th className="px-4 py-3 text-end">{t('incidents.table_actions')}</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {isLoading ? (
                <tr>
                  <td colSpan={8} className="py-8 text-center text-muted-foreground text-sm">
                    Loading incidents...
                  </td>
                </tr>
              ) : !data || data.items.length === 0 ? (
                <tr>
                  <td colSpan={8} className="py-12 text-center text-muted-foreground text-sm">
                    <ShieldAlert className="h-8 w-8 mx-auto mb-2 text-muted-foreground/50" />
                    {t('incidents.empty')}
                  </td>
                </tr>
              ) : (
                data.items.map((inc: IncidentResponse) => (
                  <tr
                    key={inc.incident_id}
                    className="hover:bg-accent/40 transition-colors"
                  >
                    {/* Keyframe thumbnail */}
                    <td className="px-4 py-3">
                      <div className="h-12 w-20 rounded overflow-hidden border border-border bg-black/60 shrink-0">
                        {inc.keyframe_url ? (
                          <img
                            src={inc.keyframe_url}
                            alt={inc.incident_id}
                            className="h-full w-full object-cover"
                          />
                        ) : (
                          <div className="h-full w-full flex items-center justify-center text-[10px] text-muted-foreground">
                            N/A
                          </div>
                        )}
                      </div>
                    </td>

                    {/* ID */}
                    <td className="px-4 py-3 font-mono font-bold text-xs text-foreground">
                      {inc.incident_id}
                    </td>

                    {/* Timestamp */}
                    <td className="px-4 py-3 font-mono text-xs text-muted-foreground">
                      {inc.start_timestamp.replace('T', ' ')}
                    </td>

                    {/* Class */}
                    <td className="px-4 py-3 font-semibold text-foreground text-xs">
                      {t(`classes.${inc.class_name}`, inc.class_name)}
                    </td>

                    {/* Confidence */}
                    <td className="px-4 py-3 font-mono text-xs font-semibold text-cyan-400">
                      {(inc.max_confidence * 100).toFixed(0)}%
                    </td>

                    {/* Severity */}
                    <td className="px-4 py-3">
                      <Badge
                        variant={getSeverityBadgeVariant(inc.severity) as any}
                        className="text-[10px] uppercase font-mono"
                      >
                        {t(`incidents.severity_${inc.severity}`, inc.severity)}
                      </Badge>
                    </td>

                    {/* Status */}
                    <td className="px-4 py-3">
                      <Badge
                        variant={getStatusBadgeVariant(inc.status) as any}
                        className="text-[10px] uppercase font-mono"
                      >
                        {t(`incidents.status_${inc.status}`, inc.status)}
                      </Badge>
                    </td>

                    {/* Actions */}
                    <td className="px-4 py-3 text-end">
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => setSelectedIncidentId(inc.incident_id)}
                        className="h-8 gap-1.5 text-xs text-primary hover:text-primary hover:bg-primary/10"
                      >
                        <Eye className="h-3.5 w-3.5" />
                        <span>{t('incidents.view_details')}</span>
                      </Button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination Bar */}
        {data && data.total_pages > 1 && (
          <div className="p-4 border-t border-border flex items-center justify-between">
            <span className="text-xs text-muted-foreground font-mono">
              Page {data.page} of {data.total_pages} ({data.total} total)
            </span>
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                disabled={page <= 1}
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                className="h-8 px-2.5 text-xs border-border"
              >
                {isArabic ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
              </Button>
              <Button
                variant="outline"
                size="sm"
                disabled={page >= data.total_pages}
                onClick={() => setPage((p) => p + 1)}
                className="h-8 px-2.5 text-xs border-border"
              >
                {isArabic ? <ChevronLeft className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
              </Button>
            </div>
          </div>
        )}
      </Card>

      {/* Shared Incident Drawer */}
      <IncidentDrawer
        incidentId={selectedIncidentId}
        isOpen={Boolean(selectedIncidentId)}
        onClose={() => setSelectedIncidentId(null)}
        incidentList={data?.items || []}
        onSelectIncident={(id) => setSelectedIncidentId(id)}
      />
    </div>
  )
}
