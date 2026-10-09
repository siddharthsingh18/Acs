import React from 'react'
import { useTranslation } from 'react-i18next'
import { useQuery } from '@tanstack/react-query'
import {
  BarChart3,
  Calendar,
  Layers,
  Gauge,
  Clock,
  ShieldAlert,
} from 'lucide-react'
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Cell,
} from 'recharts'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { api } from '@/api/client'

export const AnalyticsPage: React.FC = () => {
  const { t } = useTranslation()

  const { data: stats, isLoading } = useQuery({
    queryKey: ['stats'],
    queryFn: () => api.getStats(),
  })

  // Format Class Data for Bar Chart
  const classChartData = stats?.by_class
    ? Object.entries(stats.by_class).map(([className, count]) => ({
        name: t(`classes.${className}`, className),
        rawName: className,
        count: Number(count) || 0,
      }))
    : []

  // Palette for chart bars
  const BAR_COLORS = ['#06b6d4', '#38bdf8', '#f43f5e', '#f59e0b', '#10b981', '#8b5cf6', '#ec4899']

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight flex items-center gap-2.5">
          <BarChart3 className="h-7 w-7 text-primary" />
          {t('analytics.header')}
        </h1>
        <p className="text-sm text-muted-foreground mt-1">
          {t('analytics.subtitle')}
        </p>
      </div>

      {/* KPI Cards Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Incidents Today */}
        <Card className="border-border p-5 flex flex-col justify-between">
          <div className="flex items-center justify-between text-muted-foreground text-xs font-semibold">
            <span>{t('analytics.kpi_today')}</span>
            <Calendar className="h-4 w-4 text-cyan-400" />
          </div>
          <div className="mt-3">
            <span className="text-3xl font-black font-mono text-foreground">
              {stats?.incidents_today ?? 0}
            </span>
          </div>
        </Card>

        {/* Total Incidents */}
        <Card className="border-border p-5 flex flex-col justify-between">
          <div className="flex items-center justify-between text-muted-foreground text-xs font-semibold">
            <span>{t('analytics.kpi_total')}</span>
            <Layers className="h-4 w-4 text-primary" />
          </div>
          <div className="mt-3">
            <span className="text-3xl font-black font-mono text-foreground">
              {stats?.total_incidents ?? 0}
            </span>
          </div>
        </Card>

        {/* Avg Confidence */}
        <Card className="border-border p-5 flex flex-col justify-between">
          <div className="flex items-center justify-between text-muted-foreground text-xs font-semibold">
            <span>{t('analytics.kpi_confidence')}</span>
            <Gauge className="h-4 w-4 text-emerald-400" />
          </div>
          <div className="mt-3">
            <span className="text-3xl font-black font-mono text-emerald-400">
              {stats?.average_confidence
                ? `${(stats.average_confidence * 100).toFixed(1)}%`
                : '0.0%'}
            </span>
          </div>
        </Card>

        {/* Avg Review Time */}
        <Card className="border-border p-5 flex flex-col justify-between">
          <div className="flex items-center justify-between text-muted-foreground text-xs font-semibold">
            <span>{t('analytics.kpi_review_time')}</span>
            <Clock className="h-4 w-4 text-amber-400" />
          </div>
          <div className="mt-3">
            <span className="text-3xl font-black font-mono text-amber-400">
              {stats?.average_review_time_seconds
                ? `${stats.average_review_time_seconds.toFixed(1)}s`
                : '—'}
            </span>
          </div>
        </Card>
      </div>

      {/* Recharts Analytics Card */}
      <Card className="border-border">
        <CardHeader className="py-4 px-6 border-b border-border">
          <CardTitle className="text-base font-bold flex items-center gap-2">
            <ShieldAlert className="h-4 w-4 text-primary" />
            {t('analytics.chart_class_title')}
          </CardTitle>
          <CardDescription className="text-xs">
            Frequency distribution of collisions detected across CCTV feeds.
          </CardDescription>
        </CardHeader>
        <CardContent className="p-6">
          {isLoading ? (
            <div className="h-72 flex items-center justify-center text-sm text-muted-foreground font-mono">
              Loading analytics charts...
            </div>
          ) : classChartData.length === 0 ? (
            <div className="h-72 flex items-center justify-center text-sm text-muted-foreground font-mono">
              No incident data recorded yet.
            </div>
          ) : (
            <div className="h-80 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={classChartData} margin={{ top: 20, right: 30, left: 10, bottom: 40 }}>
                  <XAxis
                    dataKey="name"
                    stroke="#94a3b8"
                    fontSize={11}
                    tickLine={false}
                    interval={0}
                    angle={-25}
                    textAnchor="end"
                  />
                  <YAxis
                    stroke="#94a3b8"
                    fontSize={11}
                    tickLine={false}
                    allowDecimals={false}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#0e121d',
                      borderColor: '#1e2738',
                      borderRadius: '8px',
                      color: '#f8fafc',
                      fontSize: '12px',
                      fontFamily: 'JetBrains Mono',
                    }}
                  />
                  <Bar dataKey="count" radius={[6, 6, 0, 0]}>
                    {classChartData.map((_, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={BAR_COLORS[index % BAR_COLORS.length]}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
