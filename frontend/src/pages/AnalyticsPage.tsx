import React, { useState, useMemo } from 'react'
import { useTranslation } from 'react-i18next'
import { useQuery } from '@tanstack/react-query'
import {
  BarChart3,
  Calendar,
  Layers,
  Gauge,
  Clock,
  ShieldAlert,
  TrendingUp,
  Flame,
  Sparkles,
} from 'lucide-react'
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Cell,
  CartesianGrid,
} from 'recharts'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { AnimatedCounter } from '@/components/analytics/AnimatedCounter'
import { api } from '@/api/client'
import { useAppStore } from '@/store/appStore'
import { cn } from '@/lib/utils'

export const AnalyticsPage: React.FC = () => {
  const { t } = useTranslation()
  const { theme } = useAppStore()
  const isDark = theme === 'dark'

  // Date range filter state
  const [datePreset, setDatePreset] = useState<'all' | 'today' | '7d' | '30d'>('all')
  const [startDate, setStartDate] = useState<string>('')
  const [endDate, setEndDate] = useState<string>('')

  // Query high-level stats
  const { data: stats } = useQuery({
    queryKey: ['stats'],
    queryFn: () => api.getStats(),
  })

  // Query detailed incident records for time series and hourly analytics
  const { data: incidentList } = useQuery({
    queryKey: ['incidents_analytics', startDate, endDate],
    queryFn: () =>
      api.getIncidents({
        page: 1,
        page_size: 100,
        start_date: startDate || null,
        end_date: endDate || null,
        sort_by: 'created_at',
        order: 'asc',
      }),
  })

  // Handle Preset changes
  const applyPreset = (preset: 'all' | 'today' | '7d' | '30d') => {
    setDatePreset(preset)
    const now = new Date()

    if (preset === 'all') {
      setStartDate('')
      setEndDate('')
    } else if (preset === 'today') {
      const todayStr = now.toISOString().split('T')[0]
      setStartDate(todayStr)
      setEndDate(todayStr)
    } else if (preset === '7d') {
      const past = new Date()
      past.setDate(now.getDate() - 7)
      setStartDate(past.toISOString().split('T')[0])
      setEndDate(now.toISOString().split('T')[0])
    } else if (preset === '30d') {
      const past = new Date()
      past.setDate(now.getDate() - 30)
      setStartDate(past.toISOString().split('T')[0])
      setEndDate(now.toISOString().split('T')[0])
    }
  }

  const items = useMemo(() => incidentList?.items || [], [incidentList?.items])

  // 1. Incidents Over Time Series (Area Chart)
  const timeSeriesData = useMemo(() => {
    if (items.length === 0) {
      // Baseline sample distribution when starting clean
      return [
        { date: '08:00', count: 1 },
        { date: '10:00', count: 2 },
        { date: '12:00', count: 1 },
        { date: '14:00', count: 3 },
        { date: '16:00', count: 2 },
        { date: '18:00', count: 4 },
        { date: '20:00', count: 1 },
      ]
    }

    const dateMap: { [key: string]: number } = {}
    items.forEach((inc) => {
      const d = inc.start_timestamp.split('T')[0] || 'Unknown'
      dateMap[d] = (dateMap[d] || 0) + 1
    })

    return Object.entries(dateMap).map(([date, count]) => ({
      date: date.slice(5), // MM-DD
      count,
    }))
  }, [items])

  // 2. Class Breakdown (Bar Chart)
  const classChartData = useMemo(() => {
    if (stats?.by_class && Object.keys(stats.by_class).length > 0) {
      return Object.entries(stats.by_class).map(([className, count]) => ({
        name: t(`classes.${className}`, className.replace(/_/g, ' ')),
        rawName: className,
        count: Number(count) || 0,
      }))
    }

    // Default classes
    return [
      { name: 'Car-Car', count: 4 },
      { name: 'Car-Bike', count: 2 },
      { name: 'Bike-Bike', count: 1 },
      { name: 'Pedestrian', count: 1 },
    ]
  }, [stats?.by_class, t])

  // 3. Hour-of-Day Heatmap (24-Hour Distribution)
  const hourlyData = useMemo(() => {
    const hours = Array.from({ length: 24 }, (_, i) => ({
      hour: `${i.toString().padStart(2, '0')}:00`,
      count: 0,
      intensity: 0,
    }))

    if (items.length > 0) {
      items.forEach((inc) => {
        try {
          const timePart = inc.start_timestamp.split('T')[1] || ''
          const hourIdx = parseInt(timePart.split(':')[0], 10)
          if (!isNaN(hourIdx) && hourIdx >= 0 && hourIdx < 24) {
            hours[hourIdx].count += 1
          }
        } catch {}
      })
    } else {
      // Sample thermal curve: peak rush hours at 08:00, 13:00, 18:00
      hours[8].count = 2
      hours[9].count = 1
      hours[13].count = 2
      hours[17].count = 3
      hours[18].count = 4
      hours[19].count = 2
      hours[22].count = 1
    }

    const maxCount = Math.max(...hours.map((h) => h.count), 1)
    return hours.map((h) => ({
      ...h,
      intensity: h.count / maxCount,
    }))
  }, [items])

  // 4. Confidence Trend Line
  const confidenceTrendData = useMemo(() => {
    if (items.length === 0) {
      return [
        { index: 1, confidence: 82 },
        { index: 2, confidence: 89 },
        { index: 3, confidence: 85 },
        { index: 4, confidence: 93 },
        { index: 5, confidence: 88 },
        { index: 6, confidence: 91 },
      ]
    }

    return items.map((inc, i) => ({
      index: i + 1,
      id: inc.incident_id,
      confidence: Math.round((inc.max_confidence || 0) * 100),
    }))
  }, [items])

  // 5. Review Status Funnel (Detected -> Approved -> Dispatched)
  const funnelData = useMemo(() => {
    const total = stats?.total_incidents || (items.length > 0 ? items.length : 12)
    const approved =
      stats?.by_status?.approved ||
      items.filter((i) => i.status === 'approved').length ||
      5
    const dispatched =
      stats?.by_status?.dispatched ||
      items.filter((i) => i.status === 'dispatched').length ||
      3
    const rejected =
      stats?.by_status?.rejected ||
      items.filter((i) => i.status === 'rejected').length ||
      2

    const reviewed = Number(approved) + Number(dispatched) + Number(rejected)

    return {
      total: Number(total),
      reviewed,
      approved: Number(approved),
      dispatched: Number(dispatched),
      reviewRate: total > 0 ? Math.round((reviewed / total) * 100) : 0,
      dispatchRate: total > 0 ? Math.round((Number(dispatched) / total) * 100) : 0,
    }
  }, [stats, items])

  // Theme-aware chart colors
  const gridColor = isDark ? '#1e2738' : '#e2e8f0'
  const textColor = isDark ? '#94a3b8' : '#64748b'
  const tooltipBg = isDark ? '#0e121d' : '#ffffff'
  const tooltipBorder = isDark ? '#1e2738' : '#cbd5e1'
  const tooltipText = isDark ? '#f8fafc' : '#0f172a'

  const BAR_COLORS = [
    '#06b6d4',
    '#38bdf8',
    '#f43f5e',
    '#f59e0b',
    '#10b981',
    '#8b5cf6',
    '#ec4899',
  ]

  return (
    <div className="space-y-6">
      {/* Page Header & Date Range Picker */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight flex items-center gap-2.5">
            <BarChart3 className="h-7 w-7 text-primary" />
            {t('analytics.header')}
          </h1>
          <p className="text-sm text-muted-foreground mt-1">
            {t('analytics.subtitle')}
          </p>
        </div>

        {/* Date Range Picker Bar */}
        <div className="flex flex-wrap items-center gap-2 bg-card/80 p-2 rounded-xl border border-border">
          {/* Quick Preset Buttons */}
          <div className="flex rounded-lg border border-border p-0.5 bg-muted/40">
            {(['all', 'today', '7d', '30d'] as const).map((preset) => (
              <button
                key={preset}
                type="button"
                onClick={() => applyPreset(preset)}
                className={cn(
                  'px-2.5 py-1 text-xs font-mono font-medium rounded-md transition-all',
                  datePreset === preset
                    ? 'bg-primary text-primary-foreground shadow-xs'
                    : 'text-muted-foreground hover:text-foreground'
                )}
              >
                {preset === 'all'
                  ? 'All Time'
                  : preset === 'today'
                  ? 'Today'
                  : preset === '7d'
                  ? '7 Days'
                  : '30 Days'}
              </button>
            ))}
          </div>

          {/* Custom Date Inputs */}
          <div className="flex items-center gap-1.5 text-xs font-mono">
            <input
              type="date"
              value={startDate}
              onChange={(e) => {
                setStartDate(e.target.value)
                setDatePreset('all')
              }}
              className="px-2 py-1 rounded border border-border bg-background text-xs text-foreground outline-none"
            />
            <span className="text-muted-foreground">to</span>
            <input
              type="date"
              value={endDate}
              onChange={(e) => {
                setEndDate(e.target.value)
                setDatePreset('all')
              }}
              className="px-2 py-1 rounded border border-border bg-background text-xs text-foreground outline-none"
            />
          </div>
        </div>
      </div>

      {/* STAT CARDS WITH ANIMATED COUNTERS */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Incidents Today */}
        <Card className="border-border p-5 flex flex-col justify-between bg-card/80 shadow-xs">
          <div className="flex items-center justify-between text-muted-foreground text-xs font-semibold">
            <span>{t('analytics.kpi_today')}</span>
            <Calendar className="h-4 w-4 text-cyan-400" />
          </div>
          <div className="mt-3 flex items-baseline gap-2">
            <span className="text-3xl font-black font-mono text-foreground">
              <AnimatedCounter value={stats?.incidents_today ?? 0} />
            </span>
            <span className="text-[11px] font-mono text-cyan-400 font-semibold">
              Today
            </span>
          </div>
        </Card>

        {/* Total Recorded */}
        <Card className="border-border p-5 flex flex-col justify-between bg-card/80 shadow-xs">
          <div className="flex items-center justify-between text-muted-foreground text-xs font-semibold">
            <span>{t('analytics.kpi_total')}</span>
            <Layers className="h-4 w-4 text-primary" />
          </div>
          <div className="mt-3 flex items-baseline gap-2">
            <span className="text-3xl font-black font-mono text-foreground">
              <AnimatedCounter value={stats?.total_incidents ?? 0} />
            </span>
            <span className="text-[11px] font-mono text-muted-foreground">
              Master Log
            </span>
          </div>
        </Card>

        {/* Average Confidence */}
        <Card className="border-border p-5 flex flex-col justify-between bg-card/80 shadow-xs">
          <div className="flex items-center justify-between text-muted-foreground text-xs font-semibold">
            <span>{t('analytics.kpi_confidence')}</span>
            <Gauge className="h-4 w-4 text-emerald-400" />
          </div>
          <div className="mt-3 flex items-baseline gap-1">
            <span className="text-3xl font-black font-mono text-emerald-400">
              <AnimatedCounter
                value={(stats?.average_confidence || 0) * 100}
                decimals={1}
                suffix="%"
              />
            </span>
          </div>
        </Card>

        {/* Average Review Time */}
        <Card className="border-border p-5 flex flex-col justify-between bg-card/80 shadow-xs">
          <div className="flex items-center justify-between text-muted-foreground text-xs font-semibold">
            <span>{t('analytics.kpi_review_time')}</span>
            <Clock className="h-4 w-4 text-amber-400" />
          </div>
          <div className="mt-3 flex items-baseline gap-1">
            <span className="text-3xl font-black font-mono text-amber-400">
              {stats?.average_review_time_seconds ? (
                <AnimatedCounter
                  value={stats.average_review_time_seconds}
                  decimals={1}
                  suffix="s"
                />
              ) : (
                '—'
              )}
            </span>
            <span className="text-[11px] font-mono text-muted-foreground">
              Verification
            </span>
          </div>
        </Card>
      </div>

      {/* CHARTS GRID 1: Incidents Over Time & Collision Class Breakdown */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Incidents Over Time Area Chart (7 cols) */}
        <Card className="lg:col-span-7 border-border bg-card/90 shadow-sm">
          <CardHeader className="py-4 px-6 border-b border-border flex flex-row items-center justify-between">
            <div>
              <CardTitle className="text-base font-bold flex items-center gap-2">
                <TrendingUp className="h-4 w-4 text-primary" />
                Incidents Over Time
              </CardTitle>
              <CardDescription className="text-xs">
                Timeline frequency of detected collisions across camera surveillance.
              </CardDescription>
            </div>
            <Badge variant="outline" className="font-mono text-[10px]">
              Daily Volume
            </Badge>
          </CardHeader>
          <CardContent className="p-6">
            <div dir="ltr" className="h-72 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart
                  data={timeSeriesData}
                  margin={{ top: 10, right: 20, left: -10, bottom: 0 }}
                >
                  <defs>
                    <linearGradient id="colorIncidents" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.4} />
                      <stop offset="95%" stopColor="#06b6d4" stopOpacity={0.0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
                  <XAxis
                    dataKey="date"
                    stroke={textColor}
                    fontSize={11}
                    tickLine={false}
                  />
                  <YAxis
                    stroke={textColor}
                    fontSize={11}
                    tickLine={false}
                    allowDecimals={false}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: tooltipBg,
                      borderColor: tooltipBorder,
                      borderRadius: '8px',
                      color: tooltipText,
                      fontSize: '12px',
                      fontFamily: 'JetBrains Mono',
                    }}
                  />
                  <Area
                    type="monotone"
                    dataKey="count"
                    name="Incidents"
                    stroke="#06b6d4"
                    strokeWidth={2.5}
                    fillOpacity={1}
                    fill="url(#colorIncidents)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Breakdown by Class Bar Chart (5 cols) */}
        <Card className="lg:col-span-5 border-border bg-card/90 shadow-sm">
          <CardHeader className="py-4 px-6 border-b border-border flex flex-row items-center justify-between">
            <div>
              <CardTitle className="text-base font-bold flex items-center gap-2">
                <ShieldAlert className="h-4 w-4 text-[var(--status-alert)]" />
                {t('analytics.chart_class_title')}
              </CardTitle>
              <CardDescription className="text-xs">
                Proportion by collision classification.
              </CardDescription>
            </div>
          </CardHeader>
          <CardContent className="p-6">
            <div dir="ltr" className="h-72 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={classChartData}
                  layout="vertical"
                  margin={{ top: 10, right: 20, left: 30, bottom: 0 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke={gridColor} horizontal={false} />
                  <XAxis
                    type="number"
                    stroke={textColor}
                    fontSize={11}
                    tickLine={false}
                    allowDecimals={false}
                  />
                  <YAxis
                    type="category"
                    dataKey="name"
                    stroke={textColor}
                    fontSize={11}
                    tickLine={false}
                    width={90}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: tooltipBg,
                      borderColor: tooltipBorder,
                      borderRadius: '8px',
                      color: tooltipText,
                      fontSize: '12px',
                      fontFamily: 'JetBrains Mono',
                    }}
                  />
                  <Bar dataKey="count" name="Count" radius={[0, 4, 4, 0]}>
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
          </CardContent>
        </Card>
      </div>

      {/* CHARTS GRID 2: Hour-of-Day Heatmap & Confidence Trend */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Hour-of-Day Heatmap (24-Hour Distribution) (7 cols) */}
        <Card className="lg:col-span-7 border-border bg-card/90 shadow-sm">
          <CardHeader className="py-4 px-6 border-b border-border flex flex-row items-center justify-between">
            <div>
              <CardTitle className="text-base font-bold flex items-center gap-2">
                <Flame className="h-4 w-4 text-amber-400" />
                Hour-of-Day Collision Heatmap
              </CardTitle>
              <CardDescription className="text-xs">
                24-hour temporal distribution exposing high-risk surveillance windows.
              </CardDescription>
            </div>
            <div className="flex items-center gap-1.5 text-[10px] font-mono text-muted-foreground">
              <span>Low</span>
              <div className="flex gap-0.5">
                <span className="h-2.5 w-2.5 rounded-xs bg-cyan-950" />
                <span className="h-2.5 w-2.5 rounded-xs bg-cyan-600" />
                <span className="h-2.5 w-2.5 rounded-xs bg-amber-500" />
                <span className="h-2.5 w-2.5 rounded-xs bg-rose-500" />
              </div>
              <span>Peak</span>
            </div>
          </CardHeader>
          <CardContent className="p-6">
            <div dir="ltr" className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={hourlyData}
                  margin={{ top: 10, right: 10, left: -20, bottom: 0 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke={gridColor} vertical={false} />
                  <XAxis
                    dataKey="hour"
                    stroke={textColor}
                    fontSize={10}
                    tickLine={false}
                    interval={2}
                  />
                  <YAxis
                    stroke={textColor}
                    fontSize={10}
                    tickLine={false}
                    allowDecimals={false}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: tooltipBg,
                      borderColor: tooltipBorder,
                      borderRadius: '8px',
                      color: tooltipText,
                      fontSize: '12px',
                      fontFamily: 'JetBrains Mono',
                    }}
                  />
                  <Bar dataKey="count" name="Incidents" radius={[4, 4, 0, 0]}>
                    {hourlyData.map((entry, index) => {
                      // Thermal heatmap coloring
                      const color =
                        entry.intensity > 0.75
                          ? '#f43f5e'
                          : entry.intensity > 0.4
                          ? '#f59e0b'
                          : entry.intensity > 0.1
                          ? '#06b6d4'
                          : isDark
                          ? '#1e2738'
                          : '#cbd5e1'
                      return <Cell key={`hour-${index}`} fill={color} />
                    })}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Confidence Stability Trend (5 cols) */}
        <Card className="lg:col-span-5 border-border bg-card/90 shadow-sm">
          <CardHeader className="py-4 px-6 border-b border-border flex flex-row items-center justify-between">
            <div>
              <CardTitle className="text-base font-bold flex items-center gap-2">
                <Gauge className="h-4 w-4 text-emerald-400" />
                Average Confidence Trend
              </CardTitle>
              <CardDescription className="text-xs">
                Vision model confidence percentage across sequential events.
              </CardDescription>
            </div>
          </CardHeader>
          <CardContent className="p-6">
            <div dir="ltr" className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={confidenceTrendData}
                  margin={{ top: 10, right: 20, left: -10, bottom: 0 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
                  <XAxis
                    dataKey="index"
                    stroke={textColor}
                    fontSize={10}
                    tickLine={false}
                    label={{ value: 'Incident Index', position: 'insideBottom', offset: -5, fontSize: 10, fill: textColor }}
                  />
                  <YAxis
                    domain={[60, 100]}
                    stroke={textColor}
                    fontSize={10}
                    tickLine={false}
                    unit="%"
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: tooltipBg,
                      borderColor: tooltipBorder,
                      borderRadius: '8px',
                      color: tooltipText,
                      fontSize: '12px',
                      fontFamily: 'JetBrains Mono',
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="confidence"
                    name="Confidence"
                    stroke="#10b981"
                    strokeWidth={2}
                    dot={{ r: 3, fill: '#10b981' }}
                    activeDot={{ r: 6 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* REVIEW STATUS FUNNEL: Detected -> Reviewed/Approved -> Dispatched */}
      <Card className="border-border bg-card/90 shadow-sm">
        <CardHeader className="py-4 px-6 border-b border-border">
          <CardTitle className="text-base font-bold flex items-center gap-2">
            <Sparkles className="h-4 w-4 text-primary" />
            Human-in-the-Loop Review Funnel
          </CardTitle>
          <CardDescription className="text-xs">
            Operational pipeline conversion: Machine Detected $\to$ Officer Verified $\to$ 911 Dispatched.
          </CardDescription>
        </CardHeader>
        <CardContent className="p-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 relative">
            {/* Stage 1: Detected */}
            <div className="p-5 rounded-xl border border-border bg-card/60 space-y-3 relative">
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono font-bold text-muted-foreground uppercase">
                  Stage 1: Detected
                </span>
                <Badge variant="outline" className="font-mono text-xs">
                  100% Base
                </Badge>
              </div>
              <div className="text-3xl font-black font-mono text-foreground">
                <AnimatedCounter value={funnelData.total} />
              </div>
              <p className="text-xs text-muted-foreground">
                Total temporal collisions aggregated by ByteTrack and Event Manager.
              </p>
              <div className="h-2 w-full bg-primary/20 rounded-full overflow-hidden">
                <div className="h-full bg-primary rounded-full w-full" />
              </div>
            </div>

            {/* Stage 2: Reviewed / Authenticated */}
            <div className="p-5 rounded-xl border border-border bg-card/60 space-y-3 relative">
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono font-bold text-muted-foreground uppercase">
                  Stage 2: Reviewed
                </span>
                <Badge variant="success" className="font-mono text-xs">
                  {funnelData.reviewRate}% Verification
                </Badge>
              </div>
              <div className="text-3xl font-black font-mono text-[var(--status-ok)]">
                <AnimatedCounter value={funnelData.reviewed} />
              </div>
              <p className="text-xs text-muted-foreground">
                Incidents audited and authenticated by traffic desk officers.
              </p>
              <div className="h-2 w-full bg-muted rounded-full overflow-hidden">
                <div
                  className="h-full bg-[var(--status-ok)] rounded-full transition-all duration-500"
                  style={{ width: `${funnelData.reviewRate}%` }}
                />
              </div>
            </div>

            {/* Stage 3: Dispatched to 911 */}
            <div className="p-5 rounded-xl border border-border bg-card/60 space-y-3 relative">
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono font-bold text-muted-foreground uppercase">
                  Stage 3: Dispatched
                </span>
                <Badge variant="destructive" className="font-mono text-xs">
                  {funnelData.dispatchRate}% Dispatch
                </Badge>
              </div>
              <div className="text-3xl font-black font-mono text-[var(--status-alert)]">
                <AnimatedCounter value={funnelData.dispatched} />
              </div>
              <p className="text-xs text-muted-foreground">
                High-severity events dispatched to EMS, police, and fire rescue.
              </p>
              <div className="h-2 w-full bg-muted rounded-full overflow-hidden">
                <div
                  className="h-full bg-[var(--status-alert)] rounded-full transition-all duration-500"
                  style={{ width: `${funnelData.dispatchRate}%` }}
                />
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
