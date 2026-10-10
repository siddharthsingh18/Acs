import React, { useState, useMemo } from 'react'
import { useTranslation } from 'react-i18next'
import { useQuery } from '@tanstack/react-query'
import {
  useReactTable,
  getCoreRowModel,
  flexRender,
  type ColumnDef,
  type SortingState,
} from '@tanstack/react-table'
import {
  FileSpreadsheet,
  Download,
  Eye,
  ChevronLeft,
  ChevronRight,
  ArrowUpDown,
  ArrowUp,
  ArrowDown,
  Search,
  X,
  Calendar,
  RotateCcw,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { api, type IncidentResponse } from '@/api/client'
import { IncidentDrawer } from '@/components/incidents/IncidentDrawer'
import { EmptyState } from '@/components/common/EmptyState'
import { ErrorState } from '@/components/common/ErrorState'

export const IncidentsPage: React.FC = () => {
  const { t, i18n } = useTranslation()
  const isArabic = i18n.language === 'ar'

  // Pagination & Sorting States
  const [page, setPage] = useState<number>(1)
  const [pageSize, setPageSize] = useState<number>(10)
  const [sorting, setSorting] = useState<SortingState>([
    { id: 'created_at', desc: true },
  ])

  // Filters & Search
  const [searchQuery, setSearchQuery] = useState<string>('')
  const [filterClass, setFilterClass] = useState<string>('')
  const [filterSeverity, setFilterSeverity] = useState<string>('')
  const [filterStatus, setFilterStatus] = useState<string>('')
  const [startDate, setStartDate] = useState<string>('')
  const [endDate, setEndDate] = useState<string>('')

  // Drawer inspection state
  const [selectedIncidentId, setSelectedIncidentId] = useState<string | null>(null)

  // Derive sort_by and order for API query
  const sortBy = sorting[0]?.id || 'created_at'
  const sortOrder = sorting[0]?.desc ? 'desc' : 'asc'

  // Query Incidents with server-side pagination, sorting, and filters
  const { data, isLoading, isFetching, error, refetch } = useQuery({
    queryKey: [
      'incidents',
      page,
      pageSize,
      sortBy,
      sortOrder,
      filterClass,
      filterSeverity,
      filterStatus,
      startDate,
      endDate,
    ],
    queryFn: () =>
      api.getIncidents({
        page,
        page_size: pageSize,
        sort_by: sortBy,
        order: sortOrder,
        class: filterClass || null,
        severity: filterSeverity || null,
        status: filterStatus || null,
        start_date: startDate || null,
        end_date: endDate || null,
      }),
  })

  // Filter items by client-side search query if user searches by text
  const filteredItems = useMemo(() => {
    if (!data?.items) return []
    if (!searchQuery.trim()) return data.items

    const q = searchQuery.toLowerCase().trim()
    return data.items.filter(
      (item) =>
        item.incident_id.toLowerCase().includes(q) ||
        item.class_name.toLowerCase().includes(q) ||
        item.reviewer_name?.toLowerCase().includes(q) ||
        item.severity.toLowerCase().includes(q) ||
        item.status.toLowerCase().includes(q)
    )
  }, [data?.items, searchQuery])

  // Helper Badge styles
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

  // TanStack Table Column Definitions
  const columns = useMemo<ColumnDef<IncidentResponse>[]>(
    () => [
      {
        id: 'keyframe',
        header: () => <span>{t('live.keyframe')}</span>,
        cell: ({ row }) => (
          <div className="h-12 w-20 rounded-md overflow-hidden border border-border bg-black/60 shrink-0">
            {row.original.keyframe_url ? (
              <img
                src={row.original.keyframe_url}
                alt={row.original.incident_id}
                className="h-full w-full object-cover select-none"
              />
            ) : (
              <div className="h-full w-full flex items-center justify-center text-[10px] text-muted-foreground font-mono">
                N/A
              </div>
            )}
          </div>
        ),
      },
      {
        accessorKey: 'incident_id',
        header: ({ column }) => {
          const isSorted = column.getIsSorted()
          return (
            <button
              type="button"
              onClick={() => column.toggleSorting(isSorted === 'asc')}
              className="flex items-center gap-1.5 font-mono text-xs uppercase hover:text-foreground transition-colors"
            >
              <span>{t('incidents.table_id')}</span>
              {isSorted === 'asc' ? (
                <ArrowUp className="h-3.5 w-3.5 text-primary" />
              ) : isSorted === 'desc' ? (
                <ArrowDown className="h-3.5 w-3.5 text-primary" />
              ) : (
                <ArrowUpDown className="h-3.5 w-3.5 opacity-40" />
              )}
            </button>
          )
        },
        cell: ({ getValue }) => (
          <span className="font-mono font-bold text-xs text-foreground">
            {getValue<string>()}
          </span>
        ),
      },
      {
        accessorKey: 'start_timestamp',
        header: ({ column }) => {
          const isSorted = column.getIsSorted()
          return (
            <button
              type="button"
              onClick={() => column.toggleSorting(isSorted === 'asc')}
              className="flex items-center gap-1.5 font-mono text-xs uppercase hover:text-foreground transition-colors"
            >
              <span>{t('incidents.table_time')}</span>
              {isSorted === 'asc' ? (
                <ArrowUp className="h-3.5 w-3.5 text-primary" />
              ) : isSorted === 'desc' ? (
                <ArrowDown className="h-3.5 w-3.5 text-primary" />
              ) : (
                <ArrowUpDown className="h-3.5 w-3.5 opacity-40" />
              )}
            </button>
          )
        },
        cell: ({ getValue }) => (
          <span className="font-mono text-xs text-muted-foreground whitespace-nowrap">
            {getValue<string>()?.replace('T', ' ') || '—'}
          </span>
        ),
      },
      {
        accessorKey: 'class_name',
        header: ({ column }) => {
          const isSorted = column.getIsSorted()
          return (
            <button
              type="button"
              onClick={() => column.toggleSorting(isSorted === 'asc')}
              className="flex items-center gap-1.5 font-mono text-xs uppercase hover:text-foreground transition-colors"
            >
              <span>{t('incidents.table_class')}</span>
              {isSorted === 'asc' ? (
                <ArrowUp className="h-3.5 w-3.5 text-primary" />
              ) : isSorted === 'desc' ? (
                <ArrowDown className="h-3.5 w-3.5 text-primary" />
              ) : (
                <ArrowUpDown className="h-3.5 w-3.5 opacity-40" />
              )}
            </button>
          )
        },
        cell: ({ getValue }) => {
          const raw = getValue<string>()
          return (
            <span className="font-semibold text-foreground text-xs whitespace-nowrap">
              {t(`classes.${raw}`, raw)}
            </span>
          )
        },
      },
      {
        accessorKey: 'max_confidence',
        header: ({ column }) => {
          const isSorted = column.getIsSorted()
          return (
            <button
              type="button"
              onClick={() => column.toggleSorting(isSorted === 'asc')}
              className="flex items-center gap-1.5 font-mono text-xs uppercase hover:text-foreground transition-colors"
            >
              <span>{t('incidents.table_conf')}</span>
              {isSorted === 'asc' ? (
                <ArrowUp className="h-3.5 w-3.5 text-primary" />
              ) : isSorted === 'desc' ? (
                <ArrowDown className="h-3.5 w-3.5 text-primary" />
              ) : (
                <ArrowUpDown className="h-3.5 w-3.5 opacity-40" />
              )}
            </button>
          )
        },
        cell: ({ getValue }) => {
          const val = getValue<number>() || 0
          return (
            <div className="space-y-1 w-24">
              <span className="font-mono text-xs font-semibold text-cyan-400">
                {(val * 100).toFixed(0)}%
              </span>
              <div className="h-1.5 w-full bg-muted rounded-full overflow-hidden">
                <div
                  className="h-full bg-cyan-400 rounded-full"
                  style={{ width: `${Math.round(val * 100)}%` }}
                />
              </div>
            </div>
          )
        },
      },
      {
        accessorKey: 'duration',
        header: () => <span>Duration</span>,
        cell: ({ getValue }) => (
          <span className="font-mono text-xs text-muted-foreground">
            {(getValue<number>() || 0).toFixed(1)}s
          </span>
        ),
      },
      {
        accessorKey: 'severity',
        header: () => <span>{t('incidents.table_severity')}</span>,
        cell: ({ getValue }) => {
          const sev = getValue<string>()
          return (
            <Badge
              variant={getSeverityBadgeVariant(sev) as any}
              className="text-[10px] uppercase font-mono"
            >
              {t(`incidents.severity_${sev}`, sev)}
            </Badge>
          )
        },
      },
      {
        accessorKey: 'status',
        header: () => <span>{t('incidents.table_status')}</span>,
        cell: ({ getValue }) => {
          const st = getValue<string>()
          return (
            <Badge
              variant={getStatusBadgeVariant(st) as any}
              className="text-[10px] uppercase font-mono"
            >
              {t(`incidents.status_${st}`, st)}
            </Badge>
          )
        },
      },
      {
        id: 'actions',
        header: () => <span className="text-end">{t('incidents.table_actions')}</span>,
        cell: ({ row }) => (
          <div className="text-end">
            <Button
              variant="ghost"
              size="sm"
              onClick={(e) => {
                e.stopPropagation()
                setSelectedIncidentId(row.original.incident_id)
              }}
              className="h-8 gap-1.5 text-xs text-primary hover:text-primary hover:bg-primary/10"
            >
              <Eye className="h-3.5 w-3.5" />
              <span>{t('incidents.view_details')}</span>
            </Button>
          </div>
        ),
      },
    ],
    [t]
  )

  // TanStack Table Instance
  const table = useReactTable({
    data: filteredItems,
    columns,
    state: {
      sorting,
    },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    manualSorting: true,
    manualPagination: true,
    pageCount: data?.total_pages ?? -1,
  })

  // Reset Filters handler
  const handleResetFilters = () => {
    setFilterClass('')
    setFilterSeverity('')
    setFilterStatus('')
    setStartDate('')
    setEndDate('')
    setSearchQuery('')
    setPage(1)
  }

  const isFiltered =
    Boolean(filterClass || filterSeverity || filterStatus || startDate || endDate || searchQuery)

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

        {/* Bulk CSV Export Button */}
        <div className="flex items-center gap-2">
          <a
            href={api.getExportCsvUrl()}
            download="incidents_master.csv"
            className="inline-flex"
          >
            <Button
              variant="outline"
              className="gap-2 text-xs font-semibold border-border shadow-xs hover:border-primary/50"
            >
              <Download className="h-4 w-4 text-primary" />
              {t('incidents.export_csv')}
            </Button>
          </a>
        </div>
      </div>

      {/* Filter and Search Bar */}
      <Card className="border-border bg-card/70 backdrop-blur-md">
        <CardContent className="p-4 space-y-3">
          <div className="flex flex-wrap items-center gap-3">
            {/* Search Input */}
            <div className="relative flex-1 min-w-[200px]">
              <Search className="absolute start-3 top-2.5 h-3.5 w-3.5 text-muted-foreground" />
              <input
                type="text"
                placeholder="Search incident ID, reviewer, or class..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full ps-9 pe-8 py-1.5 rounded-lg border border-border bg-background text-xs font-mono text-foreground focus:ring-1 focus:ring-primary outline-none"
              />
              {searchQuery && (
                <button
                  type="button"
                  onClick={() => setSearchQuery('')}
                  className="absolute end-2.5 top-2.5 text-muted-foreground hover:text-foreground"
                >
                  <X className="h-3.5 w-3.5" />
                </button>
              )}
            </div>

            {/* Class Filter */}
            <select
              value={filterClass}
              onChange={(e) => {
                setFilterClass(e.target.value)
                setPage(1)
              }}
              className="px-3 py-1.5 rounded-lg border border-border bg-background text-xs text-foreground focus:ring-1 focus:ring-primary outline-none"
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
              className="px-3 py-1.5 rounded-lg border border-border bg-background text-xs text-foreground focus:ring-1 focus:ring-primary outline-none"
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
              className="px-3 py-1.5 rounded-lg border border-border bg-background text-xs text-foreground focus:ring-1 focus:ring-primary outline-none"
            >
              <option value="">{t('incidents.all_statuses')}</option>
              <option value="pending">{t('incidents.status_pending')}</option>
              <option value="approved">{t('incidents.status_approved')}</option>
              <option value="dispatched">{t('incidents.status_dispatched')}</option>
              <option value="rejected">{t('incidents.status_rejected')}</option>
            </select>
          </div>

          {/* Date Range Row */}
          <div className="flex flex-wrap items-center gap-3 pt-2 border-t border-border/60 text-xs">
            <span className="text-muted-foreground font-semibold flex items-center gap-1.5 font-mono">
              <Calendar className="h-3.5 w-3.5 text-primary" />
              Date Range:
            </span>

            <input
              type="date"
              value={startDate}
              onChange={(e) => {
                setStartDate(e.target.value)
                setPage(1)
              }}
              className="px-2.5 py-1 rounded-md border border-border bg-background text-xs font-mono text-foreground focus:ring-1 focus:ring-primary outline-none"
            />
            <span className="text-muted-foreground font-mono">to</span>
            <input
              type="date"
              value={endDate}
              onChange={(e) => {
                setEndDate(e.target.value)
                setPage(1)
              }}
              className="px-2.5 py-1 rounded-md border border-border bg-background text-xs font-mono text-foreground focus:ring-1 focus:ring-primary outline-none"
            />

            {isFiltered && (
              <Button
                variant="ghost"
                size="sm"
                onClick={handleResetFilters}
                className="h-7 px-2.5 gap-1.5 text-xs text-muted-foreground hover:text-foreground ms-auto"
              >
                <RotateCcw className="h-3 w-3" />
                <span>Reset Filters</span>
              </Button>
            )}
          </div>
        </CardContent>
      </Card>

      {error && (
        <ErrorState
          title="Incident Records Unavailable"
          message="Could not load master incident records from the backend database."
          onRetry={() => refetch()}
        />
      )}

      {/* TanStack Table Card */}
      <Card className="border-border overflow-hidden shadow-sm bg-card/90">
        <div className="overflow-x-auto">
          <table className="w-full text-sm text-start">
            <thead className="bg-muted/50 border-b border-border text-xs uppercase text-muted-foreground font-mono">
              {table.getHeaderGroups().map((headerGroup) => (
                <tr key={headerGroup.id}>
                  {headerGroup.headers.map((header) => (
                    <th
                      key={header.id}
                      className="px-4 py-3 text-start font-semibold"
                    >
                      {header.isPlaceholder
                        ? null
                        : flexRender(
                            header.column.columnDef.header,
                            header.getContext()
                          )}
                    </th>
                  ))}
                </tr>
              ))}
            </thead>

            <tbody className="divide-y divide-border">
              {isLoading ? (
                // SKELETON LOADERS
                Array.from({ length: 6 }).map((_, idx) => (
                  <tr key={`skeleton-${idx}`} className="animate-pulse">
                    <td className="px-4 py-3">
                      <div className="h-12 w-20 bg-muted/60 rounded-md" />
                    </td>
                    <td className="px-4 py-3">
                      <div className="h-4 w-24 bg-muted/60 rounded" />
                    </td>
                    <td className="px-4 py-3">
                      <div className="h-4 w-32 bg-muted/60 rounded" />
                    </td>
                    <td className="px-4 py-3">
                      <div className="h-4 w-28 bg-muted/60 rounded" />
                    </td>
                    <td className="px-4 py-3">
                      <div className="h-4 w-16 bg-muted/60 rounded" />
                    </td>
                    <td className="px-4 py-3">
                      <div className="h-4 w-12 bg-muted/60 rounded" />
                    </td>
                    <td className="px-4 py-3">
                      <div className="h-5 w-16 bg-muted/60 rounded-full" />
                    </td>
                    <td className="px-4 py-3">
                      <div className="h-5 w-20 bg-muted/60 rounded-full" />
                    </td>
                    <td className="px-4 py-3 text-end">
                      <div className="h-7 w-20 bg-muted/60 rounded ms-auto" />
                    </td>
                  </tr>
                ))
              ) : table.getRowModel().rows.length === 0 ? (
                <tr>
                  <td
                    colSpan={columns.length}
                    className="p-6 text-center"
                  >
                    <EmptyState
                      variant={isFiltered ? 'search' : 'radar'}
                      title={isFiltered ? 'No Incidents Match Search Criteria' : t('incidents.empty')}
                      description={
                        isFiltered
                          ? 'Try clearing active filters, dates, or search keywords to view logged incidents.'
                          : 'As traffic surveillance runs, automated AI collision events will populate this master directory.'
                      }
                      actionLabel={isFiltered ? 'Reset Filters' : undefined}
                      onAction={isFiltered ? handleResetFilters : undefined}
                    />
                  </td>
                </tr>
              ) : (
                table.getRowModel().rows.map((row) => (
                  <tr
                    key={row.id}
                    onClick={() => setSelectedIncidentId(row.original.incident_id)}
                    className="hover:bg-accent/40 transition-colors cursor-pointer"
                  >
                    {row.getVisibleCells().map((cell) => (
                      <td key={cell.id} className="px-4 py-3">
                        {flexRender(
                          cell.column.columnDef.cell,
                          cell.getContext()
                        )}
                      </td>
                    ))}
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        {/* Server-Side Pagination Bar */}
        {data && (
          <div className="p-4 border-t border-border flex flex-col sm:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-3 text-xs text-muted-foreground font-mono">
              <span>
                Showing {filteredItems.length} of {data.total} incidents
              </span>
              <span className="text-border">|</span>
              <div className="flex items-center gap-1.5">
                <span>Rows:</span>
                <select
                  value={pageSize}
                  onChange={(e) => {
                    setPageSize(Number(e.target.value))
                    setPage(1)
                  }}
                  className="px-2 py-0.5 rounded border border-border bg-background text-xs font-mono text-foreground outline-none"
                >
                  <option value={10}>10</option>
                  <option value={20}>20</option>
                  <option value={50}>50</option>
                </select>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <span className="text-xs text-muted-foreground font-mono me-2">
                Page {data.page} of {Math.max(1, data.total_pages)}
              </span>

              <Button
                variant="outline"
                size="sm"
                disabled={page <= 1 || isFetching}
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                className="h-8 px-2.5 text-xs border-border"
              >
                {isArabic ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
              </Button>
              <Button
                variant="outline"
                size="sm"
                disabled={page >= data.total_pages || isFetching}
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
