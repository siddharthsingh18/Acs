import React, { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  ShieldCheck,
  Search,
  Filter,
  RefreshCw,
  Clock,
  User,
  Layers,
  ChevronLeft,
  ChevronRight,
  Lock,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { api, type AuditLogItem } from '@/api/client'
import { useAuthStore } from '@/store/authStore'
import { cn } from '@/lib/utils'

export const AuditLogsPage: React.FC = () => {
  const { user } = useAuthStore()

  const [page, setPage] = useState(0)
  const [pageSize, setPageSize] = useState(25)
  const [actionFilter, setActionFilter] = useState<string>('')
  const [targetTypeFilter, setTargetTypeFilter] = useState<string>('')
  const [usernameFilter, setUsernameFilter] = useState<string>('')

  const isAdmin = user?.role === 'admin'

  const {
    data: auditData,
    isLoading,
    isFetching,
    refetch,
  } = useQuery({
    queryKey: ['audit-logs', page, pageSize, actionFilter, targetTypeFilter, usernameFilter],
    queryFn: () =>
      api.getAuditLogs({
        skip: page * pageSize,
        limit: pageSize,
        action: actionFilter || undefined,
        target_type: targetTypeFilter || undefined,
        username: usernameFilter || undefined,
      }),
    enabled: isAdmin,
    refetchInterval: 10000,
  })

  const total = auditData?.total ?? 0
  const totalPages = Math.max(1, Math.ceil(total / pageSize))
  const items = auditData?.items ?? []

  const getActionColor = (action: string) => {
    if (action.includes('approved') || action.includes('success')) {
      return 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
    }
    if (action.includes('rejected') || action.includes('failed') || action.includes('deleted')) {
      return 'bg-rose-500/10 text-rose-400 border-rose-500/30'
    }
    if (action.includes('dispatched')) {
      return 'bg-amber-500/10 text-amber-400 border-amber-500/30'
    }
    if (action.includes('created') || action.includes('start')) {
      return 'bg-cyan-500/10 text-cyan-400 border-cyan-500/30'
    }
    return 'bg-blue-500/10 text-blue-400 border-blue-500/30'
  }

  const getRoleBadge = (role: string) => {
    switch (role?.toLowerCase()) {
      case 'admin':
        return 'bg-purple-500/15 text-purple-400 border-purple-500/30'
      case 'operator':
        return 'bg-cyan-500/15 text-cyan-400 border-cyan-500/30'
      default:
        return 'bg-slate-500/15 text-slate-400 border-slate-500/30'
    }
  }

  if (!isAdmin) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] p-6 text-center space-y-4">
        <div className="h-16 w-16 rounded-full bg-amber-500/15 border border-amber-500/30 flex items-center justify-center text-amber-400 shadow-lg">
          <Lock className="h-8 w-8" />
        </div>
        <h2 className="text-xl font-bold text-foreground">Access Restricted</h2>
        <p className="text-sm text-muted-foreground max-w-md">
          The Audit Log contains immutable forensic security records and is restricted to System Administrators.
          Your current account role is <span className="font-mono text-primary font-bold uppercase">{user?.role || 'viewer'}</span>.
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-6 p-4 sm:p-6 max-w-7xl mx-auto">
      {/* Header Banner */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-border">
        <div>
          <div className="flex items-center gap-2.5">
            <div className="h-9 w-9 rounded-lg bg-primary/10 border border-primary/30 flex items-center justify-center text-primary shadow-sm">
              <ShieldCheck className="h-5 w-5" />
            </div>
            <div>
              <h1 className="text-xl sm:text-2xl font-bold tracking-tight text-foreground flex items-center gap-2">
                Security Audit Trail
                <span className="text-xs font-mono px-2 py-0.5 rounded-full bg-emerald-500/15 text-emerald-400 border border-emerald-500/30 font-normal">
                  Append-Only
                </span>
              </h1>
              <p className="text-xs sm:text-sm text-muted-foreground">
                Immutable chronological log of all approvals, rejections, emergency dispatches, settings updates, and system events.
              </p>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <Button
            variant="outline"
            size="sm"
            onClick={() => refetch()}
            disabled={isFetching}
            className="gap-2 text-xs font-mono border-border hover:border-primary/50"
          >
            <RefreshCw className={cn('h-3.5 w-3.5', isFetching && 'animate-spin text-primary')} />
            Refresh
          </Button>
        </div>
      </div>

      {/* Filter Toolbar */}
      <div className="p-4 rounded-xl border border-border bg-card/60 backdrop-blur-md grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3 shadow-xs">
        {/* Username search */}
        <div className="relative">
          <Search className="absolute start-3 top-2.5 h-4 w-4 text-muted-foreground" />
          <input
            type="text"
            placeholder="Filter by actor / username..."
            value={usernameFilter}
            onChange={(e) => {
              setUsernameFilter(e.target.value)
              setPage(0)
            }}
            className="w-full ps-9 pe-3 py-1.5 text-xs font-mono rounded-lg border border-border bg-background focus:ring-1 focus:ring-primary outline-none"
          />
        </div>

        {/* Action filter */}
        <div className="relative">
          <Filter className="absolute start-3 top-2.5 h-4 w-4 text-muted-foreground" />
          <select
            value={actionFilter}
            onChange={(e) => {
              setActionFilter(e.target.value)
              setPage(0)
            }}
            className="w-full ps-9 pe-3 py-1.5 text-xs font-mono rounded-lg border border-border bg-background focus:ring-1 focus:ring-primary outline-none appearance-none"
          >
            <option value="">All Actions</option>
            <option value="incident_approved">incident_approved</option>
            <option value="incident_rejected">incident_rejected</option>
            <option value="incident_dispatched">incident_dispatched</option>
            <option value="review_decision">review_decision</option>
            <option value="review_undo">review_undo</option>
            <option value="alert_settings_updated">alert_settings_updated</option>
            <option value="alert_test_sent">alert_test_sent</option>
            <option value="retention_settings_updated">retention_settings_updated</option>
            <option value="camera_created">camera_created</option>
            <option value="camera_updated">camera_updated</option>
            <option value="camera_deleted">camera_deleted</option>
            <option value="user_created">user_created</option>
            <option value="user_updated">user_updated</option>
            <option value="user_deleted">user_deleted</option>
            <option value="auth_login_success">auth_login_success</option>
            <option value="auth_login_failed">auth_login_failed</option>
            <option value="auth_logout">auth_logout</option>
            <option value="job_started">job_started</option>
            <option value="job_cancelled">job_cancelled</option>
          </select>
        </div>

        {/* Target Type filter */}
        <div className="relative">
          <Layers className="absolute start-3 top-2.5 h-4 w-4 text-muted-foreground" />
          <select
            value={targetTypeFilter}
            onChange={(e) => {
              setTargetTypeFilter(e.target.value)
              setPage(0)
            }}
            className="w-full ps-9 pe-3 py-1.5 text-xs font-mono rounded-lg border border-border bg-background focus:ring-1 focus:ring-primary outline-none appearance-none"
          >
            <option value="">All Target Types</option>
            <option value="incident">incident</option>
            <option value="settings">settings</option>
            <option value="alert">alert</option>
            <option value="camera">camera</option>
            <option value="user">user</option>
            <option value="job">job</option>
            <option value="auth">auth</option>
          </select>
        </div>

        {/* Page size */}
        <div className="flex items-center justify-end gap-2 text-xs font-mono text-muted-foreground">
          <span>Page size:</span>
          <select
            value={pageSize}
            onChange={(e) => {
              setPageSize(Number(e.target.value))
              setPage(0)
            }}
            className="px-2 py-1 rounded border border-border bg-background text-xs font-mono focus:ring-1 focus:ring-primary outline-none"
          >
            <option value={15}>15</option>
            <option value={25}>25</option>
            <option value={50}>50</option>
            <option value={100}>100</option>
          </select>
        </div>
      </div>

      {/* Audit Log Table */}
      <div className="rounded-xl border border-border bg-card/60 backdrop-blur-md overflow-hidden shadow-sm">
        <div className="overflow-x-auto">
          <table className="w-full text-start border-collapse text-xs">
            <thead>
              <tr className="border-b border-border bg-muted/40 text-muted-foreground font-mono uppercase text-[11px] tracking-wider">
                <th className="py-3 px-4 text-start">Timestamp (UTC)</th>
                <th className="py-3 px-4 text-start">Actor</th>
                <th className="py-3 px-4 text-start">Role</th>
                <th className="py-3 px-4 text-start">Action</th>
                <th className="py-3 px-4 text-start">Target</th>
                <th className="py-3 px-4 text-start">Client IP</th>
                <th className="py-3 px-4 text-start">Details</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/60">
              {isLoading ? (
                <tr>
                  <td colSpan={7} className="py-12 text-center text-muted-foreground font-mono">
                    <RefreshCw className="h-5 w-5 animate-spin mx-auto mb-2 text-primary" />
                    Loading immutable audit trail records...
                  </td>
                </tr>
              ) : items.length === 0 ? (
                <tr>
                  <td colSpan={7} className="py-12 text-center text-muted-foreground font-mono">
                    No matching audit records found.
                  </td>
                </tr>
              ) : (
                items.map((log: AuditLogItem) => (
                  <tr key={log.id} className="hover:bg-accent/40 transition-colors">
                    {/* Timestamp */}
                    <td className="py-3 px-4 font-mono text-muted-foreground whitespace-nowrap">
                      <div className="flex items-center gap-1.5">
                        <Clock className="h-3.5 w-3.5 text-muted-foreground/60 shrink-0" />
                        <span title={log.timestamp}>{log.timestamp.replace('T', ' ').slice(0, 19)}</span>
                      </div>
                    </td>

                    {/* Actor */}
                    <td className="py-3 px-4 font-semibold text-foreground whitespace-nowrap">
                      <div className="flex items-center gap-1.5">
                        <User className="h-3.5 w-3.5 text-primary shrink-0" />
                        <span>{log.username}</span>
                      </div>
                    </td>

                    {/* Role */}
                    <td className="py-3 px-4 whitespace-nowrap">
                      <span
                        className={cn(
                          'text-[10px] font-mono font-bold uppercase px-2 py-0.5 rounded border',
                          getRoleBadge(log.user_role)
                        )}
                      >
                        {log.user_role}
                      </span>
                    </td>

                    {/* Action */}
                    <td className="py-3 px-4 whitespace-nowrap">
                      <span
                        className={cn(
                          'text-[11px] font-mono font-medium px-2 py-0.5 rounded border',
                          getActionColor(log.action)
                        )}
                      >
                        {log.action}
                      </span>
                    </td>

                    {/* Target */}
                    <td className="py-3 px-4 font-mono text-muted-foreground whitespace-nowrap">
                      <span className="text-foreground/90 font-medium">{log.target_type}</span>
                      {log.target_id && (
                        <span className="text-muted-foreground ms-1 text-[11px]">
                          ({log.target_id})
                        </span>
                      )}
                    </td>

                    {/* IP Address */}
                    <td className="py-3 px-4 font-mono text-muted-foreground whitespace-nowrap">
                      {log.ip_address || 'unknown'}
                    </td>

                    {/* Details */}
                    <td className="py-3 px-4 max-w-xs truncate font-mono text-[11px] text-muted-foreground">
                      {log.details ? (
                        <span title={log.details} className="bg-background/80 px-2 py-0.5 rounded border border-border block truncate">
                          {log.details}
                        </span>
                      ) : (
                        <span className="text-muted-foreground/40">—</span>
                      )}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination Footer */}
        <div className="p-3 px-4 border-t border-border bg-muted/20 flex flex-col sm:flex-row items-center justify-between gap-3 text-xs font-mono text-muted-foreground">
          <div>
            Showing <span className="font-semibold text-foreground">{items.length}</span> of{' '}
            <span className="font-semibold text-foreground">{total}</span> events (Page{' '}
            <span className="font-semibold text-foreground">{page + 1}</span> of{' '}
            <span className="font-semibold text-foreground">{totalPages}</span>)
          </div>

          <div className="flex items-center gap-1.5">
            <Button
              variant="outline"
              size="sm"
              disabled={page === 0 || isLoading}
              onClick={() => setPage((p) => Math.max(0, p - 1))}
              className="h-7 px-2.5 text-xs font-mono border-border"
            >
              <ChevronLeft className="h-3.5 w-3.5 me-1" />
              Previous
            </Button>
            <Button
              variant="outline"
              size="sm"
              disabled={page >= totalPages - 1 || isLoading}
              onClick={() => setPage((p) => p + 1)}
              className="h-7 px-2.5 text-xs font-mono border-border"
            >
              Next
              <ChevronRight className="h-3.5 w-3.5 ms-1" />
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}
