import React from 'react'
import { NavLink } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import {
  Radio,
  FileSpreadsheet,
  BarChart3,
  Sliders,
  ShieldAlert,
  Info,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react'
import { useAppStore } from '@/store/appStore'
import { cn } from '@/lib/utils'

interface NavItem {
  to: string
  labelKey: string
  icon: React.ElementType
  badge?: string
}

const NAV_ITEMS: NavItem[] = [
  {
    to: '/live',
    labelKey: 'nav.live',
    icon: Radio,
    badge: 'LIVE',
  },
  {
    to: '/incidents',
    labelKey: 'nav.incidents',
    icon: FileSpreadsheet,
  },
  {
    to: '/analytics',
    labelKey: 'nav.analytics',
    icon: BarChart3,
  },
  {
    to: '/settings',
    labelKey: 'nav.settings',
    icon: Sliders,
  },
  {
    to: '/about',
    labelKey: 'nav.about',
    icon: Info,
  },
]

export const Sidebar: React.FC = () => {
  const { t, i18n } = useTranslation()
  const { sidebarCollapsed, toggleSidebar } = useAppStore()
  const isArabic = i18n.language === 'ar'

  return (
    <aside
      className={cn(
        'relative flex flex-col border-e border-border bg-card/80 backdrop-blur-md transition-all duration-300 z-40 select-none h-screen sticky top-0',
        sidebarCollapsed ? 'w-16' : 'w-64'
      )}
    >
      {/* Brand Header */}
      <div className="h-16 flex items-center px-4 border-b border-border overflow-hidden">
        <NavLink
          to="/live"
          className="flex items-center gap-3 text-foreground hover:opacity-90 transition-opacity w-full"
        >
          <div className="h-9 w-9 rounded-lg bg-primary/10 border border-primary/30 flex items-center justify-center shrink-0 text-primary shadow-[0_0_12px_rgba(6,182,212,0.25)]">
            <ShieldAlert className="h-5 w-5 text-primary" />
          </div>

          {!sidebarCollapsed && (
            <div className="flex flex-col min-w-0 transition-opacity duration-200">
              <span className="font-bold text-base tracking-tight truncate leading-tight flex items-center gap-1.5">
                ACS
                <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-primary/20 text-primary border border-primary/30 uppercase">
                  v2.0
                </span>
              </span>
              <span className="text-[11px] text-muted-foreground truncate leading-tight">
                {t('app.title')}
              </span>
            </div>
          )}
        </NavLink>
      </div>

      {/* Navigation Links */}
      <nav className="flex-1 py-4 px-2 space-y-1 overflow-y-auto">
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon
          return (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                cn(
                  'flex items-center rounded-lg px-3 py-2.5 text-sm font-medium transition-all group relative',
                  isActive
                    ? 'bg-primary/15 text-primary border border-primary/25 shadow-sm'
                    : 'text-muted-foreground hover:bg-accent/60 hover:text-foreground border border-transparent'
                )
              }
            >
              {({ isActive }) => (
                <>
                  <Icon
                    className={cn(
                      'h-5 w-5 shrink-0 transition-colors',
                      isActive
                        ? 'text-primary'
                        : 'text-muted-foreground group-hover:text-foreground'
                    )}
                  />

                  {!sidebarCollapsed && (
                    <span className="ms-3 truncate text-sm flex-1 font-sans">
                      {t(item.labelKey)}
                    </span>
                  )}

                  {!sidebarCollapsed && item.badge && (
                    <span className="ms-auto text-[10px] font-mono font-bold px-1.5 py-0.5 rounded-full bg-[var(--status-alert)]/20 text-[var(--status-alert)] border border-[var(--status-alert)]/30 animate-pulse">
                      {item.badge}
                    </span>
                  )}

                  {/* Active Indicator bar */}
                  {isActive && (
                    <span className="absolute start-0 top-1.5 bottom-1.5 w-1 bg-primary rounded-e-full" />
                  )}
                </>
              )}
            </NavLink>
          )
        })}
      </nav>

      {/* Footer / Collapse Button */}
      <div className="p-2 border-t border-border flex items-center justify-between">
        <button
          type="button"
          onClick={toggleSidebar}
          aria-label={sidebarCollapsed ? t('nav.expand') : t('nav.collapse')}
          className="w-full flex items-center justify-center gap-2 p-2 rounded-lg text-xs text-muted-foreground hover:bg-accent hover:text-foreground transition-colors"
        >
          {sidebarCollapsed ? (
            isArabic ? (
              <ChevronLeft className="h-4 w-4" />
            ) : (
              <ChevronRight className="h-4 w-4" />
            )
          ) : (
            <>
              {isArabic ? (
                <ChevronRight className="h-4 w-4" />
              ) : (
                <ChevronLeft className="h-4 w-4" />
              )}
              <span className="text-xs font-medium font-sans">
                {t('nav.collapse')}
              </span>
            </>
          )}
        </button>
      </div>
    </aside>
  )
}
