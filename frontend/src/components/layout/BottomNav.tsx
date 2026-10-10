import React from 'react'
import { NavLink } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import {
  Radio,
  FileSpreadsheet,
  BarChart3,
  Sliders,
  Info,
} from 'lucide-react'
import { cn } from '@/lib/utils'

interface BottomNavItem {
  to: string
  labelKey: string
  icon: React.ElementType
  isLive?: boolean
}

const BOTTOM_NAV_ITEMS: BottomNavItem[] = [
  {
    to: '/live',
    labelKey: 'nav.live',
    icon: Radio,
    isLive: true,
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

export const BottomNav: React.FC = () => {
  const { t } = useTranslation()

  return (
    <nav
      role="navigation"
      aria-label="Mobile Bottom Navigation"
      className="md:hidden fixed bottom-0 inset-x-0 z-40 bg-card/95 backdrop-blur-lg border-t border-border flex items-center justify-around h-16 px-1 shadow-[0_-4px_16px_rgba(0,0,0,0.2)]"
    >
      {BOTTOM_NAV_ITEMS.map((item) => {
        const Icon = item.icon
        return (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) =>
              cn(
                'flex flex-col items-center justify-center flex-1 h-full py-1 text-center select-none transition-colors relative min-w-[48px]',
                isActive
                  ? 'text-primary font-bold'
                  : 'text-muted-foreground hover:text-foreground'
              )
            }
          >
            {({ isActive }) => (
              <>
                <div className="relative">
                  <Icon
                    className={cn(
                      'h-5 w-5 transition-transform',
                      isActive ? 'scale-110 text-primary' : 'text-muted-foreground'
                    )}
                  />
                  {item.isLive && (
                    <span className="absolute -top-0.5 -end-0.5 h-2 w-2 rounded-full bg-[var(--status-alert)] animate-pulse" />
                  )}
                </div>
                <span className="text-[10px] mt-1 truncate max-w-[64px] font-sans">
                  {t(item.labelKey)}
                </span>
                {isActive && (
                  <span className="absolute top-0 inset-x-3 h-0.5 bg-primary rounded-b-full shadow-[0_0_8px_var(--primary)]" />
                )}
              </>
            )}
          </NavLink>
        )
      })}
    </nav>
  )
}
