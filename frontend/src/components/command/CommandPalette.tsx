import React, { useState, useEffect, useMemo, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import {
  Radio,
  FileSpreadsheet,
  BarChart3,
  Sliders,
  Info,
  Moon,
  Sun,
  Download,
  Bell,
  Sparkles,
  Search,
  X,
  CornerDownLeft,
  ShieldAlert,
} from 'lucide-react'
import { useAppStore } from '@/store/appStore'
import { api } from '@/api/client'
import { cn } from '@/lib/utils'

interface CommandItem {
  id: string
  title: string
  subtitle?: string
  icon: React.ElementType
  category: 'navigation' | 'actions' | 'preferences'
  shortcut?: string
  action: () => void
}

interface CommandPaletteProps {
  isOpen: boolean
  onClose: () => void
}

export const CommandPalette: React.FC<CommandPaletteProps> = ({ isOpen, onClose }) => {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const inputRef = useRef<HTMLInputElement>(null)
  const listRef = useRef<HTMLDivElement>(null)

  const {
    theme,
    toggleTheme,
    isDemoMode,
    toggleDemoMode,
  } = useAppStore()

  const [query, setQuery] = useState('')
  const [selectedIndex, setSelectedIndex] = useState(0)

  // Focus input when opened
  useEffect(() => {
    if (isOpen) {
      setQuery('')
      setSelectedIndex(0)
      setTimeout(() => inputRef.current?.focus(), 50)
    }
  }, [isOpen])

  // Build commands list
  const commands = useMemo<CommandItem[]>(() => {
    return [
      // Navigation
      {
        id: 'nav-live',
        title: t('nav.live'),
        subtitle: 'CCTV surveillance video analysis stream',
        icon: Radio,
        category: 'navigation',
        shortcut: '⌘1',
        action: () => {
          navigate('/live')
          onClose()
        },
      },
      {
        id: 'nav-incidents',
        title: t('nav.incidents'),
        subtitle: 'Master directory of detected traffic accidents',
        icon: FileSpreadsheet,
        category: 'navigation',
        shortcut: '⌘2',
        action: () => {
          navigate('/incidents')
          onClose()
        },
      },
      {
        id: 'nav-analytics',
        title: t('nav.analytics'),
        subtitle: 'Collision telemetry charts and metrics',
        icon: BarChart3,
        category: 'navigation',
        shortcut: '⌘3',
        action: () => {
          navigate('/analytics')
          onClose()
        },
      },
      {
        id: 'nav-settings',
        title: t('nav.settings'),
        subtitle: 'Inference weights, models, and alert channels',
        icon: Sliders,
        category: 'navigation',
        shortcut: '⌘4',
        action: () => {
          navigate('/settings')
          onClose()
        },
      },
      {
        id: 'nav-about',
        title: t('nav.about'),
        subtitle: 'System architecture and computer vision specs',
        icon: Info,
        category: 'navigation',
        shortcut: '⌘5',
        action: () => {
          navigate('/about')
          onClose()
        },
      },

      // Actions
      {
        id: 'act-demo',
        title: isDemoMode ? 'Disable Presentation Demo Mode' : 'Enable Presentation Demo Mode',
        subtitle: 'Replay pre-recorded incident sequence without a live backend',
        icon: Sparkles,
        category: 'actions',
        shortcut: '⌘D',
        action: () => {
          toggleDemoMode()
          onClose()
        },
      },
      {
        id: 'act-export-csv',
        title: t('incidents.export_csv'),
        subtitle: 'Download complete SQLite incident records as CSV',
        icon: Download,
        category: 'actions',
        action: () => {
          window.location.href = api.getExportCsvUrl()
          onClose()
        },
      },
      {
        id: 'act-test-alert',
        title: 'Dispatch Test Emergency Alert',
        subtitle: 'Simulate multi-channel notification to police & EMS bots',
        icon: Bell,
        category: 'actions',
        action: async () => {
          try {
            await api.testAlert()
            alert('Emergency test alert dispatched successfully!')
          } catch (e) {
            console.error('Test alert error:', e)
          }
          onClose()
        },
      },

      // Preferences
      {
        id: 'pref-theme',
        title: theme === 'dark' ? 'Switch to Light Mode' : 'Switch to Dark Mode',
        subtitle: 'Toggle control room theme appearance',
        icon: theme === 'dark' ? Sun : Moon,
        category: 'preferences',
        shortcut: '⌘T',
        action: () => {
          toggleTheme()
          onClose()
        },
      },

    ]
  }, [t, navigate, onClose, isDemoMode, toggleDemoMode, theme, toggleTheme])

  // Filter commands by search query
  const filteredCommands = useMemo(() => {
    if (!query.trim()) return commands
    const q = query.toLowerCase().trim()
    return commands.filter(
      (c) =>
        c.title.toLowerCase().includes(q) ||
        c.subtitle?.toLowerCase().includes(q) ||
        c.category.toLowerCase().includes(q)
    )
  }, [commands, query])

  // Handle keyboard navigation inside palette
  useEffect(() => {
    if (!isOpen) return

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault()
        onClose()
      } else if (e.key === 'ArrowDown') {
        e.preventDefault()
        setSelectedIndex((prev) =>
          prev < filteredCommands.length - 1 ? prev + 1 : 0
        )
      } else if (e.key === 'ArrowUp') {
        e.preventDefault()
        setSelectedIndex((prev) =>
          prev > 0 ? prev - 1 : filteredCommands.length - 1
        )
      } else if (e.key === 'Enter') {
        e.preventDefault()
        if (filteredCommands[selectedIndex]) {
          filteredCommands[selectedIndex].action()
        }
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [isOpen, filteredCommands, selectedIndex, onClose])

  if (!isOpen) return null

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Command Palette"
      className="fixed inset-0 z-50 flex items-start justify-center pt-16 sm:pt-24 px-4 bg-black/60 backdrop-blur-sm animate-in fade-in duration-150"
      onClick={onClose}
    >
      <div
        className="w-full max-w-xl bg-card border border-border rounded-xl shadow-2xl overflow-hidden animate-in zoom-in-95 duration-150"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Search Input Bar */}
        <div className="flex items-center px-4 py-3 border-b border-border bg-muted/30">
          <Search className="h-4 w-4 text-muted-foreground me-3 shrink-0" />
          <input
            ref={inputRef}
            type="text"
            placeholder="Type a command or jump to page... (e.g. live, csv, theme)"
            value={query}
            onChange={(e) => {
              setQuery(e.target.value)
              setSelectedIndex(0)
            }}
            className="w-full bg-transparent text-sm text-foreground placeholder:text-muted-foreground outline-none font-sans"
          />
          {query && (
            <button
              type="button"
              onClick={() => setQuery('')}
              className="p-1 text-muted-foreground hover:text-foreground me-1"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          )}
          <kbd className="hidden sm:inline-flex items-center px-1.5 py-0.5 text-[10px] font-mono text-muted-foreground bg-background rounded border border-border">
            ESC
          </kbd>
        </div>

        {/* Results List */}
        <div
          ref={listRef}
          className="max-h-80 overflow-y-auto p-2 divide-y divide-border/40"
        >
          {filteredCommands.length === 0 ? (
            <div className="py-8 text-center text-xs text-muted-foreground">
              No matching commands or actions found for "{query}".
            </div>
          ) : (
            filteredCommands.map((item, idx) => {
              const Icon = item.icon
              const isSelected = idx === selectedIndex

              return (
                <div
                  key={item.id}
                  onClick={item.action}
                  onMouseEnter={() => setSelectedIndex(idx)}
                  className={cn(
                    'flex items-center justify-between p-2.5 rounded-lg cursor-pointer transition-colors text-xs select-none',
                    isSelected
                      ? 'bg-primary/15 text-primary'
                      : 'text-foreground hover:bg-accent/50'
                  )}
                >
                  <div className="flex items-center gap-3 min-w-0">
                    <div
                      className={cn(
                        'h-8 w-8 rounded-md flex items-center justify-center shrink-0 border transition-colors',
                        isSelected
                          ? 'border-primary/40 bg-primary/20 text-primary'
                          : 'border-border bg-muted/40 text-muted-foreground'
                      )}
                    >
                      <Icon className="h-4 w-4" />
                    </div>

                    <div className="min-w-0">
                      <div className="font-semibold truncate flex items-center gap-2">
                        <span>{item.title}</span>
                        {item.id === 'act-demo' && isDemoMode && (
                          <span className="text-[10px] px-1 py-0.2 rounded bg-amber-500/20 text-amber-400 font-mono">
                            ACTIVE
                          </span>
                        )}
                      </div>
                      {item.subtitle && (
                        <div className="text-[11px] text-muted-foreground truncate">
                          {item.subtitle}
                        </div>
                      )}
                    </div>
                  </div>

                  <div className="flex items-center gap-2 shrink-0 ms-2">
                    {item.shortcut && (
                      <kbd className="px-1.5 py-0.5 text-[10px] font-mono text-muted-foreground bg-background rounded border border-border">
                        {item.shortcut}
                      </kbd>
                    )}
                    {isSelected && (
                      <CornerDownLeft className="h-3 w-3 text-primary animate-pulse" />
                    )}
                  </div>
                </div>
              )
            })
          )}
        </div>

        {/* Footer Shortcut Legend */}
        <div className="px-4 py-2 border-t border-border bg-muted/30 flex items-center justify-between text-[11px] text-muted-foreground font-mono">
          <div className="flex items-center gap-3">
            <span>↑↓ Navigate</span>
            <span>↵ Select</span>
            <span>Esc Close</span>
          </div>
          <div className="flex items-center gap-1.5 text-primary">
            <ShieldAlert className="h-3 w-3" />
            <span>ACS Control Room</span>
          </div>
        </div>
      </div>
    </div>
  )
}
