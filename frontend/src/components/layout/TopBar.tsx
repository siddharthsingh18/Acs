import React, { useState } from 'react'
import { useTranslation } from 'react-i18next'
import {
  Cpu,
  Globe,
  Moon,
  Sun,
  Menu,
  Search,
  Sparkles,
  User,
  LogIn,
  LogOut,
  ShieldCheck,
  ChevronDown,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { useAppStore } from '@/store/appStore'
import { useAuthStore } from '@/store/authStore'
import { LoginModal } from '@/components/auth/LoginModal'
import { cn } from '@/lib/utils'

interface TopBarProps {
  onOpenCommandPalette?: () => void
}

export const TopBar: React.FC<TopBarProps> = ({ onOpenCommandPalette }) => {
  const { t, i18n } = useTranslation()
  const {
    theme,
    toggleTheme,
    toggleSidebar,
    isConnected,
    device,
    activeModel,
    isDemoMode,
    toggleDemoMode,
  } = useAppStore()

  const { user, isAuthenticated, logout } = useAuthStore()
  const [isLoginModalOpen, setIsLoginModalOpen] = useState(false)
  const [isUserMenuOpen, setIsUserMenuOpen] = useState(false)

  const currentLang = i18n.language || 'en'
  const isArabic = currentLang === 'ar'

  const toggleLanguage = () => {
    const nextLang = isArabic ? 'en' : 'ar'
    i18n.changeLanguage(nextLang)
  }

  const getRoleBadgeClass = (role?: string) => {
    switch (role?.toLowerCase()) {
      case 'admin':
        return 'bg-purple-500/15 text-purple-400 border-purple-500/30'
      case 'operator':
        return 'bg-cyan-500/15 text-cyan-400 border-cyan-500/30'
      default:
        return 'bg-slate-500/15 text-slate-400 border-slate-500/30'
    }
  }

  return (
    <>
      <header className="h-16 border-b border-border bg-card/60 backdrop-blur-md px-3 sm:px-6 flex items-center justify-between sticky top-0 z-30 transition-colors">
        <div className="flex items-center gap-2 sm:gap-3">
          <Button
            variant="ghost"
            size="icon"
            onClick={toggleSidebar}
            aria-label={t('nav.collapse')}
            className="text-muted-foreground hover:text-foreground hidden md:inline-flex"
          >
            <Menu className="h-5 w-5" />
          </Button>

          <div className="flex items-center gap-2">
            <div className="h-2 w-2 rounded-full bg-cyan-400 animate-pulse hidden sm:block" />
            <h2 className="text-xs sm:text-sm font-semibold tracking-wide uppercase text-muted-foreground truncate max-w-[150px] sm:max-w-none">
              {t('app.control_room')}
            </h2>
          </div>

          {/* Global Command Palette Trigger */}
          <button
            type="button"
            onClick={onOpenCommandPalette}
            className="flex items-center gap-2 px-2.5 py-1 rounded-lg border border-border bg-background/60 hover:bg-accent/60 text-xs text-muted-foreground hover:text-foreground transition-all shadow-2xs ms-1 sm:ms-2 focus-visible:ring-2 focus-visible:ring-primary"
            aria-label="Open Command Palette (Cmd+K)"
          >
            <Search className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Search & Actions</span>
            <kbd className="hidden sm:inline-flex items-center px-1.5 py-0.2 text-[10px] font-mono text-muted-foreground bg-muted rounded border border-border">
              ⌘K
            </kbd>
          </button>
        </div>

        <div className="flex items-center gap-1.5 sm:gap-3">
          {/* Presentation Demo Mode Indicator Toggle */}
          <Button
            variant={isDemoMode ? 'default' : 'outline'}
            size="sm"
            onClick={toggleDemoMode}
            className={`h-7 px-2 text-[11px] font-mono gap-1 border ${
              isDemoMode
                ? 'bg-amber-500/20 text-amber-300 border-amber-500/40 hover:bg-amber-500/30'
                : 'border-border text-muted-foreground hover:text-foreground'
            }`}
            title="Toggle Presentation Demo Mode (Offline Replay)"
          >
            <Sparkles className={`h-3 w-3 ${isDemoMode ? 'text-amber-400 animate-pulse' : ''}`} />
            <span className="hidden sm:inline">
              {isDemoMode ? 'DEMO ACTIVE' : 'DEMO'}
            </span>
          </Button>

          {/* Active Model Indicator */}
          {activeModel && (
            <div className="hidden md:flex items-center gap-1.5 px-2.5 py-1 rounded-md border border-border bg-background/50 text-xs font-mono text-muted-foreground">
              <span className="text-[10px] uppercase tracking-wider text-muted-foreground/70">
                {t('topbar.active_model')}:
              </span>
              <span className="text-foreground font-semibold">{activeModel}</span>
            </div>
          )}

          {/* Compute Device Badge */}
          <Badge
            variant="outline"
            className="bg-card/80 border-border px-2.5 py-1 text-xs font-mono font-medium flex items-center gap-1.5 shadow-sm"
          >
            <Cpu className="h-3.5 w-3.5 text-primary" />
            <span className="uppercase text-[11px] tracking-wider text-foreground">
              {device || 'MPS'}
            </span>
          </Badge>

          {/* Live Backend Connection Dot */}
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full border border-border bg-background/60 text-xs font-mono">
            <span
              className={`h-2 w-2 rounded-full transition-colors ${
                isConnected
                  ? 'bg-[var(--status-ok)] shadow-[0_0_8px_rgba(16,185,129,0.7)] animate-pulse'
                  : 'bg-[var(--status-alert)] shadow-[0_0_8px_rgba(239,68,68,0.7)]'
              }`}
            />
            <span className="text-xs text-muted-foreground font-medium hidden lg:inline">
              {isConnected ? t('app.connected') : t('app.disconnected')}
            </span>
          </div>

          {/* User Account / Role Badge */}
          <div className="relative">
            {isAuthenticated && user ? (
              <div className="flex items-center gap-1.5">
                <button
                  type="button"
                  onClick={() => setIsUserMenuOpen(!isUserMenuOpen)}
                  className="flex items-center gap-2 px-2 py-1 rounded-lg border border-border bg-background/60 hover:bg-accent/60 text-xs font-mono transition-colors"
                >
                  <div className="h-6 w-6 rounded-full bg-primary/10 border border-primary/30 flex items-center justify-center text-primary">
                    <User className="h-3.5 w-3.5" />
                  </div>
                  <span className="font-semibold text-foreground hidden sm:inline">
                    {user.full_name || user.username}
                  </span>
                  <span
                    className={cn(
                      'text-[10px] font-mono font-bold uppercase px-1.5 py-0.2 rounded border',
                      getRoleBadgeClass(user.role)
                    )}
                  >
                    {user.role}
                  </span>
                  <ChevronDown className="h-3 w-3 text-muted-foreground" />
                </button>

                {isUserMenuOpen && (
                  <div className="absolute end-0 top-11 w-48 bg-card border border-border rounded-xl shadow-xl p-1.5 z-50 space-y-1 animate-in fade-in zoom-in-95">
                    <div className="px-3 py-2 border-b border-border">
                      <p className="text-xs font-bold text-foreground truncate">{user.full_name}</p>
                      <p className="text-[11px] font-mono text-muted-foreground">@{user.username}</p>
                    </div>

                    <button
                      type="button"
                      onClick={() => {
                        setIsUserMenuOpen(false)
                        setIsLoginModalOpen(true)
                      }}
                      className="w-full text-start flex items-center gap-2 px-3 py-1.5 text-xs text-muted-foreground hover:text-foreground hover:bg-accent rounded-lg transition-colors font-mono"
                    >
                      <ShieldCheck className="h-3.5 w-3.5 text-primary" />
                      Switch Account
                    </button>

                    <button
                      type="button"
                      onClick={() => {
                        setIsUserMenuOpen(false)
                        logout()
                      }}
                      className="w-full text-start flex items-center gap-2 px-3 py-1.5 text-xs text-rose-400 hover:bg-rose-500/10 rounded-lg transition-colors font-mono"
                    >
                      <LogOut className="h-3.5 w-3.5" />
                      Sign Out
                    </button>
                  </div>
                )}
              </div>
            ) : (
              <Button
                variant="outline"
                size="sm"
                onClick={() => setIsLoginModalOpen(true)}
                className="h-8 px-2.5 gap-1.5 text-xs font-medium border-border hover:border-primary/50 text-foreground"
              >
                <LogIn className="h-3.5 w-3.5 text-primary" />
                <span>Sign In</span>
              </Button>
            )}
          </div>



          {/* Theme Toggle */}
          <Button
            variant="ghost"
            size="icon"
            onClick={toggleTheme}
            aria-label={theme === 'dark' ? t('topbar.theme_light') : t('topbar.theme_dark')}
            className="h-8 w-8 text-muted-foreground hover:text-foreground"
          >
            {theme === 'dark' ? (
              <Sun className="h-4 w-4 text-amber-400 hover:rotate-45 transition-transform" />
            ) : (
              <Moon className="h-4 w-4 text-slate-700 hover:-rotate-12 transition-transform" />
            )}
          </Button>
        </div>
      </header>

      {/* Login Modal */}
      <LoginModal isOpen={isLoginModalOpen} onClose={() => setIsLoginModalOpen(false)} />
    </>
  )
}

