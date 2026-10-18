import React, { useState } from 'react'
import {
  Lock,
  User,
  Shield,
  AlertCircle,
  Sparkles,
  X,
  LogIn,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useAuthStore } from '@/store/authStore'

interface LoginModalProps {
  isOpen: boolean
  onClose: () => void
}

export const LoginModal: React.FC<LoginModalProps> = ({ isOpen, onClose }) => {
  const { login, isLoading } = useAuthStore()

  const [username, setUsername] = useState('admin')
  const [password, setPassword] = useState('AdminSecure2026!#')
  const [errorMsg, setErrorMsg] = useState<string | null>(null)

  if (!isOpen) return null

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setErrorMsg(null)
    const result = await login(username, password)
    if (result.success) {
      onClose()
    } else {
      setErrorMsg(result.error || 'Authentication failed. Please verify credentials.')
    }
  }

  const fillQuickAccount = (user: string, pass: string) => {
    setUsername(user)
    setPassword(pass)
    setErrorMsg(null)
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/60 backdrop-blur-xs transition-opacity"
        onClick={onClose}
      />

      {/* Modal Dialog */}
      <div className="relative w-full max-w-md bg-card border border-border rounded-2xl shadow-2xl overflow-hidden p-6 space-y-5 animate-in fade-in zoom-in-95 duration-200">
        <div className="flex items-center justify-between pb-3 border-b border-border">
          <div className="flex items-center gap-2.5">
            <div className="h-9 w-9 rounded-lg bg-primary/10 border border-primary/30 flex items-center justify-center text-primary shadow-sm">
              <Shield className="h-5 w-5" />
            </div>
            <div>
              <h3 className="text-base font-bold text-foreground">ACS Security Authentication</h3>
              <p className="text-xs text-muted-foreground">JWT Session & Argon2 Access Control</p>
            </div>
          </div>
          <Button
            variant="ghost"
            size="icon"
            onClick={onClose}
            className="h-8 w-8 text-muted-foreground hover:text-foreground"
          >
            <X className="h-4 w-4" />
          </Button>
        </div>

        {errorMsg && (
          <div className="flex items-start gap-2.5 p-3 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs font-mono">
            <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" />
            <span>{errorMsg}</span>
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-1.5">
            <label className="text-xs font-mono text-muted-foreground uppercase tracking-wider block">
              Username
            </label>
            <div className="relative">
              <User className="absolute start-3 top-2.5 h-4 w-4 text-muted-foreground" />
              <input
                type="text"
                required
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="Username (e.g. admin, operator)"
                className="w-full ps-9 pe-3 py-2 text-xs font-mono rounded-lg border border-border bg-background text-foreground focus:ring-1 focus:ring-primary outline-none"
              />
            </div>
          </div>

          <div className="space-y-1.5">
            <label className="text-xs font-mono text-muted-foreground uppercase tracking-wider block">
              Password
            </label>
            <div className="relative">
              <Lock className="absolute start-3 top-2.5 h-4 w-4 text-muted-foreground" />
              <input
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••••••"
                className="w-full ps-9 pe-3 py-2 text-xs font-mono rounded-lg border border-border bg-background text-foreground focus:ring-1 focus:ring-primary outline-none"
              />
            </div>
          </div>

          <Button
            type="submit"
            disabled={isLoading}
            className="w-full gap-2 text-xs font-semibold py-2.5 bg-primary hover:bg-primary/90 text-primary-foreground shadow-sm"
          >
            <LogIn className="h-4 w-4" />
            {isLoading ? 'Authenticating...' : 'Sign In'}
          </Button>
        </form>

        {/* Quick Demo Switcher */}
        <div className="pt-3 border-t border-border space-y-2">
          <div className="flex items-center gap-1.5 text-[11px] font-mono text-muted-foreground">
            <Sparkles className="h-3.5 w-3.5 text-primary" />
            <span>Preset Role Logins (Initial Seeded):</span>
          </div>
          <div className="grid grid-cols-3 gap-2">
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => fillQuickAccount('admin', 'AdminSecure2026!#')}
              className="text-[11px] font-mono h-7 px-2 border-border hover:border-purple-500/50 hover:text-purple-400"
            >
              Admin
            </Button>
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => fillQuickAccount('operator', 'OperatorSecure2026!#')}
              className="text-[11px] font-mono h-7 px-2 border-border hover:border-cyan-500/50 hover:text-cyan-400"
            >
              Operator
            </Button>
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => fillQuickAccount('viewer', 'ViewerSecure2026!#')}
              className="text-[11px] font-mono h-7 px-2 border-border hover:border-slate-500/50 hover:text-slate-400"
            >
              Viewer
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}
