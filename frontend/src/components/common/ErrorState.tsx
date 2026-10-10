import React from 'react'
import { AlertTriangle, RefreshCw } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'

interface ErrorStateProps {
  title?: string
  message?: string
  onRetry?: () => void
  className?: string
}

export const ErrorState: React.FC<ErrorStateProps> = ({
  title = 'System Telemetry Error',
  message = 'An unexpected communication error occurred while interacting with the server.',
  onRetry,
  className,
}) => {
  return (
    <div
      role="alert"
      className={cn(
        'p-6 sm:p-8 rounded-xl border border-[var(--status-alert)]/30 bg-[var(--status-alert)]/5 text-center flex flex-col items-center justify-center space-y-3',
        className
      )}
    >
      <div className="h-12 w-12 rounded-full bg-[var(--status-alert)]/10 border border-[var(--status-alert)]/30 flex items-center justify-center text-[var(--status-alert)] shadow-xs">
        <AlertTriangle className="h-6 w-6" />
      </div>

      <div className="space-y-1 max-w-md">
        <h3 className="text-sm font-bold text-foreground font-mono">{title}</h3>
        <p className="text-xs text-muted-foreground leading-relaxed">{message}</p>
      </div>

      {onRetry && (
        <Button
          variant="outline"
          size="sm"
          onClick={onRetry}
          className="mt-2 text-xs font-semibold gap-1.5 border-[var(--status-alert)]/40 hover:bg-[var(--status-alert)]/10 text-foreground"
        >
          <RefreshCw className="h-3.5 w-3.5 text-[var(--status-alert)]" />
          <span>Retry Operation</span>
        </Button>
      )}
    </div>
  )
}
