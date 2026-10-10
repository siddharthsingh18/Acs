import React from 'react'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'

export type EmptyStateVariant = 'radar' | 'search' | 'camera' | 'shield'

interface EmptyStateProps {
  variant?: EmptyStateVariant
  title: string
  description?: string
  actionLabel?: string
  onAction?: () => void
  actionIcon?: React.ReactNode
  className?: string
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  variant = 'radar',
  title,
  description,
  actionLabel,
  onAction,
  actionIcon,
  className,
}) => {
  return (
    <div
      role="status"
      aria-live="polite"
      className={cn(
        'flex flex-col items-center justify-center p-8 sm:p-12 text-center rounded-xl border border-dashed border-border bg-card/40 backdrop-blur-xs select-none',
        className
      )}
    >
      {/* Tactical SVG Illustration */}
      <div className="relative mb-5 flex items-center justify-center">
        <div className="h-20 w-20 rounded-full bg-primary/10 border border-primary/20 flex items-center justify-center relative overflow-hidden">
          {variant === 'radar' && (
            <>
              {/* Radar rings */}
              <div className="absolute inset-1 rounded-full border border-primary/20" />
              <div className="absolute inset-4 rounded-full border border-primary/30" />
              <div className="absolute inset-7 rounded-full border border-primary/40" />
              {/* Crosshair lines */}
              <div className="absolute inset-x-0 top-1/2 h-px bg-primary/30" />
              <div className="absolute inset-y-0 start-1/2 w-px bg-primary/30" />
              {/* Sweeping ray */}
              <div className="absolute inset-0 radar-sweep opacity-70" />
              {/* Center blip */}
              <div className="h-2 w-2 rounded-full bg-primary animate-ping" />
            </>
          )}

          {variant === 'search' && (
            <svg
              className="h-9 w-9 text-primary/80"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <circle cx="11" cy="11" r="8" />
              <path d="m21 21-4.3-4.3" />
              <path d="M11 8v6" strokeDasharray="2 2" />
              <path d="M8 11h6" strokeDasharray="2 2" />
            </svg>
          )}

          {variant === 'camera' && (
            <svg
              className="h-9 w-9 text-cyan-400"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z" />
              <circle cx="12" cy="13" r="3" />
              <path d="M18 10h.01" />
            </svg>
          )}

          {variant === 'shield' && (
            <svg
              className="h-9 w-9 text-emerald-400"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
              <path d="m9 12 2 2 4-4" />
            </svg>
          )}
        </div>
      </div>

      <h3 className="text-base font-bold tracking-tight text-foreground max-w-md">
        {title}
      </h3>

      {description && (
        <p className="mt-1.5 text-xs text-muted-foreground max-w-sm leading-relaxed">
          {description}
        </p>
      )}

      {actionLabel && onAction && (
        <Button
          variant="outline"
          size="sm"
          onClick={onAction}
          className="mt-5 text-xs font-semibold gap-1.5 border-border hover:border-primary/50 shadow-xs focus-visible:ring-2 focus-visible:ring-primary"
        >
          {actionIcon}
          <span>{actionLabel}</span>
        </Button>
      )}
    </div>
  )
}
