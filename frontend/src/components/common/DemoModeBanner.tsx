import React from 'react'
import { Sparkles, X } from 'lucide-react'
import { useAppStore } from '@/store/appStore'
import { Button } from '@/components/ui/button'

export const DemoModeBanner: React.FC = () => {
  const { isDemoMode, setDemoMode } = useAppStore()

  if (!isDemoMode) return null

  return (
    <div
      role="status"
      aria-live="polite"
      className="bg-amber-500/15 border-b border-amber-500/30 px-4 py-2 text-xs flex items-center justify-between gap-3 text-amber-200 backdrop-blur-md animate-in slide-in-from-top duration-200 z-50 sticky top-0"
    >
      <div className="flex items-center gap-2 min-w-0">
        <Sparkles className="h-4 w-4 text-amber-400 shrink-0 animate-pulse" />
        <span className="font-bold tracking-wider uppercase font-mono text-[11px] text-amber-400 shrink-0">
          Demo Presentation Mode:
        </span>
        <span className="truncate text-foreground/80">
          Simulating real-time CCTV stream, AI detections, and emergency response without live backend.
        </span>
      </div>

      <Button
        variant="ghost"
        size="sm"
        onClick={() => setDemoMode(false)}
        className="h-6 px-2 text-[11px] font-mono text-amber-300 hover:text-amber-100 hover:bg-amber-500/20 shrink-0 gap-1 border border-amber-500/30"
      >
        <span>Exit Demo</span>
        <X className="h-3 w-3" />
      </Button>
    </div>
  )
}
