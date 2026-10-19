import React, { useState, useEffect } from 'react'
import { Outlet } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { Sidebar } from './Sidebar'
import { TopBar } from './TopBar'
import { BottomNav } from './BottomNav'
import { CommandPalette } from '@/components/command/CommandPalette'
import { DemoModeBanner } from '@/components/common/DemoModeBanner'
import { api } from '@/api/client'
import { useAppStore } from '@/store/appStore'

export const Layout: React.FC = () => {
  const { setIsConnected, setDevice, setActiveModel, isDemoMode, setIsPublicDemo } = useAppStore()
  const [isCommandPaletteOpen, setIsCommandPaletteOpen] = useState(false)

  // Global Cmd+K / Ctrl+K keyboard shortcut listener
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault()
        setIsCommandPaletteOpen((prev) => !prev)
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

  // Heartbeat query checking backend health
  const healthQuery = useQuery({
    queryKey: ['health'],
    queryFn: () => api.getHealth(),
    refetchInterval: 10000,
    retry: 2,
  })

  // System config query
  const configQuery = useQuery({
    queryKey: ['config'],
    queryFn: () => api.getConfig(),
    staleTime: 60000,
  })

  useEffect(() => {
    if (isDemoMode) {
      setIsConnected(true)
    } else {
      setIsConnected(Boolean(healthQuery.data && healthQuery.isSuccess))
    }
  }, [healthQuery.data, healthQuery.isSuccess, isDemoMode, setIsConnected])

  useEffect(() => {
    if (configQuery.data) {
      if (configQuery.data.device_info?.device) {
        setDevice(configQuery.data.device_info.device)
      }
      if (configQuery.data.defaults?.model_path) {
        // extract basename from defaults.model_path if path is provided
        const modelName = configQuery.data.defaults.model_path.split('/').pop() || 'best.pt'
        setActiveModel(modelName)
      }
      if (configQuery.data.defaults?.public_demo) {
        setIsPublicDemo(true)
      }
    }
  }, [configQuery.data, setDevice, setActiveModel, setIsPublicDemo])

  return (
    <div className="min-h-screen flex flex-col bg-background text-foreground selection:bg-primary/20 selection:text-primary">
      {/* Demo Mode Notice Banner */}
      <DemoModeBanner />

      <div className="flex-1 flex min-w-0">
        {/* Collapsible Sidebar (Tablet & Desktop) */}
        <Sidebar />

        {/* Main Content Area */}
        <div className="flex-1 flex flex-col min-w-0 min-h-screen pb-20 md:pb-6">
          <TopBar onOpenCommandPalette={() => setIsCommandPaletteOpen(true)} />
          <main className="flex-1 p-3 sm:p-6 lg:p-8 max-w-7xl w-full mx-auto animate-in fade-in duration-300">
            <Outlet />
          </main>
        </div>
      </div>

      {/* Mobile Bottom Navigation Bar (Phone) */}
      <BottomNav />

      {/* Global Command Palette */}
      <CommandPalette
        isOpen={isCommandPaletteOpen}
        onClose={() => setIsCommandPaletteOpen(false)}
      />
    </div>
  )
}
