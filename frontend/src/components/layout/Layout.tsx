import React, { useEffect } from 'react'
import { Outlet } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { Sidebar } from './Sidebar'
import { TopBar } from './TopBar'
import { api } from '@/api/client'
import { useAppStore } from '@/store/appStore'

export const Layout: React.FC = () => {
  const { setIsConnected, setDevice, setActiveModel } = useAppStore()

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
    setIsConnected(Boolean(healthQuery.data && healthQuery.isSuccess))
  }, [healthQuery.data, healthQuery.isSuccess, setIsConnected])

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
    }
  }, [configQuery.data, setDevice, setActiveModel])

  return (
    <div className="min-h-screen flex bg-background text-foreground selection:bg-primary/20 selection:text-primary">
      {/* Collapsible Sidebar */}
      <Sidebar />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 min-h-screen">
        <TopBar />
        <main className="flex-1 p-4 sm:p-6 lg:p-8 max-w-7xl w-full mx-auto animate-in fade-in duration-300">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
