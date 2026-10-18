import React, { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  Grid2X2,
  Grid3X3,
  Square,
  Plus,
  RefreshCw,
  Video,
  Radio,
  AlertTriangle,
  ArrowLeft,
} from 'lucide-react'
import { api, type CameraItem, type CameraCreatePayload } from '@/api/client'
import { CameraTile } from './CameraTile'
import { AddCameraModal } from './AddCameraModal'
import { cn } from '@/lib/utils'

export type GridLayout = '1x1' | '2x2' | '3x3'

interface CameraWallProps {
  onSelectIncident?: (incidentId: string) => void
}

export const CameraWall: React.FC<CameraWallProps> = () => {
  const queryClient = useQueryClient()
  const [layout, setLayout] = useState<GridLayout>('2x2')
  const [focusedCameraId, setFocusedCameraId] = useState<string | null>(null)
  const [isAddModalOpen, setIsAddModalOpen] = useState<boolean>(false)

  // Fetch camera list
  const { data: cameras = [], isLoading, refetch } = useQuery({
    queryKey: ['cameras'],
    queryFn: () => api.getCameras(),
    refetchInterval: 5000,
  })

  // Add Camera Mutation
  const addCameraMutation = useMutation({
    mutationFn: (payload: CameraCreatePayload) => api.createCamera(payload),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['cameras'] })
    },
  })

  // Delete Camera Mutation
  const deleteCameraMutation = useMutation({
    mutationFn: (id: string) => api.deleteCamera(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['cameras'] })
      if (focusedCameraId) setFocusedCameraId(null)
    },
  })

  const handleFocus = (camera: CameraItem) => {
    if (focusedCameraId === camera.id) {
      setFocusedCameraId(null)
    } else {
      setFocusedCameraId(camera.id)
    }
  }

  const handleDelete = async (camera: CameraItem) => {
    if (window.confirm(`Are you sure you want to remove camera '${camera.name}'?`)) {
      await deleteCameraMutation.mutateAsync(camera.id)
    }
  }

  // Active incidents count across camera list
  const activeIncidentsCount = cameras.filter(
    (c) => c.health?.is_incident_active
  ).length
  const connectedCount = cameras.filter((c) => c.health?.connected).length

  // Filtered cameras based on focus mode and grid layout
  const focusedCamera = cameras.find((c) => c.id === focusedCameraId)

  // Maximum cameras visible based on layout
  const maxTiles = layout === '1x1' ? 1 : layout === '2x2' ? 4 : 9
  const displayedCameras = focusedCamera
    ? [focusedCamera]
    : cameras.slice(0, maxTiles)

  return (
    <div className="space-y-4">
      {/* Tactical Camera Wall Control Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 p-3.5 rounded-xl border border-border bg-card/80 backdrop-blur-md shadow-sm">
        {/* Left: Summary KPIs */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5 px-3 py-1 rounded-lg bg-primary/10 border border-primary/20 text-xs font-bold text-primary">
            <Radio className="h-3.5 w-3.5 animate-pulse" />
            <span>WALL FEED</span>
          </div>

          <div className="flex items-center gap-2 text-xs font-mono text-muted-foreground">
            <span>
              Total: <strong className="text-foreground">{cameras.length}</strong>
            </span>
            <span className="text-border">|</span>
            <span className="text-emerald-400">
              Active: <strong>{connectedCount}</strong>
            </span>
            {activeIncidentsCount > 0 && (
              <>
                <span className="text-border">|</span>
                <span className="flex items-center gap-1 text-rose-400 font-bold animate-pulse">
                  <AlertTriangle className="h-3.5 w-3.5" />
                  {activeIncidentsCount} ALARM{activeIncidentsCount > 1 ? 'S' : ''}
                </span>
              </>
            )}
          </div>
        </div>

        {/* Right: Layout Switcher & Action Buttons */}
        <div className="flex items-center gap-2">
          {focusedCamera && (
            <button
              onClick={() => setFocusedCameraId(null)}
              className="flex items-center gap-1 px-3 py-1.5 rounded-lg border border-border bg-muted/40 hover:bg-muted text-xs font-semibold text-foreground transition-colors"
            >
              <ArrowLeft className="h-3.5 w-3.5" />
              <span>Back to Wall</span>
            </button>
          )}

          {/* Grid Layout Selector: 1x1, 2x2, 3x3 */}
          <div className="flex items-center rounded-lg border border-border bg-muted/40 p-0.5">
            <button
              onClick={() => {
                setLayout('1x1')
                if (!focusedCameraId && cameras.length > 0) {
                  setFocusedCameraId(cameras[0].id)
                }
              }}
              title="1x1 Focused Layout"
              className={cn(
                'flex items-center gap-1 px-2.5 py-1 rounded-md text-xs font-bold transition-all',
                layout === '1x1' || focusedCameraId
                  ? 'bg-card text-foreground shadow-sm'
                  : 'text-muted-foreground hover:text-foreground'
              )}
            >
              <Square className="h-3.5 w-3.5" />
              <span>1x1</span>
            </button>

            <button
              onClick={() => {
                setLayout('2x2')
                setFocusedCameraId(null)
              }}
              title="2x2 Quad Grid"
              className={cn(
                'flex items-center gap-1 px-2.5 py-1 rounded-md text-xs font-bold transition-all',
                layout === '2x2' && !focusedCameraId
                  ? 'bg-card text-foreground shadow-sm'
                  : 'text-muted-foreground hover:text-foreground'
              )}
            >
              <Grid2X2 className="h-3.5 w-3.5" />
              <span>2x2</span>
            </button>

            <button
              onClick={() => {
                setLayout('3x3')
                setFocusedCameraId(null)
              }}
              title="3x3 High-Density Wall"
              className={cn(
                'flex items-center gap-1 px-2.5 py-1 rounded-md text-xs font-bold transition-all',
                layout === '3x3' && !focusedCameraId
                  ? 'bg-card text-foreground shadow-sm'
                  : 'text-muted-foreground hover:text-foreground'
              )}
            >
              <Grid3X3 className="h-3.5 w-3.5" />
              <span>3x3</span>
            </button>
          </div>

          {/* Refresh List Button */}
          <button
            onClick={() => refetch()}
            className="p-1.5 rounded-lg border border-border bg-muted/30 hover:bg-muted text-muted-foreground hover:text-foreground transition-colors"
            title="Refresh Camera Feeds"
          >
            <RefreshCw className="h-4 w-4" />
          </button>

          {/* Add Camera Button */}
          <button
            onClick={() => setIsAddModalOpen(true)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-primary hover:bg-primary/90 text-primary-foreground text-xs font-bold shadow transition-colors"
          >
            <Plus className="h-3.5 w-3.5" />
            <span>Add Camera</span>
          </button>
        </div>
      </div>

      {/* Main Grid View */}
      {isLoading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {[1, 2, 3, 4].map((i) => (
            <div
              key={i}
              className="aspect-video w-full rounded-xl border border-border bg-muted/20 animate-pulse flex items-center justify-center text-xs font-mono text-muted-foreground"
            >
              Loading camera telemetry...
            </div>
          ))}
        </div>
      ) : displayedCameras.length === 0 ? (
        <div className="flex flex-col items-center justify-center p-12 rounded-2xl border border-dashed border-border bg-card/40 text-center space-y-3">
          <Video className="h-12 w-12 text-muted-foreground/60" />
          <h3 className="text-base font-bold text-foreground">No Cameras Configured</h3>
          <p className="text-xs text-muted-foreground max-w-sm">
            Register your first RTSP network stream, USB webcam index, or continuous video file to monitor traffic.
          </p>
          <button
            onClick={() => setIsAddModalOpen(true)}
            className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-primary text-primary-foreground text-xs font-bold shadow"
          >
            <Plus className="h-4 w-4" />
            <span>Add First Camera</span>
          </button>
        </div>
      ) : (
        <div
          className={cn(
            'grid gap-4 transition-all duration-300',
            layout === '1x1' || focusedCameraId
              ? 'grid-cols-1 max-w-5xl mx-auto'
              : layout === '2x2'
              ? 'grid-cols-1 md:grid-cols-2'
              : 'grid-cols-1 md:grid-cols-2 xl:grid-cols-3'
          )}
        >
          {displayedCameras.map((cam) => (
            <CameraTile
              key={cam.id}
              camera={cam}
              isFocused={focusedCameraId === cam.id}
              onFocus={handleFocus}
              onDelete={handleDelete}
            />
          ))}
        </div>
      )}

      {/* Modal Dialog for Registering Camera */}
      <AddCameraModal
        isOpen={isAddModalOpen}
        onClose={() => setIsAddModalOpen(false)}
        onAddCamera={async (payload) => {
          await addCameraMutation.mutateAsync(payload)
        }}
      />
    </div>
  )
}
