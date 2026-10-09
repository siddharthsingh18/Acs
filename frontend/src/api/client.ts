import createClient from 'openapi-fetch'
import type { components, paths } from './schema'

/**
 * Typed OpenAPI client connected via Vite dev proxy or relative path in production.
 */
export const apiClient = createClient<paths>({
  baseUrl: '',
})

// Schema type exports
export type HealthResponse = components['schemas']['HealthResponse']
export type ConfigResponse = components['schemas']['SystemConfigResponse']
export type VisionClassInfo = components['schemas']['ClassItem']
export type IncidentResponse = components['schemas']['IncidentResponse']
export type IncidentListResponse = components['schemas']['IncidentListResponse']
export type IncidentStatusUpdateRequest = components['schemas']['IncidentStatusUpdateRequest']
export type StatsResponse = components['schemas']['StatsResponse']
export type AlertSettingsResponse = components['schemas']['AlertSettingsResponse']
export type AlertSettingsUpdateRequest = components['schemas']['AlertSettingsUpdateRequest']
export type AlertTestResponse = components['schemas']['AlertTestResponse']
export type JobCreateResponse = components['schemas']['JobCreatedResponse']
export type JobCancelResponse = components['schemas']['JobCancelResponse']

/**
 * WebSocket streaming message definitions
 */
export interface StreamFrameMessage {
  type: 'frame'
  index: number
  total: number
  fps: number
  jpeg_base64: string
}

export interface StreamIncidentMessage {
  type: 'incident'
  incident_id: string
  class_name: string
  max_confidence: number
  start_timestamp: string
  end_timestamp: string
  start_time: number
  end_time: number
  duration: number
  keyframe_url: string
}

export interface StreamDoneMessage {
  type: 'done'
  summary: {
    status: string
    frames_processed: number
    incidents_detected: number
    total_frames: number
    fps: number
    job_id: string
  }
}

export interface StreamErrorMessage {
  type: 'error'
  message: string
}

export type JobStreamMessage =
  | StreamFrameMessage
  | StreamIncidentMessage
  | StreamDoneMessage
  | StreamErrorMessage

// Typed query and mutation functions for use with TanStack Query
export const api = {
  getHealth: async () => {
    const { data, error } = await apiClient.GET('/api/health')
    if (error) throw error
    return data
  },

  getConfig: async () => {
    const { data, error } = await apiClient.GET('/api/config')
    if (error) throw error
    return data
  },

  getStats: async () => {
    const { data, error } = await apiClient.GET('/api/stats')
    if (error) throw error
    return data
  },

  getAlertSettings: async () => {
    const { data, error } = await apiClient.GET('/api/alerts/settings')
    if (error) throw error
    return data
  },

  updateAlertSettings: async (settings: AlertSettingsUpdateRequest) => {
    const { data, error } = await apiClient.PUT('/api/alerts/settings', {
      body: settings,
    })
    if (error) throw error
    return data
  },

  testAlerts: async (dryRun: boolean = false, lang: string = 'en', channels?: string[]) => {
    const { data, error } = await apiClient.POST('/api/alerts/test', {
      body: {
        dry_run: dryRun,
        lang,
        channels,
      },
    })
    if (error) throw error
    return data
  },

  getIncidents: async (params?: {
    page?: number
    page_size?: number
    class?: string | null
    start_date?: string | null
    end_date?: string | null
    severity?: string | null
    status?: string | null
    sort_by?: string
    order?: string
  }) => {
    const { data, error } = await apiClient.GET('/api/incidents', {
      params: { query: params },
    })
    if (error) throw error
    return data
  },

  getIncident: async (incidentId: string) => {
    const { data, error } = await apiClient.GET('/api/incidents/{incident_id}', {
      params: { path: { incident_id: incidentId } },
    })
    if (error) throw error
    return data
  },

  updateIncidentStatus: async (
    incidentId: string,
    status: 'approved' | 'rejected' | 'dispatched',
    reviewerName: string = 'Traffic Desk Officer'
  ) => {
    const { data, error } = await apiClient.POST(
      '/api/incidents/{incident_id}/status',
      {
        params: { path: { incident_id: incidentId } },
        body: { status, reviewer_name: reviewerName },
      }
    )
    if (error) throw error
    return data
  },

  generateDispatchReport: async (incidentId: string, lang: string = 'en') => {
    const { data, error } = await apiClient.POST(
      '/api/incidents/{incident_id}/dispatch-report',
      {
        params: {
          path: { incident_id: incidentId },
          query: { lang },
        },
      }
    )
    if (error) throw error
    return data
  },

  createJobWithSample: async (options?: {
    model_name?: string
    confidence?: number
    use_injury_model?: boolean
  }) => {
    const response = await fetch('/api/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        source: 'sample',
        model_name: options?.model_name,
        confidence: options?.confidence,
        use_injury_model: options?.use_injury_model,
      }),
    })
    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: response.statusText }))
      throw new Error(err.detail || 'Failed to start job')
    }
    return (await response.json()) as JobCreateResponse
  },

  createJobWithUpload: async (
    file: File,
    options?: {
      model_name?: string
      confidence?: number
      use_injury_model?: boolean
    }
  ) => {
    const formData = new FormData()
    formData.append('file', file)
    if (options?.model_name) formData.append('model_name', options.model_name)
    if (options?.confidence !== undefined)
      formData.append('confidence', options.confidence.toString())
    if (options?.use_injury_model !== undefined)
      formData.append('use_injury_model', options.use_injury_model.toString())

    const response = await fetch('/api/jobs', {
      method: 'POST',
      body: formData,
    })
    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: response.statusText }))
      throw new Error(err.detail || 'Failed to upload video and start job')
    }
    return (await response.json()) as JobCreateResponse
  },

  cancelJob: async (jobId: string) => {
    const { data, error } = await apiClient.POST('/api/jobs/{job_id}/cancel', {
      params: { path: { job_id: jobId } },
    })
    if (error) throw error
    return data
  },

  getExportCsvUrl: () => '/api/incidents/export.csv',
  getIncidentPdfUrl: (incidentId: string) => `/api/incidents/${incidentId}/pdf`,
}
