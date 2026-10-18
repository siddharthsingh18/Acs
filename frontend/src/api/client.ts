import createClient from 'openapi-fetch'
import type { components, paths } from './schema'

/**
 * Typed OpenAPI client connected via Vite dev proxy or relative path in production.
 */
export const apiClient = createClient<paths>({
  baseUrl: '',
})

/**
 * Fetch wrapper that automatically injects the Authorization Bearer header.
 */
export const authFetch = (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
  const token = localStorage.getItem('acs_access_token')
  const headers = new Headers(init?.headers || {})
  if (token && !headers.has('Authorization')) {
    headers.set('Authorization', `Bearer ${token}`)
  }
  return fetch(input, { ...init, headers })
}

apiClient.use({
  async onRequest({ request }) {
    const token = localStorage.getItem('acs_access_token')
    if (token) {
      request.headers.set('Authorization', `Bearer ${token}`)
    }
    return request
  },
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
export type AlertDeliveryResponse = components['schemas']['AlertDeliveryResponse']
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
  clip_url?: string | null
  clip_annotated_url?: string | null
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

export interface CameraHealth {
  connected: boolean
  fps: number
  last_frame_age: number
  dropped_frames: number
  reconnect_count: number
  is_incident_active: boolean
  active_incident_id?: string | null
  last_accident_class?: string | null
  last_accident_confidence?: number | null
}

export interface CameraItem {
  id: string
  name: string
  source_type: 'file' | 'webcam' | 'rtsp' | 'http'
  source_url: string
  location_lat?: number | null
  location_lng?: number | null
  enabled: boolean
  confidence_override?: number | null
  created_at: number
  updated_at: number
  health?: CameraHealth | null
}

export interface CameraCreatePayload {
  name: string
  source_url: string
  location_lat?: number | null
  location_lng?: number | null
  enabled?: boolean
  confidence_override?: number | null
}

export interface CameraStreamFrameMessage {
  type: 'frame'
  camera_id: string
  camera_name: string
  frame_number: number
  timestamp: number
  fps: number
  incident_active: boolean
  active_incident_id?: string | null
  frame: string
  detections_count: number
}

export interface CameraStreamIncidentMessage {
  type: 'incident'
  camera_id: string
  camera_name: string
  incident_id: string
  class_name: string
  confidence: number
  start_timestamp: string
  end_timestamp: string
  keyframe_url: string
}

export type CameraStreamMessage =
  | CameraStreamFrameMessage
  | CameraStreamIncidentMessage
  | { type: 'heartbeat' | 'status'; camera_id: string; connected?: boolean; health?: CameraHealth }

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

  testAlert: async (dryRun: boolean = false) => {
    return api.testAlerts(dryRun)
  },

  getVapidPublicKey: async () => {
    const res = await authFetch('/api/alerts/vapid-public-key')
    if (!res.ok) throw new Error('Failed to fetch VAPID public key')
    return (await res.json()) as { public_key: string }
  },

  subscribeWebPush: async (subscription: { endpoint: string; keys: { p256dh: string; auth: string }; user_agent?: string }) => {
    const res = await authFetch('/api/alerts/push/subscribe', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(subscription),
    })
    if (!res.ok) throw new Error('Failed to subscribe to Web Push')
    return await res.json()
  },

  unsubscribeWebPush: async (endpoint: string) => {
    const res = await authFetch('/api/alerts/push/unsubscribe', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ endpoint }),
    })
    if (!res.ok) throw new Error('Failed to unsubscribe from Web Push')
    return await res.json()
  },

  getIncidentAlertDeliveries: async (incidentId: string) => {
    const res = await authFetch(`/api/incidents/${incidentId}/alert-deliveries`)
    if (!res.ok) throw new Error('Failed to fetch incident alert deliveries')
    return (await res.json()) as AlertDeliveryResponse[]
  },

  acknowledgeIncident: async (incidentId: string, notes?: string) => {
    const res = await authFetch(`/api/incidents/${incidentId}/ack`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ notes }),
    })
    if (!res.ok) throw new Error('Failed to acknowledge incident')
    return (await res.json()) as IncidentResponse
  },

  sendIncidentAlert: async (incidentId: string, channels?: string[], dryRun?: boolean) => {
    const query = new URLSearchParams()
    if (channels && channels.length) {
      channels.forEach(ch => query.append('channels', ch))
    }
    if (dryRun !== undefined) {
      query.append('dry_run', dryRun ? 'true' : 'false')
    }
    const qStr = query.toString() ? `?${query.toString()}` : ''
    const res = await authFetch(`/api/incidents/${incidentId}/alert${qStr}`, {
      method: 'POST',
    })
    if (!res.ok) throw new Error('Failed to dispatch incident alert')
    return await res.json()
  },

  triggerEscalationCheck: async () => {
    const res = await authFetch('/api/alerts/escalate-check', {
      method: 'POST',
    })
    if (!res.ok) throw new Error('Failed to execute escalation check')
    return await res.json()
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
    const response = await authFetch('/api/jobs', {
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

    const response = await authFetch('/api/jobs', {
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

  // Camera Management Endpoints
  getCameras: async (): Promise<CameraItem[]> => {
    const response = await authFetch('/api/cameras')
    if (!response.ok) throw new Error('Failed to fetch cameras')
    return (await response.json()) as CameraItem[]
  },

  getCamera: async (id: string): Promise<CameraItem> => {
    const response = await authFetch(`/api/cameras/${id}`)
    if (!response.ok) throw new Error(`Failed to fetch camera ${id}`)
    return (await response.json()) as CameraItem
  },

  createCamera: async (payload: CameraCreatePayload): Promise<CameraItem> => {
    const response = await authFetch('/api/cameras', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: response.statusText }))
      throw new Error(err.detail || 'Failed to create camera')
    }
    return (await response.json()) as CameraItem
  },

  updateCamera: async (id: string, payload: Partial<CameraCreatePayload>): Promise<CameraItem> => {
    const response = await authFetch(`/api/cameras/${id}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: response.statusText }))
      throw new Error(err.detail || `Failed to update camera ${id}`)
    }
    return (await response.json()) as CameraItem
  },

  deleteCamera: async (id: string): Promise<void> => {
    const response = await authFetch(`/api/cameras/${id}`, {
      method: 'DELETE',
    })
    if (!response.ok) throw new Error(`Failed to delete camera ${id}`)
  },

  getCameraStreamWsUrl: (id: string): string => {
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    return `${proto}//${window.location.host}/api/cameras/${id}/stream`
  },

  getRetentionSettings: async (): Promise<RetentionSettings> => {
    const response = await authFetch('/api/retention')
    if (!response.ok) throw new Error('Failed to fetch retention settings')
    return (await response.json()) as RetentionSettings
  },

  updateRetentionSettings: async (payload: {
    retention_days?: number
    max_disk_usage_mb?: number
    trigger_cleanup?: boolean
    privacy_blur_enabled?: boolean
    privacy_preserve_unblurred_original?: boolean
    incident_retention_days?: number
  }): Promise<RetentionSettings> => {
    const response = await authFetch('/api/retention', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
    if (!response.ok) throw new Error('Failed to update retention settings')
    return (await response.json()) as RetentionSettings
  },

  // Review Queue & Active Learning Endpoints
  getReviewQueue: async (includeReviewed = false): Promise<ReviewQueueResponse> => {
    const response = await authFetch(`/api/review-queue?include_reviewed=${includeReviewed}`)
    if (!response.ok) throw new Error('Failed to fetch review queue')
    return (await response.json()) as ReviewQueueResponse
  },

  getReviewStats: async (): Promise<ReviewStatsResponse> => {
    const response = await authFetch('/api/review-queue/stats')
    if (!response.ok) throw new Error('Failed to fetch review stats')
    return (await response.json()) as ReviewStatsResponse
  },

  submitReviewDecision: async (
    incidentId: string,
    payload: ReviewDecisionPayload
  ): Promise<ReviewFeedbackResponse> => {
    const response = await authFetch(`/api/review-queue/${incidentId}/decision`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: response.statusText }))
      throw new Error(err.detail || 'Failed to submit review decision')
    }
    return (await response.json()) as ReviewFeedbackResponse
  },

  undoReviewDecision: async (incidentId: string): Promise<IncidentResponse> => {
    const response = await authFetch(`/api/review-queue/${incidentId}/undo`, {
      method: 'POST',
    })
    if (!response.ok) throw new Error('Failed to undo review decision')
    return (await response.json()) as IncidentResponse
  },

  // Audit Logs (Admin only)
  getAuditLogs: async (params?: {
    skip?: number
    limit?: number
    action?: string
    target_type?: string
    username?: string
  }): Promise<AuditLogListResponse> => {
    const query = new URLSearchParams()
    if (params?.skip !== undefined) query.set('skip', String(params.skip))
    if (params?.limit !== undefined) query.set('limit', String(params.limit))
    if (params?.action) query.set('action', params.action)
    if (params?.target_type) query.set('target_type', params.target_type)
    if (params?.username) query.set('username', params.username)

    const qs = query.toString() ? `?${query.toString()}` : ''
    const response = await authFetch(`/api/audit-logs${qs}`)
    if (!response.ok) throw new Error('Failed to fetch audit logs')
    return (await response.json()) as AuditLogListResponse
  },

  // User Management (Admin only)
  getUsers: async (): Promise<UserItem[]> => {
    const response = await authFetch('/api/users')
    if (!response.ok) throw new Error('Failed to fetch users')
    return (await response.json()) as UserItem[]
  },

  createUser: async (payload: {
    username: string
    password: string
    full_name: string
    role: string
  }): Promise<UserItem> => {
    const response = await authFetch('/api/users', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: 'Failed to create user' }))
      throw new Error(err.detail || 'Failed to create user')
    }
    return (await response.json()) as UserItem
  },

  updateUser: async (
    id: string,
    payload: { full_name?: string; role?: string; is_active?: boolean; password?: string }
  ): Promise<UserItem> => {
    const response = await authFetch(`/api/users/${id}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: 'Failed to update user' }))
      throw new Error(err.detail || 'Failed to update user')
    }
    return (await response.json()) as UserItem
  },

  deleteUser: async (id: string): Promise<void> => {
    const response = await authFetch(`/api/users/${id}`, {
      method: 'DELETE',
    })
    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: 'Failed to delete user' }))
      throw new Error(err.detail || 'Failed to delete user')
    }
  },

  // Incident Privacy & Retention Operations
  exportIncidentBundle: async (incidentId: string): Promise<Blob> => {
    const response = await authFetch(`/api/incidents/${incidentId}/export`)
    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: 'Failed to export incident bundle' }))
      throw new Error(err.detail || 'Failed to export incident bundle')
    }
    return await response.blob()
  },

  deleteIncident: async (
    incidentId: string
  ): Promise<{ status: string; incident_id: string; deleted_files_count: number }> => {
    const response = await authFetch(`/api/incidents/${incidentId}`, {
      method: 'DELETE',
    })
    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: 'Failed to delete incident' }))
      throw new Error(err.detail || 'Failed to delete incident')
    }
    return await response.json()
  },

  getUnredactedKeyframeUrl: (incidentId: string): string => {
    return `/api/incidents/${incidentId}/unredacted-keyframe`
  },

  getUnredactedClipUrl: (incidentId: string): string => {
    return `/api/incidents/${incidentId}/unredacted-clip`
  },

  runRetentionCleanup: async (params?: {
    retention_days?: number
    max_disk_mb?: number
  }): Promise<{
    expired_incidents_deleted: number
    expired_clips_deleted: number
    disk_cap_clips_deleted: number
    freed_disk_mb: number
    remaining_disk_mb: number
  }> => {
    const query = new URLSearchParams()
    if (params?.retention_days !== undefined) query.set('retention_days', String(params.retention_days))
    if (params?.max_disk_mb !== undefined) query.set('max_disk_mb', String(params.max_disk_mb))
    const qs = query.toString() ? `?${query.toString()}` : ''
    const response = await authFetch(`/api/incidents/retention/cleanup${qs}`, {
      method: 'POST',
    })
    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: 'Failed to execute retention cleanup' }))
      throw new Error(err.detail || 'Failed to execute retention cleanup')
    }
    return await response.json()
  },
}


export interface RetentionSettings {
  retention_days: number
  max_disk_usage_mb: number
  current_disk_usage_mb: number
  total_clips_count: number
  clips_dir: string
  privacy_blur_enabled?: boolean
  privacy_preserve_unblurred_original?: boolean
  incident_retention_days?: number
}

export interface ReviewQueueItem {
  incident_id: string
  class_name: string
  max_confidence: number
  image_path: string
  keyframe_url: string
  clip_path?: string | null
  clip_url?: string | null
  clip_annotated_path?: string | null
  clip_annotated_url?: string | null
  verification_status?: string | null
  vlm_confidence?: number | null
  vlm_reason?: string | null
  physics_passed?: boolean | null
  uncertainty_reason: string
  start_timestamp: string
  severity?: string | null
  severity_score?: number | null
  duration: number
}

export interface ReviewQueueResponse {
  items: ReviewQueueItem[]
  total: number
}

export interface ReviewStatsResponse {
  total_reviewed: number
  accidents_approved: number
  hard_negatives: number
  wrong_class_count: number
  staging_pending_export: number
}

export interface ReviewDecisionPayload {
  decision: 'accident' | 'not_accident' | 'wrong_class'
  corrected_class?: string | null
  corrected_boxes?: any | null
  reviewer_name?: string
  notes?: string | null
}

export interface ReviewFeedbackResponse {
  id?: number | null
  incident_id: string
  decision: string
  original_class: string
  corrected_class?: string | null
  confidence: number
  image_path: string
  clip_path?: string | null
  clip_annotated_path?: string | null
  reviewer_name: string
  reviewed_at: string
  notes?: string | null
  corrected_boxes?: string | null
  is_exported: boolean
  exported_at?: string | null
  created_at: number
}

export interface AuditLogItem {
  id: number
  timestamp: string
  epoch: number
  user_id?: string | null
  username: string
  user_role: string
  action: string
  target_type: string
  target_id?: string | null
  ip_address?: string | null
  user_agent?: string | null
  details?: string | null
}

export interface AuditLogListResponse {
  items: AuditLogItem[]
  total: number
  skip: number
  limit: number
}

export interface UserItem {
  id: string
  username: string
  full_name: string
  role: 'viewer' | 'operator' | 'admin'
  is_active: boolean
  created_at: number
  last_login_at?: number | null
}


