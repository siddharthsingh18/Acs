import React, { useState, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useForm, Controller } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import {
  Sliders,
  Bell,
  Cpu,
  Send,
  CheckCircle2,
  Save,
  Radio,
  Clock,
  ShieldCheck,
  ShieldAlert,
  Lock,
  Globe,
  Sun,
  Moon,
  AlertTriangle,
  Check,
  Camera as CameraIcon,
  MapPin,
  Plus,
  Trash2,
  HardDrive,
  EyeOff,
  FolderArchive,
  Info,
  Smartphone,
} from 'lucide-react'
import {
  isPushSupported,
  getPushSubscription,
  subscribeToPushNotifications,
  unsubscribeFromPushNotifications,
} from '@/lib/pushNotifications'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import {
  api,
  type AlertSettingsUpdateRequest,
  type CameraCreatePayload,
} from '@/api/client'
import { CameraLocationPicker } from '@/components/map/CameraLocationPicker'
import { AddCameraModal } from '@/components/live/AddCameraModal'
import { useAppStore } from '@/store/appStore'
import { useAuthStore } from '@/store/authStore'
import { cn } from '@/lib/utils'

// Zod validation schema with strict boundary constraints
export const settingsFormSchema = z.object({
  // Emergency Alerts
  alerts_enabled: z.boolean(),
  alerts_dry_run: z.boolean(),
  channels: z.array(z.string()).min(1, 'Select at least one notification delivery channel'),
  escalation_enabled: z.boolean(),
  escalation_delay_minutes: z
    .number()
    .min(0.5, 'Minimum escalation delay is 0.5 minutes (30s)')
    .max(60.0, 'Maximum escalation delay is 60 minutes'),

  // Vision & Inference Defaults
  model_path: z.string().min(1, 'Primary YOLO weights checkpoint is required'),
  confidence_threshold: z
    .number()
    .min(0.05, 'Minimum confidence threshold is 0.05 (5%)')
    .max(1.0, 'Maximum confidence threshold is 1.0 (100%)'),

  // Temporal State Machine & Filters
  debounce_frames: z
    .number()
    .int('Debounce frames must be an integer')
    .min(1, 'Debounce threshold must be at least 1 frame')
    .max(30, 'Debounce threshold cannot exceed 30 frames'),
  cooldown_seconds: z
    .number()
    .min(1.0, 'Cooldown period must be at least 1.0 second')
    .max(60.0, 'Cooldown period cannot exceed 60.0 seconds'),
  timeout_seconds: z
    .number()
    .min(1.0, 'Inactivity timeout must be at least 1.0 second')
    .max(15.0, 'Inactivity timeout cannot exceed 15.0 seconds'),

  // Preferences
  language: z.enum(['en', 'ar']),
  theme: z.enum(['dark', 'light']),
})

export type SettingsFormData = z.infer<typeof settingsFormSchema>

interface TestResultState {
  status: 'idle' | 'testing' | 'success' | 'error'
  message?: string
  results?: Record<string, boolean>
  timestamp?: string
}

export const SettingsPage: React.FC = () => {
  const { t, i18n } = useTranslation()
  const queryClient = useQueryClient()
  const { activeModel, setActiveModel, theme, setTheme } = useAppStore()
  const { user } = useAuthStore()
  const isAdmin = user ? user.role === 'admin' : true

  const [saveSuccessMsg, setSaveSuccessMsg] = useState<string | null>(null)
  const [testState, setTestState] = useState<TestResultState>({ status: 'idle' })

  // System config query
  const { data: config } = useQuery({
    queryKey: ['config'],
    queryFn: () => api.getConfig(),
  })

  // Alert settings query
  const { data: alertSettings } = useQuery({
    queryKey: ['alertSettings'],
    queryFn: () => api.getAlertSettings(),
  })

  // Camera Geolocation & Management state
  const [isAddCameraModalOpen, setIsAddCameraModalOpen] = useState(false)
  const [editingCameraId, setEditingCameraId] = useState<string | null>(null)
  const [editingCoords, setEditingCoords] = useState<{ lat: number; lng: number } | null>(null)
  const [cameraFeedbackMsg, setCameraFeedbackMsg] = useState<string | null>(null)

  // Cameras query
  const { data: cameras = [] } = useQuery({
    queryKey: ['cameras'],
    queryFn: () => (api.getCameras ? api.getCameras() : Promise.resolve([])),
  })

  // Camera coordinates update mutation
  const updateCameraCoordsMutation = useMutation({
    mutationFn: async ({ id, lat, lng }: { id: string; lat: number | null; lng: number | null }) => {
      return api.updateCamera(id, { location_lat: lat, location_lng: lng })
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['cameras'] })
      setCameraFeedbackMsg(t('map.save_location', 'Camera location updated and saved.'))
      setTimeout(() => setCameraFeedbackMsg(null), 3500)
      setEditingCameraId(null)
      setEditingCoords(null)
    },
  })

  // Camera create mutation
  const createCameraMutation = useMutation({
    mutationFn: async (payload: CameraCreatePayload) => {
      return api.createCamera(payload)
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['cameras'] })
      setCameraFeedbackMsg('New camera registered successfully.')
      setTimeout(() => setCameraFeedbackMsg(null), 3500)
    },
  })

  // Camera delete mutation
  const deleteCameraMutation = useMutation({
    mutationFn: async (id: string) => {
      return api.deleteCamera(id)
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['cameras'] })
      setCameraFeedbackMsg('Camera removed successfully.')
      setTimeout(() => setCameraFeedbackMsg(null), 3500)
    },
  })

  // Privacy & Retention Policy Query and Mutations
  const { data: retentionData, refetch: refetchRetention } = useQuery({
    queryKey: ['retentionSettings'],
    queryFn: () => api.getRetentionSettings(),
  })

  const [privacyBlur, setPrivacyBlur] = useState(true)
  const [preserveOriginal, setPreserveOriginal] = useState(true)
  const [incidentRetentionDays, setIncidentRetentionDays] = useState(30)
  const [clipRetentionDays, setClipRetentionDays] = useState(30)
  const [maxDiskMb, setMaxDiskMb] = useState(5000)
  const [retentionFeedback, setRetentionFeedback] = useState<string | null>(null)
  const [cleanupResult, setCleanupResult] = useState<string | null>(null)
  const [isCleaningUp, setIsCleaningUp] = useState(false)

  useEffect(() => {
    if (retentionData) {
      setPrivacyBlur(retentionData.privacy_blur_enabled ?? true)
      setPreserveOriginal(retentionData.privacy_preserve_unblurred_original ?? true)
      setIncidentRetentionDays(retentionData.incident_retention_days ?? 30)
      setClipRetentionDays(retentionData.retention_days ?? 30)
      setMaxDiskMb(retentionData.max_disk_usage_mb ?? 5000)
    }
  }, [retentionData])

  const updateRetentionMutation = useMutation({
    mutationFn: (payload: {
      privacy_blur_enabled?: boolean
      privacy_preserve_unblurred_original?: boolean
      incident_retention_days?: number
      retention_days?: number
      max_disk_usage_mb?: number
    }) => api.updateRetentionSettings(payload),
    onSuccess: () => {
      setRetentionFeedback('Privacy & retention policies saved successfully.')
      setTimeout(() => setRetentionFeedback(null), 4000)
      queryClient.invalidateQueries({ queryKey: ['retentionSettings'] })
    },
    onError: (err: any) => {
      setRetentionFeedback(`Failed to save retention settings: ${err?.message || 'Error'}`)
    },
  })

  const handleRunCleanup = async () => {
    try {
      setIsCleaningUp(true)
      const res = await api.runRetentionCleanup()
      setCleanupResult(
        `Cleanup executed: purged ${res.expired_incidents_deleted} expired incidents, ${res.expired_clips_deleted} expired clips, and freed ${res.freed_disk_mb?.toFixed(1) ?? '0'} MB.`
      )
      refetchRetention()
      setTimeout(() => setCleanupResult(null), 7000)
    } catch (err: any) {
      setCleanupResult(`Cleanup error: ${err.message || 'Failed'}`)
    } finally {
      setIsCleaningUp(false)
    }
  }

  const handleSaveRetention = () => {
    updateRetentionMutation.mutate({
      privacy_blur_enabled: privacyBlur,
      privacy_preserve_unblurred_original: preserveOriginal,
      incident_retention_days: incidentRetentionDays,
      retention_days: clipRetentionDays,
      max_disk_usage_mb: maxDiskMb,
    })
  }

  // React Hook Form initialized with zod validation
  const {
    control,
    handleSubmit,
    setValue,
    watch,
    formState: { errors, isSubmitting },
  } = useForm<SettingsFormData>({
    resolver: zodResolver(settingsFormSchema),
    defaultValues: {
      alerts_enabled: true,
      alerts_dry_run: false,
      channels: ['console'],
      escalation_enabled: true,
      escalation_delay_minutes: 5.0,
      model_path: activeModel || 'best.pt',
      confidence_threshold: 0.72,
      debounce_frames: Number(localStorage.getItem('acs_debounce_frames')) || 5,
      cooldown_seconds: 10.0,
      timeout_seconds: Number(localStorage.getItem('acs_timeout_seconds')) || 3.0,
      language: (i18n.language?.startsWith('ar') ? 'ar' : 'en'),
      theme: theme === 'light' ? 'light' : 'dark',
    },
  })

  // Device push notification registration state
  const [isDeviceSubscribed, setIsDeviceSubscribed] = useState(false)
  const [pushLoading, setPushLoading] = useState(false)
  const [pushStatusMsg, setPushStatusMsg] = useState<string | null>(null)

  useEffect(() => {
    if (isPushSupported()) {
      getPushSubscription().then((sub) => {
        setIsDeviceSubscribed(Boolean(sub))
      })
    }
  }, [])

  const handleToggleDevicePush = async () => {
    setPushLoading(true)
    setPushStatusMsg(null)
    if (isDeviceSubscribed) {
      const res = await unsubscribeFromPushNotifications()
      if (res.success) {
        setIsDeviceSubscribed(false)
        setPushStatusMsg('Device push notifications disabled.')
      } else {
        setPushStatusMsg('Failed to unsubscribe: ' + (res.error || 'Error'))
      }
    } else {
      const res = await subscribeToPushNotifications()
      if (res.success) {
        setIsDeviceSubscribed(true)
        setPushStatusMsg('Device subscribed! You will receive push notifications on this phone/browser.')
      } else {
        setPushStatusMsg('Failed to subscribe: ' + (res.error || 'Permission denied or unsupported'))
      }
    }
    setPushLoading(false)
    setTimeout(() => setPushStatusMsg(null), 4000)
  }

  // Populate form defaults when queries resolve
  useEffect(() => {
    if (alertSettings) {
      setValue('alerts_enabled', alertSettings.enabled)
      setValue('alerts_dry_run', alertSettings.dry_run)
      if (alertSettings.escalation_enabled !== undefined && alertSettings.escalation_enabled !== null) {
        setValue('escalation_enabled', alertSettings.escalation_enabled)
      }
      if (alertSettings.escalation_delay_minutes !== undefined && alertSettings.escalation_delay_minutes !== null) {
        setValue('escalation_delay_minutes', alertSettings.escalation_delay_minutes)
      }
      if (alertSettings.channels && alertSettings.channels.length > 0) {
        setValue('channels', alertSettings.channels)
      }
    }
    if (config?.defaults) {
      if (config.defaults.model_path) setValue('model_path', config.defaults.model_path)
      if (config.defaults.confidence_threshold)
        setValue('confidence_threshold', config.defaults.confidence_threshold)
      if (config.defaults.cooldown_seconds)
        setValue('cooldown_seconds', config.defaults.cooldown_seconds)
    }
  }, [alertSettings, config, setValue])

  // Save Settings Mutation
  const updateMutation = useMutation({
    mutationFn: (data: AlertSettingsUpdateRequest) => api.updateAlertSettings(data),
    onSuccess: () => {
      setSaveSuccessMsg(t('settings.saved_success', 'Settings saved and applied successfully.'))
      setTimeout(() => setSaveSuccessMsg(null), 4000)
      queryClient.invalidateQueries({ queryKey: ['alertSettings'] })
      queryClient.invalidateQueries({ queryKey: ['config'] })
    },
  })

  // Send Test Alert Mutation
  const testMutation = useMutation({
    mutationFn: async () => {
      const currentDryRun = watch('alerts_dry_run')
      const currentChannels = watch('channels')
      const currentLang = watch('language')
      return api.testAlerts(currentDryRun, currentLang, currentChannels)
    },
    onMutate: () => {
      setTestState({ status: 'testing' })
    },
    onSuccess: (data) => {
      setTestState({
        status: 'success',
        message: data.message,
        results: (data.results || {}) as Record<string, boolean>,
        timestamp: new Date().toLocaleTimeString(),
      })
    },
    onError: (err: any) => {
      setTestState({
        status: 'error',
        message: err?.message || 'Failed to dispatch test alert packet.',
        timestamp: new Date().toLocaleTimeString(),
      })
    },
  })

  const onSubmit = (values: SettingsFormData) => {
    // 1. Update backend settings
    updateMutation.mutate({
      enabled: values.alerts_enabled,
      dry_run: values.alerts_dry_run,
      channels: values.channels,
      escalation_enabled: values.escalation_enabled,
      escalation_delay_minutes: values.escalation_delay_minutes,
      model_path: values.model_path,
      confidence_threshold: values.confidence_threshold,
      cooldown_seconds: values.cooldown_seconds,
    })

    // 2. Persist client-side application state
    setActiveModel(values.model_path)
    localStorage.setItem('acs_debounce_frames', values.debounce_frames.toString())
    localStorage.setItem('acs_cooldown_seconds', values.cooldown_seconds.toString())
    localStorage.setItem('acs_timeout_seconds', values.timeout_seconds.toString())
    localStorage.setItem('acs_confidence_threshold', values.confidence_threshold.toString())

    // 3. Sync language and theme if changed
    if (values.language !== i18n.language) {
      i18n.changeLanguage(values.language)
    }
    if (values.theme !== theme) {
      setTheme(values.theme)
    }
  }

  const selectedChannels = watch('channels') || []
  const watchedConfidence = watch('confidence_threshold') || 0.72
  const watchedCooldown = watch('cooldown_seconds') || 10.0
  const watchedDebounce = watch('debounce_frames') || 5
  const watchedTimeout = watch('timeout_seconds') || 3.0

  const toggleChannel = (channelId: string) => {
    const current = watch('channels') || []
    if (current.includes(channelId)) {
      if (current.length > 1) {
        setValue(
          'channels',
          current.filter((c) => c !== channelId),
          { shouldValidate: true, shouldDirty: true }
        )
      }
    } else {
      setValue('channels', [...current, channelId], {
        shouldValidate: true,
        shouldDirty: true,
      })
    }
  }

  const isChannelConfigured = (channelId: string): boolean => {
    if (channelId === 'console') return true
    const status = alertSettings?.channel_status?.[channelId]
    if (status) return status.configured
    return false
  }

  const ALL_CHANNELS = [
    {
      id: 'webpush',
      title: 'Web Push (PWA Notifications)',
      descriptionKey: 'settings.channel_webpush',
      fallbackDesc: 'Delivers instant browser and lockscreen phone notifications with one-click acknowledgment.',
      secretHelp: 'VAPID_PUBLIC_KEY & VAPID_PRIVATE_KEY (Auto-generated)',
    },
    {
      id: 'telegram',
      title: 'Telegram Alert Bot',
      descriptionKey: 'settings.channel_telegram',
      fallbackDesc: 'Dispatches emergency incident cards with keyframe photos to Telegram channel.',
      secretHelp: 'TELEGRAM_BOT_TOKEN & TELEGRAM_CHAT_ID',
    },
    {
      id: 'email',
      title: 'SMTP Dispatch Email',
      descriptionKey: 'settings.channel_email',
      fallbackDesc: 'Transmits MIME multipart dispatch summaries with keyframe attachments.',
      secretHelp: 'SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_TO',
    },
    {
      id: 'console',
      title: 'Control Room Console',
      descriptionKey: 'settings.channel_console',
      fallbackDesc: 'Streams timestamped alert banners to terminal and structured application loggers.',
      secretHelp: null,
    },
  ]

  return (
    <form onSubmit={handleSubmit(onSubmit)} className="space-y-6 max-w-7xl mx-auto pb-12">
      {/* Page Header & Global Save Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-border pb-4">
        <div>
          <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight flex items-center gap-2.5 text-foreground">
            <Sliders className="h-7 w-7 text-primary" />
            {t('settings.header', 'System & Emergency Alert Configuration')}
          </h1>
          <p className="text-xs sm:text-sm text-muted-foreground mt-1">
            {t(
              'settings.subtitle',
              'Manage detection models, inference thresholds, temporal state machine, and emergency dispatch channels.'
            )}
          </p>
        </div>

        <div className="flex items-center gap-3">
          {isAdmin && (
            <a href="/audit">
              <Button type="button" variant="outline" size="sm" className="gap-1.5 text-xs font-semibold">
                <ShieldCheck className="h-4 w-4 text-primary" />
                <span>Audit Trail</span>
              </Button>
            </a>
          )}
          <Button
            type="submit"
            disabled={isSubmitting || updateMutation.isPending || (user ? !isAdmin : false)}
            className="gap-2 text-xs font-bold px-5 shadow-sm"
          >
            <Save className="h-4 w-4" />
            {updateMutation.isPending
              ? t('settings.saving', 'Saving...')
              : t('settings.save_settings', 'Save Settings')}
          </Button>
        </div>
      </div>

      {/* Role restriction banner */}
      {user && !isAdmin && (
        <div className="p-3.5 rounded-lg bg-amber-500/10 border border-amber-500/25 text-amber-300 text-xs flex items-center justify-between gap-2.5 shadow-sm">
          <div className="flex items-center gap-2">
            <Lock className="h-4 w-4 shrink-0 text-amber-400" />
            <span>
              Operating in <strong>read-only mode</strong> ({user.role.toUpperCase()}). Modifying cameras, dispatch credentials, or thresholds requires <strong>Administrator</strong> privileges.
            </span>
          </div>
          <Badge variant="outline" className="text-[10px] uppercase font-mono border-amber-500/40 text-amber-400">
            {user.role}
          </Badge>
        </div>
      )}

      {/* Global Success / Validation Banner */}
      {saveSuccessMsg && (
        <div className="p-3.5 rounded-lg bg-[var(--status-ok)]/15 border border-[var(--status-ok)]/30 text-[var(--status-ok)] text-xs flex items-center gap-2.5 font-medium shadow-sm animate-in fade-in">
          <CheckCircle2 className="h-4 w-4 shrink-0" />
          <span>{saveSuccessMsg}</span>
        </div>
      )}

      {errors.channels && (
        <div className="p-3 rounded-lg bg-[var(--status-alert)]/15 border border-[var(--status-alert)]/30 text-[var(--status-alert)] text-xs flex items-center gap-2 font-medium">
          <AlertTriangle className="h-4 w-4 shrink-0" />
          <span>{errors.channels.message}</span>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* ========================================================================= */}
        {/* SECTION 1: EMERGENCY ALERT CHANNELS & TEST DISPATCH                      */}
        {/* ========================================================================= */}
        <Card className="border-border shadow-sm">
          <CardHeader className="py-4 px-6 border-b border-border bg-muted/20">
            <div className="flex items-center justify-between">
              <CardTitle className="text-base font-bold flex items-center gap-2">
                <Bell className="h-4 w-4 text-[var(--status-alert)]" />
                {t('settings.alerts_settings', 'Emergency Alert Dispatch')}
              </CardTitle>
              <Badge variant="outline" className="text-[10px] font-mono">
                Live Multi-Channel
              </Badge>
            </div>
            <CardDescription className="text-xs">
              {t(
                'settings.alerts_desc',
                'Configure automated notification channels, simulation mode, and test connectivity.'
              )}
            </CardDescription>
          </CardHeader>

          <CardContent className="p-6 space-y-6">
            {/* Master Alert Enable & Dry-Run Toggles */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 p-4 rounded-xl border border-border bg-card/60">
              <div className="space-y-1.5">
                <label className="flex items-center gap-3 cursor-pointer">
                  <Controller
                    control={control}
                    name="alerts_enabled"
                    render={({ field }) => (
                      <input
                        type="checkbox"
                        checked={field.value}
                        onChange={(e) => field.onChange(e.target.checked)}
                        className="rounded border-border text-primary focus:ring-primary h-4 w-4"
                      />
                    )}
                  />
                  <span className="text-xs font-bold text-foreground">
                    {t('settings.enable_alerts', 'Enable Emergency Alerts')}
                  </span>
                </label>
                <p className="text-[11px] text-muted-foreground">
                  Master switch to dispatch automated notifications when accidents occur.
                </p>
              </div>

              <div className="space-y-1.5">
                <label className="flex items-center gap-3 cursor-pointer">
                  <Controller
                    control={control}
                    name="alerts_dry_run"
                    render={({ field }) => (
                      <input
                        type="checkbox"
                        checked={field.value}
                        onChange={(e) => field.onChange(e.target.checked)}
                        className="rounded border-border text-primary focus:ring-primary h-4 w-4"
                      />
                    )}
                  />
                  <div className="flex items-center gap-1.5">
                    <span className="text-xs font-bold text-foreground">
                      {t('settings.dry_run', 'Dry-Run Mode')}
                    </span>
                    <Badge variant="secondary" className="text-[9px] uppercase px-1.5 py-0">
                      Demo
                    </Badge>
                  </div>
                </label>
                <p className="text-[11px] text-muted-foreground">
                  Simulates alert generation in logs without emitting live external network messages.
                </p>
              </div>
            </div>

            {/* Operator PWA Device Push Notification Link */}
            <div className="p-4 rounded-xl border border-indigo-500/30 bg-indigo-950/20 space-y-3">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <div className="flex items-start gap-3">
                  <div className="h-9 w-9 rounded-lg bg-indigo-500/20 border border-indigo-500/30 flex items-center justify-center text-indigo-400 shrink-0">
                    <Smartphone className="h-4.5 w-4.5" />
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <h4 className="text-xs font-bold text-foreground">Operator Device Push Notifications (PWA)</h4>
                      {isDeviceSubscribed ? (
                        <Badge className="bg-emerald-500/20 text-emerald-300 border-emerald-500/30 text-[9px] font-mono">
                          DEVICE SUBSCRIBED
                        </Badge>
                      ) : (
                        <Badge variant="outline" className="text-[9px] font-mono text-muted-foreground border-border">
                          NOT LINKED
                        </Badge>
                      )}
                    </div>
                    <p className="text-[11px] text-muted-foreground">
                      Receive instant real-time alerts with keyframe images and one-click acknowledgment on your phone or browser lockscreen.
                    </p>
                  </div>
                </div>

                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  disabled={pushLoading}
                  onClick={handleToggleDevicePush}
                  className={cn(
                    "shrink-0 gap-2 text-xs font-semibold",
                    isDeviceSubscribed
                      ? "border-red-500/30 text-red-400 hover:bg-red-500/10 hover:border-red-400"
                      : "border-indigo-500/40 text-indigo-300 bg-indigo-500/10 hover:bg-indigo-500/20 hover:border-indigo-400"
                  )}
                >
                  <Smartphone className="h-3.5 w-3.5" />
                  {pushLoading
                    ? 'Connecting...'
                    : isDeviceSubscribed
                    ? 'Unlink Device Push'
                    : 'Link Device for Web Push'}
                </Button>
              </div>
              {pushStatusMsg && (
                <div className="p-2.5 rounded-lg bg-background/50 border border-border text-[11px] font-mono text-indigo-300 flex items-center gap-2">
                  <Info className="h-3.5 w-3.5 text-indigo-400 shrink-0" />
                  <span>{pushStatusMsg}</span>
                </div>
              )}
            </div>

            {/* Multi-Tier Alert Escalation Policy */}
            <div className="p-4 rounded-xl border border-border bg-card/60 space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Clock className="h-4 w-4 text-amber-400" />
                  <h4 className="text-xs font-bold text-foreground">Multi-Tier Alert Escalation Policy</h4>
                </div>
                <Badge variant="outline" className="text-[10px] font-mono text-amber-400 border-amber-500/40 bg-amber-500/10">
                  Auto-Escalation
                </Badge>
              </div>
              <p className="text-[11px] text-muted-foreground">
                If an on-call operator does not acknowledge an alert within the designated threshold, the incident automatically escalates to Tier 2 channels (Email & Secondary dispatch).
              </p>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-1">
                <div className="space-y-1.5">
                  <label className="flex items-center gap-3 cursor-pointer">
                    <Controller
                      control={control}
                      name="escalation_enabled"
                      render={({ field }) => (
                        <input
                          type="checkbox"
                          checked={field.value}
                          onChange={(e) => field.onChange(e.target.checked)}
                          className="rounded border-border text-primary focus:ring-primary h-4 w-4"
                        />
                      )}
                    />
                    <span className="text-xs font-bold text-foreground">Enable Escalation Timer</span>
                  </label>
                  <p className="text-[10px] text-muted-foreground">
                    Automatically re-routes unacknowledged alerts to Tier 2 contacts.
                  </p>
                </div>

                <div className="space-y-1.5">
                  <label className="text-xs font-bold text-foreground block">
                    Escalation Delay (Minutes)
                  </label>
                  <Controller
                    control={control}
                    name="escalation_delay_minutes"
                    render={({ field }) => (
                      <div className="flex items-center gap-2">
                        <input
                          type="number"
                          step="0.5"
                          min="0.5"
                          max="60"
                          value={field.value}
                          onChange={(e) => field.onChange(parseFloat(e.target.value) || 5.0)}
                          className="w-24 px-2.5 py-1 text-xs font-mono rounded-lg border border-border bg-background text-foreground"
                        />
                        <span className="text-xs text-muted-foreground">minutes before Tier 2 dispatch</span>
                      </div>
                    )}
                  />
                </div>
              </div>
            </div>

            {/* Mandatory Regulatory Disclaimer */}
            <div className="p-3 rounded-lg bg-amber-500/10 border border-amber-500/30 text-amber-300 text-xs flex items-center gap-2.5">
              <AlertTriangle className="h-4 w-4 text-amber-400 shrink-0" />
              <span>⚠️ DISCLAIMER: Alerts support human decisions and must not trigger emergency dispatch automatically.</span>
            </div>

            {/* Channels Checklist with Configured / Not Configured Badges */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <label className="text-xs font-bold text-foreground block">
                  {t('settings.channels', 'Notification Delivery Channels')}
                </label>
                <span className="text-[11px] text-muted-foreground">
                  {selectedChannels.length} active
                </span>
              </div>

              <div className="space-y-2.5">
                {ALL_CHANNELS.map((ch) => {
                  const isChecked = selectedChannels.includes(ch.id)
                  const configured = isChannelConfigured(ch.id)

                  return (
                    <div
                      key={ch.id}
                      className={cn(
                        'p-3.5 rounded-xl border transition-all flex flex-col sm:flex-row sm:items-center justify-between gap-3',
                        isChecked
                          ? 'border-primary/40 bg-primary/5 shadow-xs'
                          : 'border-border bg-card hover:bg-muted/30'
                      )}
                    >
                      <div className="flex items-start gap-3">
                        <input
                          type="checkbox"
                          checked={isChecked}
                          onChange={() => toggleChannel(ch.id)}
                          className="rounded border-border text-primary focus:ring-primary h-4 w-4 mt-0.5"
                          id={`channel-${ch.id}`}
                        />
                        <div className="space-y-0.5">
                          <label
                            htmlFor={`channel-${ch.id}`}
                            className="text-xs font-bold text-foreground cursor-pointer block"
                          >
                            {ch.title}
                          </label>
                          <p className="text-[11px] text-muted-foreground">
                            {t(ch.descriptionKey, ch.fallbackDesc)}
                          </p>
                          {ch.secretHelp && (
                            <div className="flex items-center gap-1 text-[10px] text-muted-foreground font-mono pt-0.5">
                              <Lock className="h-2.5 w-2.5 text-muted-foreground/70" />
                              <span>{ch.secretHelp} (protected in .env)</span>
                            </div>
                          )}
                        </div>
                      </div>

                      {/* Security Status Badge: NEVER SHOW SECRET VALUES */}
                      <div className="shrink-0 sm:self-center pl-7 sm:pl-0">
                        {configured ? (
                          <Badge
                            variant="success"
                            className="gap-1 text-[10px] font-mono px-2 py-0.5"
                          >
                            <ShieldCheck className="h-3 w-3" />
                            {t('settings.configured', 'Configured')}
                          </Badge>
                        ) : (
                          <Badge
                            variant="outline"
                            className="gap-1 text-[10px] font-mono px-2 py-0.5 text-amber-500 border-amber-500/30 bg-amber-500/10"
                          >
                            <ShieldAlert className="h-3 w-3" />
                            {t('settings.not_configured', 'Not Configured')}
                          </Badge>
                        )}
                      </div>
                    </div>
                  )
                })}
              </div>

              {/* Secrets Protection Notice */}
              <div className="p-3 rounded-lg bg-muted/40 border border-border/80 flex items-start gap-2.5 text-[11px] text-muted-foreground">
                <Lock className="h-3.5 w-3.5 text-primary shrink-0 mt-0.5" />
                <span>
                  {t(
                    'settings.secrets_notice',
                    'Secret credentials (bot tokens, API keys, SMTP passwords) are loaded securely from server environment variables (.env) and are never displayed in this console.'
                  )}
                </span>
              </div>
            </div>

            {/* Test Alert Dispatch with Live Results */}
            <div className="pt-4 border-t border-border space-y-3">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <div>
                  <h4 className="text-xs font-bold text-foreground">
                    {t('settings.test_alert', 'Send Test Emergency Alert')}
                  </h4>
                  <p className="text-[11px] text-muted-foreground">
                    {t(
                      'settings.test_alert_desc',
                      'Dispatches a synthetic test incident to verify network delivery across all enabled channels.'
                    )}
                  </p>
                </div>

                <Button
                  type="button"
                  variant="outline"
                  onClick={() => testMutation.mutate()}
                  disabled={testMutation.isPending || (user ? !isAdmin : false)}
                  className="shrink-0 gap-2 text-xs font-semibold border-cyan-500/40 text-cyan-400 hover:bg-cyan-500/10 hover:border-cyan-400"
                >
                  <Send className="h-3.5 w-3.5" />
                  {testMutation.isPending
                    ? t('settings.testing_alert', 'Transmitting test...')
                    : t('settings.test_alert', 'Send Test Alert')}
                </Button>
              </div>

              {/* Live Test Alert Feedback */}
              {testState.status === 'testing' && (
                <div className="p-3.5 rounded-lg border border-cyan-500/30 bg-cyan-950/20 text-cyan-400 text-xs flex items-center gap-2.5 font-mono animate-pulse">
                  <Radio className="h-4 w-4 animate-spin text-cyan-400" />
                  <span>Transmitting test alert packet across active notification sockets...</span>
                </div>
              )}

              {testState.status === 'success' && testState.results && (
                <div className="p-4 rounded-xl border border-cyan-500/30 bg-cyan-950/20 text-xs space-y-2.5 shadow-sm animate-in fade-in">
                  <div className="flex items-center justify-between border-b border-cyan-500/20 pb-2">
                    <span className="font-bold text-cyan-400 flex items-center gap-2">
                      <Radio className="h-3.5 w-3.5 text-cyan-400 animate-pulse" />
                      {t('settings.live_results', 'Live Dispatch Results')}
                    </span>
                    <span className="text-[10px] font-mono text-muted-foreground">
                      {testState.timestamp}
                    </span>
                  </div>

                  <p className="text-foreground font-mono text-[11px]">{testState.message}</p>

                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 pt-1">
                    {Object.entries(testState.results).map(([channel, success]) => (
                      <div
                        key={channel}
                        className={cn(
                          'flex items-center justify-between p-2 rounded-lg border text-[11px] font-mono',
                          success
                            ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-400'
                            : 'border-amber-500/30 bg-amber-500/10 text-amber-400'
                        )}
                      >
                        <span className="capitalize font-semibold">{channel}</span>
                        <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-background/50">
                          {success ? 'OK / SENT' : 'FAILED / SKIPPED'}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {testState.status === 'error' && (
                <div className="p-3.5 rounded-lg border border-[var(--status-alert)]/30 bg-[var(--status-alert)]/10 text-[var(--status-alert)] text-xs flex items-center gap-2 font-mono">
                  <AlertTriangle className="h-4 w-4 shrink-0" />
                  <span>{testState.message}</span>
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        {/* ========================================================================= */}
        {/* SECTION 2: VISION INFERENCE ENGINE & CONFIDENCE DEFAULTS                  */}
        {/* ========================================================================= */}
        <div className="space-y-6">
          <Card className="border-border shadow-sm">
            <CardHeader className="py-4 px-6 border-b border-border bg-muted/20">
              <div className="flex items-center justify-between">
                <CardTitle className="text-base font-bold flex items-center gap-2">
                  <Cpu className="h-4 w-4 text-cyan-400" />
                  {t('settings.model_settings', 'Vision Inference Engine')}
                </CardTitle>
                <Badge variant="outline" className="text-[10px] font-mono uppercase">
                  {config?.device_info?.device || 'MPS'}
                </Badge>
              </div>
              <CardDescription className="text-xs">
                {t(
                  'settings.model_desc',
                  'YOLO weights checkpoint selection, hardware compute device, and detection thresholds.'
                )}
              </CardDescription>
            </CardHeader>

            <CardContent className="p-6 space-y-5">
              {/* Primary Model Select */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <label className="text-xs font-bold text-foreground block">
                    {t('settings.model_select', 'Primary YOLO Weights Checkpoint')}
                  </label>
                  <span className="text-[11px] text-muted-foreground font-mono">
                    {config?.available_models?.length || 1} available
                  </span>
                </div>

                <Controller
                  control={control}
                  name="model_path"
                  render={({ field }) => (
                    <select
                      value={field.value}
                      onChange={(e) => field.onChange(e.target.value)}
                      className="w-full px-3.5 py-2.5 rounded-lg border border-border bg-background text-xs font-mono text-foreground focus:ring-1 focus:ring-primary outline-none transition-colors"
                    >
                      {config?.available_models && config.available_models.length > 0 ? (
                        config.available_models.map((m: string) => (
                          <option key={m} value={m}>
                            {m} {m === 'best.pt' ? '(Recommended / Production)' : ''}
                          </option>
                        ))
                      ) : (
                        <option value="best.pt">best.pt</option>
                      )}
                    </select>
                  )}
                />
                <p className="text-[11px] text-muted-foreground">
                  Loaded onto hardware accelerator for collision bounding box localization.
                </p>
                {errors.model_path && (
                  <p className="text-[11px] text-[var(--status-alert)] font-medium">
                    {errors.model_path.message}
                  </p>
                )}
              </div>

              {/* Confidence Threshold Slider + Number Input */}
              <div className="space-y-2.5 pt-2 border-t border-border">
                <div className="flex items-center justify-between">
                  <label className="text-xs font-bold text-foreground">
                    {t('settings.confidence_threshold', 'Detection Confidence Threshold')}
                  </label>
                  <div className="flex items-center gap-1.5">
                    <Badge variant="secondary" className="font-mono text-cyan-400 text-xs px-2 py-0.5">
                      {(watchedConfidence * 100).toFixed(0)}% ({watchedConfidence.toFixed(2)})
                    </Badge>
                  </div>
                </div>

                <Controller
                  control={control}
                  name="confidence_threshold"
                  render={({ field }) => (
                    <div className="space-y-2">
                      <div className="flex items-center gap-3">
                        <input
                          type="range"
                          min="0.05"
                          max="1.0"
                          step="0.01"
                          value={field.value}
                          onChange={(e) => field.onChange(parseFloat(e.target.value))}
                          className="w-full h-1.5 bg-muted rounded-lg appearance-none cursor-pointer accent-primary"
                        />
                        <input
                          type="number"
                          min="0.05"
                          max="1.0"
                          step="0.01"
                          value={field.value}
                          onChange={(e) => field.onChange(parseFloat(e.target.value) || 0.05)}
                          className="w-20 px-2 py-1 text-center font-mono text-xs rounded border border-border bg-background"
                        />
                      </div>
                      <div className="flex justify-between text-[10px] text-muted-foreground font-mono">
                        <span>5% (High Recall)</span>
                        <span>72% (Default)</span>
                        <span>100% (High Precision)</span>
                      </div>
                    </div>
                  )}
                />

                <p className="text-[11px] text-muted-foreground">
                  {t(
                    'settings.confidence_helper',
                    'Minimum detection probability required for an object box to be considered in collision tracking (default 0.72).'
                  )}
                </p>
                {errors.confidence_threshold && (
                  <p className="text-[11px] text-[var(--status-alert)] font-medium">
                    {errors.confidence_threshold.message}
                  </p>
                )}
              </div>

              {/* Hardware Device Info Details */}
              <div className="p-3.5 rounded-xl border border-border bg-muted/20 space-y-2 text-xs font-mono">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Accelerator:</span>
                  <span className="font-bold text-foreground">
                    {config?.device_info?.device_name || 'Apple Silicon GPU (MPS)'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Device Identifier:</span>
                  <span className="font-bold text-cyan-400 uppercase">
                    {config?.device_info?.device || 'mps'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Secondary Injury Model:</span>
                  <span className="text-foreground">
                    {config?.defaults?.injury_model_available
                      ? 'Available (injury.pt)'
                      : 'Disabled (injury.pt not present)'}
                  </span>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* ========================================================================= */}
          {/* SECTION 3: TEMPORAL STATE MACHINE: COOLDOWN & DEBOUNCE                    */}
          {/* ========================================================================= */}
          <Card className="border-border shadow-sm">
            <CardHeader className="py-4 px-6 border-b border-border bg-muted/20">
              <div className="flex items-center justify-between">
                <CardTitle className="text-base font-bold flex items-center gap-2">
                  <Clock className="h-4 w-4 text-primary" />
                  {t('settings.cooldown_settings', 'Temporal State Machine & Debounce')}
                </CardTitle>
                <Badge variant="outline" className="text-[10px] font-mono">
                  State Filters
                </Badge>
              </div>
              <CardDescription className="text-xs">
                {t(
                  'settings.cooldown_desc',
                  'Tune consecutive frame thresholds, collision cooldown intervals, and event termination delays.'
                )}
              </CardDescription>
            </CardHeader>

            <CardContent className="p-6 space-y-5">
              {/* Debounce Threshold: Consecutive Frames */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <label className="text-xs font-bold text-foreground">
                    {t('settings.debounce_frames', 'Debounce Threshold (Consecutive Frames)')}
                  </label>
                  <Badge variant="secondary" className="font-mono text-primary text-xs px-2 py-0.5">
                    {watchedDebounce} frames (~{(watchedDebounce / 25).toFixed(2)}s @ 25fps)
                  </Badge>
                </div>

                <Controller
                  control={control}
                  name="debounce_frames"
                  render={({ field }) => (
                    <div className="space-y-2">
                      <div className="flex items-center gap-3">
                        <input
                          type="range"
                          min="1"
                          max="30"
                          step="1"
                          value={field.value}
                          onChange={(e) => field.onChange(parseInt(e.target.value, 10))}
                          className="w-full h-1.5 bg-muted rounded-lg appearance-none cursor-pointer accent-primary"
                        />
                        <input
                          type="number"
                          min="1"
                          max="30"
                          step="1"
                          value={field.value}
                          onChange={(e) => field.onChange(parseInt(e.target.value, 10) || 1)}
                          className="w-20 px-2 py-1 text-center font-mono text-xs rounded border border-border bg-background"
                        />
                      </div>
                      <div className="flex justify-between text-[10px] text-muted-foreground font-mono">
                        <span>1 frame (Instant)</span>
                        <span>5 frames (Standard)</span>
                        <span>30 frames (High Inertia)</span>
                      </div>
                    </div>
                  )}
                />

                <p className="text-[11px] text-muted-foreground">
                  {t(
                    'settings.debounce_helper',
                    'Number of consecutive accident detection frames required to confirm an incident (default 5). Eliminates single-frame false alarms and noise.'
                  )}
                </p>
                {errors.debounce_frames && (
                  <p className="text-[11px] text-[var(--status-alert)] font-medium">
                    {errors.debounce_frames.message}
                  </p>
                )}
              </div>

              {/* Cooldown Period (Seconds) */}
              <div className="space-y-2 pt-2 border-t border-border">
                <div className="flex items-center justify-between">
                  <label className="text-xs font-bold text-foreground">
                    {t('settings.cooldown_seconds', 'Incident Cooldown (Seconds)')}
                  </label>
                  <Badge variant="secondary" className="font-mono text-cyan-400 text-xs px-2 py-0.5">
                    {watchedCooldown.toFixed(1)}s
                  </Badge>
                </div>

                <Controller
                  control={control}
                  name="cooldown_seconds"
                  render={({ field }) => (
                    <div className="space-y-2">
                      <div className="flex items-center gap-3">
                        <input
                          type="range"
                          min="1.0"
                          max="60.0"
                          step="0.5"
                          value={field.value}
                          onChange={(e) => field.onChange(parseFloat(e.target.value))}
                          className="w-full h-1.5 bg-muted rounded-lg appearance-none cursor-pointer accent-primary"
                        />
                        <input
                          type="number"
                          min="1.0"
                          max="60.0"
                          step="0.5"
                          value={field.value}
                          onChange={(e) => field.onChange(parseFloat(e.target.value) || 1.0)}
                          className="w-20 px-2 py-1 text-center font-mono text-xs rounded border border-border bg-background"
                        />
                      </div>
                      <div className="flex justify-between text-[10px] text-muted-foreground font-mono">
                        <span>1.0s (Minimal)</span>
                        <span>10.0s (Default)</span>
                        <span>60.0s (Extended)</span>
                      </div>
                    </div>
                  )}
                />

                <p className="text-[11px] text-muted-foreground">
                  {t(
                    'settings.cooldown_helper',
                    'Minimum quiet period after an accident concludes before a new incident can be triggered. Prevents duplicate alerts for the same collision event.'
                  )}
                </p>
                {errors.cooldown_seconds && (
                  <p className="text-[11px] text-[var(--status-alert)] font-medium">
                    {errors.cooldown_seconds.message}
                  </p>
                )}
              </div>

              {/* Inactivity Timeout (Seconds) */}
              <div className="space-y-2 pt-2 border-t border-border">
                <div className="flex items-center justify-between">
                  <label className="text-xs font-bold text-foreground">
                    {t('settings.timeout_seconds', 'Inactivity Timeout (Seconds)')}
                  </label>
                  <Badge variant="secondary" className="font-mono text-primary text-xs px-2 py-0.5">
                    {watchedTimeout.toFixed(1)}s
                  </Badge>
                </div>

                <Controller
                  control={control}
                  name="timeout_seconds"
                  render={({ field }) => (
                    <div className="space-y-2">
                      <div className="flex items-center gap-3">
                        <input
                          type="range"
                          min="1.0"
                          max="15.0"
                          step="0.5"
                          value={field.value}
                          onChange={(e) => field.onChange(parseFloat(e.target.value))}
                          className="w-full h-1.5 bg-muted rounded-lg appearance-none cursor-pointer accent-primary"
                        />
                        <input
                          type="number"
                          min="1.0"
                          max="15.0"
                          step="0.5"
                          value={field.value}
                          onChange={(e) => field.onChange(parseFloat(e.target.value) || 1.0)}
                          className="w-20 px-2 py-1 text-center font-mono text-xs rounded border border-border bg-background"
                        />
                      </div>
                      <div className="flex justify-between text-[10px] text-muted-foreground font-mono">
                        <span>1.0s (Short)</span>
                        <span>3.0s (Default)</span>
                        <span>15.0s (Long)</span>
                      </div>
                    </div>
                  )}
                />

                <p className="text-[11px] text-muted-foreground">
                  {t(
                    'settings.timeout_helper',
                    'Duration without accident detections before an active event closes and triggers dispatch report generation.'
                  )}
                </p>
                {errors.timeout_seconds && (
                  <p className="text-[11px] text-[var(--status-alert)] font-medium">
                    {errors.timeout_seconds.message}
                  </p>
                )}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* ========================================================================= */}
      {/* SECTION: CCTV SURVEILLANCE CAMERAS & GEOLOCATION MAPPING                  */}
      {/* ========================================================================= */}
      <Card className="border-border shadow-sm">
        <CardHeader className="py-4 px-6 border-b border-border bg-muted/20">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div>
              <CardTitle className="text-base font-bold flex items-center gap-2">
                <CameraIcon className="h-4 w-4 text-primary" />
                {t('map.manage_cameras', 'CCTV Camera Geolocation & Network')}
              </CardTitle>
              <CardDescription className="text-xs mt-1">
                {t(
                  'map.location_picker_title',
                  'Click on the interactive map to place and position CCTV cameras for the spatial incident replay map.'
                )}
              </CardDescription>
            </div>
            <Button
              type="button"
              variant="outline"
              size="sm"
              disabled={user ? !isAdmin : false}
              onClick={() => setIsAddCameraModalOpen(true)}
              className="gap-1.5 text-xs font-semibold shrink-0"
            >
              <Plus className="h-3.5 w-3.5" />
              <span>Register Camera</span>
            </Button>
          </div>
        </CardHeader>

        <CardContent className="p-6 space-y-4">
          {cameraFeedbackMsg && (
            <div className="p-3 rounded-lg bg-[var(--status-ok)]/15 border border-[var(--status-ok)]/30 text-[var(--status-ok)] text-xs flex items-center gap-2 font-medium">
              <CheckCircle2 className="h-4 w-4 shrink-0" />
              <span>{cameraFeedbackMsg}</span>
            </div>
          )}

          {cameras.length === 0 ? (
            <div className="text-center py-8 border border-dashed border-border rounded-xl">
              <CameraIcon className="h-8 w-8 mx-auto text-muted-foreground opacity-50 mb-2" />
              <p className="text-xs text-muted-foreground">
                No cameras registered yet. Click &quot;Register Camera&quot; to add a camera and set its GPS location.
              </p>
            </div>
          ) : (
            <div className="space-y-3">
              {cameras.map((cam) => {
                const isEditingThis = editingCameraId === cam.id
                const hasCoordinates =
                  typeof cam.location_lat === 'number' && typeof cam.location_lng === 'number'

                return (
                  <div
                    key={cam.id}
                    className="p-4 rounded-xl border border-border bg-card/60 space-y-3 transition-colors"
                  >
                    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2.5">
                      <div className="space-y-1">
                        <div className="flex items-center gap-2">
                          <span className="font-bold text-sm text-foreground">{cam.name}</span>
                          <Badge
                            variant={cam.health?.connected ? 'success' : 'outline'}
                            className="text-[10px] font-mono px-2 py-0.2"
                          >
                            {cam.health?.connected ? 'ONLINE' : 'OFFLINE'}
                          </Badge>
                          <Badge variant="secondary" className="text-[10px] font-mono uppercase px-1.5 py-0">
                            {cam.source_type}
                          </Badge>
                        </div>
                        <div className="flex items-center gap-3 text-[11px] text-muted-foreground font-mono">
                          <span>ID: {cam.id}</span>
                          <span className="text-border">|</span>
                          {hasCoordinates ? (
                            <span className="text-cyan-400 flex items-center gap-1">
                              <MapPin className="h-3 w-3" />
                              {cam.location_lat?.toFixed(5)}, {cam.location_lng?.toFixed(5)}
                            </span>
                          ) : (
                            <span className="text-amber-400">
                              {t('map.no_location_set', 'No GPS coordinates set. Click map to place marker.')}
                            </span>
                          )}
                        </div>
                      </div>

                      <div className="flex items-center gap-2 self-start sm:self-center">
                        <Button
                          type="button"
                          variant={isEditingThis ? 'secondary' : 'outline'}
                          size="sm"
                          onClick={() => {
                            if (isEditingThis) {
                              setEditingCameraId(null)
                              setEditingCoords(null)
                            } else {
                              setEditingCameraId(cam.id)
                              setEditingCoords(
                                hasCoordinates
                                  ? { lat: cam.location_lat!, lng: cam.location_lng! }
                                  : null
                              )
                            }
                          }}
                          className="h-7 px-2.5 text-xs gap-1.5"
                        >
                          <MapPin className="h-3 w-3 text-primary" />
                          <span>
                            {isEditingThis
                              ? 'Close Map'
                              : hasCoordinates
                              ? t('map.edit_location', 'Edit Location')
                              : t('map.save_location', 'Set Geolocation')}
                          </span>
                        </Button>

                        <Button
                          type="button"
                          variant="ghost"
                          size="sm"
                          onClick={async () => {
                            if (window.confirm(`Delete camera '${cam.name}'?`)) {
                              await deleteCameraMutation.mutateAsync(cam.id)
                            }
                          }}
                          className="h-7 w-7 p-0 text-muted-foreground hover:text-rose-400"
                          title="Delete Camera"
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </Button>
                      </div>
                    </div>

                    {/* Collapsible Interactive Click-to-Place Map Picker */}
                    {isEditingThis && (
                      <div className="pt-3 border-t border-border space-y-3 animate-in fade-in duration-200">
                        <div className="flex items-center justify-between">
                          <p className="text-xs text-muted-foreground">
                            {t('map.click_to_place', 'Click anywhere on the map to position the camera marker.')}
                          </p>
                          <Button
                            type="button"
                            size="sm"
                            disabled={!editingCoords || updateCameraCoordsMutation.isPending}
                            onClick={() => {
                              updateCameraCoordsMutation.mutate({
                                id: cam.id,
                                lat: editingCoords?.lat ?? null,
                                lng: editingCoords?.lng ?? null,
                              })
                            }}
                            className="h-7 px-3 text-xs font-bold gap-1 shadow-xs"
                          >
                            <Save className="h-3 w-3" />
                            <span>
                              {updateCameraCoordsMutation.isPending
                                ? 'Saving...'
                                : t('map.save_location', 'Save Location')}
                            </span>
                          </Button>
                        </div>

                        <CameraLocationPicker
                          lat={editingCoords?.lat ?? cam.location_lat}
                          lng={editingCoords?.lng ?? cam.location_lng}
                          onChange={(coords) => setEditingCoords(coords)}
                          height="h-64"
                        />
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          )}
        </CardContent>
      </Card>

      {/* ========================================================================= */}
      {/* SECTION 4: LANGUAGE AND THEME PREFERENCES                                 */}
      {/* ========================================================================= */}
      <Card className="border-border shadow-sm">
        <CardHeader className="py-4 px-6 border-b border-border bg-muted/20">
          <CardTitle className="text-base font-bold flex items-center gap-2">
            <Globe className="h-4 w-4 text-cyan-400" />
            {t('settings.preferences_header', 'Interface & Localization Preferences')}
          </CardTitle>
          <CardDescription className="text-xs">
            {t(
              'settings.preferences_desc',
              'Configure display language, bidirectional text layout (LTR / RTL), and color theme.'
            )}
          </CardDescription>
        </CardHeader>

        <CardContent className="p-6">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
            {/* Language Preference */}
            <div className="space-y-3">
              <label className="text-xs font-bold text-foreground block">
                {t('settings.language_select', 'Display Language')}
              </label>
              <div className="p-3.5 rounded-xl border border-primary bg-primary/10 shadow-xs flex items-center justify-between">
                <div>
                  <div className="text-xs font-bold text-foreground">English</div>
                  <div className="text-[10px] text-muted-foreground">System Language</div>
                </div>
                <Check className="h-4 w-4 text-primary" />
              </div>
              <p className="text-[11px] text-muted-foreground">
                System language is locked to English.
              </p>
            </div>

            {/* Theme Preference */}
            <div className="space-y-3">
              <label className="text-xs font-bold text-foreground block">
                {t('settings.theme_select', 'Visual Theme')}
              </label>

              <Controller
                control={control}
                name="theme"
                render={({ field }) => (
                  <div className="grid grid-cols-2 gap-3">
                    <button
                      type="button"
                      onClick={() => {
                        field.onChange('dark')
                        setTheme('dark')
                      }}
                      className={cn(
                        'p-3.5 rounded-xl border transition-all cursor-pointer flex items-center justify-between',
                        field.value === 'dark'
                          ? 'border-cyan-500 bg-cyan-950/20 text-cyan-400 shadow-xs'
                          : 'border-border bg-card hover:bg-muted/40'
                      )}
                    >
                      <div className="flex items-center gap-2">
                        <Moon className="h-4 w-4 text-cyan-400" />
                        <div className="text-left">
                          <div className="text-xs font-bold text-foreground">Dark</div>
                          <div className="text-[10px] text-muted-foreground">Cyber Cyan</div>
                        </div>
                      </div>
                      {field.value === 'dark' && <Check className="h-4 w-4 text-cyan-400" />}
                    </button>

                    <button
                      type="button"
                      onClick={() => {
                        field.onChange('light')
                        setTheme('light')
                      }}
                      className={cn(
                        'p-3.5 rounded-xl border transition-all cursor-pointer flex items-center justify-between',
                        field.value === 'light'
                          ? 'border-primary bg-primary/10 shadow-xs'
                          : 'border-border bg-card hover:bg-muted/40'
                      )}
                    >
                      <div className="flex items-center gap-2">
                        <Sun className="h-4 w-4 text-amber-500" />
                        <div className="text-left">
                          <div className="text-xs font-bold text-foreground">Light</div>
                          <div className="text-[10px] text-muted-foreground">Daylight Mode</div>
                        </div>
                      </div>
                      {field.value === 'light' && <Check className="h-4 w-4 text-primary" />}
                    </button>
                  </div>
                )}
              />
              <p className="text-[11px] text-muted-foreground">
                Controls canvas background, contrast ratios, and color palette tokens across all pages.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* ========================================================================= */}
      {/* SECTION 5: PRIVACY PROTECTION & DATA RETENTION                            */}
      {/* ========================================================================= */}
      <Card className="border-border shadow-sm">
        <CardHeader className="py-4 px-6 border-b border-border bg-muted/20">
          <div className="flex items-center justify-between">
            <CardTitle className="text-base font-bold flex items-center gap-2">
              <ShieldCheck className="h-4 w-4 text-emerald-400" />
              <span>Privacy Protection & CCTV Retention Policy</span>
            </CardTitle>
            <Badge variant="outline" className="text-[10px] font-mono border-emerald-500/40 text-emerald-400 bg-emerald-500/10">
              Surveillance Compliant
            </Badge>
          </div>
          <CardDescription className="text-xs">
            Automated face & license plate blurring, access-controlled vault for unblurred footage, and lifecycle cleanup.
          </CardDescription>
        </CardHeader>

        <CardContent className="p-6 space-y-6">
          {retentionFeedback && (
            <div className="p-3.5 rounded-lg bg-[var(--status-ok)]/15 border border-[var(--status-ok)]/30 text-[var(--status-ok)] text-xs flex items-center gap-2 font-medium">
              <CheckCircle2 className="h-4 w-4 shrink-0" />
              <span>{retentionFeedback}</span>
            </div>
          )}

          {cleanupResult && (
            <div className="p-3.5 rounded-lg bg-cyan-500/15 border border-cyan-500/30 text-cyan-300 text-xs flex items-center gap-2 font-mono">
              <HardDrive className="h-4 w-4 shrink-0 text-cyan-400" />
              <span>{cleanupResult}</span>
            </div>
          )}

          {/* Privacy Redaction Toggles */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 p-4 rounded-xl border border-border bg-card/60">
            {/* Automated Blur Toggle */}
            <div className="space-y-1.5">
              <label className="flex items-center gap-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={privacyBlur}
                  disabled={!isAdmin}
                  onChange={(e) => setPrivacyBlur(e.target.checked)}
                  className="rounded border-border text-primary focus:ring-primary h-4 w-4"
                />
                <span className="text-xs font-bold text-foreground flex items-center gap-1.5">
                  <EyeOff className="h-3.5 w-3.5 text-emerald-400" />
                  Blur Faces & License Plates
                </span>
              </label>
              <p className="text-[11px] text-muted-foreground">
                Applies edge-aware Gaussian blur to all human faces and vehicle license plates on stored keyframes, clips, and PDF reports.
              </p>
            </div>

            {/* Unredacted Original Vault Toggle */}
            <div className="space-y-1.5">
              <label className="flex items-center gap-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={preserveOriginal}
                  disabled={!isAdmin}
                  onChange={(e) => setPreserveOriginal(e.target.checked)}
                  className="rounded border-border text-primary focus:ring-primary h-4 w-4"
                />
                <span className="text-xs font-bold text-foreground flex items-center gap-1.5">
                  <Lock className="h-3.5 w-3.5 text-amber-400" />
                  Vault Unblurred Originals
                </span>
              </label>
              <p className="text-[11px] text-muted-foreground">
                Saves unmodified originals into the access-controlled vault (<code className="text-[10px] text-primary">media/unredacted/</code>). Accessible strictly to Administrators for authorized legal/court reviews.
              </p>
            </div>
          </div>

          {/* Retention Thresholds */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="space-y-1.5">
              <label className="text-xs font-bold text-foreground flex items-center gap-1.5">
                <Clock className="h-3.5 w-3.5 text-primary" />
                Incident Record Retention
              </label>
              <div className="flex items-center gap-2">
                <input
                  type="number"
                  min={1}
                  max={3650}
                  disabled={!isAdmin}
                  value={incidentRetentionDays}
                  onChange={(e) => setIncidentRetentionDays(Number(e.target.value))}
                  className="w-full px-3 py-1.5 text-xs rounded-lg border border-border bg-background text-foreground font-mono"
                />
                <span className="text-xs text-muted-foreground font-mono">days</span>
              </div>
              <p className="text-[10px] text-muted-foreground">
                Automatic deletion of incident records, review notes, and media.
              </p>
            </div>

            <div className="space-y-1.5">
              <label className="text-xs font-bold text-foreground flex items-center gap-1.5">
                <FolderArchive className="h-3.5 w-3.5 text-primary" />
                Clip Age Retention
              </label>
              <div className="flex items-center gap-2">
                <input
                  type="number"
                  min={1}
                  max={365}
                  disabled={!isAdmin}
                  value={clipRetentionDays}
                  onChange={(e) => setClipRetentionDays(Number(e.target.value))}
                  className="w-full px-3 py-1.5 text-xs rounded-lg border border-border bg-background text-foreground font-mono"
                />
                <span className="text-xs text-muted-foreground font-mono">days</span>
              </div>
              <p className="text-[10px] text-muted-foreground">
                Maximum age before standalone MP4 video clips are purged.
              </p>
            </div>

            <div className="space-y-1.5">
              <label className="text-xs font-bold text-foreground flex items-center gap-1.5">
                <HardDrive className="h-3.5 w-3.5 text-primary" />
                Max Disk Usage Cap
              </label>
              <div className="flex items-center gap-2">
                <input
                  type="number"
                  min={100}
                  max={1000000}
                  step={100}
                  disabled={!isAdmin}
                  value={maxDiskMb}
                  onChange={(e) => setMaxDiskMb(Number(e.target.value))}
                  className="w-full px-3 py-1.5 text-xs rounded-lg border border-border bg-background text-foreground font-mono"
                />
                <span className="text-xs text-muted-foreground font-mono">MB</span>
              </div>
              <p className="text-[10px] text-muted-foreground">
                Triggers LRU deletion if stored clips exceed disk limit.
              </p>
            </div>
          </div>

          {/* Storage Metrics & Manual Purge */}
          <div className="p-4 rounded-xl border border-border bg-muted/20 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div className="space-y-1 text-xs">
              <div className="font-semibold text-foreground flex items-center gap-2">
                <HardDrive className="h-4 w-4 text-cyan-400" />
                <span>Disk Storage:</span>
                <span className="font-mono text-cyan-300">
                  {retentionData ? `${retentionData.current_disk_usage_mb.toFixed(1)} MB` : 'Loading...'}
                </span>
                <span className="text-muted-foreground">/</span>
                <span className="font-mono text-muted-foreground">{maxDiskMb} MB cap</span>
              </div>
              <div className="text-[11px] text-muted-foreground font-mono">
                Stored Clips: {retentionData?.total_clips_count ?? 0} files | Vault Dir: {retentionData?.clips_dir || 'media/clips'}
              </div>
            </div>

            <div className="flex items-center gap-2">
              <Button
                type="button"
                variant="outline"
                size="sm"
                disabled={isCleaningUp || !isAdmin}
                onClick={handleRunCleanup}
                className="gap-1.5 text-xs border-cyan-500/30 text-cyan-400 hover:bg-cyan-950/20"
              >
                <Trash2 className="h-3.5 w-3.5 text-cyan-400" />
                <span>{isCleaningUp ? 'Purging...' : 'Execute Retention Cleanup Now'}</span>
              </Button>

              <Button
                type="button"
                size="sm"
                disabled={updateRetentionMutation.isPending || !isAdmin}
                onClick={handleSaveRetention}
                className="gap-1.5 text-xs font-bold bg-emerald-600 hover:bg-emerald-500 text-white"
              >
                <Save className="h-3.5 w-3.5" />
                <span>Save Retention Policy</span>
              </Button>
            </div>
          </div>

          {/* Legal Compliance Disclaimer */}
          <div className="p-3.5 rounded-lg bg-blue-950/30 border border-blue-500/30 text-blue-300 text-xs flex items-start gap-2.5">
            <Info className="h-4 w-4 text-blue-400 shrink-0 mt-0.5" />
            <div className="space-y-1">
              <span className="font-semibold text-blue-200">Legal Compliance & Surveillance Privacy Notice:</span>
              <p className="text-[11px] text-blue-300/90 leading-relaxed">
                CCTV video streams inherently record identifiable individuals and motor vehicles. Real-world deployments must strictly comply with applicable municipal, national, and international privacy laws (including GDPR, CCPA, and statutory surveillance mandates), as well as the camera facility owner's internal data governance policies.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Add Camera Modal with Map Location Picker */}
      <AddCameraModal
        isOpen={isAddCameraModalOpen}
        onClose={() => setIsAddCameraModalOpen(false)}
        onAddCamera={async (payload) => {
          await createCameraMutation.mutateAsync(payload)
        }}
      />
    </form>
  )
}

export default SettingsPage
